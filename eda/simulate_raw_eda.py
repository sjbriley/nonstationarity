import os
import time
import numpy as np
import pandas as pd
import pandas as pd
import glob
import neurokit2 as nk
import sys

from logging.config import dictConfig
import json
with open('logging.json', 'r') as read_file:
    contents = json.load(read_file)
dictConfig(contents)
import logging
LOGGER = logging.getLogger()

# Parameters (match paper)
N_SUBJECTS = 10
BASELINE_SECONDS = 1174
STRESS_SECONDS = 664
SCR_BASELINE_RANGE = (1, 5)
SCR_STRESS_RANGE = (6, 20)

def main():
    sample_type = sys.argv[1]
    if sample_type == 'chest':
        output_dir = "data/chest/simulated/eda_raw"
        sample_rate = 12
        LOGGER.debug('Generating simulated EDA data for chest with sample rate %s to %s', sample_rate, output_dir)
    else:
        output_dir = "data/wrist/simulated/eda_raw"
        sample_rate = 4
        LOGGER.debug('Generating simulated EDA data for wrist with sample rate %s to %s', sample_rate, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_dfs = []
    path = "data/wrist/wesad/*/*_eda.csv" if 'wrist' == sample_type else "data/chest/wesad/*/*_eda.csv"
    LOGGER.debug(f'Pulling mean & std from real data at {path}')
    for fpath in glob.glob(path):
        df = pd.read_csv(fpath)
        all_dfs.append(df)
    real_full = pd.concat(all_dfs, ignore_index=True)
    real_base   = real_full.loc[real_full["label"] == 0, "eda_signal"]
    real_stress = real_full.loc[real_full["label"] == 1, "eda_signal"]

    MEAN_BASE, STD_BASE = real_base.mean(), real_base.std()
    MEAN_STRESS, STD_STRESS = real_stress.mean(), real_stress.std()
    LOGGER.debug("Real EDA baseline stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                 real_base.mean(), real_base.std(), real_base.min(), real_base.max())
    LOGGER.debug("Real EDA stress   stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                 real_stress.mean(), real_stress.std(), real_stress.min(), real_stress.max())


    for subj in range(1, N_SUBJECTS + 1):
        # simulate baseline and stress segments
        seed = int(time.time() * 1000) % (2**32)
        np.random.seed(seed)
        eda_base = nk.eda_simulate(
            duration=BASELINE_SECONDS,
            sampling_rate=sample_rate,
            scr_number=np.random.randint(*SCR_BASELINE_RANGE)
        )
        eda_stress = nk.eda_simulate(
            duration=STRESS_SECONDS,
            sampling_rate=sample_rate,
            scr_number=np.random.randint(*SCR_STRESS_RANGE)
        )

        eda_base = (eda_base - eda_base.mean()) / eda_base.std() * STD_BASE + MEAN_BASE
        eda_stress = (eda_stress - eda_stress.mean()) / eda_stress.std() * STD_STRESS + MEAN_STRESS
        eda = np.clip(np.concatenate([eda_base, eda_stress]), a_min=0, a_max=None)
        eda = np.concatenate([
            np.full_like(eda_base, MEAN_BASE),
            np.full_like(eda_stress, MEAN_STRESS)
        ])
        label = np.concatenate([
            np.full_like(eda_base, 0, dtype=int),   # 0 = baseline
            np.full_like(eda_stress, 1, dtype=int)    # 1 = stress
        ])

        df = pd.DataFrame({"eda_signal": eda, "label": label})
        # Log simulated EDA summary statistics
        LOGGER.debug("Sim S%s EDA stats: mean=%.3f, std=%.3f, min=%.3f, max=%.3f",
                     subj, df['eda_signal'].mean(), df['eda_signal'].std(),
                     df['eda_signal'].min(), df['eda_signal'].max())

        LOGGER.debug(f"Final simulated label distribution for {subj}:")
        LOGGER.debug(df['label'].value_counts())

        LOGGER.debug('Writing simulated data to %s', f"{output_dir}/S{subj}_eda.csv")
        df.to_csv(f"{output_dir}/S{subj}_eda.csv", index=False)


if __name__ == '__main__':
    main()