import os
import numpy as np
import pandas as pd
import neurokit2 as nk
import sys


# Parameters (match paper)
N_SUBJECTS = 10
BASELINE_SECONDS = 1174
STRESS_SECONDS = 664
SCR_BASELINE_RANGE = (1, 6)
SCR_STRESS_RANGE = (6, 21)

def main():
    sample_type = sys.argv[1]
    if sample_type == 'chest':
        output_dir = "data/chest/simulated/eda_raw"
        sample_rate = 700
    else:
        output_dir = "data/wrist/simulated/eda_raw"
        sample_rate = 4
    os.makedirs(output_dir, exist_ok=True)

    for subj in range(1, N_SUBJECTS + 1):
        # simulate baseline and stress segments
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
        eda = np.concatenate([eda_base, eda_stress])
        label = np.concatenate([
                np.zeros_like(eda_base,  dtype=int),   # 0 = baseline
                np.ones_like(eda_stress, dtype=int)    # 1 = stress
            ])

        df = pd.DataFrame({"eda_signal": eda, "label": label})
        """
        >>> import pandas as pd, glob, numpy as np, json, pathlib, os
        stress_vals = np.concatenate([pd.read_csv(f)['eda_signal']
                                    for f in glob.glob('data/wesad/*/*_eda.csv')
                                    if '_eda.csv' in f])
        >>> real_mean = stress_vals.mean()
        >>> print(real_mean)
        1.3347262513238476
        """
        # REAL_MEAN = 1.3347262513238476
        # df['eda_signal'] *= REAL_MEAN / df['eda_signal'].mean()
        df.to_csv(f"{output_dir}/S{subj}_eda.csv", index=False)


if __name__ == '__main__':
    main()