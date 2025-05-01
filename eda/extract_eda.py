import os
import pickle
import pandas as pd
import numpy as np
import sys

RAW_ROOT = 'data/wesad_raw'


def main():
    sample_type = sys.argv[1]
    if sample_type == 'chest':
        output_dir = "data/chest/wesad"
    else:
        output_dir = "data/wrist/wesad"
    os.makedirs(output_dir, exist_ok=True)

    for subj in os.listdir(RAW_ROOT):
        subj_dir = os.path.join(RAW_ROOT, subj)
        if not os.path.isdir(subj_dir):
            continue
        # find the .pkl file
        pkl_files = [f for f in os.listdir(subj_dir) if f.endswith('.pkl')]
        if not pkl_files:
            continue
        pkl_path = os.path.join(subj_dir, pkl_files[0])

        # load the pickled object
        with open(pkl_path, 'rb') as f:
            obj = pickle.load(f, encoding='latin1')

        # Expect a dict with nested wrist EDA under obj['signal']['wrist']['EDA']
        if not isinstance(obj, dict):
            raise ValueError(f"Expected pickle object to be dict, got {type(obj)}")

        try:
            if sample_type != 'chest':
                eda_4hz = np.asarray(obj['signal']['wrist']['EDA']).flatten()
            else:
                eda_4hz = np.asarray(obj['signal']['chest']['EDA']).flatten()
        except Exception as exc:
            raise ValueError(f"Could not find EDA data: {exc}")

        try:
            lbl_raw = np.asarray(obj['label']).astype(int).flatten()
        except Exception:
            raise ValueError("Could not find wrist label at obj['label']")

        # labels need to be downsized 700->4hz to match wrist data at 4hz
        if sample_type != 'chest':
            lbl_4hz = lbl_raw[::175]

        if len(eda_4hz) != len(lbl_4hz):
            raise RuntimeError(f"Warning: lengths do not match; EDA={len(eda_4hz)}, label={len(lbl_4hz)}")

        # convert to DataFrame with the expected column name
        eda = pd.DataFrame({'eda_signal': eda_4hz, 'label': lbl_4hz})

        # Map original WESAD labels: 0=neutral, 1=stress, 2 (or 3)=amusement
        # Collapse both neutral and amusement into baseline (0), stress remains 1
        eda['label'] = eda['label'].apply(lambda x: 1 if x == 1 else 0)

        # Write out as data/wesad/S*/S*_eda.csv
        out_dir = os.path.join(output_dir, subj)
        os.makedirs(out_dir, exist_ok=True)
        print(f'Writing to {os.path.join(out_dir, f"{subj}_eda.csv")}')
        eda.to_csv(os.path.join(out_dir, f"{subj}_eda.csv"), index=False)

if __name__ == '__main__':
    main()
