import os
import pickle
import pandas as pd
import numpy as np

RAW_ROOT = 'data/wesad_raw'
OUT_ROOT = 'data/wesad'


def main():
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

        # Convert to DataFrame
        if isinstance(obj, pd.DataFrame):
            df_all = obj
        elif isinstance(obj, dict):
            # find first array-like value
            arr = None
            for v in obj.values():
                if isinstance(v, (np.ndarray, list)):
                    arr = np.array(v)
                    break
            if arr is None:
                raise ValueError(f"No array in pickle for {subj}")
            # ensure shape [samples, channels]
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                arr = arr.T
            df_all = pd.DataFrame(arr)
        else:
            arr = np.array(obj)
            df_all = pd.DataFrame(arr)

        if isinstance(obj, dict) and 'EDA' in obj:
            eda_4hz = np.asarray(obj['EDA'])

        # 2.   Wrist CSV layout    (DataFrame → column named 'EDA')
        elif isinstance(df_all, pd.DataFrame) and any(str(c).lower() == 'eda' for c in df_all.columns):
            eda_4hz = df_all[[c for c in df_all.columns if str(c).lower() == 'eda'][0]].values.flatten()
        # 3.   Otherwise we must have loaded the 700 Hz chest trace; down-sample 175:1
        else:
            # ensure 1-D array
            flat = df_all.values.flatten()
            eda_4hz = flat[::175]          # 700 Hz / 175 ≈ 4 Hz

        if isinstance(obj, dict) and 'label' in obj:
            lbl_raw = np.asarray(obj['label']).astype(int)
        elif isinstance(df_all, pd.DataFrame) and 'label' in df_all.columns:
            lbl_raw = df_all['label'].values.astype(int)
        else:
            raise ValueError("‘label’ array not found in pickle")

        lbl_4hz = lbl_raw[::175] if len(lbl_raw) > len(eda_4hz) else lbl_raw
        if len(lbl_4hz) != len(eda_4hz):
            raise ValueError("label/EDA length mismatch after down-sampling")

        # convert to DataFrame with the expected column name
        eda = pd.DataFrame({'eda_signal': eda_4hz, 'label': lbl_4hz})
        eda.loc[eda['label'] == 3, 'label'] = 1
        # Locate the EDA column
        # eda_col = next((c for c in df_all.columns if 'eda' in str(c).lower()), None)

        # if eda_col is None:
        #     # fallback: assume 3rd channel
        #     eda_col = df_all.columns[2] if df_all.shape[1] >= 3 else df_all.columns[0]
        # eda = df_all[[eda_col]].rename(columns={eda_col: 'eda_signal'})

        # Write out as data/wesad/S*/S*_eda.csv
        out_dir = os.path.join(OUT_ROOT, subj)
        os.makedirs(out_dir, exist_ok=True)
        print(f'Writing to {os.path.join(out_dir, f"{subj}_eda.csv")}')
        eda.to_csv(os.path.join(out_dir, f"{subj}_eda.csv"), index=False)

if __name__ == '__main__':
    main()
