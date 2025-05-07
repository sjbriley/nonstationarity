import os
import pickle
import pandas as pd
import numpy as np
import sys

RAW_ROOT = 'data/wesad_raw'
from logging.config import dictConfig
import json
with open('logging.json', 'r') as read_file:
    contents = json.load(read_file)
dictConfig(contents)
import logging
LOGGER = logging.getLogger()

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
            LOGGER.debug('Not extracting EDA data from %s', subj_dir)
            continue
        LOGGER.debug('Extracting EDA data from %s', subj_dir)
        pkl_path = os.path.join(subj_dir, pkl_files[0])

        # load the pickled object
        with open(pkl_path, 'rb') as f:
            obj = pickle.load(f, encoding='latin1')

        if not isinstance(obj, dict):
            raise ValueError(f"Expected pickle object to be dict, got {type(obj)}")

        try:
            if sample_type != 'chest':
                LOGGER.debug('Loading wrist data from signal:wrist:EDA')
                eda_4hz = np.asarray(obj['signal']['wrist']['EDA']).flatten()
            else:
                LOGGER.debug('Loading chest data from signal:chest:EDA')
                eda_4hz = np.asarray(obj['signal']['chest']['EDA']).flatten()
                LOGGER.debug('Downsizing chest data from 700hz to 12hz')
                eda_4hz = eda_4hz[::58]
        except Exception as exc:
            raise ValueError(f"Could not find EDA data: {exc}")

        try:
            lbl_raw = np.asarray(obj['label']).astype(int).flatten()
        except Exception:
            raise ValueError("Could not find wrist label at obj['label']")

        # labels need to be downsized 700->4hz to match wrist data at 4hz
        if sample_type != 'chest':
            LOGGER.debug('Downsizing labels from 700->4 hz to match 4hz wrist EDA data.')
            lbl_4hz = lbl_raw[::175]
        else:
            LOGGER.debug('Downsizing labels from 700->12 hz to match 4hz chest EDA data.')
            lbl_4hz = lbl_raw[::58]

        if len(eda_4hz) != len(lbl_4hz):
            raise RuntimeError(f"Warning: lengths do not match; EDA={len(eda_4hz)}, label={len(lbl_4hz)}")

        # convert to DataFrame with the expected column name
        eda = pd.DataFrame({'eda_signal': eda_4hz, 'label': lbl_4hz})

        # include amusement (3)
        eda = eda[eda['label'].isin([1, 2, 3])]
        eda['label'] = eda['label'].map({1:0, 3:0, 2:1})

        LOGGER.debug("Final label distribution:")
        LOGGER.debug(eda['label'].value_counts())


        # Write out as data/wesad/S*/S*_eda.csv
        out_dir = os.path.join(output_dir, subj)
        os.makedirs(out_dir, exist_ok=True)
        LOGGER.debug(f'Writing to {os.path.join(out_dir, f"{subj}_eda.csv")}')
        eda.to_csv(os.path.join(out_dir, f"{subj}_eda.csv"), index=False)

if __name__ == '__main__':
    main()
