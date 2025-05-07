# Nonstationarity-Simulation

This repository contains the implementation code for Simulation of Time Series with Nonstationarity. This approach involves learning models of nonstationarity in real data and augmenting simulated data with these properties to generate realistic synthetic datasets.

Adedolapo Aishat Toye, Louis Gomez and Samantha Kleinberg
CHIL 2024.

## Replication

/data/wesad_raw should contain all files
all other directories under data/ can be removed
all files under results/ and models/ can be removed


### Automated

The script `run_full_pipeline.sh` automates all below scripts and generates results. The only requirements are to have XGBoost installed and virtualenv with dependencies resolved.

`run_full_pipeline_chest_extension.sh` is our extension to analyze the chest recorded data.

### Overview

`/data/simulated` contains data generated with neurokit2 through simulate_raw_eda.py.
`/data/simulated/eda_raw` is the raw data that is simulated.
`/data/simulated/eda_meansdshift_constant` is the baseline-constant synthetic data.
`/data/simulated/eda_aug` is the augmented data after learning from real data.

### Environment Setup
```
brew install libomp
```

```
python3.11 -m venv virtualenv
./virtualenv/bin/pip install --upgrade pip
./virtualenv/bin/pip install -r requirements.txt
```


### Data Setup
Extract WESAD dataset to:
```NONSTATIONARITY-SIMULATION/data/wesad_raw/*```

### Prepare CSV's

Extract wrist eda data to `data/wrist/wesad` from WESAD pickle files for 15 subjects.

Columns: `eda_signal, label`

```./virtualenv/bin/python eda/extract_eda.py```

### Simulate EDA Data
Generate simulated raw data and output to `data/wrist/simulated/eda_raw` for 10 subjects.

Columns: `eda_signal, label`

```
./virtualenv/bin/python eda/simulate_raw_eda.py
```

### Generate Baselines
Four baselines which add nonstationarity to data are used to compare reuslts. Run 4 baselines for MeanShift-Constant, MeanSDShift-Constant, MeanShift-Varying, and MeanSDShift-Varying.

Input dir: `data/wrist/simulated/eda_raw/*.csv`

Output dir: `data/wrist/simulated/<type>/`

- Columns: `eda_signal_old, label, PtID, duration, type_of_change, mean_change, std_change, cpd, eda_signal, std_change_new, eda_signal_new_original`

CPD Details output dir: `data/wrist/simulated/cpd_details/<cpd_baseline_sim_<type>/`

- Columns: `CPD_Index,duration,type_of_change,mean_change,std_change`

meanshift constant
```
./virtualenv/bin/python eda/baselines.py \
  WESAD \
  "data/simulated/eda_raw/*.csv" \
  data/simulated/eda_meanshift_constant/ \
  data/eda/cpd_baseline_meanshift_constant/ \
  1
```
meandshift constant
```
./virtualenv/bin/python eda/baselines.py \
  WESAD \
  "data/simulated/eda_raw/*.csv" \
  data/simulated/eda_meansdshift_constant/ \
  data/eda/cpd_baseline_meansdshift_constant/ \
  2
```
meanshift varying
```
./virtualenv/bin/python eda/baselines.py \
  WESAD \
  "data/simulated/eda_raw/*.csv" \
  data/simulated/eda_meanshift_varying/ \
  data/eda/cpd_baseline_meanshift_varying/ \
  3
```
meandshift varying
```
./virtualenv/bin/python eda/baselines.py \
  WESAD \
  "data/simulated/eda_raw/*.csv" \
  data/simulated/eda_meansdshift_varying/ \
  data/eda/cpd_baseline_meansdshift_varying/ \
  4
```

### Learn Nonstationarity from Real Data

Input dir: `data/wrist/wesad`

Outputs cpd annotations to `data/wrist/eda/cpd_annotations/S#.csv`

- columns: `eda_signal, label, cpd, duration_samples, duration, diff, mean_change, std_change, type_of_change`

Outputs X_train features to `data/wrist/eda/cpd_annotations/S#_features.csv`

Outputs classifier_predictions to `models/classifier_predictions/S#.csv`

- columns: `y_actual,y_pred,y_prob`

Outputs final combined csv to `/models/WESAD_classifier_data.csv`

- columns: same as X_train above

Outputs bkps_counts to `data/wrist/eda/bkps_counts.csv`

- columns: `PtID, before_checking, after_checking`
- Shows number of changepoints before and after checking for sig

Outputs pickle of distribution to `models/wrist/wesad_cp_model.pkl`

- columns:
  - mdl: the cpd model
  - `mdl, type_of_change, mean_change, std_change, duration, eda_diff`


Learns from patients specified in `data/WESAD_learning_ids.csv` where each patients data is located at `data/wrist/wesad/S#/S#_eda.csv`.


```
./virtualenv/bin/python eda/learning_nonstationarity.py \
  WESAD \
  data/wesad/ \
  data/eda/cpd_annotations/ \
  5 \
  data/eda/bkps_counts.csv  \
  models/wesad_cp_model.pkl
```

### Augment Simulated EDA Data

Input dir: `data/wrist/simulated/eda_raw/*_eda.csv` and `models/wrist/wesad_cp_model.pkl` for model

Output dir: `data/wrist/simulated/eda_aug/S#.csv`

- columns: `eda_signal_old, label, PtID, duration, type_of_change, mean_change, std_change, cpd, eda_signal, mean_change_new, std_change_new, eda_signal_new_original`

Output cpd details to `data/wrist/eda/cpd_aug_details/`

- columns: `CPD_Index,duration,type_of_change,mean_change,std_change`

thresold: 0.1

```
./virtualenv/bin/python eda/simulating.py \
  WESAD \
  "data/simulated/eda_raw/" \
  data/simulated/eda_aug/ \
  models/wesad_cp_model.pkl \
  data/eda/cpd_aug_details/ \
  0.12
```

### Classification task

Input dir: `data/wesad` or `/data/simulated/<type>/`

Output dir: `results/<type>/`


Real data
```
./virtualenv/bin/python eda/classification.py \
       WESAD \
       data/wesad/ \
       results/real/
```

Augmented synthetic data
```
./virtualenv/bin/python eda/classification.py \
       sim_aug_WESAD \
       data/simulated/eda_aug/  \
       results/sim_aug/
```

MeanShift-Constant
```
./virtualenv/bin/python eda/classification.py \
       sim_WESAD \
       data/simulated/eda_meanshift_constant/ \
       results/sim_msc/
```

Mean+SDShift-Constant
```
./virtualenv/bin/python eda/classification.py \
       sim_WESAD \
       data/simulated/eda_meansdshift_constant/ \
       results/sim_msdc/
```

MeanShift-Varying
```
./virtualenv/bin/python eda/classification.py \
       sim_WESAD \
       data/simulated/eda_meanshift_varying/ \
       results/sim_msv/
```

Mean+SDShift-Varying
```
./virtualenv/bin/python eda/classification.py \
       sim_WESAD \
       data/simulated/eda_meansdshift_varying/ \
       results/sim_msdv/
```

### Comparing against real data & baselines
```./virtualenv/bin/python compare_results.py```