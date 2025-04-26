# Nonstationarity-Simulation

This repository contains the implementation code for Simulation of Time Series with Nonstationarity. This approach involves learning models of nonstationarity in real data and augmenting simulated data with these properties to generate realistic synthetic datasets.

Adedolapo Aishat Toye, Louis Gomez and Samantha Kleinberg
CHIL 2024.

## Replication

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

### Simulate Base EDA Data
```
./virtualenv/bin/python eda/simulate_raw_eda.py
```

### Generate Baselines
Run 4 baselines for MeanShift-Constant, MeanSDShift-Constant, MeanShift-Varying, and MeanSDShift-Varying

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

### Prepare CSV's
```./virtualenv/bin/python eda/extract_eda.py```

### Learn Nonstationarity from Real Data
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

```
./virtualenv/bin/python eda/simulating.py \
  WESAD \
  "data/simulated/eda_raw/" \
  data/simulated/eda_aug/ \
  models/wesad_cp_model.pkl \
  data/eda/cpd_aug_details/ \
  0.1
```

### Classification task
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
       sim_msc_WESAD \
       data/simulated/eda_meanshift_constant/ \
       results/sim_msc/
```

Mean+SDShift-Constant
```
python eda/classification.py \
       sim_msdc_WESAD \
       data/simulated/eda_meansdshift_constant/ \
       results/sim_msdc/
```

MeanShift-Varying
```
./virtualenv/bin/python eda/classification.py \
       sim_msv_WESAD \
       data/simulated/eda_meanshift_varying/ \
       results/sim_msv/
```

Mean+SDShift-Varying
```
./virtualenv/bin/python eda/classification.py \
       sim_msdv_WESAD \
       data/simulated/eda_meansdshift_varying/ \
       results/sim_msdv/
```

### Comparing against real data & baselines

