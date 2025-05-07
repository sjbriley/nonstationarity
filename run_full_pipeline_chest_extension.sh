#!/usr/bin/env bash
# ==========================================================
#  run_full_pipeline.sh
#  End-to-end rebuild for the WESAD non-stationarity project
# ==========================================================

set -e
VENV="./virtualenv/bin/python"

# ───────────────────────────────
# 1. CLEAN SLATE
# ───────────────────────────────
echo ">> Cleaning previous artifacts"
rm -rf data/chest            \
       models/chest                \
       results/chest
mkdir -p models/chest results/chest data/chest

# ───────────────────────────────
# 2. REAL DATA  →  data/wesad/
# ───────────────────────────────
echo ">> Extracting real EDA CSVs"
${VENV} eda/extract_eda.py chest

# ───────────────────────────────
# 3. RAW SYNTHETIC  → data/simulated/eda_raw/
# ───────────────────────────────
echo ">> Generating raw synthetic EDA"
# sample rate=4 for chest
${VENV} eda/simulate_raw_eda.py chest

# ───────────────────────────────
# 4. FOUR SHIFT BASELINES
# ───────────────────────────────
echo ">> Generating baselines (MSC, MSDC, MSV, MSDV)"

${VENV} eda/baselines.py WESAD "data/chest/simulated/eda_raw/*.csv" data/chest/simulated/eda_meanshift_constant/   data/chest/simulated/cpd_details/cpd_baseline_sim_msc/  1
${VENV} eda/baselines.py WESAD "data/chest/simulated/eda_raw/*.csv" data/chest/simulated/eda_meansdshift_constant/ data/chest/simulated/cpd_details/cpd_baseline_sim_msdc/ 2
${VENV} eda/baselines.py WESAD "data/chest/simulated/eda_raw/*.csv" data/chest/simulated/eda_meanshift_varying/    data/chest/simulated/cpd_details/cpd_baseline_sim_msv/  3
${VENV} eda/baselines.py WESAD "data/chest/simulated/eda_raw/*.csv" data/chest/simulated/eda_meansdshift_varying/  data/chest/simulated/cpd_details/cpd_baseline_sim_msdv/ 4

# ───────────────────────────────
# 5. LEARN CPD MODEL  → models/wesad_cp_model.pkl
# ───────────────────────────────
echo ">> Learning non-stationarity model"
${VENV} eda/learning_nonstationarity.py \
        WESAD \
        data/chest/wesad \
        data/chest/eda/cpd_annotations/ \
        5 \
        data/chest/eda/bkps_counts.csv  \
        models/chest/wesad_cp_model.pkl

# ───────────────────────────────
# 6. AUGMENT SYNTHETIC  → data/simulated/eda_aug/
# ───────────────────────────────
echo ">> Augmenting synthetic EDA with learned CPDs"
${VENV} eda/simulating.py \
        WESAD \
        data/chest/simulated/eda_raw/ \
        data/chest/simulated/eda_aug/ \
        models/chest/wesad_cp_model.pkl \
        data/chest/eda/cpd_aug_details/ \
        0.1

# ───────────────────────────────
# 7. CLASSIFICATION  (6 runs)
# ───────────────────────────────
echo ">> Classification on real + 5 synthetic sets"
${VENV} eda/classification.py WESAD         data/chest/wesad/                              results/chest/real/
${VENV} eda/classification.py sim_aug_WESAD data/chest/simulated/eda_aug/                  results/chest/sim_aug/
${VENV} eda/classification.py sim_WESAD     data/chest/simulated/eda_meanshift_constant/   results/chest/sim_msc/
${VENV} eda/classification.py sim_WESAD     data/chest/simulated/eda_meansdshift_constant/ results/chest/sim_msdc/
${VENV} eda/classification.py sim_WESAD     data/chest/simulated/eda_meanshift_varying/    results/chest/sim_msv/
${VENV} eda/classification.py sim_WESAD     data/chest/simulated/eda_meansdshift_varying/  results/chest/sim_msdv/
${VENV} eda/classification.py sim_WESAD     data/wrist/simulated/eda_raw/                  results/wrist/sim

# ───────────────────────────────
# 8. SUMMARY TABLE
# ───────────────────────────────
echo ">> Building comparison table"
${VENV} compare_results.py "results/chest"

echo "Pipeline completed.  See results/chest/summary_f1_by_model.csv"
