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
rm -rf data/wrist            \
       models/wrist                \
       results/wrist
mkdir -p models/wrist results/wrist data/wrist

# ───────────────────────────────
# 2. REAL DATA  →  data/wesad/
# ───────────────────────────────
echo ">> Extracting real EDA CSVs"
${VENV} eda/extract_eda.py wrist

# ───────────────────────────────
# 3. RAW SYNTHETIC  → data/simulated/eda_raw/
# ───────────────────────────────
echo ">> Generating raw synthetic EDA"
# sample rate=4 for wrist
${VENV} eda/simulate_raw_eda.py wrist

# ───────────────────────────────
# 4. FOUR SHIFT BASELINES
# ───────────────────────────────
echo ">> Generating baselines (MSC, MSDC, MSV, MSDV)"

${VENV} eda/baselines.py WESAD "data/wrist/simulated/eda_raw/*.csv" data/wrist/simulated/eda_meanshift_constant/   data/wrist/simulated/cpd_details/cpd_baseline_sim_msc/  1
${VENV} eda/baselines.py WESAD "data/wrist/simulated/eda_raw/*.csv" data/wrist/simulated/eda_meansdshift_constant/ data/wrist/simulated/cpd_details/cpd_baseline_sim_msdc/ 2
${VENV} eda/baselines.py WESAD "data/wrist/simulated/eda_raw/*.csv" data/wrist/simulated/eda_meanshift_varying/    data/wrist/simulated/cpd_details/cpd_baseline_sim_msv/  3
${VENV} eda/baselines.py WESAD "data/wrist/simulated/eda_raw/*.csv" data/wrist/simulated/eda_meansdshift_varying/  data/wrist/simulated/cpd_details/cpd_baseline_sim_msdv/ 4

# ───────────────────────────────
# 5. LEARN CPD MODEL  → models/wesad_cp_model.pkl
# ───────────────────────────────
echo ">> Learning non-stationarity model"
${VENV} eda/learning_nonstationarity.py \
        WESAD \
        data/wrist/wesad \
        data/wrist/eda/cpd_annotations/ \
        5 \
        data/wrist/eda/bkps_counts.csv  \
        models/wrist/wesad_cp_model.pkl

# ───────────────────────────────
# 6. AUGMENT SYNTHETIC  → data/simulated/eda_aug/
# ───────────────────────────────
echo ">> Augmenting synthetic EDA with learned CPDs"
${VENV} eda/simulating.py \
        WESAD \
        data/wrist/simulated/eda_raw/ \
        data/wrist/simulated/eda_aug/ \
        models/wrist/wesad_cp_model.pkl \
        data/wrist/eda/cpd_aug_details/ \
        0.03

# ───────────────────────────────
# 7. CLASSIFICATION  (6 runs)
# ───────────────────────────────
echo ">> Classification on real + 5 synthetic sets"
${VENV} eda/classification.py WESAD         data/wrist/wesad/                              results/wrist/real/
${VENV} eda/classification.py sim_aug_WESAD data/wrist/simulated/eda_aug/                  results/wrist/sim_aug/
${VENV} eda/classification.py sim_WESAD     data/wrist/simulated/eda_meanshift_constant/   results/wrist/sim_msc/
${VENV} eda/classification.py sim_WESAD     data/wrist/simulated/eda_meansdshift_constant/ results/wrist/sim_msdc/
${VENV} eda/classification.py sim_WESAD     data/wrist/simulated/eda_meanshift_varying/    results/wrist/sim_msv/
${VENV} eda/classification.py sim_WESAD     data/wrist/simulated/eda_meansdshift_varying/  results/wrist/sim_msdv/

# ───────────────────────────────
# 8. SUMMARY TABLE
# ───────────────────────────────
echo ">> Building comparison table"
${VENV} compare_results.py "results/wrist"

echo "Pipeline completed.  See results/wrist/summary_f1_by_model.csv"
