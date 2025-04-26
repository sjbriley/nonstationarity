#!/usr/bin/env bash
# ==========================================================
#  run_full_pipeline.sh
#  End-to-end rebuild for the WESAD non-stationarity project
# ==========================================================

set -e                      # stop on first error
VENV="./virtualenv/bin/python"

# ───────────────────────────────
# CONFIGURABLE CONSTANTS
# ───────────────────────────────
REAL_MEAN=1.3347262513238476           # <- put the mean stress EDA you measured
WIN_SIZE=10                  # seconds per window (classification.py)

# ───────────────────────────────
# 1. CLEAN SLATE
# ───────────────────────────────
echo ">> Cleaning previous artifacts"
rm -rf data/wesad            \
       data/simulated        \
       data/eda              \
       models                \
       results
mkdir  -p models results

# ───────────────────────────────
# 2. REAL DATA  →  data/wesad/
# ───────────────────────────────
echo ">> Extracting real EDA CSVs"
${VENV} eda/extract_eda.py           # relabel amusement inside the script

# ───────────────────────────────
# 3. RAW SYNTHETIC  → data/simulated/eda_raw/
# ───────────────────────────────
echo ">> Generating raw synthetic EDA"
SED="sed -i ''"                      # macOS in-place sed
${SED} "s/REAL_MEAN *=.*/REAL_MEAN = ${REAL_MEAN}/" eda/simulate_raw_eda.py
${VENV} eda/simulate_raw_eda.py

# ───────────────────────────────
# 4. FOUR SHIFT BASELINES
# ───────────────────────────────
echo ">> Generating baselines (MSC, MSDC, MSV, MSDV)"
${SED} "s/REAL_MEAN *=.*/REAL_MEAN = ${REAL_MEAN}/" eda/baselines.py

for mode in 1 2 3 4; do
  case $mode in
    1) out=data/simulated/eda_meanshift_constant/  ; tag=sim_msc ;;
    2) out=data/simulated/eda_meansdshift_constant/; tag=sim_msdc;;
    3) out=data/simulated/eda_meanshift_varying/   ; tag=sim_msv ;;
    4) out=data/simulated/eda_meansdshift_varying/ ; tag=sim_msdv;;
  esac
  ${VENV} eda/baselines.py \
        WESAD \
        "data/simulated/eda_raw/*.csv" \
        ${out} \
        data/eda/cpd_baseline_${tag}/ \
        ${mode}
done

# ───────────────────────────────
# 5. LEARN CPD MODEL  → models/wesad_cp_model.pkl
# ───────────────────────────────
echo ">> Learning non-stationarity model"
${VENV} eda/learning_nonstationarity.py \
        WESAD \
        data/wesad/ \
        data/eda/cpd_annotations/ \
        5 \
        data/eda/bkps_counts.csv  \
        models/wesad_cp_model.pkl

# ───────────────────────────────
# 6. AUGMENT SYNTHETIC  → data/simulated/eda_aug/
# ───────────────────────────────
echo ">> Augmenting synthetic EDA with learned CPDs"
${VENV} eda/simulating.py \
        WESAD \
        data/simulated/eda_raw/ \
        data/simulated/eda_aug/ \
        models/wesad_cp_model.pkl \
        data/eda/cpd_aug_details/ \
        0.1

# ───────────────────────────────
# 7. CLASSIFICATION  (6 runs)
# ───────────────────────────────
echo ">> Classification on real + 5 synthetic sets"
py="eda/classification.py"
export PYTHONPATH=.                              # allow module imports
for ds in real sim_aug sim_msc sim_msdc sim_msv sim_msdv; do
  case $ds in
    real)      name=WESAD            src=data/wesad/                       ;;
    sim_aug)   name=sim_aug_WESAD    src=data/simulated/eda_aug/           ;;
    sim_msc)   name=sim_WESAD        src=data/simulated/eda_meanshift_constant/  ;;
    sim_msdc)  name=sim_WESAD        src=data/simulated/eda_meansdshift_constant/;;
    sim_msv)   name=sim_WESAD        src=data/simulated/eda_meanshift_varying/   ;;
    sim_msdv)  name=sim_WESAD        src=data/simulated/eda_meansdshift_varying/ ;;
  esac
  ${SED} "s/window_size *=.*/window_size = ${WIN_SIZE}/" eda/classification.py
  ${VENV} ${py} ${name} ${src} results/${ds}/
done

# ───────────────────────────────
# 8. SUMMARY TABLE
# ───────────────────────────────
echo ">> Building comparison table"
${VENV} compare_results.py

echo "✅  Pipeline completed.  See results/summary_f1_by_model.csv"