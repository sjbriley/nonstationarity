#!/usr/bin/env bash
# ==========================================================
#  run_full_pipeline.sh
#  End-to-end rebuild for the WESAD non-stationarity project with chest data
# ==========================================================

set -e
VENV="./virtualenv/bin/python"

# ───────────────────────────────
# 1. CLEAN SLATE
# ───────────────────────────────
echo ">> Cleaning previous artifacts"
rm -rf data/wesad_chest            \
       data/wesad        \
       data/simulated        \
       data/eda              \
       models                \
       results/chest
mkdir  -p models results/chest/

# ───────────────────────────────
# 2. REAL DATA  →  data/wesad/
# ───────────────────────────────
echo ">> Extracting real EDA CSVs"
${VENV} eda/extract_chest_eda.py chest

# ───────────────────────────────
# 3. RAW SYNTHETIC
# ───────────────────────────────
echo ">> Generating raw synthetic EDA"
# sample rate=700 for chest
${VENV} eda/simulate_raw_eda.py chest

# ───────────────────────────────
# 4. FOUR SHIFT BASELINES
# ───────────────────────────────
echo ">> Generating baselines (MSC, MSDC, MSV, MSDV)"

for mode in 1 2 3 4; do
  case $mode in
    1) out=data/simulated/eda_meanshift_constant_chest/  ; tag=sim_msc ;;
    2) out=data/simulated/eda_meansdshift_constant_chest/; tag=sim_msdc;;
    3) out=data/simulated/eda_meanshift_varying_chest/   ; tag=sim_msv ;;
    4) out=data/simulated/eda_meansdshift_varying_chest/ ; tag=sim_msdv;;
  esac
  ${VENV} eda/baselines.py \
        WESAD \
        "data/simulated/eda_raw/*.csv" \
        ${out} \
        data/eda/cpd_baseline_${tag}_chest/ \
        ${mode}
done
${VENV} eda/baselines.py WESAD "data/simulated/eda_raw/*.csv" data/simulated/eda_meanshift_constant_chest/   data/eda/cpd_baseline_sim_msc_chest/  1
${VENV} eda/baselines.py WESAD "data/simulated/eda_raw/*.csv" data/simulated/eda_meansdshift_constant_chest/ data/eda/cpd_baseline_sim_msdc_chest/ 2
${VENV} eda/baselines.py WESAD "data/simulated/eda_raw/*.csv" data/simulated/eda_meanshift_varying_chest/    data/eda/cpd_baseline_sim_msv_chest/  3
${VENV} eda/baselines.py WESAD "data/simulated/eda_raw/*.csv" data/simulated/eda_meansdshift_varying_chest/  data/eda/cpd_baseline_sim_msdv_chest/ 4


# ───────────────────────────────
# 5. LEARN MODEL
# ───────────────────────────────
echo ">> Learning non-stationarity model"
${VENV} eda/learning_nonstationarity.py \
        WESAD \
        data/wesad/chest/ \
        data/eda/chest_cpd_annotations/ \
        10 \
        data/eda/chest_bkps_counts.csv  \
        models/chest/wesad_chest_cp_model.pkl

# ───────────────────────────────
# 6. AUGMENT SYNTHETIC  → data/simulated/eda_aug/
# ───────────────────────────────
echo ">> Augmenting synthetic EDA with learned CPDs"
${VENV} eda/simulating.py \
        WESAD_chest \
        data/simulated/eda_raw/ \
        data/simulated/eda_chest_aug/ \
        models/wesad_chest_cp_model.pkl \
        data/eda/chest_cpd_aug_details/ \
        0.1

# ───────────────────────────────
# 7. CLASSIFICATION  (6 runs)
# ───────────────────────────────
echo ">> Classification on real + 5 synthetic sets"
py="eda/classification.py"
export PYTHONPATH=.
for ds in real sim_aug sim_msc sim_msdc sim_msv sim_msdv; do
  case $ds in
    real)      name=WESAD_chest            src=data/wesad_chest/                       ;;
    sim_aug)   name=sim_aug_WESAD_chest    src=data/simulated/eda_chest_aug/           ;;
    sim_msc)   name=sim_WESAD_chest        src=data/simulated/eda_meanshift_constant_chest/  ;;
    sim_msdc)  name=sim_WESAD_chest        src=data/simulated/eda_meansdshift_constant_chest/;;
    sim_msv)   name=sim_WESAD_chest        src=data/simulated/eda_meanshift_varying_chest/   ;;
    sim_msdv)  name=sim_WESAD_chest        src=data/simulated/eda_meansdshift_varying_chest/ ;;
  esac
  ${VENV} ${py} ${name} ${src} results/chest/${ds}/
done

# ───────────────────────────────
# 8. SUMMARY TABLE
# ───────────────────────────────
echo ">> Building comparison table"
${VENV} compare_results.py "results/chest"

echo "Pipeline completed.  See results/chest//summary_f1_by_model.csv"
