#!/usr/bin/env bash

# Train the paper's main KDH-CAD configurations. Final test evaluation is
# deliberately disabled; use test_main_results.sh with the selected checkpoint.

set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/paths.sh"

PYTHON_BIN=${PYTHON_BIN:-python}
EPOCHS=${KDH_CAD_EPOCHS:-40}
BATCH_SIZE=${KDH_CAD_BATCH_SIZE:-8}
LR=${KDH_CAD_LR:-1e-4}
OUTPUT_ROOT="${KDH_CAD_RESULTS_ROOT}/training/main_results"
mkdir -p "${OUTPUT_ROOT}"

train_target() {
    local name=$1 embeddings=$2 split_dir=$3
    "${PYTHON_BIN}" "${REPO_ROOT}/code/train.py" \
        --model_type qwen --classifier_mode cross_attn \
        --use_shift yes --shift_init_mode data --use_shift_scale no \
        --qwen_path "${KDH_CAD_QWEN_PATH}" --embed_base_path "${embeddings}" \
        --train_json "${split_dir}/train.json" \
        --valid_json "${split_dir}/valid.json" \
        --test_json "${split_dir}/test.json" \
        --batch_size "${BATCH_SIZE}" --lr "${LR}" --epochs "${EPOCHS}" \
        --best_model_path "${OUTPUT_ROOT}/${name}.pth" \
        --exp_name "kdhcad_${name}" --swanlab_mode disabled --skip_test yes
}

case ${KDH_CAD_TRAIN_TARGET:-all} in
    tmcad)
        train_target tmcad_best "${KDH_CAD_TMCAD_EMBEDDINGS}" "${KDH_CAD_TMCAD_SPLITS}/shot_100"
        ;;
    fabwave)
        train_target fabwave_best "${KDH_CAD_FABWAVE_EMBEDDINGS}" "${KDH_CAD_FABWAVE_SPLIT_DIR}"
        ;;
    all)
        train_target tmcad_best "${KDH_CAD_TMCAD_EMBEDDINGS}" "${KDH_CAD_TMCAD_SPLITS}/shot_100"
        train_target fabwave_best "${KDH_CAD_FABWAVE_EMBEDDINGS}" "${KDH_CAD_FABWAVE_SPLIT_DIR}"
        ;;
    *)
        echo "KDH_CAD_TRAIN_TARGET must be tmcad, fabwave, or all." >&2
        exit 2
        ;;
esac
