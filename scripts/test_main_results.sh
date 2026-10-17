#!/usr/bin/env bash

# Test only: load the released best checkpoint(s), then report classification
# and retrieval metrics. No training is performed here.

set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/paths.sh"

PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_ROOT="${KDH_CAD_RESULTS_ROOT}/tests/main_results"
TMCAD_SPLIT="${KDH_CAD_TMCAD_SPLITS}/shot_100"

: "${KDH_CAD_TMCAD_CKPT:?Set KDH_CAD_TMCAD_CKPT in configs/paths.env}"
test -f "${KDH_CAD_TMCAD_CKPT}"
mkdir -p "${OUTPUT_ROOT}"

echo '=== TMCAD classification ==='
"${PYTHON_BIN}" "${REPO_ROOT}/code/infer.py" \
    --checkpoint "${KDH_CAD_TMCAD_CKPT}" \
    --output-dir "${OUTPUT_ROOT}/tmcad" \
    --model_type qwen --classifier_mode "${KDH_CAD_TMCAD_CLASSIFIER_MODE}" \
    --use_shift yes --shift_init_mode data --use_shift_scale no \
    --qwen_path "${KDH_CAD_QWEN_PATH}" \
    --embed_base_path "${KDH_CAD_TMCAD_EMBEDDINGS}" \
    --train_json "${TMCAD_SPLIT}/train.json" \
    --valid_json "${TMCAD_SPLIT}/valid.json" \
    --test_json "${TMCAD_SPLIT}/test.json" \
    --batch_size 8

echo '=== TMCAD model retrieval ==='
"${PYTHON_BIN}" "${REPO_ROOT}/code/retrieval_eval.py" \
    --model_path "${KDH_CAD_TMCAD_CKPT}" \
    --gallery_json "${TMCAD_SPLIT}/train.json" \
    --query_json "${TMCAD_SPLIT}/test.json" \
    --model_type qwen --classifier_mode "${KDH_CAD_TMCAD_CLASSIFIER_MODE}" \
    --use_shift yes --shift_init_mode data --use_shift_scale no \
    --qwen_path "${KDH_CAD_QWEN_PATH}" \
    --embed_base_path "${KDH_CAD_TMCAD_EMBEDDINGS}" \
    --batch_size 8

if [[ -n "${KDH_CAD_FABWAVE_CKPT:-}" ]]; then
    test -f "${KDH_CAD_FABWAVE_CKPT}"
    echo '=== FabWave classification ==='
    "${PYTHON_BIN}" "${REPO_ROOT}/code/infer.py" \
        --checkpoint "${KDH_CAD_FABWAVE_CKPT}" \
        --output-dir "${OUTPUT_ROOT}/fabwave" \
        --model_type qwen --classifier_mode "${KDH_CAD_FABWAVE_CLASSIFIER_MODE}" \
        --use_shift yes --shift_init_mode data --use_shift_scale no \
        --qwen_path "${KDH_CAD_QWEN_PATH}" \
        --embed_base_path "${KDH_CAD_FABWAVE_EMBEDDINGS}" \
        --train_json "${KDH_CAD_FABWAVE_SPLIT_DIR}/train.json" \
        --valid_json "${KDH_CAD_FABWAVE_SPLIT_DIR}/valid.json" \
        --test_json "${KDH_CAD_FABWAVE_SPLIT_DIR}/test.json" \
        --batch_size 8
else
    echo 'Skipping FabWave: KDH_CAD_FABWAVE_CKPT is not configured.'
fi
