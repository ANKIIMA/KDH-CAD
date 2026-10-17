#!/usr/bin/env bash

# Shared external-asset setup for public KDH-CAD experiment entrypoints.
# The caller may point KDH_CAD_PATHS_FILE at any shell-compatible env file.

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PATHS_FILE=${KDH_CAD_PATHS_FILE:-"${REPO_ROOT}/configs/paths.env"}

if [[ ! -f "${PATHS_FILE}" ]]; then
    echo "Missing ${PATHS_FILE}. Copy configs/paths.env.example to configs/paths.env and set local paths." >&2
    exit 2
fi

# shellcheck disable=SC1090
source "${PATHS_FILE}"

: "${KDH_CAD_ASSET_ROOT:?KDH_CAD_ASSET_ROOT must be set}"
: "${KDH_CAD_MODEL_ROOT:?KDH_CAD_MODEL_ROOT must be set}"
: "${KDH_CAD_RESULTS_ROOT:?KDH_CAD_RESULTS_ROOT must be set}"

resolve_from_repo() {
    local path=$1
    if [[ -z "${path}" || "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${REPO_ROOT}/${path}"
    fi
}

KDH_CAD_ASSET_ROOT=$(resolve_from_repo "${KDH_CAD_ASSET_ROOT}")
KDH_CAD_MODEL_ROOT=$(resolve_from_repo "${KDH_CAD_MODEL_ROOT}")
KDH_CAD_RESULTS_ROOT=$(resolve_from_repo "${KDH_CAD_RESULTS_ROOT}")

KDH_CAD_TMCAD_SPLITS=${KDH_CAD_TMCAD_SPLITS:-"${KDH_CAD_ASSET_ROOT}/json_newPath"}
KDH_CAD_FABWAVE_SPLITS=${KDH_CAD_FABWAVE_SPLITS:-"${KDH_CAD_ASSET_ROOT}/data/Febwave_jsons_merged"}
KDH_CAD_TMCAD_EMBEDDINGS=${KDH_CAD_TMCAD_EMBEDDINGS:-"${KDH_CAD_ASSET_ROOT}/data/test0122"}
KDH_CAD_FABWAVE_EMBEDDINGS=${KDH_CAD_FABWAVE_EMBEDDINGS:-"${KDH_CAD_ASSET_ROOT}/data/FabwaveAnchor"}
KDH_CAD_QWEN_PATH=${KDH_CAD_QWEN_PATH:-"${KDH_CAD_MODEL_ROOT}/qwen3vl2b"}
KDH_CAD_CHECKPOINT_ROOT=${KDH_CAD_CHECKPOINT_ROOT:-"${KDH_CAD_ASSET_ROOT}/data/ckpt"}
KDH_CAD_TMCAD_CLASSIFIER_MODE=${KDH_CAD_TMCAD_CLASSIFIER_MODE:-cross_attn}
KDH_CAD_FABWAVE_CLASSIFIER_MODE=${KDH_CAD_FABWAVE_CLASSIFIER_MODE:-cross_attn}
KDH_CAD_FABWAVE_SPLIT_DIR=${KDH_CAD_FABWAVE_SPLIT_DIR:-"${KDH_CAD_FABWAVE_SPLITS}/tail_50"}

for variable in \
    KDH_CAD_TMCAD_SPLITS KDH_CAD_FABWAVE_SPLITS \
    KDH_CAD_TMCAD_EMBEDDINGS KDH_CAD_FABWAVE_EMBEDDINGS \
    KDH_CAD_QWEN_PATH KDH_CAD_CHECKPOINT_ROOT \
    KDH_CAD_TMCAD_CKPT KDH_CAD_FABWAVE_CKPT KDH_CAD_FABWAVE_SPLIT_DIR; do
    value=${!variable:-}
    if [[ -n "${value}" ]]; then
        printf -v "${variable}" '%s' "$(resolve_from_repo "${value}")"
    fi
done

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
