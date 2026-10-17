#!/usr/bin/env bash

set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/paths.sh"

MODEL_ID=Qwen/Qwen3-VL-2B-Instruct
mkdir -p "${KDH_CAD_QWEN_PATH}"
modelscope download --model "${MODEL_ID}" --local_dir "${KDH_CAD_QWEN_PATH}"
