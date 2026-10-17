#!/usr/bin/env bash

# Download published KDH-CAD checkpoints from direct HTTP(S) links.

set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/lib/paths.sh"

LINKS_FILE=${KDH_CAD_DOWNLOAD_CONFIG:-"${REPO_ROOT}/configs/download.env"}
if [[ ! -f "${LINKS_FILE}" ]]; then
    echo "Missing ${LINKS_FILE}. Copy configs/download.env.example and fill in direct links." >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${LINKS_FILE}"

DOWNLOAD_DIR=${KDH_CAD_DOWNLOAD_DIR:-"${KDH_CAD_ASSET_ROOT}/downloads"}
mkdir -p "${DOWNLOAD_DIR}" "${KDH_CAD_ASSET_ROOT}"

download() {
    local label=$1 url=$2 destination=$3 expected_sha=$4
    if [[ -z "${url}" ]]; then
        echo "Skipping ${label}: no direct URL configured."
        return
    fi
    echo "Downloading ${label}..."
    curl --fail --location --continue-at - --output "${destination}" "${url}"
    if [[ -n "${expected_sha}" ]]; then
        echo "${expected_sha}  ${destination}" | sha256sum --check --status
        echo "Verified SHA256 for ${label}."
    fi
}

download "TMCAD checkpoint" "${KDH_CAD_TMCAD_CKPT_URL:-}" \
    "${KDH_CAD_ASSET_ROOT}/${KDH_CAD_TMCAD_CKPT_FILE}" "${KDH_CAD_TMCAD_CKPT_SHA256:-}"
download "FabWave checkpoint" "${KDH_CAD_FABWAVE_CKPT_URL:-}" \
    "${KDH_CAD_ASSET_ROOT}/${KDH_CAD_FABWAVE_CKPT_FILE}" "${KDH_CAD_FABWAVE_CKPT_SHA256:-}"
