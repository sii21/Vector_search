#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
ARCHIVE_PATH="${DATA_DIR}/laion-small-clip.tgz"

DATA_URL="https://storage.googleapis.com/ann-filtered-benchmark/datasets/laion-small-clip.tgz"

mkdir -p "${DATA_DIR}"

if [ -f "${DATA_DIR}/vectors.npy" ]; then
  echo "vectors.npy already exists in ${DATA_DIR}, skip download."
  exit 0
fi

if [ ! -f "${ARCHIVE_PATH}" ]; then
  echo "Downloading laion-small-clip.tgz..."
  curl -L "${DATA_URL}" -o "${ARCHIVE_PATH}"
else
  echo "Archive already downloaded: ${ARCHIVE_PATH}"
fi

echo "Unpacking archive..."
tar -xzf "${ARCHIVE_PATH}" -C "${DATA_DIR}"

echo "Done. Check ${DATA_DIR} for vectors.npy, payloads.jsonl, tests.jsonl."
