#!/usr/bin/env bash
# Download selected SA-1B tar shards from a Hugging Face dataset mirror.
# Order: INPUT_TSV, OUTPUT_DIR, HF_REPO(optional, default: ssbai/sa1b)

set -euo pipefail

INPUT_TSV="$1"
OUTPUT_DIR="$2"
HF_REPO="${3:-${SA1B_HF_REPO:-ssbai/sa1b}}"
HF_BIN="${HF_BIN:-hf}"

mkdir -p "${OUTPUT_DIR}"

includes=()
expected_count=0
while IFS=$'\t' read -r file_name url; do
  if [[ "${file_name}" == "file_name" && "${url:-}" == "cdn_link" ]]; then
    continue
  fi
  [[ -z "${file_name:-}" ]] && continue
  includes+=(--include "${file_name}")
  expected_count=$((expected_count+1))
done < "${INPUT_TSV}"

if [ "${expected_count}" -eq 0 ]; then
  echo "ERROR: no SA-1B filenames found in ${INPUT_TSV}" >&2
  exit 1
fi

echo "Downloading ${expected_count} SA-1B shard(s) from Hugging Face dataset ${HF_REPO}"
"${HF_BIN}" download "${HF_REPO}" \
  --repo-type dataset \
  --local-dir "${OUTPUT_DIR}" \
  "${includes[@]}"

missing=0
for include_arg in "${includes[@]}"; do
  if [[ "${include_arg}" == --include ]]; then
    continue
  fi
  if [ ! -s "${OUTPUT_DIR}/${include_arg}" ]; then
    echo "ERROR: missing or empty shard after HF download: ${OUTPUT_DIR}/${include_arg}" >&2
    missing=$((missing+1))
  fi
done

if [ "${missing}" -ne 0 ]; then
  exit 1
fi

printf '%s\n' "${includes[@]}" \
  | awk 'prev == "--include" { print $0 } { prev = $0 }' \
  > "${OUTPUT_DIR}/downloaded_ok.txt"
echo "Success list saved to: ${OUTPUT_DIR}/downloaded_ok.txt"
