#!/usr/bin/env bash
# Download selected SA-1B tar shards from a Hugging Face dataset mirror.
# Order: INPUT_TSV, OUTPUT_DIR, HF_REPO(optional, default: ssbai/sa1b)

set -euo pipefail

INPUT_TSV="$1"
OUTPUT_DIR="$2"
HF_REPO="${3:-${SA1B_HF_REPO:-ssbai/sa1b}}"
HF_BIN="${HF_BIN:-hf}"
HF_DOWNLOAD_BACKEND="${HF_DOWNLOAD_BACKEND:-hf}"
HF_REPO_URL="${SA1B_HF_REPO_URL:-https://huggingface.co/datasets/${HF_REPO}}"
HF_GIT_DIR="${SA1B_HF_GIT_DIR:-${OUTPUT_DIR}/../hf_git_sa1b}"

mkdir -p "${OUTPUT_DIR}"

includes=()
files=()
expected_count=0
while IFS=$'\t' read -r file_name url; do
  if [[ "${file_name}" == "file_name" && "${url:-}" == "cdn_link" ]]; then
    continue
  fi
  [[ -z "${file_name:-}" ]] && continue
  includes+=(--include "${file_name}")
  files+=("${file_name}")
  expected_count=$((expected_count+1))
done < "${INPUT_TSV}"

if [ "${expected_count}" -eq 0 ]; then
  echo "ERROR: no SA-1B filenames found in ${INPUT_TSV}" >&2
  exit 1
fi

echo "Downloading ${expected_count} SA-1B shard(s) from Hugging Face dataset ${HF_REPO} via ${HF_DOWNLOAD_BACKEND}"
case "${HF_DOWNLOAD_BACKEND}" in
  hf)
    "${HF_BIN}" download "${HF_REPO}" \
      --repo-type dataset \
      --local-dir "${OUTPUT_DIR}" \
      "${includes[@]}"
    ;;
  git)
    mkdir -p "$(dirname "${HF_GIT_DIR}")"
    git_args=()
    if [ -n "${HF_TOKEN:-}" ]; then
      git_args+=(-c "http.extraHeader=Authorization: Bearer ${HF_TOKEN}")
    fi
    if [ -d "${HF_GIT_DIR}/.git" ]; then
      git "${git_args[@]}" -C "${HF_GIT_DIR}" fetch --depth 1 origin
      git -C "${HF_GIT_DIR}" sparse-checkout set --no-cone "${files[@]}"
      git -C "${HF_GIT_DIR}" checkout --force FETCH_HEAD
    else
      git "${git_args[@]}" clone --filter=blob:none --no-checkout "${HF_REPO_URL}" "${HF_GIT_DIR}"
      git -C "${HF_GIT_DIR}" sparse-checkout init --no-cone
      git -C "${HF_GIT_DIR}" sparse-checkout set --no-cone "${files[@]}"
      git -C "${HF_GIT_DIR}" checkout
    fi
    for file_name in "${files[@]}"; do
      mkdir -p "${OUTPUT_DIR}/$(dirname "${file_name}")"
      cp -f "${HF_GIT_DIR}/${file_name}" "${OUTPUT_DIR}/${file_name}"
    done
    ;;
  *)
    echo "ERROR: unsupported HF_DOWNLOAD_BACKEND=${HF_DOWNLOAD_BACKEND}; use hf or git." >&2
    exit 1
    ;;
esac

missing=0
for include_arg in "${includes[@]}"; do
  if [[ "${include_arg}" == --include ]]; then
    continue
  fi
  if [ ! -s "${OUTPUT_DIR}/${include_arg}" ]; then
    echo "ERROR: missing or empty shard after HF download: ${OUTPUT_DIR}/${include_arg}" >&2
    missing=$((missing+1))
  elif [ "$(wc -c < "${OUTPUT_DIR}/${include_arg}")" -lt 1000000 ]; then
    echo "ERROR: shard is unexpectedly small after HF download: ${OUTPUT_DIR}/${include_arg}" >&2
    echo "Install git-xet or git-lfs support, then rerun the download." >&2
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
