#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  echo "Usage: $0 HF_REPO_URL FILE_PATH OUTPUT_DIR [CLONE_DIR]" >&2
  exit 2
fi

HF_REPO_URL="$1"
FILE_PATH="$2"
OUTPUT_DIR="$3"
CLONE_DIR="${4:-}"

if [ -z "${CLONE_DIR}" ]; then
  safe_name="$(printf '%s' "${HF_REPO_URL}" | sed -E 's#^https?://##; s#[^A-Za-z0-9._-]+#_#g')"
  CLONE_DIR="${OUTPUT_DIR}/../hf_git/${safe_name}"
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${CLONE_DIR}")"

git_args=()
if [ -n "${HF_TOKEN:-}" ]; then
  git_args+=(-c "http.extraHeader=Authorization: Bearer ${HF_TOKEN}")
fi

if [ -d "${CLONE_DIR}/.git" ]; then
  echo "Updating Hugging Face git repo: ${CLONE_DIR}"
  git "${git_args[@]}" -C "${CLONE_DIR}" fetch --depth 1 origin
  git -C "${CLONE_DIR}" checkout --force FETCH_HEAD
else
  echo "Cloning Hugging Face git repo: ${HF_REPO_URL}"
  git "${git_args[@]}" clone --depth 1 "${HF_REPO_URL}" "${CLONE_DIR}"
fi

src="${CLONE_DIR}/${FILE_PATH}"
dst="${OUTPUT_DIR}/$(basename "${FILE_PATH}")"

if [ ! -f "${src}" ]; then
  echo "ERROR: ${FILE_PATH} was not found in ${CLONE_DIR}" >&2
  exit 1
fi

cp -f "${src}" "${dst}"

size_bytes="$(wc -c < "${dst}")"
if [ "${size_bytes}" -lt 1000000 ]; then
  echo "ERROR: ${dst} is unexpectedly small (${size_bytes} bytes)." >&2
  echo "Install git-xet or git-lfs support, then rerun the download." >&2
  exit 1
fi

echo "Downloaded file saved to: ${dst}"
