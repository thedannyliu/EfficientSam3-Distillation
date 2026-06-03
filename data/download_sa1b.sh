#!/usr/bin/env bash
# Order: INPUT_TSV, OUTPUT_DIR, CONCURRENCY(optional, default: 4)
# usage: bash data/download_sa1b.sh data/sa-1b-10p.txt data/sa-1b-10p 8
# Successfully downloaded filenames are recorded in downloaded_ok.txt

set -uo pipefail

INPUT_TSV="$1"
OUTPUT_DIR="$2"
CONCURRENCY="${3:-4}"   # Number of parallel downloads, default: 4

mkdir -p "$OUTPUT_DIR"

# File to record successfully downloaded filenames
SUCCESS_LIST="$OUTPUT_DIR/downloaded_ok.txt"
FAIL_LIST="$OUTPUT_DIR/downloaded_failed.txt"
: > "$SUCCESS_LIST"   # Truncate/create the file
: > "$FAIL_LIST"

i=0
expected_count=0

download_one() {
  local file_name="$1"
  local url="$2"
  local idx="$3"

  local dest="$OUTPUT_DIR/$file_name"

  echo "[$idx] START $file_name"

  # Wrap wget in an if-block so failures won't stop the main script
  if wget -c --tries=5 --timeout=30 -O "$dest" "$url"; then
    if [ -s "$dest" ]; then
      echo "[$idx] DONE  $file_name"
      echo "$file_name" >> "$SUCCESS_LIST"
    else
      echo "[$idx] FAIL  $file_name (empty download)" >&2
      rm -f "$dest"
      echo "$file_name" >> "$FAIL_LIST"
    fi
  else
    echo "[$idx] FAIL  $file_name" >&2
    rm -f "$dest"
    echo "$file_name" >> "$FAIL_LIST"
  fi
}

# Read TSV line by line: file_name<TAB>cdn_link
while IFS=$'\t' read -r file_name url; do
  # Skip header
  if [[ "$file_name" == "file_name" && "$url" == "cdn_link" ]]; then
    continue
  fi

  # Skip empty lines
  [[ -z "${file_name:-}" || -z "${url:-}" ]] && continue

  expected_count=$((expected_count+1))
  download_one "$file_name" "$url" "$i" &   # Run in background (parallel)
  i=$((i+1))

  # Control concurrency: wait after every CONCURRENCY jobs
  if (( i % CONCURRENCY == 0 )); then
    wait
  fi
done < "$INPUT_TSV"

# Wait for remaining background jobs
wait

success_count=$(wc -l < "$SUCCESS_LIST" 2>/dev/null || echo 0)
fail_count=$(wc -l < "$FAIL_LIST" 2>/dev/null || echo 0)
echo "Total downloaded: $success_count file(s)."
echo "Success list saved to: $SUCCESS_LIST"
if [ "$success_count" -ne "$expected_count" ] || [ "$fail_count" -ne 0 ]; then
  echo "ERROR: expected $expected_count files, downloaded $success_count, failed $fail_count." >&2
  echo "Failure list saved to: $FAIL_LIST" >&2
  exit 1
fi
