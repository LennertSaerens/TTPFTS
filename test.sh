#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="$ROOT_DIR/envs.txt"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Could not find envs.txt at: $ENV_FILE" >&2
  exit 1
fi

ENVIRONMENTS="$(awk '
  NF {
    gsub(/^[[:space:]]+|[[:space:]]+$/, "", $0)
    if ($0 != "") {
      if (seen) {
        printf ","
      }
      printf "%s", $0
      seen = 1
    }
  }
' "$ENV_FILE")"

if [[ -z "$ENVIRONMENTS" ]]; then
  echo "No environments found in $ENV_FILE" >&2
  exit 1
fi

python3 "$ROOT_DIR/momab_main.py" \
  --algorithms TTPFTS_GKV,CPFTS_GKV,RPFTS_GKV \
  --environments "$ENVIRONMENTS" \
  --num_runs 100

