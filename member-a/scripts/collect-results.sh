#!/bin/bash
# Run on your Mac after the remote COMPLETE marker exists. No model weights copied.
set -euo pipefail
SC4001_HOST="${1:?Usage: bash scripts/collect-results.sh username@host}"
SC4001_DEST="${2:-formal-results}"
mkdir -p "$SC4001_DEST"
ssh -p 22 "$SC4001_HOST" 'set -e; cd sc4001-a; test -f results/A-v1/COMPLETE; tar -czf - results/A-v1 runs/A-v1/*/test/metrics.json runs/A-v1/*/test/*_predictions.jsonl runs/A-v1/*/run_config.json runs/A-v1/*/validation.json runs/A-v1/*/parameters.json runs/A-v1/*/history.json runs/A-v1/*/source runs/A-v1/selection-*.json runs/A-v1/attempts.jsonl logs/formal_[0-9]*.out logs/formal_[0-9]*.err' | tar -xzf - -C "$SC4001_DEST"
printf 'Collected completed formal records into %s\n' "$SC4001_DEST"
