#!/usr/bin/env bash
set -euo pipefail

# Increase file descriptor limit for multi-worker tensor sharing
ulimit -n 65535

MODE="${1:-train}"
shift || true

case "${MODE}" in
  train|validate|check-runtime|sweep)
    ;;
  *)
    echo "unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac

FAMILY="${KINEXAR_FAMILY:-${KINEXAR_DEFAULT_FAMILY:-full}}"

case "${MODE}" in
    train)
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python -u scripts/train_gcp.py --family "${FAMILY}" "$@"
    ;;
  validate)
    python scripts/validate_gcp.py --family "${FAMILY}" "$@"
    ;;
  check-runtime)
    python scripts/check_runtime.py --family "${FAMILY}" "$@"
    ;;
  sweep)
    python scripts/sweep_hparams.py --family "${FAMILY}" "$@"
    ;;
esac
