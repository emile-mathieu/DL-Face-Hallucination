#!/bin/bash
set -euo pipefail

# Submit the 3 core jobs with dependencies:
# 1) Train Basic CNN
# 2) Train BiChannel CNN
# 3) Evaluate Classical methods (after both trainings succeed)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASIC_SCRIPT="${SCRIPT_DIR}/slurm_train_basic.sh"
BICHANNEL_SCRIPT="${SCRIPT_DIR}/slurm_train_bichannel.sh"
CLASSICAL_SCRIPT="${SCRIPT_DIR}/slurm_eval_baselines.sh"

for f in "$BASIC_SCRIPT" "$BICHANNEL_SCRIPT" "$CLASSICAL_SCRIPT"; do
  if [[ ! -f "$f" ]]; then
    echo "Error: missing script: $f" >&2
    exit 1
  fi
done

if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: sbatch is not available in PATH." >&2
  exit 1
fi

echo "Submitting Basic CNN training..."
BASIC_JOB_ID="$(sbatch --parsable "$BASIC_SCRIPT")"
echo "  Basic job id: $BASIC_JOB_ID"

echo "Submitting BiChannel CNN training..."
BICHANNEL_JOB_ID="$(sbatch --parsable "$BICHANNEL_SCRIPT")"
echo "  BiChannel job id: $BICHANNEL_JOB_ID"

echo "Submitting Classical evaluation after both training jobs succeed..."
CLASSICAL_JOB_ID="$(sbatch --parsable --dependency=afterok:${BASIC_JOB_ID}:${BICHANNEL_JOB_ID} "$CLASSICAL_SCRIPT")"
echo "  Classical job id: $CLASSICAL_JOB_ID"

echo

echo "Submission summary:"
echo "  Basic CNN:      $BASIC_JOB_ID"
echo "  BiChannel CNN:  $BICHANNEL_JOB_ID"
echo "  Classical Eval: $CLASSICAL_JOB_ID (afterok:${BASIC_JOB_ID}:${BICHANNEL_JOB_ID})"

echo

echo "Use this command to monitor:"
echo "  squeue -j ${BASIC_JOB_ID},${BICHANNEL_JOB_ID},${CLASSICAL_JOB_ID}"
