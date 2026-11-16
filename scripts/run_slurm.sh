#!/usr/bin/env bash
#SBATCH --job-name=pauli-moments
#SBATCH --output=pauli-moments-%j.out
#SBATCH --error=pauli-moments-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

# Portable Slurm example adapted from the author's original TACC launcher.
# Set cluster-specific partition, account, QOS, and time limits yourself.
# Submit from the repository root: JOB_INDEX=1 sbatch scripts/run_slurm.sh
set -euo pipefail

: "${JOB_INDEX:?Set JOB_INDEX=1,2,... to distinguish independent jobs}"
: "${SLURM_JOB_ID:?Run this example using sbatch}"

WORK_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$WORK_DIR"

export L="${L:-8}"
export NS="${NS:-100}"
export CHECKPOINT_LAYERS="${CHECKPOINT_LAYERS:-4}"
export INITIAL_OPERATOR_MODE="${INITIAL_OPERATOR_MODE:-random_pauli}"
export BACKEND="${BACKEND:-dict}"
export PROGRESS_EVERY="${PROGRESS_EVERY:-25}"
export JOB_INDEX
export SEED_JOB_STRIDE="${SEED_JOB_STRIDE:-1000000}"
export RESUME="${RESUME:-true}"
export PARALLEL_REALIZATIONS=true
export JULIA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OUTPUT_DIR="${OUTPUT_DIR:-$WORK_DIR/results}"
mkdir -p "$OUTPUT_DIR"

JULIA_BIN="${JULIA_BIN:-julia}"
"$JULIA_BIN" --project="$WORK_DIR" --threads="$JULIA_NUM_THREADS" \
  "$WORK_DIR/src/simulate_variance.jl"
