#!/bin/bash

# ===============================
# SLURM Job Configuration Example
# ===============================
#SBATCH --account=lp_inspiremed
#SBATCH --clusters=genius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --time=25:00:00
#SBATCH --partition=gpu_a100

set -euo pipefail

# Function to display usage information
usage() {
    cat <<EOF
Usage: $0 [--conda_path <conda_path>] [--env <env_name>] [--bundle_dir <bundle_directory>] [additional MONAI bundle arguments]

Runs multi-GPU model training using MONAI Bundle framework with distributed data parallel.

Options:
  --conda_path      Path to conda installation to prepend to PATH [optional]
  --env             Name of conda environment to activate [optional]
  --bundle_dir      Path to the MONAI bundle directory [default: one level up from script location]
  -h, --help        Show this help message

MONAI Bundle Arguments:
  Any additional arguments, such as --data_dir, --fold, and --batch_size, will be passed to the 'monai.bundle run' command.

Examples:
  # Run training with multi-GPU setup
  $0 --env <env_name> --bundle_dir /path/to/bundle --data_dir /path/to/data --fold 0 --batch_size 2

SLURM Submission Example:
  sbatch --job-name=nnunet_brats23_multigpu train_multigpu.sh --conda_path /path/to/conda --env <env_name> --data_dir /path/to/data --fold 0 --batch_size 2

EOF
    exit 1
}

# =======================
# Argument Parsing
# =======================
ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
    --conda_path)
        CONDA_PATH="$2"
        shift 2
        ;;
    --env)
        ENV="$2"
        shift 2
        ;;
    --bundle_dir)
        BUNDLE_DIR="$2"
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    *)
        ARGS+=("$1")
        shift
        ;;
    esac
done

# Set default bundle directory
SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_DIR="${BUNDLE_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"

# =======================
# Conda Environment Setup
# =======================
if [[ -n "${CONDA_PATH:-}" ]]; then
    export PATH="$CONDA_PATH/bin:$PATH"
fi

if [[ -n "${ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"
    conda activate "$ENV"
fi

# =======================
# Environment Variables
# =======================
echo "Bundle root directory: $BUNDLE_DIR"
export PYTHONPATH="$BUNDLE_DIR"

# seems to resolve some multiprocessing issues with certain libraries
export OMP_NUM_THREADS=1
CKPT=none

# =======================
# Multi-GPU Environment Setup
# =======================
# need to change this if you have multiple nodes or not 2 GPUs
WORLD_SIZE=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Running on $WORLD_SIZE GPUs"

# =======================
# Main Execution
# =======================
PYTHON="torchrun --standalone --nnodes=1 --nproc_per_node=$WORLD_SIZE"

$PYTHON -m monai.bundle run \
    --meta_file $BUNDLE_DIR/configs/metadata.json \
    --logging_file "$BUNDLE_DIR/configs/logging.conf" \
    --config_file "['$BUNDLE_DIR/configs/train.yaml','$BUNDLE_DIR/configs/train_multigpu.yaml']" \
    --bundle_root $BUNDLE_DIR \
    "${ARGS[@]}"
