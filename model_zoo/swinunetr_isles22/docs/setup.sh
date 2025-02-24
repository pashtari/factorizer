#!/bin/bash

# ========================
# Environment Setup Script
# ========================

set -euo pipefail

# Function to display usage information
usage() {
    cat <<EOF
Usage: $0 [--device <cpu|cuda>] [--conda_path <conda_path>] [--env <env_name>] [--python_version <version>] [-h|--help]

Sets up the development environment.

Options:
  -d, --device          Device to use (cpu or cuda) [default: cpu]
  -c, --conda_path      Path to conda installation to prepend to PATH [optional]
  -e, --env             Name of conda environment [optional]
  -p, --python_version  Python version to use [default: latest stable]
  -h, --help            Show this help message

Prerequisites:
  Conda must be installed and available.

Examples:
  # Install the CPU requirements in the current environment.
  $0 --device cpu

  # Create a conda environment for CPU with the latest Python:
  $0 --device cpu --env cpupenv

  # Create a conda environment for CUDA with Python 3.12:
  $0 --device cuda conda_path /path/to/conda --env gpuenv --python_version 3.12

EOF
    exit 1
}

# Function to log messages
log() {
    local level="$1"
    shift
    case "$level" in
    info)
        echo -e "\033[1;32m[INFO] $*\033[0m" # Green text for information
        ;;
    warn)
        echo -e "\033[1;33m[WARNING] $*\033[0m" # Yellow text for warnings
        ;;
    error)
        echo -e "\033[1;31m[ERROR] $*\033[0m" # Red text for errors
        ;;
    *)
        echo -e "\033[1;37m[LOG] $*\033[0m" # Default gray text
        ;;
    esac
}

# =======================
# Argument Parsing
# =======================
DEVICE="cpu"
PYTHON_VERSION=""
while [[ $# -gt 0 ]]; do
    case "$1" in
    -d | --device)
        DEVICE="$2"
        shift 2
        ;;
    -c | --conda_path)
        CONDA_PATH="$2"
        shift 2
        ;;
    -e | --env)
        ENV="$2"
        shift 2
        ;;
    -p | --python_version)
        PYTHON_VERSION="$2"
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    *)
        log error "Unknown option: $1"
        usage
        ;;
    esac
done

# =======================
# Conda Configuration
# =======================
if [[ -n "${CONDA_PATH:-}" ]]; then
    log info "Adding conda to PATH..."
    export PATH="$CONDA_PATH/bin:$PATH"
fi

# =======================
# Environment Setup
# =======================
if [[ -n "${ENV:-}" ]]; then
    eval "$(conda shell.bash hook)"

    if conda env list | grep -q "$ENV"; then
        log warn "Environment '$ENV' already exists. Skipping creation."
    else
        log info "Creating conda environment..."
        conda create -y -n "$ENV" python="$PYTHON_VERSION"
    fi
    conda activate "$ENV"
fi

# =======================
# Dependency Installation
# =======================
SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ "$DEVICE" == "cuda" ]]; then
    log info "Installing GPU requirements from requirements_cuda.txt..."
    pip install -r requirements_cuda.txt
else
    log info "Installing CPU requirements from requirements.txt..."
    pip install -r requirements.txt
fi

log info "Setup completed successfully!"
