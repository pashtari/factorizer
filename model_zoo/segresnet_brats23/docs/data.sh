#!/bin/bash

# ===================================
# Data Download and Extraction Script
# ===================================

set -euo pipefail

# Function to display usage information
usage() {
    cat <<EOF
Usage: $0 [--data_dir <data_directory>] [-h|--help]

Downloads and extracts the BraTS2023-GLI data files from Synapse.

Options:
  --data_dir    Path to data directory [required]
  -h, --help    Show this help message

Prerequisites:
  - Valid Synapse credentials
  - Python installed

Examples:
  # Specify data directory as a positional argument:
  $0 /path/to/data

  # Specify data directory with the --data_dir option:
  $0 --data_dir /path/to/data

EOF
    exit 1
}

# Function to log messages with color and level
log() {
    local level="$1"
    shift
    case "$level" in
    info)
        echo -e "\033[1;32m[INFO] $*\033[0m" # Green for info messages
        ;;
    error)
        echo -e "\033[1;31m[ERROR] $*\033[0m" # Red for errors
        ;;
    *)
        echo -e "\033[1;37m[LOG] $*\033[0m" # Default gray for general logs
        ;;
    esac
}

# =====================
# Argument Parsing
# =====================
DATA_DIR=""
while [[ $# -gt 0 ]]; do
    case "$1" in
    --data_dir)
        DATA_DIR="$2"
        shift 2
        ;;
    -h | --help)
        usage
        ;;
    *)
        if [[ -z "$DATA_DIR" ]]; then
            DATA_DIR="$1"
            shift
        else
            log error "Unknown option: $1"
            usage
        fi
        ;;
    esac
done

# =====================
# Validation and Setup
# =====================
if [[ -z "$DATA_DIR" ]]; then
    log error "Data directory not provided."
    usage
fi

# Ensure pip is available
log info "Ensuring pip is installed and updated..."
python -m ensurepip --upgrade

# Install or upgrade Synapse client
log info "Installing/upgrading synapseclient..."
pip install --upgrade synapseclient

# Log into Synapse with error handling
log info "Logging into Synapse..."
if ! synapse login; then
    log error "Synapse login failed. Check your credentials and try again."
    exit 1
fi

# =====================
# Data Files Download
# =====================
# Create and navigate to data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Synapse file IDs and zip filenames
MAPPING_FILE_ID="syn51615438"
VALIDATION_FILE_ID="syn51514110"
TRAINING_FILE_ID="syn51514132"
VALIDATION_ZIP="ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData.zip"
TRAINING_ZIP="ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip"

# Download files with logging
log info "Downloading mapping file..."
synapse get "$MAPPING_FILE_ID"

log info "Downloading validation data..."
synapse get "$VALIDATION_FILE_ID" -o "$VALIDATION_ZIP"

log info "Downloading training data..."
synapse get "$TRAINING_FILE_ID" -o "$TRAINING_ZIP"

# =====================
# Data Files Extraction
# =====================
# Extract validation data if present
log info "Extracting validation data..."
python -c "import zipfile; zipfile.ZipFile('$VALIDATION_ZIP', 'r').extractall('.')"

# Extract training data if present
log info "Extracting training data..."
python -c "import zipfile; zipfile.ZipFile('$TRAINING_ZIP', 'r').extractall('.')"

# Clean up zip files
log info "Cleaning up downloaded zip files..."
rm -f "$DATA_DIR/$VALIDATION_ZIP" "$DATA_DIR/$TRAINING_ZIP"

log info "Download and extraction completed successfully!"
