#!/bin/bash

# ===================================
# Data Download and Extraction Script
# ===================================

set -euo pipefail

# Function to display usage information
usage() {
    cat <<EOF
Usage: $0 [--data_dir <data_directory>] [-h|--help]

Downloads and extracts the ISLES'22 data file.

Options:
  --data_dir    Path to data directory [required]
  -h, --help    Show this help message

Prerequisites:
  - unzip

Examples:
  # Specify data directory as a positional argument:
  $0 <data_directory>

  # Specify data directory with the --data_dir option:
  $0 --data_dir <data_directory>

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

# Ensure unzip is available
log info "Check if unzip is installed..."

if ! command -v unzip > /dev/null 2>&1; then
    echo "unzip is not installed. Please install it first"
    exit 1
fi

# =====================
# Data Files Download
# =====================
# Create and navigate to data directory
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Download files with logging
log info "Downloading data files..."
wget "https://zenodo.org/records/7960856/files/ISLES-2022.zip"

# =====================
# Data Files Extraction
# =====================
log info "Extracting data files..."
unzip "ISLES-2022.zip"
mv "ISLES-2022" "ISLES22"

# Clean up zip files
log info "Cleaning up downloaded zip file..."
rm "ISLES-2022.zip"

log info "Download and extraction completed successfully!"