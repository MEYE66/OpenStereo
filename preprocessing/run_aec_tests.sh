#!/bin/bash

# Auto Exposure Control Test Script
# Tests gradient, mixed, and semantic metrics with pid controller on val dataset

set -e  # Exit on error

# Configuration
SOURCE_ROOT="${HOME}/dataset/ADEC/bbb"
OUTPUT_ROOT="${HOME}/dataset/ADEC/carla_ae"
CONTROLLER="pid"
OPTIMIZER="nelder_mead"
SPLITS="val"
CONDA_ENV="openstereo"
N_WORKERS=8

# Metrics to test
METRICS=("gradient" "mixed" "semantic")

echo "=========================================="
echo "Auto Exposure Control Test Script"
echo "=========================================="
echo "Source: ${SOURCE_ROOT}/${SPLITS}"
echo "Output: ${OUTPUT_ROOT}"
echo "Controller: ${CONTROLLER}"
echo "Optimizer: ${OPTIMIZER}"
echo "Metrics: ${METRICS[@]}"
echo "Workers: ${N_WORKERS}"
echo "=========================================="
echo

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}"

# Run tests for each metric
for metric in "${METRICS[@]}"; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running AEC with metric=${metric}..."
    conda run -n "${CONDA_ENV}" python multi_process_ae.py \
        --source-root "${SOURCE_ROOT}" \
        --output-root "${OUTPUT_ROOT}" \
        --splits ${SPLITS} \
        --controller ${CONTROLLER} \
        --optimizer ${OPTIMIZER} \
        --metric ${metric} \
        --n-workers ${N_WORKERS} \
        --overwrite
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Completed metric=${metric}"
    echo
done

echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
echo "Output directories:"
for metric in "${METRICS[@]}"; do
    echo "  - ${OUTPUT_ROOT}/ae_${CONTROLLER}_${OPTIMIZER}_${metric}_real/"
done
echo "=========================================="
