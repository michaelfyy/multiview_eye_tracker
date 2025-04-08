#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the initial config file
INITIAL_CONFIG="configs/ue2_multiview_regression.yaml"

# --- Run Training ---
echo "Starting training..."
# Execute training script. Capture its standard output AND show it on terminal (using tee).
# Use process substitution <(...) to capture only the last line into RUN_DIR variable.
# This avoids needing temporary files.
RUN_DIR=$(python scripts/train_multiview.py --config "$INITIAL_CONFIG" | tee /dev/tty | tail -n 1)

# Validate RUN_DIR (basic check: is it non-empty and looks like a path?)
if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "ERROR: Failed to get valid run directory from training script output."
  echo "Output was: $RUN_DIR"
  exit 1
fi

echo "Training finished. Run directory: $RUN_DIR"

# --- Construct path to the config file saved within the run directory ---
RUN_CONFIG_PATH="$RUN_DIR/config.yaml"

# Check if the run-specific config file exists
if [[ ! -f "$RUN_CONFIG_PATH" ]]; then
  echo "ERROR: Config file not found in the run directory: $RUN_CONFIG_PATH"
  exit 1
fi

echo "Run-specific config path: $RUN_CONFIG_PATH"

# --- Run Prediction ---
echo "Starting prediction using the completed training run..."
python scripts/predict_multiview.py --config "$RUN_CONFIG_PATH"
# Add optional arguments to predict if needed, e.g.:
# python scripts/predict_multiview.py --config "$RUN_CONFIG_PATH" --checkpoint best --device cuda

echo "Prediction finished."
echo "Workflow complete."