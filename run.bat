@echo off
setlocal

REM Define the initial config file
set INITIAL_CONFIG=configs\ue2_multiview_regression.yaml

REM --- Run Training ---
echo Starting training...

REM Execute python script and capture the last line of output into RUN_DIR
REM The 'for /f' loop iterates over the output lines; the variable gets overwritten
REM so the last value assigned is the last line printed by the script.
for /f "delims=" %%a in ('python scripts/train_multiview.py --config "%INITIAL_CONFIG%"') do (
    set "RUN_DIR=%%a"
)

REM Basic validation: Check if RUN_DIR was set
if not defined RUN_DIR (
    echo ERROR: Failed to get run directory from training script output.
    exit /b 1
)

REM Optional: Check if the captured path actually exists as a directory
if not exist "%RUN_DIR%\" (
    echo ERROR: Training output path "%RUN_DIR%" is not a valid directory.
    exit /b 1
)

echo Training finished. Run directory: %RUN_DIR%

REM --- Construct path to the config file saved within the run directory ---
set "RUN_CONFIG_PATH=%RUN_DIR%\config.yaml"

REM Check if the run-specific config file exists
if not exist "%RUN_CONFIG_PATH%" (
  echo ERROR: Config file not found in the run directory: %RUN_CONFIG_PATH%
  exit /b 1
)

echo Run-specific config path: %RUN_CONFIG_PATH%

REM --- Run Prediction ---
echo Starting prediction using the completed training run...
python scripts/predict_multiview.py --config "%RUN_CONFIG_PATH%"
REM Add optional arguments to predict if needed, e.g.:
REM python scripts/predict_multiview.py --config "%RUN_CONFIG_PATH%" --checkpoint best --device cuda

REM Check for errors during prediction (optional)
if errorlevel 1 (
    echo ERROR: Prediction script failed.
    exit /b 1
)

echo Prediction finished.
echo Workflow complete.

endlocal
exit /b 0