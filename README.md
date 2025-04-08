# Multi-View Eye Tracking with Deep Learning

This project implements a deep learning pipeline for predicting 2D pupil center, 3D pupil center, and 3D gaze endpoint using synchronized input from multiple cameras (typically 4 endoscope cameras). It leverages state-of-the-art Vision Transformer (ViT) backbones and configurable components for feature fusion and prediction heads.

## Features

- **Multi-View Input:** Processes synchronized frames from 4 cameras.
- **Configurable Backbones:** Supports various backbones like:
  - `EfficientNet` (e.g., `efficientnet_b0`, via Timm)
  - `MobileViT` (e.g., `mobilevit_s`, via Timm)
  - `EfficientViT` (e.g., `efficientvit_m0`, via Timm)
  - `Swin Transformer` (e.g., `swin_tiny_patch4_window7_224`, via Timm)
  - `DINOv2` (e.g., `dinov2_vits14`, via Torch Hub)
  - Easily extendable to other backbones available in `timm` or custom implementations.
- **Configurable Fusion:** Includes options for combining features from multiple views:
  - `none`: No explicit fusion (features concatenated).
  - `concat`: Simple concatenation.
  - `attention`: Multi-head self-attention fusion using Transformer Encoder layers.
- **Configurable 2D Head:** Predicts 2D pupil center using:
  - `regression`: Direct MLP regression to coordinates.
  - `heatmap`: Predicts a 2D Gaussian heatmap, often more accurate for localization.
- **3D Prediction:** Predicts 3D pupil center (iris center) and 3D gaze endpoint (derived from pupil center and gaze vector) using an MLP head.
- **Training:** Includes training script with support for:
  - Multi-task loss balancing.
  - Learning Rate Schedulers (`CosineAnnealingLR`, `StepLR`, `ReduceLROnPlateau`).
  - Checkpointing (saving best/latest models).
  - Resuming training from checkpoints.
  - Early stopping.
- **Prediction Pipeline:** Script to run inference on multi-view videos:
  - Loads 4 synchronized videos.
  - Processes frames and runs model inference.
  - Optionally loads 2D ground truth annotations (e.g., from CVAT XML) for comparison.
  - Saves annotated videos showing 2D pupil predictions vs. ground truth.
  - Generates plots for per-frame pixel error and error histograms.
  - Saves raw frame-by-frame predictions (2D & 3D) to a JSON file.
  - Saves overall evaluation metrics (timing, average errors) to a JSON file.

## Project Structure

```plaintext
.
├── configs/                # YAML configuration files for experiments
│   └── ue2_multiview.yaml
├── datasets/               # Dataset classes and transforms
│   ├── ue2_multiview.py
│   └── transforms.py
├── dataloaders/            # Dataloader creation logic
│   └── dataloader.py
├── models/                 # Model definitions (backbones, heads, fusion, base)
│   ├── __init__.py
│   ├── backbones.py
│   ├── base_multiview_model.py
│   ├── fusion_modules.py
│   └── heads.py
├── scripts/                # Training and prediction scripts
│   ├── train_multiview.py
│   └── predict_multiview.py
├── utils/                  # Utility functions (config, logger, heatmap, video, timer)
│   ├── config.py
│   ├── heatmap_utils.py
│   ├── logger.py
│   ├── timer.py
│   └── video_helper.py
├── losses/                 # Loss function implementations
│   └── losses.py
├── metrics/                # Metric calculation functions
│   └── metrics.py
├── runs/                   # Default output directory for training runs (logs, checkpoints)
├── predictions/            # Default output directory for prediction runs (videos, plots, jsons)
├── run.sh                  # Linux/macOS run script
├── run.bat                 # Windows run script
└── requirements.txt        # Python dependencies
```

## Usage

```bash
python scripts/train_multiview.py --config configs/ue2_multiview.yaml
python scripts/predict_multiview.py --config runs/YOUR_RUN_NAME/config.yaml
```
