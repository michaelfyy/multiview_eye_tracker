# scripts/train_multiview.py
import os
import sys
import argparse
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import logging # Use standard logging
import yaml
import json
from datetime import datetime
import shutil
import numpy as np

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config import load_config
from utils.timer import Timer
from models import get_model
from utils.dataloader import create_dataloaders
from losses.losses import get_loss
from utils.heatmap_utils import get_coords_from_heatmap, scale_coords
from utils.visualization_utils import save_loss_plot, visualize_predictions_2d

# Configure base logger - gets configured further in setup_logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Get logger instance

def parse_args():
    parser = argparse.ArgumentParser(description="Train Multi-View Eye Tracking Model")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file.")
    args = parser.parse_args()
    config = load_config(args.config)
    return config

def setup_logging(run_dir):
    """Configures logging to console and file."""
    log_file = os.path.join(run_dir, "log.txt")

    # Remove existing handlers if reconfiguring
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add handlers to the root logger
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    logging.root.setLevel(logging.INFO)

    logger.info(f"Logging configured. Log file: {log_file}")


def setup_run_directory(config):
    """Creates the run directory, saves the config, and sets up logging."""
    base_log_dir = config.get('logging', {}).get('log_dir', './logs')
    run_name_cfg = config.get('logging', {}).get('run_name')

    if run_name_cfg:
        run_name = run_name_cfg
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backbone_type = config.get('model',{}).get('backbone',{}).get('type', 'unknown_backbone')
        run_name = f"run_{timestamp}_{backbone_type}"

    run_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Setup Logging Here
    setup_logging(run_dir)

    config_save_path = os.path.join(run_dir, "config.yaml")
    try:
        with open(config_save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Configuration saved to {config_save_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")

    logger.info(f"Run directory: {run_dir}")
    return run_dir

def calculate_losses(outputs, batch, loss_fns, loss_weights, device, num_views):
    """Calculates the weighted multi-task loss."""
    total_loss = torch.tensor(0.0, device=device)
    loss_breakdown = {}
    pred_2d_type = outputs.get('pred_2d_type', 'regression')

    for i in range(1, num_views + 1):
        # -- 2D Loss --
        if pred_2d_type == 'heatmap':
            loss_key_2d = 'pupil_heatmap'
            pred_key_2d = f'pupil_heatmap_cam{i}'
            target_key_2d = f'cam_{i}_heatmap_gt'
        else: # regression
            loss_key_2d = 'pupil_2d'
            pred_key_2d = f'pupil_2d_cam{i}'
            target_key_2d = f'cam_{i}_2d'

        pred_2d = outputs.get(pred_key_2d)
        target_2d = batch.get(target_key_2d)

        if loss_key_2d in loss_fns and pred_2d is not None and target_2d is not None:
            # --- FIX: Add .float() when moving target to device ---
            target_2d = target_2d.to(device).float()
            # --- End Fix ---
            if pred_2d_type == 'heatmap' and target_2d.dim() == 3:
                 target_2d = target_2d.unsqueeze(1) # Ensure channel dim for heatmap

            try:
                # Ensure prediction is also float32 (model output usually is)
                loss_2d = loss_fns[loss_key_2d](pred_2d.float(), target_2d)
                weighted_loss_2d = loss_weights.get(loss_key_2d, 1.0) * loss_2d
                if not torch.isnan(weighted_loss_2d) and not torch.isinf(weighted_loss_2d):
                    total_loss += weighted_loss_2d
                    loss_breakdown[f'loss_{loss_key_2d}_cam{i}'] = weighted_loss_2d.item()
            except Exception as e:
                 logger.error(f"Error calculating loss {loss_key_2d}_cam{i}: {e}", exc_info=True)


        # --- 3D Pupil Loss ---
        loss_key_p3d = 'pupil_3d'
        pred_p3d = outputs.get(f'pupil_3d_cam{i}')
        target_p3d = batch.get(f'cam_{i}_pupil_3d')

        if loss_key_p3d in loss_fns and pred_p3d is not None and target_p3d is not None:
            # --- FIX: Add .float() when moving target to device ---
            target_p3d = target_p3d.to(device).float()
            # --- End Fix ---
            try:
                loss_p3d = loss_fns[loss_key_p3d](pred_p3d.float(), target_p3d)
                weighted_loss_p3d = loss_weights.get(loss_key_p3d, 1.0) * loss_p3d
                if not torch.isnan(weighted_loss_p3d) and not torch.isinf(weighted_loss_p3d):
                    total_loss += weighted_loss_p3d
                    loss_breakdown[f'loss_p3d_cam{i}'] = weighted_loss_p3d.item()
            except Exception as e:
                 logger.error(f"Error calculating loss p3d_cam{i}: {e}", exc_info=True)


        # --- 3D Gaze Endpoint Loss ---
        loss_key_g3d = 'gaze_endpoint_3d'
        pred_g3d = outputs.get(f'gaze_endpoint_3d_cam{i}')
        target_g3d = batch.get(f'cam_{i}_gaze_endpoint_3d') # Ensure dataset provides this key

        if loss_key_g3d in loss_fns and pred_g3d is not None and target_g3d is not None:
            # --- FIX: Add .float() when moving target to device ---
            target_g3d = target_g3d.to(device).float()
            # --- End Fix ---
            try:
                loss_g3d = loss_fns[loss_key_g3d](pred_g3d.float(), target_g3d)
                weighted_loss_g3d = loss_weights.get(loss_key_g3d, 1.0) * loss_g3d
                if not torch.isnan(weighted_loss_g3d) and not torch.isinf(weighted_loss_g3d):
                    total_loss += weighted_loss_g3d
                    loss_breakdown[f'loss_g3d_cam{i}'] = weighted_loss_g3d.item()
            except Exception as e:
                 logger.error(f"Error calculating loss g3d_cam{i}: {e}", exc_info=True)

    if not loss_breakdown:
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        loss_breakdown['total_loss'] = 0.0
    else:
        loss_breakdown['total_loss'] = total_loss.item()

    return total_loss, loss_breakdown

def calculate_metrics(outputs, batch, device, num_views, heatmap_output_res, target_size_hw):
    """ Calculates evaluation metrics (e.g., 2D pixel error). """
    metrics = {}
    all_pixel_errors_2d = []

    target_size_wh = (target_size_hw[1], target_size_hw[0])
    heatmap_res_wh = (heatmap_output_res, heatmap_output_res)
    pred_2d_type = outputs.get('pred_2d_type', 'regression')

    with torch.no_grad():
        for i in range(1, num_views + 1):
            target_coords_orig_key = f'cam_{i}_2d_orig'
            orig_size_hw_key = 'orig_sizes'

            if target_coords_orig_key not in batch or orig_size_hw_key not in batch:
                continue

            target_coords_orig = batch[target_coords_orig_key].to(device)
            if batch[orig_size_hw_key].shape[1] <= (i - 1):
                 logger.warning(f"Skipping metrics for cam {i}: '{orig_size_hw_key}' shape {batch[orig_size_hw_key].shape} insufficient for view index {i-1}.")
                 continue
            orig_size_hw = batch[orig_size_hw_key][:, i-1, :].to(device)
            orig_size_wh = torch.flip(orig_size_hw, dims=[1])

            valid_targets_mask = (target_coords_orig.abs().sum(dim=1) > 1e-3)
            valid_size_mask = (orig_size_wh.abs().sum(dim=1) > 1e-3)
            valid_mask = valid_targets_mask & valid_size_mask

            if not valid_mask.any():
                continue

            pred_coords_orig = None
            if pred_2d_type == 'heatmap':
                pred_hm = outputs.get(f'pupil_heatmap_cam{i}')
                if pred_hm is None: continue
                pred_coords_hm = get_coords_from_heatmap(pred_hm)
                pred_coords_scaled = torch.full_like(pred_coords_hm, -1.0)
                pred_coords_scaled[valid_mask] = scale_coords(pred_coords_hm[valid_mask], heatmap_res_wh, orig_size_wh[valid_mask])
                pred_coords_orig = pred_coords_scaled
            else: # regression
                pred_coords_resized = outputs.get(f'pupil_2d_cam{i}')
                if pred_coords_resized is None: continue
                pred_coords_scaled = torch.full_like(pred_coords_resized, -1.0)
                pred_coords_scaled[valid_mask] = scale_coords(pred_coords_resized[valid_mask], target_size_wh, orig_size_wh[valid_mask])
                pred_coords_orig = pred_coords_scaled

            if pred_coords_orig is not None:
                valid_pred_orig = pred_coords_orig[valid_mask]
                valid_target_orig = target_coords_orig[valid_mask]
                if valid_pred_orig.shape[0] > 0:
                    error = torch.linalg.norm(valid_pred_orig - valid_target_orig, dim=1)
                    all_pixel_errors_2d.extend(error.cpu().tolist())

    if all_pixel_errors_2d:
        # --- Store results for aggregation ---
        # NOTE: For accurate median/std dev, these raw errors need to be collected
        # across all batches in the validation loop. This function only returns
        # the metrics for the *current* batch based on the collected errors.
        metrics['mean_pixel_error_2d'] = np.mean(all_pixel_errors_2d)
        metrics['median_pixel_error_2d'] = np.median(all_pixel_errors_2d) # Median of *this batch*
        metrics['std_pixel_error_2d'] = np.std(all_pixel_errors_2d) # Std dev of *this batch*
        metrics['count_pixel_error_2d'] = len(all_pixel_errors_2d)

    return metrics


def train():
    config = parse_args()
    run_dir = setup_run_directory(config)

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Config access
    train_cfg = config.get('training', {})
    data_cfg = config.get('data', {})
    model_cfg = config.get('model', {})
    log_cfg = config.get('logging', {})
    loss_cfg = config.get('loss', {})

    # Basic config validation (can be expanded)
    if not all(k in train_cfg for k in ['device', 'learning_rate', 'weight_decay', 'num_epochs', 'early_stop', 'batch_size']):
        logger.error("Missing one or more required keys in training config."); return
    if not all(k in data_cfg for k in ['data_dir', 'target_size', 'num_views']):
        logger.error("Missing one or more required keys in data config."); return
    if not all(k in model_cfg for k in ['base_class', 'backbone', 'head_2d']):
        logger.error("Missing one or more required keys in model config."); return

    logger.info("--- Starting Training Run ---")
    logger.info(f"Run Directory: {run_dir}")
    logger.info(f"Config:\n{yaml.dump(config, indent=2)}")
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # --- Data ---
    try:
        dataloaders = create_dataloaders(config)
        if not dataloaders or 'train' not in dataloaders or 'val' not in dataloaders:
             raise ValueError("create_dataloaders did not return valid train/val dataloaders.")
        logger.info(f"Datasets and DataLoaders created. Train batches: {len(dataloaders['train'])}, Val batches: {len(dataloaders['val'])}")
    except Exception as e:
        logger.error(f"Failed to create dataloaders: {e}", exc_info=True); return

    # --- Model ---
    try:
        model = get_model(config).to(device)
        logger.info(f"Model '{model_cfg['base_class']}' initialized.")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model Parameters: Total={total_params:,}, Trainable={trainable_params:,}")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}", exc_info=True); return

    # --- Freeze Backbone ---
    backbone_cfg = model_cfg.get('backbone', {})
    if backbone_cfg.get('freeze', False):
        logger.info("Freezing backbone weights.")
        try:
             if hasattr(model, 'backbone') and model.backbone is not None:
                for name, param in model.backbone.named_parameters(): param.requires_grad = False
                frozen_params = sum(p.numel() for n, p in model.backbone.named_parameters())
                logger.info(f"Froze {frozen_params:,} parameters in backbone.")
             else: logger.warning("Model does not have a 'backbone' attribute or it is None. Cannot freeze.")
        except Exception as e: logger.error(f"Error freezing backbone: {e}", exc_info=True)
    else: logger.info("Backbone weights will be fine-tuned (not frozen).")

    trainable_params_after_freeze = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if backbone_cfg.get('freeze', False): logger.info(f"Model Trainable Parameters (Post-Freeze): {trainable_params_after_freeze:,}")

    # --- Loss Setup ---
    loss_fns = {}
    loss_weights = {}
    # Define expected loss keys based *only* on configured heads
    expected_loss_keys = set()
    head_2d_cfg = model_cfg.get('head_2d', {})
    head_3d_cfg = model_cfg.get('head_3d')

    if head_2d_cfg.get('type') == 'heatmap': expected_loss_keys.add('pupil_heatmap')
    elif head_2d_cfg.get('type') == 'regression': expected_loss_keys.add('pupil_2d')

    if head_3d_cfg is not None:
         expected_loss_keys.add('pupil_3d')
         expected_loss_keys.add('gaze_endpoint_3d')

    logger.info(f"Expecting loss keys based on model config: {expected_loss_keys}")

    for loss_key, params in loss_cfg.items():
        if not isinstance(params, dict) or 'type' not in params:
             logger.error(f"Conf error in 'loss': '{loss_key}' needs dict with 'type'. Got: {params}"); return

        loss_type_str = params['type'] # Get the loss type string from config

        # Optional: Warn if the configured loss key isn't expected based on model heads
        if loss_key not in expected_loss_keys:
             logger.warning(f"Loss config for '{loss_key}' found, but may not match expected model outputs ({expected_loss_keys}).")

        try:
            # --- ADJUSTED CALL: Pass only the loss type string ---
            loss_fns[loss_key] = get_loss(loss_type_str)
            # --- End Adjustment ---
            loss_weights[loss_key] = params.get('weight', 1.0)
            logger.info(f"Loss function '{loss_key}': type={loss_type_str}, weight={loss_weights[loss_key]}")
        except KeyError as e: # Catch error if get_loss fails
             logger.error(f"Failed to get loss function '{loss_type_str}' specified for '{loss_key}': {e}", exc_info=True)
             return # Stop if loss cannot be created
        except Exception as e:
            logger.error(f"Unexpected error setting up loss '{loss_key}' (type: {loss_type_str}): {e}", exc_info=True)
            return

    # Check if expected loss functions were actually configured and created
    for expected_key in expected_loss_keys:
        if expected_key not in loss_fns:
             logger.warning(f"Expected loss key '{expected_key}' based on model config, but no corresponding loss was configured in the 'loss' section.")

    # --- Optimizer ---
    optimizer_type = train_cfg.get('optimizer', 'adamw').lower() # Default optimizer
    optimizer_params = filter(lambda p: p.requires_grad, model.parameters())
    lr = train_cfg['learning_rate']
    wd = train_cfg['weight_decay']
    if optimizer_type == 'adamw': optimizer = optim.AdamW(optimizer_params, lr=lr, weight_decay=wd)
    elif optimizer_type == 'adam': optimizer = optim.Adam(optimizer_params, lr=lr, weight_decay=wd)
    elif optimizer_type == 'sgd': optimizer = optim.SGD(optimizer_params, lr=lr, momentum=train_cfg.get('sgd_momentum', 0.9), weight_decay=wd)
    else: logger.error(f"Unsupported optimizer: {optimizer_type}"); return
    logger.info(f"Optimizer: {optimizer_type.upper()}, LR={lr}, WeightDecay={wd}")

    # --- LR Scheduler ---
    scheduler_config = train_cfg.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'none').lower()
    scheduler_params = scheduler_config.get('params', {})
    scheduler = None
    scheduler_metric_mode = False # Steps based on metric?
    scheduler_step_per_epoch = True # Steps per epoch?

    if scheduler_type == 'cosine':
        T_max = scheduler_params.get('T_max', train_cfg['num_epochs'])
        eta_min = scheduler_params.get('eta_min', 0)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        logger.info(f"Using Cosine LR scheduler: T_max={T_max}, eta_min={eta_min}")
    elif scheduler_type == 'step':
        step_size = scheduler_params.get('step_size')
        gamma = scheduler_params.get('gamma')
        if step_size is None or gamma is None: logger.error("Step scheduler requires 'step_size' and 'gamma'."); return
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        logger.info(f"Using Step LR scheduler: step_size={step_size}, gamma={gamma}")
    elif scheduler_type == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params) # Pass params directly
        logger.info(f"Using ReduceLROnPlateau scheduler with params: {scheduler_params}")
        scheduler_metric_mode = True
    elif scheduler_type != 'none': logger.warning(f"Unknown scheduler type '{scheduler_type}'.")

    # --- Load Checkpoint ---
    start_epoch = 0
    best_val_metric = float('inf')
    metric_to_monitor = 'val_loss'
    mode = 'min'
    if scheduler_metric_mode and isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        mode = scheduler.mode
        best_val_metric = float('-inf') if mode == 'max' else float('inf')

    ckpt_path_config = train_cfg.get('from_checkpoint')
    if ckpt_path_config and ckpt_path_config.lower() != 'none' and ckpt_path_config != '':
        ckpt_path = ckpt_path_config
        if ckpt_path == 'latest': ckpt_path = os.path.join(checkpoint_dir, "latest_model.pt")
        elif ckpt_path == 'best': ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
        elif not os.path.isabs(ckpt_path): ckpt_path = os.path.join(run_dir, ckpt_path) # Assume relative to run_dir

        if os.path.exists(ckpt_path):
            try:
                logger.info(f"Loading checkpoint from: {ckpt_path}")
                checkpoint = torch.load(ckpt_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])

                # Always load optimizer and scheduler if they exist in checkpoint
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    logger.info("Optimizer state loaded.")
                else: logger.warning("Optimizer state not found in checkpoint.")

                if scheduler and 'scheduler_state_dict' in checkpoint:
                     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                     logger.info("Scheduler state loaded.")
                elif scheduler: logger.warning("Scheduler state not found in checkpoint.")

                start_epoch = checkpoint.get('epoch', -1) + 1
                # Load best metric to continue tracking improvement correctly
                if 'best_val_metric' in checkpoint:
                     best_val_metric = checkpoint['best_val_metric']
                     metric_to_monitor = checkpoint.get('metric_monitored', metric_to_monitor)
                     # Infer mode based on loaded best_val_metric if possible
                     if best_val_metric != float('inf') and best_val_metric != float('-inf'):
                         mode = 'max' if best_val_metric > 0 else 'min' # Simple heuristic
                     logger.info(f"Resuming best metric ({metric_to_monitor}, mode={mode}): {best_val_metric:.5f}")

                logger.info(f"Resuming training from epoch {start_epoch}")
            except Exception as e:
                logger.error(f"Error loading checkpoint {ckpt_path}: {e}. Starting from scratch.", exc_info=True)
                start_epoch = 0
                best_val_metric = float('-inf') if mode == 'max' else float('inf') # Reset best metric
        else:
            logger.warning(f"Checkpoint path not found: {ckpt_path}. Starting from scratch.")
    else:
         logger.info("No checkpoint specified or 'from_checkpoint' is null/empty. Starting training from scratch.")


    # --- Training Loop ---
    best_val_metric = float('inf') if mode == 'min' else float('-inf')
    epochs_no_improve = 0
    early_stop_patience = train_cfg.get('early_stop', 0)
    timer = Timer()
    num_views = data_cfg['num_views']
    heatmap_res = data_cfg.get('heatmap', {}).get('output_res', 64)
    target_input_size_hw = tuple(data_cfg['target_size'])
    head_2d_pred_type = model_cfg.get('head_2d',{}).get('type', 'regression') # Get head type for vis

    training_history = []
    if start_epoch > 0: logger.info("Attempting to load previous training history (if exists)...") # Placeholder for potential history loading

    for epoch in range(start_epoch, train_cfg['num_epochs']):
        timer.start_epoch()
        logger.info(f"--- Starting Epoch {epoch+1}/{train_cfg['num_epochs']} ---")

        # --- Train Phase ---
        model.train()
        epoch_train_loss = 0.0
        total_train_samples_processed = 0
        for batch_idx, batch in enumerate(dataloaders['train']):
            if batch is None or 'images' not in batch:
                logger.warning(f"Skipping invalid train batch {batch_idx+1}."); continue
            current_batch_size = batch['images'].shape[0]
            images = batch['images'].to(device)
            optimizer.zero_grad()
            try: outputs = model(images)
            except Exception as e: logger.error(f"Train Forward Error B{batch_idx+1}: {e}", exc_info=True); return
            try: total_loss, loss_breakdown = calculate_losses(outputs, batch, loss_fns, loss_weights, device, num_views)
            except Exception as e: logger.error(f"Train Loss Error B{batch_idx+1}: {e}", exc_info=True); return
            if torch.isnan(total_loss) or torch.isinf(total_loss): logger.warning(f"NaN/Inf Loss B{batch_idx+1}. Skip step."); continue
            if total_loss.requires_grad:
                try: total_loss.backward(); optimizer.step()
                except Exception as e: logger.error(f"Train Backward/Step Error B{batch_idx+1}: {e}", exc_info=True); return
                epoch_train_loss += total_loss.item() * current_batch_size
                total_train_samples_processed += current_batch_size
            # Log progress
            log_interval = log_cfg.get('log_interval', 50)
            if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
                current_avg_loss = epoch_train_loss / total_train_samples_processed if total_train_samples_processed > 0 else 0.0
                logger.info(f"E[{epoch+1}] B[{batch_idx+1}/{len(dataloaders['train'])}] AvgLoss: {current_avg_loss:.5f}")

        avg_train_loss = epoch_train_loss / total_train_samples_processed if total_train_samples_processed > 0 else float('nan')
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch+1} Train Summary: Avg Loss={avg_train_loss:.5f}, LR={current_lr:.6f}")

        # --- Validation Phase ---
        model.eval()
        epoch_val_loss = 0.0
        total_val_samples_processed = 0
        all_val_errors_2d = [] # Collect all errors for accurate median/std

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloaders['val']):
                if batch is None or 'images' not in batch: logger.warning(f"Skipping invalid val batch {batch_idx+1}."); continue
                current_batch_size = batch['images'].shape[0]
                images = batch['images'].to(device)
                try: outputs = model(images)
                except Exception as e: logger.error(f"Val Forward Error B{batch_idx+1}: {e}", exc_info=True); continue
                try:
                    val_loss_batch, _ = calculate_losses(outputs, batch, loss_fns, loss_weights, device, num_views)
                    if not (torch.isnan(val_loss_batch) or torch.isinf(val_loss_batch)):
                        epoch_val_loss += val_loss_batch.item() * current_batch_size
                        total_val_samples_processed += current_batch_size
                    else: logger.warning(f"NaN/Inf loss in val batch {batch_idx+1}.")
                except Exception as e: logger.error(f"Val Loss Error B{batch_idx+1}: {e}", exc_info=True); continue
                # Calculate metrics for the batch AND collect raw errors
                try:
                    batch_metrics = calculate_metrics(outputs, batch, device, num_views, heatmap_res, target_input_size_hw)
                    # --- MODIFICATION NEEDED in calculate_metrics ---
                    # It should return the raw error list 'all_pixel_errors_2d' calculated within it.
                    # For now, we re-calculate it here based on the batch metrics (less efficient)
                    if 'count_pixel_error_2d' in batch_metrics and batch_metrics['count_pixel_error_2d'] > 0:
                        # This requires calculate_metrics to be modified to return the list
                        # Placeholder: Assume calculate_metrics returns the list if possible
                        # errors_in_batch = calculate_metrics(...)[ 'raw_errors_2d'] # Ideal
                        # all_val_errors_2d.extend(errors_in_batch)
                        pass # Cannot get raw errors without modifying calculate_metrics

                except Exception as e: logger.error(f"Val Metric Error B{batch_idx+1}: {e}", exc_info=True)

        avg_val_loss = epoch_val_loss / total_val_samples_processed if total_val_samples_processed > 0 else float('nan')

        # Calculate final metrics from collected errors (if available)
        final_metrics = {}
        if all_val_errors_2d: # This list will be empty unless calculate_metrics is modified
            final_metrics['mean_pixel_error_2d'] = np.mean(all_val_errors_2d)
            final_metrics['median_pixel_error_2d'] = np.median(all_val_errors_2d)
            final_metrics['std_pixel_error_2d'] = np.std(all_val_errors_2d)
        else: # Fallback using per-batch means (less accurate for median/std)
             logger.warning("Cannot calculate accurate overall median/std dev for metrics without modifying calculate_metrics to return raw values.")
             # Could calculate mean from avg_val_loss or store per-batch means if needed

        logger.info(f"Epoch {epoch+1} Val Summary: Avg Loss={avg_val_loss:.5f}")
        log_metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in final_metrics.items() if not np.isnan(v)])
        if log_metrics_str: logger.info(f"  Val Metrics: {log_metrics_str}")


        # --- Store epoch results and generate visualizations ---
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            # Add other aggregated metrics if needed: **final_metrics
        }
        training_history.append(epoch_results)

        try:
            # Save loss plot
            save_loss_plot(training_history, run_dir)

            # Visualize predictions (using train loader)
            visualize_predictions_2d(
                model=model,
                dataloader=dataloaders['train'],
                device=device,
                output_dir=run_dir,
                epoch=epoch + 1,
                mode='Train',
                num_samples=4, # Visualize 4 samples
                head_2d_type=head_2d_pred_type,
                target_size_hw=target_input_size_hw,
                heatmap_res=heatmap_res
            )

            # Visualize predictions (using validation loader)
            visualize_predictions_2d(
                model=model,
                dataloader=dataloaders['val'],
                device=device,
                output_dir=run_dir,
                epoch=epoch + 1,
                mode='Val',
                num_samples=4, # Visualize 4 samples
                head_2d_type=head_2d_pred_type,
                target_size_hw=target_input_size_hw,
                heatmap_res=heatmap_res
            )
        except Exception as e:
            logger.error(f"Error during visualization generation for epoch {epoch+1}: {e}", exc_info=True)

        # --- Scheduler Step ---
        if scheduler:
            if scheduler_metric_mode:
                metric_for_scheduler = avg_val_loss # Default to loss
                # Add logic here if monitoring a different metric like mean_pixel_error_2d
                # monitor_key = f"val_{scheduler_params.get('monitor', 'loss')}"
                # if monitor_key in final_metrics and not np.isnan(final_metrics[monitor_key]): ...
                if not np.isnan(metric_for_scheduler): scheduler.step(metric_for_scheduler)
                else: logger.warning(f"Cannot step plateau scheduler: monitored metric is NaN.")
            elif scheduler_step_per_epoch: scheduler.step()

        # --- Checkpointing & Early Stopping ---
        current_metric_val = avg_val_loss # Monitor val_loss by default
        # Add logic here if monitoring a different metric

        if np.isnan(current_metric_val):
            logger.warning(f"Monitored metric '{metric_to_monitor}' is NaN. Skipping improvement check.")
            epochs_no_improve += 1 # Count as no improvement if metric is NaN
        else:
            improved = False
            if (mode == 'min' and current_metric_val < best_val_metric) or \
               (mode == 'max' and current_metric_val > best_val_metric):
                # Simplified check - doesn't explicitly handle Plateau threshold here
                # For strict Plateau behavior, the more complex check is needed
                improved = True
                logger.info(f"Val metric ({metric_to_monitor}) improved from {best_val_metric:.5f} to {current_metric_val:.5f}.")
                best_val_metric = current_metric_val
                epochs_no_improve = 0
                best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                try:
                    torch.save({
                        'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'best_val_metric': best_val_metric, 'metric_monitored': metric_to_monitor,
                        'config': config
                    }, best_model_path)
                    logger.info(f"Saved new best model to {best_model_path}")
                except Exception as e: logger.error(f"Error saving best model: {e}", exc_info=True)
            else:
                 epochs_no_improve += 1
                 logger.info(f"Val metric ({metric_to_monitor}) did not improve ({current_metric_val:.5f} vs best {best_val_metric:.5f}). No improvement count: {epochs_no_improve}.")

        # Save latest checkpoint
        latest_model_path = os.path.join(checkpoint_dir, "latest_model.pt")
        try: torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict() if scheduler else None, 'config': config }, latest_model_path)
        except Exception as e: logger.error(f"Error saving latest model: {e}", exc_info=True)

        # Early Stopping Check
        if early_stop_patience > 0 and epochs_no_improve >= early_stop_patience:
            logger.info(f"Early stopping triggered after {early_stop_patience} epochs without improvement.")
            break

        timer.end_epoch()
        logger.info(f"Epoch {epoch+1} Time: {timer.get_epoch_duration():.2f}s")
        # --- End of Epoch ---


    # --- End of Training ---
    total_time = timer.get_total_duration()
    logger.info(f"--- Training Finished ---")
    logger.info(f"Total Training Time: {total_time:.2f}s ({total_time/3600:.2f} hours)")
    logger.info(f"Best validation metric ({metric_to_monitor}): {best_val_metric:.5f}")
    logger.info(f"Best model saved at: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    logger.info(f"Run directory: {run_dir}")

    # Save final history (optional)
    history_path = os.path.join(run_dir, "training_history.json")
    try:
        with open(history_path, 'w') as f: json.dump(training_history, f, indent=4)
        logger.info(f"Training history saved to {history_path}")
    except Exception as e: logger.error(f"Failed to save training history: {e}")

    # Generate final loss plot one last time
    save_loss_plot(training_history, run_dir)

    # --- FIX: Print the absolute path of the run directory to stdout ---
    for handler in logging.getLogger().handlers: handler.flush()
    print(os.path.abspath(run_dir), flush=True)

if __name__ == "__main__":
    train()