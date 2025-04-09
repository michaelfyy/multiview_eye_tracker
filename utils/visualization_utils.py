# utils/visualization_utils.py
import os
import logging
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import math
from PIL import Image
import torch.nn.functional as F

# Assuming these utils exist and work as expected
from .heatmap_utils import get_coords_from_heatmap, scale_coords

logger = logging.getLogger(__name__)

# (save_loss_plot function remains the same)
def save_loss_plot(history, output_dir):
    """ Saves a plot of training and validation loss curves. """
    if not history: logger.warning("Cannot save loss plot: history is empty."); return
    epochs=[item['epoch'] for item in history]; train_losses=[item.get('train_loss', float('nan')) for item in history]; val_losses=[item.get('val_loss', float('nan')) for item in history]
    plt.figure(figsize=(10, 6)); plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-'); plt.plot(epochs, val_losses, label='Validation Loss', marker='x', linestyle='--')
    valid_losses=[l for l in train_losses + val_losses if l is not None and not (np.isnan(l) or np.isinf(l))]
    if valid_losses:
        min_loss=min(valid_losses)*0.9; max_loss=max(valid_losses)*1.1
        if max_loss - min_loss < 1e-5: max_loss=min_loss + 1.0
        if max_loss <= 0 : max_loss=1.0
        if min_loss > 0 and min_loss < 1e-5: min_loss=0
        if not (np.isnan(min_loss) or np.isnan(max_loss) or np.isinf(min_loss) or np.isinf(max_loss)): plt.ylim(bottom=max(0, min_loss), top=max_loss)
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training and Validation Loss"); plt.legend(); plt.grid(True, alpha=0.5); plt.tight_layout()
    plot_path = os.path.join(output_dir, "loss_plot.png");
    try: plt.savefig(plot_path); logger.info(f"Loss plot saved to {plot_path}")
    except Exception as e: logger.error(f"Failed to save loss plot: {e}")
    plt.close()


def visualize_predictions_2d(model, dataloader, device, output_dir, epoch, mode='Val', num_samples=4,
                             head_2d_type='regression',
                             target_size_hw=(224, 224), heatmap_res=64):
    """ Generates visualizations, correctly handling axis indexing. """
    model.eval(); batch = None
    try: batch = next(iter(dataloader)); assert batch is not None and 'images' in batch
    except Exception as e: logger.error(f"Error getting batch for viz: {e}"); return

    images = batch['images'].to(device)
    batch_size, num_views = images.shape[0], images.shape[1]
    actual_samples_to_vis = min(num_samples, batch_size)
    if actual_samples_to_vis == 0: return

    with torch.no_grad():
        try: outputs = model(images[:actual_samples_to_vis])
        except Exception as e: logger.error(f"Viz inference error: {e}", exc_info=True); return

    target_size_wh = (target_size_hw[1], target_size_hw[0])
    heatmap_res_wh = (heatmap_res, heatmap_res)

    cols_per_view = 2 if head_2d_type == 'heatmap' else 1
    total_cols = num_views * cols_per_view
    fig, axes = plt.subplots(actual_samples_to_vis, total_cols, figsize=(total_cols * 3, actual_samples_to_vis * 3.5), squeeze=False)

    for i in range(actual_samples_to_vis): # Iterate samples
        for j in range(num_views): # Iterate views
            cam_id = j + 1
            img_ax_idx = j * cols_per_view # Index for image+points axis
            # --- FIX: Define hm_ax_idx correctly within the scope needed ---
            # Calculate heatmap axis index IF needed
            if head_2d_type == 'heatmap':
                hm_ax_idx = img_ax_idx + 1
            # --- End Fix ---
            img_ax = axes[i, img_ax_idx]

            # Get Image & Denormalize
            img_tensor = images[i, j].cpu().permute(1, 2, 0)
            mean=torch.tensor([0.485, 0.456, 0.406]); std=torch.tensor([0.229, 0.224, 0.225])
            img_vis_rgb = torch.clamp(img_tensor * std + mean, 0, 1).numpy()
            img_to_draw_on = cv2.cvtColor((img_vis_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            # Get GT/Pred Coords (in target_size_hw pixel space)
            gt_point_vis = None; pred_point_vis = None; heatmap_to_show = None

            # GT Coords
            gt_key = f'cam_{cam_id}_2d'
            gt_coords_resized_tensor = batch.get(gt_key)
            if gt_coords_resized_tensor is not None:
                 gt_coords_resized = gt_coords_resized_tensor[i].cpu().numpy()
                 if not np.isnan(gt_coords_resized).any() and np.abs(gt_coords_resized).sum() > 1e-3:
                      gt_point_vis = tuple(gt_coords_resized)

            # Predicted Coords
            if head_2d_type == 'heatmap':
                hm_key = f'pupil_heatmap_cam{cam_id}'; pred_hm_logits = outputs.get(hm_key)
                if pred_hm_logits is not None:
                    pred_hm_logits = pred_hm_logits[i]
                    heatmap_to_show = torch.sigmoid(pred_hm_logits).squeeze().cpu().numpy()
                    pred_coords_hm = get_coords_from_heatmap(pred_hm_logits).cpu()
                    if pred_coords_hm is not None and pred_coords_hm.numel() > 0:
                         pred_point_vis_tensor = scale_coords(pred_coords_hm.squeeze(0), heatmap_res_wh, target_size_wh)
                         if pred_point_vis_tensor is not None: # Check scale_coords result
                            pred_point_vis = tuple(pred_point_vis_tensor.numpy())

            else: # Regression
                reg_key = f'pupil_2d_cam{cam_id}'; pred_coords_resized = outputs.get(reg_key)
                if pred_coords_resized is not None:
                    pred_coords_resized_np = pred_coords_resized[i].cpu().numpy()
                    if not np.isnan(pred_coords_resized_np).any():
                         pred_point_vis = tuple(pred_coords_resized_np)


            # --- Visualization: Draw on BGR image with OpenCV ---
            # Draw GT point (lime green circle - thicker)
            if gt_point_vis is not None and not np.isnan(gt_point_vis).any():
                try:
                    pt_x = int(round(np.clip(gt_point_vis[0], 0, target_size_wh[0]-1)))
                    pt_y = int(round(np.clip(gt_point_vis[1], 0, target_size_wh[1]-1)))
                    cv2.circle(img_to_draw_on, (pt_x, pt_y), radius=7, color=(0, 255, 0), thickness=2)
                except Exception as draw_e: logger.warning(f"Error drawing GT {cam_id}: {draw_e}")

            # Draw Predicted point (bright red cross - thicker)
            if pred_point_vis is not None and not np.isnan(pred_point_vis).any():
                try:
                    pt_x = int(round(np.clip(pred_point_vis[0], 0, target_size_wh[0]-1)))
                    pt_y = int(round(np.clip(pred_point_vis[1], 0, target_size_wh[1]-1)))
                    cv2.drawMarker(img_to_draw_on, (pt_x, pt_y), color=(0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2)
                except Exception as draw_e: logger.warning(f"Error drawing Pred {cam_id}: {draw_e}")

            # Display final image using matplotlib axis
            img_ax.imshow(cv2.cvtColor(img_to_draw_on, cv2.COLOR_BGR2RGB))
            img_ax.set_title(f"Sample:{i+1} Cam:{cam_id}")
            img_ax.axis('off')

            # Visualize heatmap on separate axis if applicable
            if head_2d_type == 'heatmap':
                 # --- FIX: Use hm_ax_idx defined earlier ---
                 hm_ax = axes[i, hm_ax_idx]
                 # --- End Fix ---
                 if heatmap_to_show is not None:
                      im = hm_ax.imshow(heatmap_to_show, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
                      hm_ax.set_title(f"S:{i+1} C:{cam_id} Heatmap")
                 else: hm_ax.text(0.5, 0.5, 'No Heatmap', ha='center', va='center')
                 hm_ax.axis('off')

    fig.suptitle(f"{mode} Epoch {epoch} - 2D Predictions (GT Lime O, Pred Red X)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    vis_path = os.path.join(output_dir, f"{mode.lower()}_epoch_{epoch}_predictions_2d.png")
    try: plt.savefig(vis_path); logger.info(f"Prediction visualization saved to {vis_path}")
    except Exception as e: logger.error(f"Failed to save prediction visualization: {e}")
    plt.close(fig)