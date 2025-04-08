# scripts/predict_multiview.py
import os
import sys
import time
import json
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import glob
import argparse
import logging
import yaml
from PIL import Image

# --- Add project root to sys.path ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.config import load_config
from models import get_model
from utils.video_helper import extract_frames, parse_annotations
from datasets.transforms import get_transform
from utils.heatmap_utils import get_coords_from_heatmap, scale_coords

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_predict_args():
    """ Parses arguments specifically for prediction. """
    parser = argparse.ArgumentParser(description="Run Multi-View Eye Tracking Prediction on Videos")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration file used for training AND prediction.")
    parser.add_argument('--checkpoint', type=str, default=None, help="Optional: Path to checkpoint or 'best'/'latest' to override config.")
    parser.add_argument('--device', type=str, default=None, help="Optional: Device ('cuda', 'cpu') to override config.")
    args = parser.parse_args()
    return args

# --- get_camera_id_from_video_filename, load_multiview_videos, preprocess_multiview_frame, annotate_frame remain the same ---
# (Copy from previous response if needed)
def get_camera_id_from_video_filename(filename):
    base = os.path.splitext(os.path.basename(filename))[0].lower()
    if base.startswith("e") and base[1:].isdigit(): return f"cam_{base[1:]}"
    if base.startswith("cam"):
        num_part = base.replace("cam", "").replace("_", "")
        if num_part.isdigit(): return f"cam_{num_part}"
    logger.warning(f"Could not determine camera ID from filename: {filename}")
    return None

def load_multiview_videos(video_root, num_views=4):
    supported_exts = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_files = []
    for ext in supported_exts: video_files.extend(glob.glob(os.path.join(video_root, ext)))
    video_paths = {}
    for vf in video_files:
        cam_id = get_camera_id_from_video_filename(vf)
        if cam_id:
            if cam_id in video_paths: logger.warning(f"Duplicate video for {cam_id}. Using {vf}")
            video_paths[cam_id] = vf
    expected_cams = {f"cam_{i}" for i in range(1, num_views + 1)}
    found_cams = set(video_paths.keys())
    if found_cams != expected_cams:
        missing = expected_cams - found_cams; extra = found_cams - expected_cams
        error_msg = f"Video mismatch: Expected {num_views} views ({expected_cams}). "
        if missing: error_msg += f"Missing: {missing}. "
        if extra: error_msg += f"Found unexpected: {extra}. "
        error_msg += f"Check files in {video_root}"
        logger.error(error_msg)
        if len(found_cams) == 0: raise ValueError("No valid video files found.")
    logger.info(f"Found video paths: {video_paths}")
    return video_paths

def preprocess_multiview_frame(frames_rgb, transform):
    processed_views = []; orig_sizes = []
    for frame_rgb in frames_rgb:
        if frame_rgb is None: raise ValueError("Encountered None frame.")
        orig_h, orig_w, _ = frame_rgb.shape; orig_sizes.append((orig_h, orig_w))
        try:
            img_pil = Image.fromarray(frame_rgb); processed_frame = transform(img_pil)
            processed_views.append(processed_frame)
        except Exception as e: logger.error(f"Transform error: {e}"); raise
    if not processed_views: return None, []
    return torch.stack(processed_views, dim=0), orig_sizes

def annotate_frame(frame_rgb, gt_point, pred_point, gt_color=(0, 255, 0), pred_color=(0, 0, 255)):
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR); radius = 5; thickness = -1
    if gt_point is not None and not any(np.isnan(p) for p in gt_point):
        try: gt_point_int = (int(round(gt_point[0])), int(round(gt_point[1]))); cv2.circle(frame_bgr, gt_point_int, radius, gt_color, thickness)
        except Exception as e: logger.warning(f"Could not draw GT point {gt_point}: {e}")
    if pred_point is not None and not any(np.isnan(p) for p in pred_point):
        try: pred_point_int = (int(round(pred_point[0])), int(round(pred_point[1]))); cv2.drawMarker(frame_bgr, pred_point_int, pred_color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
        except Exception as e: logger.warning(f"Could not draw Pred point {pred_point}: {e}")
    return frame_bgr


def find_checkpoint(config, checkpoint_arg, config_path):
    """ Finds the checkpoint path based on arg ('best', 'latest', specific path, or from config)."""
    ckpt_to_load = checkpoint_arg # CLI arg takes precedence
    if not ckpt_to_load:
        # If no CLI arg, use the path from config file
        ckpt_to_load = config.get('prediction', {}).get('checkpoint', 'best') # Default to 'best' if not in config

    if ckpt_to_load and ckpt_to_load not in ['best', 'latest']:
        if os.path.exists(ckpt_to_load):
            return ckpt_to_load # Absolute or relative path provided and exists
        else:
            # Try interpreting relative to config file's run dir
            run_dir_base = os.path.dirname(config_path) # Dir where config is
            potential_path = os.path.join(run_dir_base, "checkpoints", ckpt_to_load)
            if os.path.exists(potential_path):
                return potential_path
            else:
                logger.error(f"Specified checkpoint path not found: {checkpoint_arg} or {potential_path}")
                return None

    # Resolve 'best' or 'latest' relative to the run directory containing the config
    run_dir_base = os.path.dirname(config_path)
    if not os.path.isdir(run_dir_base):
        logger.error(f"Cannot resolve '{ckpt_to_load}'. Directory of config file not found: {run_dir_base}")
        return None

    checkpoint_subdir = os.path.join(run_dir_base, "checkpoints")
    if not os.path.isdir(checkpoint_subdir):
        logger.error(f"Checkpoints subdirectory not found in run directory: {checkpoint_subdir}")
        return None

    ckpt_filename = f"{ckpt_to_load}_model.pt"
    potential_path = os.path.join(checkpoint_subdir, ckpt_filename)

    if os.path.exists(potential_path):
        return potential_path
    else:
        logger.error(f"Checkpoint '{ckpt_filename}' not found in {checkpoint_subdir}")
        return None


def predict_and_evaluate():
    args = parse_predict_args()
    try:
        # Load config first to get base settings
        config = load_config(args.config)
        pred_cfg = config.get('prediction', {}) # Get prediction specific settings
        if not pred_cfg:
            logger.error("`prediction` section not found in config file.")
            return
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        return
    except Exception as e:
        logger.error(f"Error loading configuration file {args.config}: {e}")
        return

    # --- Setup Output Directory ---
    output_dir_cfg = pred_cfg.get('output_dir')
    if output_dir_cfg and os.path.isabs(output_dir_cfg):
         predictions_output_root = output_dir_cfg
    elif output_dir_cfg: # Relative path specified in config
         # Make it relative to the config file's directory
         predictions_output_root = os.path.abspath(os.path.join(os.path.dirname(args.config), output_dir_cfg))
    else:
        # Default to predictions subdir within the run directory containing the config
        run_dir_base = os.path.dirname(args.config)
        predictions_output_root = os.path.join(run_dir_base, "predictions", f"predict_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    try:
        os.makedirs(predictions_output_root, exist_ok=True)
        log_file = os.path.join(predictions_output_root, "predict_log.txt")
        # Setup logger to file AND console
        log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        log_handler_file = logging.FileHandler(log_file)
        log_handler_file.setFormatter(log_formatter)
        log_handler_console = logging.StreamHandler(sys.stdout)
        log_handler_console.setFormatter(log_formatter)
        # Get root logger and add handlers
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO) # Set root level
        # Remove default handlers if any to avoid duplicate messages
        for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
        root_logger.addHandler(log_handler_file)
        root_logger.addHandler(log_handler_console)

    except Exception as e:
        logging.error(f"Failed to setup output directory/logger at {predictions_output_root}: {e}. Exiting.")
        return

    logger.info("--- Starting Prediction Run ---")
    logger.info(f"Using config: {args.config}")
    logger.info(f"Output directory: {predictions_output_root}")

    # --- Device Setup ---
    if args.device: # CLI override
        device_str = args.device
    elif pred_cfg.get('device'): # Config override
        device_str = pred_cfg['device']
    else: # Fallback to training device or default
        device_str = config.get('training', {}).get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    try:
        device = torch.device(device_str)
        logger.info(f"Using device: {device}")
        # Test CUDA if selected
        if device.type == 'cuda':
             if not torch.cuda.is_available(): logger.warning("CUDA selected but not available! Falling back to CPU.")
             else: logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(device)}")
    except Exception as e:
         logger.error(f"Invalid device specified ('{device_str}'): {e}. Using CPU.")
         device = torch.device('cpu')


    # --- Load Model ---
    try:
        model = get_model(config).to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}", exc_info=True)
        return

    # --- Load Checkpoint ---
    checkpoint_path = find_checkpoint(config, args.checkpoint, args.config) # Pass config path for context
    if not checkpoint_path: return

    try:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint: model.load_state_dict(checkpoint['model_state_dict'])
        else: model.load_state_dict(checkpoint)
        logger.info("Model weights loaded successfully.")
        ckpt_epoch = checkpoint.get('epoch', 'N/A'); ckpt_loss = checkpoint.get('val_loss', 'N/A')
        logger.info(f"Checkpoint Epoch: {ckpt_epoch}, Val Loss: {ckpt_loss}")
    except Exception as e:
        logger.error(f"Error loading checkpoint weights from {checkpoint_path}: {e}", exc_info=True)
        return

    # --- Load Videos ---
    video_root = pred_cfg.get('video_root')
    if not video_root or not os.path.isdir(video_root):
        logger.error(f"Invalid or missing 'prediction.video_root' in config: {video_root}")
        return
    num_views = config.get('data', {}).get('num_views', 4)
    try:
        video_paths = load_multiview_videos(video_root, num_views)
        sorted_cams = sorted(video_paths.keys())
        logger.info(f"Processing videos for cameras: {sorted_cams}")
        if len(video_paths) != num_views:
             logger.warning(f"Found {len(video_paths)} videos, but config expects {num_views}. Proceeding with found cameras.")
             # num_views = len(video_paths) # Decide if num_views should be adapted dynamically
    except Exception as e:
        logger.error(f"Error loading videos from {video_root}: {e}", exc_info=True)
        return

    # --- Load Annotations ---
    annotations = {}
    annotations_folder = pred_cfg.get('annotations_folder', 'annotations') # Default subfolder
    annotations_base_dir = os.path.join(video_root, annotations_folder)
    logger.info(f"Looking for annotations in: {annotations_base_dir}")
    for cam in sorted_cams:
        video_file = video_paths.get(cam) # Use .get for safety if num_views mismatch
        if not video_file: continue
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        annot_file = os.path.join(annotations_base_dir, f"{video_name}_annotations.xml")
        if os.path.exists(annot_file):
            try:
                annotations[cam] = parse_annotations(annot_file)
                logger.info(f"Loaded {len(annotations[cam])} annotated frames for {cam} from {annot_file}")
            except Exception as e:
                logger.error(f"Error parsing annotation file {annot_file}: {e}", exc_info=True)
                annotations[cam] = {}
        else:
            annotations[cam] = {}
            logger.info(f"No annotation file found for {cam} at {annot_file}")

    # --- Extract Frames ---
    logger.info("Extracting frames from videos...")
    frames_dict_rgb = {}
    min_frames = float('inf')
    for cam in sorted_cams:
        if cam not in video_paths: continue
        try:
            frames_dict_rgb[cam] = extract_frames(video_paths[cam])
            if not frames_dict_rgb[cam]:
                 logger.error(f"No frames extracted for {cam}. Cannot proceed.")
                 return
            min_frames = min(min_frames, len(frames_dict_rgb[cam]))
            logger.info(f"Extracted {len(frames_dict_rgb[cam])} frames for {cam}.")
        except Exception as e:
             logger.error(f"Error extracting frames for {cam} from {video_paths[cam]}: {e}", exc_info=True)
             return
    num_frames = int(min_frames)
    if num_frames == 0 or num_frames == float('inf'):
         logger.error("No frames extracted or inconsistent frame counts.")
         return
    logger.info(f"Processing {num_frames} synchronized frames per camera.")

    # --- Prepare Transforms ---
    target_size = tuple(config['data']['target_size'])
    heatmap_res = config['data'].get('heatmap', {}).get('output_res', 64)
    prediction_transform = get_transform(target_size=target_size, is_train=False)

    # --- Run Inference ---
    logger.info("Starting inference...")
    predictions_all_frames = []
    inference_times = []

    for frame_idx in range(num_frames):
        start_time = time.monotonic()
        if any(frame_idx >= len(frames_dict_rgb.get(cam, [])) for cam in sorted_cams):
             logger.warning(f"Frame index {frame_idx} out of bounds. Stopping early.")
             num_frames = frame_idx
             break
        current_frames_rgb = [frames_dict_rgb[cam][frame_idx] for cam in sorted_cams if cam in frames_dict_rgb]
        if len(current_frames_rgb) != len(sorted_cams): # Check if all cameras provided a frame
             logger.warning(f"Mismatch in frame availability at index {frame_idx}. Skipping.")
             predictions_all_frames.append(None)
             continue

        try:
            input_tensor, orig_sizes_hw = preprocess_multiview_frame(current_frames_rgb, prediction_transform)
            if input_tensor is None: raise ValueError("Preprocessing returned None.")
            input_batch = input_tensor.unsqueeze(0).to(device)
        except Exception as e:
             logger.error(f"Error preprocessing frame {frame_idx}: {e}")
             predictions_all_frames.append(None)
             continue

        with torch.no_grad():
            try: outputs = model(input_batch)
            except Exception as e:
                 logger.error(f"Inference error frame {frame_idx}: {e}", exc_info=True)
                 predictions_all_frames.append(None); continue
        inference_times.append(time.monotonic() - start_time)

        # Post-process (same logic as before)
        frame_predictions = {"frame_index": frame_idx}
        pred_2d_type = outputs.get('pred_2d_type', 'regression')
        for i, cam in enumerate(sorted_cams):
            cam_preds = {}; orig_h, orig_w = orig_sizes_hw[i]; orig_size_wh = (orig_w, orig_h)
            pred_coords_orig = np.array([np.nan, np.nan])
            try:
                if pred_2d_type == 'heatmap':
                    pred_hm = outputs.get(f'pupil_heatmap_cam{i+1}')
                    if pred_hm is not None:
                         pred_coords_hm = get_coords_from_heatmap(pred_hm.squeeze(0)).cpu()
                         heatmap_res_wh = (heatmap_res, heatmap_res)
                         pred_coords_orig = scale_coords(pred_coords_hm, heatmap_res_wh, orig_size_wh).numpy()
                else: # regression
                    pred_coords_resized = outputs.get(f'pupil_2d_cam{i+1}')
                    if pred_coords_resized is not None:
                         pred_coords_resized = pred_coords_resized.squeeze(0).cpu()
                         target_size_wh = (target_size[1], target_size[0])
                         pred_coords_orig = scale_coords(pred_coords_resized, target_size_wh, orig_size_wh).numpy()
                cam_preds['pupil_2d_orig'] = pred_coords_orig.tolist()
            except Exception as e: logger.warning(f"Error post-proc 2D frame {frame_idx}, cam {cam}: {e}"); cam_preds['pupil_2d_orig'] = [np.nan, np.nan]
            try:
                 pupil_3d = outputs.get(f'pupil_3d_cam{i+1}'); gaze_ep_3d = outputs.get(f'gaze_endpoint_3d_cam{i+1}')
                 cam_preds['pupil_3d'] = pupil_3d.squeeze(0).cpu().tolist() if pupil_3d is not None else [np.nan]*3
                 cam_preds['gaze_endpoint_3d'] = gaze_ep_3d.squeeze(0).cpu().tolist() if gaze_ep_3d is not None else [np.nan]*3
            except Exception as e: logger.warning(f"Error post-proc 3D frame {frame_idx}, cam {cam}: {e}"); cam_preds['pupil_3d'] = [np.nan]*3; cam_preds['gaze_endpoint_3d'] = [np.nan]*3
            frame_predictions[cam] = cam_preds
        predictions_all_frames.append(frame_predictions)
        if (frame_idx + 1) % 50 == 0: logger.info(f"Processed frame {frame_idx + 1}/{num_frames}")

    # --- Analysis & Saving Results ---
    if not inference_times: logger.error("No frames processed."); return
    total_inference_time = sum(inference_times)
    avg_time_per_frame = total_inference_time / len(inference_times)
    logger.info(f"Inference complete. Total time: {total_inference_time:.2f}s for {len(inference_times)} frames. Avg time/frame: {avg_time_per_frame*1000:.2f}ms ({1/avg_time_per_frame if avg_time_per_frame > 0 else 0:.1f} FPS)")

    raw_predictions_path = os.path.join(predictions_output_root, "predictions_per_frame.json")
    valid_predictions = [p for p in predictions_all_frames if p is not None]
    try:
        with open(raw_predictions_path, "w") as f: json.dump(valid_predictions, f, indent=4)
        logger.info(f"Raw predictions saved: {raw_predictions_path}")
    except Exception as e: logger.error(f"Error saving raw predictions: {e}")

    logger.info("Annotating videos and calculating metrics...")
    annotated_frames_all_cams = {cam: [] for cam in sorted_cams}
    per_frame_errors_all_cams = {cam: [] for cam in sorted_cams}
    overall_metrics = {
        "config_file": args.config, "checkpoint": checkpoint_path, "num_cams_processed": len(sorted_cams),
        "total_frames_processed": len(valid_predictions), "total_inference_time_sec": total_inference_time,
        "avg_inference_time_ms": avg_time_per_frame * 1000, "per_camera": {}
    }
    num_processed_frames = len(valid_predictions)

    for frame_idx in range(num_processed_frames):
        frame_preds = valid_predictions[frame_idx]
        if frame_preds is None: continue
        for i, cam in enumerate(sorted_cams):
             if frame_idx >= len(frames_dict_rgb.get(cam, [])): continue # Check bounds
             orig_frame_rgb = frames_dict_rgb[cam][frame_idx]
             pred_data = frame_preds.get(cam, {}); pred_2d_orig = np.array(pred_data.get('pupil_2d_orig', [np.nan, np.nan]))
             gt_point_orig = None; frame_annotations = annotations.get(cam, {}).get(frame_idx)
             if frame_annotations:
                 try: gt_point_orig = frame_annotations[0][1]
                 except IndexError: logger.warning(f"Annotation format unexpected frame {frame_idx}, cam {cam}")
             error = np.nan
             if gt_point_orig is not None and not np.isnan(pred_2d_orig).any():
                 try: error = np.linalg.norm(pred_2d_orig - np.array(gt_point_orig))
                 except Exception as e: logger.warning(f"Error calculating distance frame {frame_idx}, cam {cam}: {e}")
             per_frame_errors_all_cams[cam].append(error)
             try: annotated_frame_bgr = annotate_frame(orig_frame_rgb, gt_point_orig, pred_2d_orig); annotated_frames_all_cams[cam].append(annotated_frame_bgr)
             except Exception as e: logger.error(f"Error annotating frame {frame_idx}, cam {cam}: {e}"); annotated_frames_all_cams[cam].append(cv2.cvtColor(orig_frame_rgb, cv2.COLOR_RGB2BGR))

    frame_rate = pred_cfg.get('frame_rate', 30) # Get frame rate from config
    for cam in sorted_cams:
        # --- Save Annotated Video ---
        video_out_path = os.path.join(predictions_output_root, f"{cam}_annotated.mp4"); frames_to_write = annotated_frames_all_cams.get(cam, [])
        if frames_to_write:
            first_valid_frame = next((f for f in frames_to_write if f is not None and f.ndim == 3), None)
            if first_valid_frame is not None:
                height, width, _ = first_valid_frame.shape; fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                try:
                    out_video = cv2.VideoWriter(video_out_path, fourcc, frame_rate, (width, height))
                    for frame_bgr in frames_to_write:
                        if frame_bgr is not None and frame_bgr.ndim == 3 and frame_bgr.shape[0] == height and frame_bgr.shape[1] == width: out_video.write(frame_bgr)
                        else: logger.warning(f"Skipping invalid frame during video write for {cam}")
                    out_video.release(); logger.info(f"Annotated video saved: {video_out_path}")
                except Exception as e: logger.error(f"Error writing video {cam}: {e}")
            else: logger.warning(f"No valid frames to write video for {cam}")

        # --- Metrics & Plots ---
        valid_errors = [e for e in per_frame_errors_all_cams.get(cam, []) if not np.isnan(e)]
        cam_metrics = {"num_annotated_frames": len(valid_errors)}; mean_err, median_err, std_err = None, None, None
        if valid_errors:
            mean_err = float(np.mean(valid_errors)); median_err = float(np.median(valid_errors)); std_err = float(np.std(valid_errors))
            cam_metrics.update({"mean_pixel_error": mean_err, "median_pixel_error": median_err, "std_pixel_error": std_err})
            logger.info(f"Metrics {cam}: Mean Px Err={mean_err:.2f}, Median Px Err={median_err:.2f}")
            try: # Plot timeline
                plt.figure(figsize=(12, 5)); plt.plot(range(num_processed_frames), per_frame_errors_all_cams.get(cam, []), marker='.', linestyle='-', label=f"{cam} Px Error"); plt.xlabel("Frame"); plt.ylabel("Px Error"); plt.title(f"Per-Frame Px Error ({cam})"); plt.grid(True, alpha=0.5); plt.legend(); plt.tight_layout()
                plot_path = os.path.join(predictions_output_root, f"{cam}_px_error_timeline.png"); plt.savefig(plot_path); plt.close(); logger.info(f"Saved plot: {plot_path}")
            except Exception as e: logger.error(f"Error plotting timeline {cam}: {e}")
            try: # Plot histogram
                plt.figure(figsize=(8, 5)); plt.hist(valid_errors, bins=30, edgecolor='black', alpha=0.7); plt.xlabel("Px Error"); plt.ylabel("Frequency"); plt.title(f"Px Error Hist ({cam}) Mean={mean_err:.2f}"); plt.grid(True, axis='y', alpha=0.5); plt.tight_layout()
                hist_path = os.path.join(predictions_output_root, f"{cam}_px_error_hist.png"); plt.savefig(hist_path); plt.close(); logger.info(f"Saved histogram: {hist_path}")
            except Exception as e: logger.error(f"Error plotting histogram {cam}: {e}")
        else: logger.info(f"No valid GT for {cam} to calculate/plot errors.")
        overall_metrics["per_camera"][cam] = cam_metrics

    # --- Save Overall Evaluation ---
    overall_eval_path = os.path.join(predictions_output_root, "overall_evaluation.json")
    try:
        class NpEncoder(json.JSONEncoder):
             def default(self, obj):
                 if isinstance(obj, np.integer): return int(obj)
                 if isinstance(obj, np.floating): return float(obj) if not np.isnan(obj) else None # Handle NaN
                 if isinstance(obj, np.ndarray): return obj.tolist()
                 return super(NpEncoder, self).default(obj)
        with open(overall_eval_path, "w") as f: json.dump(overall_metrics, f, indent=4, cls=NpEncoder)
        logger.info(f"Overall evaluation saved: {overall_eval_path}")
    except Exception as e: logger.error(f"Error saving overall eval JSON: {e}")

    logger.info(f"--- Prediction Run Finished ---")
    print(f"\nPrediction complete. Results saved to: {predictions_output_root}") # Also print to console

    # Close log file if using custom Logger class
    # This requires Logger class to have a close method
    # if isinstance(sys.stdout, Logger): sys.stdout.close()


if __name__ == "__main__":
    predict_and_evaluate()