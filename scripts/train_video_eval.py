import os
import sys
import json
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import shutil
import glob

# Ensure parent directory is in the path.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from utils.config import parse_args
from utils.dataloader import load_data
from utils.logger import Logger
from models import get_model
from losses.losses import get_loss
from metrics.metrics import compute_pixel_error  # used for training statistics
# Import video evaluation helpers.
from utils.video_helper import extract_frames, parse_annotations
from datasets.transforms import inverse_transform_2d
import torchvision.transforms.v2 as transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_frame(frame):
    """
    Preprocess a raw image (numpy array) into a tensor (resized to 224x224, normalized).
    Also returns the original image size.
    """
    orig_size = (frame.shape[0], frame.shape[1])
    frame_tensor = torch.tensor(frame)
    frame_tensor = transforms.functional.convert_image_dtype(frame_tensor, torch.float)
    frame_tensor = frame_tensor.permute(2, 0, 1)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToImage(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    processed_frame = preprocess(frame_tensor)
    return processed_frame, orig_size

def get_camera_id_from_video(video_path):
    base = os.path.splitext(os.path.basename(video_path))[0]
    if base.startswith("e") and len(base) > 1:
        num = base[1:]
        return f"cam_{num}"
    return None

def evaluate_on_eye_videos(model, config, camera_filter=None):
    """
    Evaluates the current model on the actual eye videos.
    For each video in config['data']['video_root'] (optionally filtered by camera),
    the corresponding annotation file (from config['data']['annotations_folder']) is loaded.
    Only frames with index >= 160 and with a ground truth annotation (i.e. where the eye is visible)
    are considered. Returns the overall mean pixel error.
    """
    video_root = config['data']['video_root']
    annotations_folder = config['data']['annotations_folder']
    video_files = glob.glob(os.path.join(video_root, "*.mp4"))
    video_errors = []
    for video_file in video_files:
        if camera_filter is not None:
            vid_cam = get_camera_id_from_video(video_file)
            if vid_cam != camera_filter:
                continue
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        annot_file = os.path.join(video_root, annotations_folder, f"{base_name}_annotations.xml")
        if not os.path.exists(annot_file):
            continue
        frames = extract_frames(video_file)
        if len(frames) == 0:
            continue
        annotations = parse_annotations(annot_file)
        processed_frames = []
        orig_sizes = []
        for frame in frames:
            proc_frame, orig_size = preprocess_frame(frame)
            processed_frames.append(proc_frame)
            orig_sizes.append(orig_size)
        inputs = torch.stack(processed_frames).to(device)
        with torch.no_grad():
            predictions = model(inputs)
        pupil_preds = predictions['pupil']
        errors = []
        for i in range(len(frames)):
            if i < 160:
                continue
            if i in annotations:
                gt_point = torch.tensor(annotations[i][0][1], dtype=torch.float)
                pred_resized = pupil_preds[i].cpu()
                orig_size = orig_sizes[i]
                pred_orig = inverse_transform_2d(pred_resized, orig_size, target_size=(224,224))
                error = torch.norm(pred_orig - gt_point, p=2).item()
                errors.append(error)
        if errors:
            video_errors.append(sum(errors)/len(errors))
    if video_errors:
        overall_error = sum(video_errors)/len(video_errors)
    else:
        overall_error = float('inf')
    return overall_error

def train():
    config = parse_args()
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    early_stop_patience = config['training'].get('early_stop', 5)
    
    # In train_video_eval.py, we use video evaluation (ignoring the validation loader) for all datasets.
    if config['data']['dataset'] == "ue2_separate":
        dataloaders_dict = load_data(config)  # only used for training loader
        for cam, loaders in dataloaders_dict.items():
            run_dir = os.path.join(config['logging']['run_dir'], f"run_{run_timestamp}_{cam}")
            os.makedirs(run_dir, exist_ok=True)
            logger = Logger(run_dir, config)
            logger.log({"message": f"Training (video eval) model for camera {cam}"})
            
            model = get_model(config).to(device)
            if config['training'].get('from_checkpoint', False):
                # Use individual checkpoint field.
                if cam == "cam_1":
                    cp = config['training'].get('checkpoint_cam_1', None)
                elif cam == "cam_2":
                    cp = config['training'].get('checkpoint_cam_2', None)
                elif cam == "cam_3":
                    cp = config['training'].get('checkpoint_cam_3', None)
                elif cam == "cam_4":
                    cp = config['training'].get('checkpoint_cam_4', None)
                else:
                    cp = None
                if cp == "latest":
                    cp = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}_{cam}.pt")
                if not cp or not os.path.exists(cp):
                    raise ValueError(f"from_checkpoint is True but checkpoint for {cam} is not provided or does not exist.")
                model.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
                logger.log({"message": f"Loaded model weights from checkpoint for {cam}: {cp}"})
            logger.log({"message": f"Model {config['model']} loaded for camera {cam}."})
            
            criterion_pupil = get_loss(config['loss']['pupil'], 'pupil')
            optimizer = optim.Adam(
                model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
            
            best_video_error = float('inf')
            best_epoch_metrics = None
            metrics = {"epoch": [], "train_loss": []}
            no_improvement_count = 0
            num_epochs = config['training']['num_epochs']
            
            for epoch in range(num_epochs):
                model.train()
                train_running_loss = 0.0
                train_samples = 0
                for batch in loaders['train']:
                    images = batch['image'].to(device)
                    pupil_labels = batch['pupil'].to(device)
                    orig_sizes = batch['orig_size'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion_pupil(outputs['pupil'], pupil_labels)
                    loss.backward()
                    optimizer.step()
                    
                    batch_size = images.size(0)
                    train_running_loss += loss.item() * batch_size
                    train_samples += batch_size
                train_loss = train_running_loss / train_samples
                metrics['epoch'].append(epoch + 1)
                metrics['train_loss'].append(train_loss)
                logger.log({"epoch": epoch+1, "train_loss": train_loss})
                print(f"Camera {cam} | Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}")
                
                model.eval()
                video_error = evaluate_on_eye_videos(model, config, camera_filter=cam)
                logger.log({"epoch": epoch+1, "video_mean_pixel_error": video_error})
                print(f"Camera {cam} | Epoch {epoch+1} - Video Mean Pixel Error (frames>=160 & visible): {video_error:.6f}")
                
                if video_error < best_video_error:
                    best_video_error = video_error
                    best_epoch_metrics = {"epoch": epoch+1, "train_loss": train_loss, "video_mean_pixel_error": video_error}
                    no_improvement_count = 0
                    checkpoint_path = os.path.join(run_dir, "best_model.pt")
                    torch.save(model.state_dict(), checkpoint_path)
                    logger.log({"message": "Best model saved based on video evaluation", "epoch": epoch+1, "video_mean_pixel_error": video_error})
                    common_ckpt = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}_{cam}.pt")
                    shutil.copy(checkpoint_path, common_ckpt)
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= early_stop_patience:
                        print(f"Early stopping triggered for camera {cam} at epoch {epoch+1}")
                        logger.log({"message": f"Early stopping triggered for camera {cam} at epoch {epoch+1}"})
                        break
                        
                with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                    json.dump(metrics, f, indent=4)
                plt.figure()
                plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Training Loss for {cam}")
                plt.legend()
                plt.savefig(os.path.join(run_dir, "loss_plot.png"))
                plt.close()
            
            logger.log({"Best Epoch Metrics for " + cam: best_epoch_metrics})
            print(f"Camera {cam} training complete. Best Epoch: {best_epoch_metrics['epoch']}, Best Video Mean Pixel Error: {best_video_error:.6f}")
            logger.log({"best video mean pixel error": f"{best_video_error:.6f}"})
            
    else:
        # For syntheseyes and ue2_combined (or any dataset not ue2_separate), use the training loader for training
        # and video evaluation for evaluation.
        dataloaders = load_data(config)
        run_dir = os.path.join(config['logging']['run_dir'], f"run_{run_timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        logger = Logger(run_dir, config)
        model = get_model(config).to(device)
        if config['training'].get('from_checkpoint', False):
            cp = config['training'].get('checkpoint_path', None)
            if cp == "latest":
                cp = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}.pt")
            if not cp or not os.path.exists(cp):
                raise ValueError("from_checkpoint is True but checkpoint_path is not provided or does not exist.")
            model.load_state_dict(torch.load(cp, map_location=device, weights_only=True))
            logger.log({"message": f"Loaded model weights from checkpoint: {cp}"})
        logger.log({"message": f"Model {config['model']} loaded."})
        print(f"Using device: {device}")
        criterion_pupil = get_loss(config['loss']['pupil'], 'pupil')
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        best_video_error = float('inf')
        best_epoch_metrics = None
        metrics = {"epoch": [], "train_loss": []}
        no_improvement_count = 0
        num_epochs = config['training']['num_epochs']
        for epoch in range(num_epochs):
            model.train()
            train_running_loss = 0.0
            train_samples = 0
            for batch in dataloaders['train']:
                images = batch['image'].to(device)
                pupil_labels = batch['pupil'].to(device)
                orig_sizes = batch['orig_size'].to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion_pupil(outputs['pupil'], pupil_labels)
                loss.backward()
                optimizer.step()
                batch_size = images.size(0)
                train_running_loss += loss.item() * batch_size
                train_samples += batch_size
            train_loss = train_running_loss / train_samples
            metrics['epoch'].append(epoch + 1)
            metrics['train_loss'].append(train_loss)
            logger.log({"epoch": epoch+1, "train_loss": train_loss})
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.6f}")
            
            model.eval()
            # Video evaluation for all datasets.
            video_error = evaluate_on_eye_videos(model, config)
            logger.log({"epoch": epoch+1, "video_mean_pixel_error": video_error})
            print(f"Epoch {epoch+1} - Video Mean Pixel Error (frames>=160 & visible): {video_error:.6f}")
            
            if video_error < best_video_error:
                best_video_error = video_error
                best_epoch_metrics = {"epoch": epoch+1, "train_loss": train_loss, "video_mean_pixel_error": video_error}
                no_improvement_count = 0
                checkpoint_path = os.path.join(run_dir, "best_model.pt")
                torch.save(model.state_dict(), checkpoint_path)
                logger.log({"message": "Best model saved based on video evaluation", "epoch": epoch+1, "video_mean_pixel_error": video_error})
                common_ckpt = os.path.join(config['logging']['run_dir'], f"latest_best_model_{config['data']['dataset']}.pt")
                shutil.copy(checkpoint_path, common_ckpt)
            else:
                no_improvement_count += 1
                if no_improvement_count >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    logger.log({"message": f"Early stopping triggered at epoch {epoch+1}"})
                    break
            
            with open(os.path.join(run_dir, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=4)
            plt.figure()
            plt.plot(metrics['epoch'], metrics['train_loss'], label='Train Loss')
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.legend()
            plt.savefig(os.path.join(run_dir, "loss_plot.png"))
            plt.close()
            
        logger.log({"Best Epoch Metrics (lowest video mean pixel error)": best_epoch_metrics})
        print("Training complete. Best Epoch: {}, Best Video Mean Pixel Error: {:.6f}".format(best_epoch_metrics["epoch"], best_video_error))
        logger.log({"best video mean pixel error": f"{best_video_error:.6f}"})

if __name__ == "__main__":
    train()
