# datasets/ue2_multiview.py
import os
import json
import logging
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

# Ensure utils is importable (e.g., if utils is in the project root)
from utils.heatmap_utils import generate_heatmap

logger = logging.getLogger(__name__)

# Helper to parse tuple strings like "(1.0, 2.0, 3.0)" - kept local for simplicity
def parse_tuple_string(s):
    """ Parses a string like "(1.0, 2.0, ...)" into a tuple of floats. Handles potential errors. """
    if not isinstance(s, str):
        logger.debug(f"parse_tuple_string expected string, got {type(s)}. Returning zeros.")
        return (0.0, 0.0, 0.0) # Default for safety
    try:
        # Remove parentheses and split by comma
        s = s.strip().strip("()")
        parts = s.split(',')
        # Convert each part to float, handling potential whitespace
        return tuple(float(x.strip()) for x in parts)
    except Exception as e:
        logger.warning(f"Could not parse tuple string: '{s}'. Error: {e}. Returning zeros.")
        return (0.0, 0.0, 0.0)

class UE2MultiviewDataset(Dataset):
    """
    PyTorch Dataset for UE2 Multiview eye tracking data.
    Assumes images ({frame}.jpg) and JSON ({frame}.json) files are in the same directory.
    """
    def __init__(self, json_files, data_dir, transform=None, target_size=(224, 224), num_views=4, heatmap_config=None):
        """
        Args:
            json_files (list): List of absolute paths to the JSON ground truth files.
            data_dir (str): Root directory containing the JSON and image files.
            transform (callable, optional): Transformation to be applied on each image.
            target_size (tuple): Desired output size (height, width) for the images.
            num_views (int): Number of camera views (usually 4).
            heatmap_config (dict, optional): Configuration for heatmap generation {'enabled', 'sigma', 'output_res'}.
        """
        self.json_files = json_files
        self.data_dir = data_dir # Root directory where images and jsons reside
        self.transform = transform
        self.target_size = target_size # Expected (H, W)
        self.num_views = num_views
        self.cam_ids = [f"cam_{i}" for i in range(1, num_views + 1)] # ["cam_1", ..., "cam_4"]

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        # Heatmap configuration
        self.heatmap_enabled = heatmap_config.get('enabled', False) if heatmap_config else False
        if self.heatmap_enabled:
            if not all(k in heatmap_config for k in ['sigma', 'output_res']):
                raise ValueError("Heatmap config must include 'sigma' and 'output_res' when enabled.")
            self.heatmap_sigma = heatmap_config['sigma']
            self.heatmap_output_res = heatmap_config['output_res']
            logger.info(f"Heatmap generation enabled: sigma={self.heatmap_sigma}, res={self.heatmap_output_res}x{self.heatmap_output_res}")
        else:
             logger.info("Heatmap generation disabled.")

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        try:
            with open(json_path, 'r') as f:
                json_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {json_path}: {e}")
            return None # Signal error to collate_fn

        # Extract frame number (base name without extension)
        frame_num = os.path.splitext(os.path.basename(json_path))[0]

        all_images = []
        all_orig_sizes = []
        ground_truths = {}
        valid_sample = True

        for cam_id in self.cam_ids:
            # Construct image path (assuming jpg extension, adjust if needed)
            img_filename = f"{frame_num}_{cam_id}.jpg"
            img_path = os.path.join(self.data_dir, img_filename)

            if not os.path.exists(img_path):
                 logger.warning(f"Image file not found: {img_path}. Skipping sample {idx}.")
                 valid_sample = False
                 break # Skip this whole sample if one image is missing

             # Load image using PIL
            try:
                image = Image.open(img_path).convert('RGB')
            except Exception as e:
                logger.error(f"Error opening image file {img_path}: {e}")
                valid_sample = False
                break

            orig_w, orig_h = image.size
            # Store original size as (H, W)
            all_orig_sizes.append(torch.tensor([orig_h, orig_w], dtype=torch.float32))

            # Apply transforms (which should include ToTensor)
            if self.transform:
                image_tensor = self.transform(image)
            else:
                # Fallback if no transform provided (apply basic ToTensor)
                image_tensor = T.ToTensor()(image)

            all_images.append(image_tensor)

            # --- Ground Truth Processing ---
            cam_data = json_data.get('cameras', {}).get(cam_id) # Use .get for safer access

            if cam_data:
                # 2D Pupil Center (derived from iris landmarks)
                iris_landmarks_2d_str = cam_data.get('iris_2d', [])
                iris_landmarks_2d = np.array([parse_tuple_string(p)[:2] for p in iris_landmarks_2d_str])
                iris_landmarks_2d[:, 1] = 480 - iris_landmarks_2d[:, 1]
                pupil_center_2d = np.mean(iris_landmarks_2d, axis=0)

                # 3D Pupil Center (Iris Center)
                # Nested structure check based on old code (data->cameras->cam_id->ground_truth->iris_center)
                gt_data = cam_data.get("ground_truth")
                if gt_data:
                     iris_center_str = gt_data.get("iris_center")
                     if iris_center_str:
                         pupil_center_3d = parse_tuple_string(iris_center_str)

                     gaze_vector_str = gt_data.get("gaze_vector")
                     if gaze_vector_str:
                         gaze_vector_3d = parse_tuple_string(gaze_vector_str)
                else:
                    # Check if iris_center is directly under cam_data as in newer json?
                    iris_center_str = cam_data.get('iris_center')
                    if iris_center_str:
                        pupil_center_3d = parse_tuple_string(iris_center_str)
                    gaze_vector_str = cam_data.get('gaze_vector')
                    if gaze_vector_str:
                        gaze_vector_3d = parse_tuple_string(gaze_vector_str)


            # --- Store GT Tensors ---
            # Store original image space 2D coords (useful for metrics)
            ground_truths[f'{cam_id}_2d_orig'] = torch.from_numpy(pupil_center_2d)

            # Store target 2D coords (for regression head), scaled to target_size
            # Note: Scaling might not be ideal if aspect ratio changes significantly.
            # Consider padding/cropping strategies in transforms if AR is an issue.
            if pupil_center_2d is not None:
                scale_x = self.target_size[1] / orig_w # target_w / orig_w
                scale_y = self.target_size[0] / orig_h # target_h / orig_h
                pupil_center_resized = torch.tensor([pupil_center_2d[0] * scale_x, pupil_center_2d[1] * scale_y], dtype=torch.float32)
            else:
                pupil_center_resized = torch.tensor([0.0, 0.0], dtype=torch.float32) # Use default or NaN?
            ground_truths[f'{cam_id}_2d'] = pupil_center_resized

            # Generate heatmap if enabled
            if self.heatmap_enabled:
                if pupil_center_2d is not None:
                    # Scale coords to heatmap resolution (output_res x output_res)
                    hm_scale_x = self.heatmap_output_res / orig_w
                    hm_scale_y = self.heatmap_output_res / orig_h
                    # Center coords for heatmap generation func (expects x, y)
                    pupil_center_hm = (pupil_center_2d[0] * hm_scale_x, pupil_center_2d[1] * hm_scale_y)
                    heatmap = generate_heatmap(pupil_center_hm, self.heatmap_output_res, self.heatmap_sigma)
                else:
                    # Generate empty heatmap if no pupil center
                    heatmap = torch.zeros((self.heatmap_output_res, self.heatmap_output_res), dtype=torch.float32)
                # Store heatmap GT, shape (H_h, W_h) -> will be unsqueezed later if needed
                ground_truths[f'{cam_id}_heatmap_gt'] = heatmap

            # Store 3D GT (Pupil Center + Gaze Endpoint)
            ground_truths[f'{cam_id}_pupil_3d'] = torch.tensor(pupil_center_3d, dtype=torch.float32)
            ground_truths[f'{cam_id}_gaze_endpoint_3d'] = torch.tensor(gaze_vector_3d, dtype=torch.float32)


        if not valid_sample:
             return None # Signal error to collate_fn

        # Stack images: (views, C, H, W)
        images_tensor = torch.stack(all_images)
        # Stack original sizes: (views, 2) -> (H, W)
        orig_sizes_tensor = torch.stack(all_orig_sizes)

        sample = {
            'images': images_tensor,
            'orig_sizes': orig_sizes_tensor,
            'json_path': json_path # Keep for reference/debugging if needed
        }
        sample.update(ground_truths) # Add all ground truth tensors: cam_1_2d, cam_1_heatmap_gt, cam_1_pupil_3d, etc.
        return sample
    
# --- Custom Collate Function ---
def collate_fn(batch):
    """ Filters out None samples before collating """
    batch = [b for b in batch if b is not None]
    if not batch:
         return None # Return None if the entire batch was invalid
    try:
         return torch.utils.data.dataloader.default_collate(batch)
    except Exception as e:
         logger.error(f"Error in default_collate: {e}")
         # You might want to inspect the batch here if errors occur
         # for i, item in enumerate(batch):
         #     for key, value in item.items():
         #         if isinstance(value, torch.Tensor):
         #             print(f"Item {i}, key {key}, shape {value.shape}")
         #         else:
         #             print(f"Item {i}, key {key}, type {type(value)}")
         return None # Skip batch if collating fails