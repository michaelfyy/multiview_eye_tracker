# utils/metrics.py

import torch
import torch.nn.functional as F
import math
from datasets.transforms import inverse_transform_2d

def compute_mse_error(pred, target):
    """
    Computes mean squared error between prediction and target.
    """
    return torch.mean((pred - target) ** 2).item()

def compute_pixel_error(pred, target, orig_sizes, target_size=(224,224)):
    """
    Computes the average Euclidean pixel error between predicted and target pupil centers in original image coordinates.
    
    Uses the inverse transformation from transforms.py to convert coordinates.
    
    Args:
        pred: Tensor of shape (batch, 2) with predicted pupil center in resized coordinates.
        target: Tensor of shape (batch, 2) with ground truth pupil center in resized coordinates.
        orig_sizes: List of tuples, each tuple is (orig_h, orig_w) for each sample.
        target_size: Tuple for the resized image dimensions (default: (224,224)).
        
    Returns:
        Average pixel error in original image coordinates.
    """
    errors = []
    batch_size = pred.shape[0]
    for i in range(batch_size):
        pred_orig = inverse_transform_2d(pred[i], orig_sizes[i], target_size)
        target_orig = inverse_transform_2d(target[i], orig_sizes[i], target_size)
        error = torch.norm(pred_orig - target_orig, p=2)
        errors.append(error)
    return torch.mean(torch.stack(errors)).item()

def compute_angular_error(pred, target):
    """
    Computes the mean angular error (in degrees) between predicted and target gaze vectors.
    Both pred and target should be nonzero and of shape (batch, D).
    """
    pred_norm = F.normalize(pred, dim=1)
    target_norm = F.normalize(target, dim=1)
    # Dot product to get cosine similarity, clamped for stability.
    cos_sim = torch.sum(pred_norm * target_norm, dim=1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    angles = torch.acos(cos_sim)  # in radians
    mean_angle_rad = torch.mean(angles).item()
    return mean_angle_rad * (180.0 / math.pi)

def compute_cosine_similarity(pred, target):
    """
    Computes the average cosine similarity between predicted and target vectors.
    """
    pred_norm = F.normalize(pred, dim=1)
    target_norm = F.normalize(target, dim=1)
    cosine_sim = torch.sum(pred_norm * target_norm, dim=1)
    return torch.mean(cosine_sim).item()

def compute_std(predictions: torch.Tensor) -> list:
    """
    Computes the standard deviation of predictions along the batch dimension.
    
    Args:
        predictions (torch.Tensor): Tensor of shape (N, D), where N is the number of samples.
        
    Returns:
        List containing the standard deviation for each dimension.
    """
    return torch.std(predictions, dim=0).tolist()