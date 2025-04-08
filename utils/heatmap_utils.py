# utils/heatmap_utils.py
import numpy as np
import torch
import math
import logging

logger = logging.getLogger(__name__)

# (generate_heatmap and get_coords_from_heatmap remain the same as fetched)
def generate_heatmap(center, output_res, sigma):
    """
    Generates a 2D Gaussian heatmap.

    Args:
        center (tuple or list or np.array): (x, y) coordinates of the center in the output_res space.
                                            Can be None if the keypoint is not visible.
        output_res (int): The desired output resolution (height and width) of the heatmap.
        sigma (float): Standard deviation of the Gaussian blob.

    Returns:
        torch.Tensor: The generated heatmap tensor shape (output_res, output_res).
    """
    size = output_res
    heatmap = np.zeros((size, size), dtype=np.float32)

    if center is None or any(math.isnan(c) for c in center): # Check all elements in center
        # logger.debug("generate_heatmap: center is None or NaN, returning zero map.")
        return torch.from_numpy(heatmap)

    center_x, center_y = center

    # Generate grid
    x, y = np.meshgrid(np.arange(size), np.arange(size))

    # Calculate Gaussian exponent efficiently
    exponent = -(((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

    # Apply exponent - values very far from center will effectively be exp(-inf) -> 0
    heatmap = np.exp(exponent)

    return torch.from_numpy(heatmap)


def get_coords_from_heatmap(heatmap):
    """
    Finds the coordinates of the maximum value in a heatmap tensor.

    Args:
        heatmap (torch.Tensor): Heatmap tensor, shape (batch, 1, H, W) or (1, H, W) or (H, W).

    Returns:
        torch.Tensor: Coordinates (x, y) of the maximum(s). Shape (batch, 2) or (2,).
                      Returns coordinates relative to the heatmap dimensions (W, H order).
    """
    input_device = heatmap.device # Remember original device
    heatmap = heatmap.float() # Ensure float for calculations

    if heatmap.dim() == 4:  # Batch dimension (B, C, H, W)
        batch_size, _, h, w = heatmap.shape
        # Flatten spatial dimensions: (B, C, H*W) -> (B, H*W) assuming C=1
        flat_heatmap = heatmap.view(batch_size, -1)
        # Find the index of the maximum value along the flattened spatial dimension
        _, idx = torch.max(flat_heatmap, dim=1) # Shape (B,)
        # Convert flat index back to 2D coordinates (W, H)
        coords_x = (idx % w) # Use integer division result directly for indices
        coords_y = (idx // w)
        # Stack as (x, y) pairs
        coords = torch.stack([coords_x, coords_y], dim=1).to(input_device) # Shape (batch, 2)
    elif heatmap.dim() <= 3:  # Single heatmap (C, H, W) or (H, W)
        if heatmap.dim() == 3: # Squeeze channel dim if present
            heatmap = heatmap.squeeze(0)
        if heatmap.dim() != 2: # Ensure it's 2D now
             raise ValueError(f"Unexpected heatmap shape after squeeze: {heatmap.shape}")
        h, w = heatmap.shape
        flat_heatmap = heatmap.view(-1)
        _, idx = torch.max(flat_heatmap, dim=0) # Shape (1,)
        coords_x = (idx % w)
        coords_y = (idx // w)
        coords = torch.stack([coords_x, coords_y]).to(input_device) # Shape (2,) -> Use stack
    else:
        raise ValueError(f"Unsupported heatmap shape: {heatmap.shape}")

    return coords.float() # Return as float


def scale_coords(coords, source_wh, target_wh):
    """
    Scales coordinates from a source resolution to a target resolution.
    Handles both single and batched inputs.

    Args:
        coords (torch.Tensor): Coordinates tensor, shape (2,) for single or (B, 2) for batch. Assumed (x, y).
        source_wh (tuple[int, int]): Source resolution tuple (Width, Height).
        target_wh (torch.Tensor or tuple/list): Target resolution.
                                               Shape (2,) for single (Width, Height) or
                                               Shape (B, 2) for batch (Width, Height).

    Returns:
        torch.Tensor: Scaled coordinates tensor, same shape as input coords.
    """
    input_device = coords.device
    input_dtype = coords.dtype

    # Ensure source_wh is tensor for broadcasting
    source_w, source_h = source_wh
    # Create source_size tensor ONCE, prevents repeated creation if function called often
    source_size = torch.tensor([source_w, source_h], device=input_device, dtype=input_dtype)
    # Prevent division by zero
    source_size = torch.where(source_size == 0, torch.tensor(1.0, device=input_device, dtype=input_dtype), source_size)


    # Ensure target_wh is tensor
    if not torch.is_tensor(target_wh):
        target_wh = torch.tensor(target_wh, device=input_device, dtype=input_dtype)

    # Check if input is batched or single
    if coords.dim() == 2: # Batched input coords: (B, 2)
        if target_wh.dim() == 1: # Single target size for all in batch
            target_wh = target_wh.unsqueeze(0).expand(coords.shape[0], -1) # Expand (2,) -> (1, 2) -> (B, 2)
        elif target_wh.dim() == 2: # Batched target size
             if target_wh.shape[0] != coords.shape[0]:
                 raise ValueError(f"Batch size mismatch: coords {coords.shape[0]}, target_wh {target_wh.shape[0]}")
        else:
             raise ValueError(f"Invalid target_wh shape for batched coords: {target_wh.shape}")

        # Prevent division by zero for target size
        target_wh = torch.where(target_wh == 0, torch.tensor(1.0, device=input_device, dtype=input_dtype), target_wh)

        # Calculate scale factors element-wise: (B, 2) / (1, 2) -> (B, 2)
        # Scale = Target / Source
        scale = target_wh / source_size.unsqueeze(0)
        scaled_coords = coords * scale # Element-wise multiplication

    elif coords.dim() == 1: # Single input coord: (2,)
        if target_wh.dim() == 2: # Received batch of target sizes? Use the first one? Or error?
             logger.warning(f"scale_coords received single coord but batched target_wh {target_wh.shape}. Using first target size.")
             target_wh = target_wh[0]
        elif target_wh.dim() != 1:
             raise ValueError(f"Invalid target_wh shape for single coord: {target_wh.shape}")

        # Prevent division by zero
        target_wh = torch.where(target_wh == 0, torch.tensor(1.0, device=input_device, dtype=input_dtype), target_wh)

        # Calculate scale factors element-wise: (2,) / (2,) -> (2,)
        # Scale = Target / Source
        scale = target_wh / source_size
        scaled_coords = coords * scale # Element-wise multiplication
    else:
        raise ValueError(f"Unsupported coords shape in scale_coords: {coords.shape}")

    return scaled_coords