# losses/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

# --- Core Loss Functions ---

def cosine_similarity_loss(pred, target):
    """ Loss based on cosine similarity: 1 - cos_sim(pred, target). """
    # Ensure inputs are float and add small epsilon for numerical stability if needed
    pred = pred.float()
    target = target.float()
    # Prevent division by zero if norms are zero, return 0 loss? or 1? Depends on context.
    # Let's assume inputs should have non-zero norm for this loss.
    cosine_sim = F.cosine_similarity(pred, target, dim=1, eps=1e-8)
    loss = 1.0 - cosine_sim
    return torch.mean(loss)

def angular_loss(pred, target):
    """ Mean angular error (radians) between prediction and target vectors. """
    pred = pred.float()
    target = target.float()
    pred_norm = F.normalize(pred, p=2, dim=1, eps=1e-8)
    target_norm = F.normalize(target, p=2, dim=1, eps=1e-8)
    # Clamp to avoid NaN from acos due to floating point inaccuracies
    cosine_sim = torch.sum(pred_norm * target_norm, dim=1).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    angles = torch.acos(cosine_sim)
    return torch.mean(angles)

# --- Loss Registry (Maps type string directly to loss function/class) ---
# Uses instantiated nn.Module losses where appropriate
_loss_registry = {
    'mse': nn.MSELoss(),
    'l1': nn.L1Loss(),
    'smooth_l1': nn.SmoothL1Loss(),
    'cosine': cosine_similarity_loss, # Custom function
    'angular': angular_loss         # Custom function
    # Add other losses here if needed, e.g.:
    # 'bce': nn.BCELoss(),
    # 'bce_logits': nn.BCEWithLogitsLoss(),
}

def get_loss(loss_type: str):
    """
    Retrieves the loss function instance based on its type name.

    Args:
        loss_type (str): The name of the loss function (e.g., 'mse', 'l1', 'cosine').

    Returns:
        A callable loss function instance (nn.Module or function).

    Raises:
        KeyError: If the requested loss_type is not found in the registry.
    """
    loss_type = loss_type.lower() # Ensure case-insensitivity
    try:
        loss_fn = _loss_registry[loss_type]
        logger.debug(f"Retrieved loss function for type '{loss_type}'.")
        return loss_fn
    except KeyError:
        available = list(_loss_registry.keys())
        logger.error(f"Loss type '{loss_type}' not found in registry. Available options: {available}")
        raise KeyError(f"Loss type '{loss_type}' not found. Available: {available}")