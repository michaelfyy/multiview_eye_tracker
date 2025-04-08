import os
import logging
import torch
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split # Alternative splitting

# Ensure datasets and transforms are importable
from datasets.ue2_multiview import UE2MultiviewDataset, collate_fn
from datasets.transforms import get_transform

logger = logging.getLogger(__name__)

def create_dataloaders(config):
    """
    Creates train and validation dataloaders for the UE2 Multiview dataset.

    Args:
        config (dict): The main configuration dictionary.

    Returns:
        dict: A dictionary containing 'train' and 'val' DataLoaders.
    """
    data_cfg = config['data']
    train_cfg = config['training']

    data_dir = data_cfg['data_dir']
    target_size = tuple(data_cfg['target_size']) # Ensure tuple HxW
    num_views = data_cfg['num_views']
    heatmap_config = data_cfg.get('heatmap')

    # --- Find JSON files ---
    # Assumes JSON files are directly in data_dir
    try:
        all_json_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith('.json')]
        # Sort files numerically based on frame number (filename without extension)
        all_json_files = sorted(all_json_files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
        if not all_json_files:
            raise FileNotFoundError(f"No JSON files found in {data_dir}")
        logger.info(f"Found {len(all_json_files)} total JSON files.")
    except Exception as e:
        logger.error(f"Error finding JSON files in {data_dir}: {e}")
        raise

    # --- Train/Validation Split ---
    # Using sklearn's train_test_split for potentially stratified splitting if needed later
    train_split_ratio = data_cfg.get('train_split', 0.8) # Default to 80/20 split
    if not (0 < train_split_ratio < 1):
        raise ValueError(f"train_split must be between 0 and 1, got {train_split_ratio}")

    train_jsons, val_jsons = train_test_split(
        all_json_files,
        train_size=train_split_ratio,
        shuffle=True,
        random_state=config.get('seed', 42) # Use seed from config if available
    )
    logger.info(f"Dataset split: {len(train_jsons)} train, {len(val_jsons)} validation samples.")

    # --- Get Transforms ---
    train_transform = get_transform(target_size=target_size, is_train=True)
    val_transform = get_transform(target_size=target_size, is_train=False)

    # --- Create Datasets ---
    train_dataset = UE2MultiviewDataset(
        json_files=train_jsons,
        data_dir=data_dir,
        transform=train_transform,
        target_size=target_size,
        num_views=num_views,
        heatmap_config=heatmap_config
    )
    val_dataset = UE2MultiviewDataset(
        json_files=val_jsons,
        data_dir=data_dir,
        transform=val_transform,
        target_size=target_size,
        num_views=num_views,
        heatmap_config=heatmap_config
    )

    # --- Create DataLoaders ---
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True, # Shuffle training data
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=data_cfg.get('pin_memory', True),
        collate_fn=collate_fn, # Use custom collate to handle potential None samples
        drop_last=True # Consider dropping last incomplete batch for consistency
    )
    dataloaders['val'] = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'] * 2, # Often use larger batch for validation
        shuffle=False, # No need to shuffle validation data
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=data_cfg.get('pin_memory', True),
        collate_fn=collate_fn
    )

    logger.info("Train and Validation DataLoaders created.")
    return dataloaders