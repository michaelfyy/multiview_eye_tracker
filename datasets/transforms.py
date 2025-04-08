# datasets/transforms.py
import logging
from torchvision import transforms as T

logger = logging.getLogger(__name__)

def get_transform(target_size=(224, 224), is_train=True):
    """
    Gets standard transformations for image models.

    Args:
        target_size (tuple): Target HxW size for image resizing.
        is_train (bool): Whether to apply training-specific augmentations.

    Returns:
        callable: A torchvision transform composition.
    """
    # Ensure target_size is in (height, width) format
    if not isinstance(target_size, (list, tuple)) or len(target_size) != 2:
        raise ValueError(f"target_size must be a tuple or list of (height, width), got {target_size}")

    # ImageNet mean/std - common starting point
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Basic transforms: Resize, ToTensor, Normalize
    transform_list = [
        T.Resize(target_size), # Takes (h, w)
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ]

    if is_train:
        # Add augmentations for training - insert before ToTensor
        augmentations = [
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.3),
            # Add other augmentations like RandomRotation, RandomHorizontalFlip if needed
            # T.RandomRotation(degrees=5),
        ]
        # Insert augmentations after Resize but before ToTensor
        transform_list.insert(1, T.Compose(augmentations))
        logger.debug(f"Created training transforms for size {target_size}")
    else:
        logger.debug(f"Created validation/test transforms for size {target_size}")

    return T.Compose(transform_list)