# models/__init__.py
import logging
import torch.nn as nn # <-- Import nn
from .base_multiview_model import ConfigurableMultiViewModel
from .backbones import get_backbone
from .fusion_modules import ConcatFusion, AttentionFusion
from .heads import RegressionHead, HeatmapHead

logger = logging.getLogger(__name__)

def get_model(config):
    """ Builds the multi-view model based on the configuration dictionary. """
    model_config = config['model']
    data_config = config['data']
    num_views = data_config.get('num_views', 4)

    logger.info(f"Building model '{model_config['base_class']}' for {num_views} views.")

    # --- 1. Get Backbone ---
    # (Same as before)
    bb_type = model_config['backbone']['type']; bb_pretrained = model_config['backbone']['pretrained']
    bb_kwargs = model_config['backbone'].get('kwargs', {})
    try: backbone, feature_dim, num_patches, patch_size, grid_size = get_backbone(bb_type, bb_pretrained, **bb_kwargs)
    except Exception as e: logger.error(f"Failed backbone init {bb_type}: {e}"); raise

    # --- 2. Get Fusion Module ---
    # (Same as before)
    fusion_type = model_config['fusion']['type']; fusion_params = model_config['fusion'].get('params', {})
    fusion_module = None; fused_feature_dim = feature_dim * num_views
    if fusion_type == 'concat': fusion_module = ConcatFusion(); logger.info("Using ConcatFusion.")
    elif fusion_type == 'attention': fusion_module = AttentionFusion(dim=feature_dim, num_views=num_views, **fusion_params); fused_feature_dim = feature_dim; logger.info(f"Using AttentionFusion: {fusion_params}")
    elif fusion_type != 'none': raise ValueError(f"Unsupported fusion type: {fusion_type}")
    else: logger.info("No fusion module selected.")


    # --- 3. Get 2D Head ---
    head_2d_config = model_config['head_2d']
    head_2d_type = head_2d_config['type']
    head_2d_params = head_2d_config.get('params', {})
    use_separate_heads = head_2d_config.get('use_separate_heads', False) # Get the new flag
    head_2d = None

    if head_2d_type == 'regression':
        # --- FIX: Create single or multiple heads ---
        if use_separate_heads:
            logger.info(f"Using SEPARATE RegressionHead for each of {num_views} views.")
            # Create a ModuleList of heads
            head_2d = nn.ModuleList([
                RegressionHead(in_features=feature_dim, out_features=2, **head_2d_params)
                for _ in range(num_views)
            ])
            logger.info(f"Separate head params (per head): {head_2d_params}")
        else:
            logger.info(f"Using SHARED RegressionHead for 2D with params: {head_2d_params}")
            # Create a single shared head
            head_2d = RegressionHead(in_features=feature_dim, out_features=2, **head_2d_params)
        # --- End Fix ---

    elif head_2d_type == 'heatmap':
        if use_separate_heads:
             logger.warning("`use_separate_heads: true` is only supported for `head_2d.type: regression`. Using shared HeatmapHead.")
             use_separate_heads = False # Force shared for heatmap

        heatmap_output_res = data_config['heatmap']['output_res']
        head_2d = HeatmapHead(in_features=feature_dim, output_res=heatmap_output_res, **head_2d_params)
        logger.info(f"Using SHARED HeatmapHead for 2D with params: {head_2d_params}, output_res={heatmap_output_res}")
    else:
        raise ValueError(f"Unsupported 2D head type: {head_2d_type}")

    # --- 4. Get 3D Head ---
    # (Same as before)
    head_3d_config = model_config.get('head_3d'); head_3d = None
    if head_3d_config:
        head_3d_type = head_3d_config['type']; head_3d_params = head_3d_config.get('params', {})
        if head_3d_type == 'mlp':
             head_3d_in_features = fused_feature_dim
             head_3d = RegressionHead(in_features=head_3d_in_features, out_features=num_views * 6, **head_3d_params)
             logger.info(f"Using RegressionHead (MLP) for 3D: params={head_3d_params}, in_features={head_3d_in_features}")
        else: raise ValueError(f"Unsupported 3D head type: {head_3d_type}")
    else: logger.info("No 3D head configured.")


    # --- 5. Instantiate Base Model ---
    base_class_name = model_config.get('base_class', 'ConfigurableMultiViewModel')
    if base_class_name == 'ConfigurableMultiViewModel':
        # --- FIX: Pass separate heads flag/info if needed ---
        # The head_2d argument will now be either a single Module or a ModuleList
        model = ConfigurableMultiViewModel(
            backbone=backbone,
            fusion_module=fusion_module,
            head_2d=head_2d, # Pass the single head OR the ModuleList
            head_3d=head_3d,
            num_views=num_views,
            head_2d_type=head_2d_type,
            use_separate_2d_heads=use_separate_heads # Pass flag to base model
        )
        # --- End Fix ---
    else: raise ValueError(f"Unsupported base_class: {base_class_name}")

    logger.info(f"Model '{base_class_name}' built successfully.")
    return model