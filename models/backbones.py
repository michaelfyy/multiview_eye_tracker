# models/backbones.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
import math

logger = logging.getLogger(__name__)

# --- Flag to track if the DINOv2 mismatch warning has been logged ---
_logged_dinov2_mismatch_warning = False
# --------------------------------------------------------------------

# --- Helper to get features from TIMM models ---
# (get_timm_features function remains the same)
def get_timm_features(model, x):
    """Extracts features (CLS token, patch features) from a TIMM Vision Transformer model."""
    B = x.shape[0]
    # This implementation assumes a standard ViT structure in TIMM.
    # May need adjustments for different TIMM model families (like EfficientViT, MobileViT).
    try:
        x = model.patch_embed(x)
        if hasattr(model, 'cls_token') and model.cls_token is not None: # Check if cls_token exists and is not None
            cls_token = model.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            pos_embed = model.pos_embed
            # Handle interpolation if input size differs from pretraining size
            if x.shape[1] != pos_embed.shape[1]:
                num_patches = x.shape[1] - 1 # Exclude CLS token
                # Calculate num_extra_tokens based on actual pos_embed shape
                num_extra_tokens = pos_embed.shape[1] - model.patch_embed.num_patches - 1 if hasattr(model.patch_embed, 'num_patches') else 0
                orig_size_sq = model.patch_embed.num_patches if hasattr(model.patch_embed, 'num_patches') else -1
                new_size_sq = num_patches

                # Check if patch numbers are perfect squares before attempting sqrt
                if orig_size_sq > 0 and new_size_sq > 0 and \
                   math.isqrt(orig_size_sq)**2 == orig_size_sq and \
                   math.isqrt(new_size_sq)**2 == new_size_sq:
                    orig_size = math.isqrt(orig_size_sq)
                    new_size = math.isqrt(new_size_sq)
                    if orig_size != new_size:
                        logger.info(f"Interpolating TIMM pos embedding from {orig_size}x{orig_size} to {new_size}x{new_size}") # Use INFO level
                        # Ensure num_extra_tokens calculation is robust
                        if num_extra_tokens < 0: num_extra_tokens = 0 # Safety check
                        extra_tokens = pos_embed[:, :num_extra_tokens+1] # CLS and any other leading tokens
                        pos_tokens = pos_embed[:, num_extra_tokens+1:] # Grid tokens
                        # Reshape based on original grid size
                        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, model.embed_dim).permute(0, 3, 1, 2)
                        pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                        pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    # Apply interpolated or original pos_embed
                    x = model.pos_drop(x + pos_embed)
                else:
                     logger.warning(f"Cannot interpolate TIMM pos embedding: Patch numbers ({orig_size_sq}, {new_size_sq}) invalid or not perfect squares. Applying original pos_embed without interpolation.")
                     if x.shape[1] == pos_embed.shape[1]: x = model.pos_drop(x + pos_embed)
                     else: logger.error(f"Pos embed shape mismatch ({pos_embed.shape[1]}) vs input ({x.shape[1]}) and cannot interpolate."); x = model.pos_drop(x)
            else: # Input size matches pretraining size
                 x = model.pos_drop(x + pos_embed)
        else: # Models without CLS token
             if hasattr(model, 'pos_embed') and model.pos_embed is not None and x.shape[1] == model.pos_embed.shape[1]: x = x + model.pos_embed
             x = model.pos_drop(x)

        # Pass through transformer blocks
        if hasattr(model, 'blocks'): x = model.blocks(x)
        elif hasattr(model, 'stages'): x = model.stages(x)
        else:
             logger.warning(f"Could not find 'blocks' or 'stages' in {type(model).__name__}. Trying 'forward_features'.")
             if hasattr(model, 'forward_features'): x = model.forward_features(x)
             else: raise AttributeError(f"Feature extraction method not found in {type(model).__name__}.")

        # Final normalization
        if hasattr(model, 'norm'): x = model.norm(x)

        # --- Extract features ---
        if hasattr(model, 'cls_token') and model.cls_token is not None: cls_feat, patch_feat = x[:, 0], x[:, 1:]
        else: logger.debug(f"No CLS token in {type(model).__name__}. Using mean patch token."); cls_feat, patch_feat = x.mean(dim=1), x

        # --- Reshape patch features to grid if possible ---
        patch_feat_grid = None
        if hasattr(model, 'patch_embed'):
             grid_size = None
             if hasattr(model.patch_embed, 'num_patches') and model.patch_embed.num_patches > 0:
                 num_p = model.patch_embed.num_patches; grid_dim_f = math.sqrt(num_p);
                 if grid_dim_f == int(grid_dim_f): grid_size = (int(grid_dim_f), int(grid_dim_f))
             elif hasattr(model.patch_embed, 'grid_size'): grid_size = model.patch_embed.grid_size
             if grid_size and patch_feat.dim() == 3:
                 B, N, D = patch_feat.shape; H_p, W_p = grid_size
                 if N == H_p * W_p:
                     try: patch_feat_grid = patch_feat.permute(0, 2, 1).reshape(B, D, H_p, W_p)
                     except Exception as reshape_e: logger.warning(f"Failed TIMM grid reshape: {reshape_e}. Shape: {patch_feat.shape}, Grid: {H_p}x{W_p}")
                 else: logger.warning(f"TIMM patch count mismatch: Feat {N}, Grid {H_p*W_p}. Cannot reshape.")
             elif grid_size is None: logger.debug("TIMM grid size unknown. Patch features not reshaped.")

        return {'cls_token': cls_feat, 'patch_features': patch_feat_grid if patch_feat_grid is not None else patch_feat}
    except AttributeError as e: logger.error(f"Attr error in TIMM model {type(model).__name__}: {e}.", exc_info=True); raise
    except Exception as e: logger.error(f"Generic error in TIMM features {type(model).__name__}: {e}", exc_info=True); raise


# --- Specific Backbone Loaders ---
# (get_timm_backbone function remains the same)
def get_timm_backbone(backbone_type='mobilevit_s', pretrained=True, **kwargs):
    """Loads a backbone from TIMM and prepares it for feature extraction."""
    try:
        model = timm.create_model(backbone_type, pretrained=pretrained, **kwargs)
        model.eval()
        feature_dim = getattr(model, 'embed_dim', getattr(model, 'num_features', None))
        if feature_dim is None:
            try: last = list(model.children())[-1]; feature_dim = getattr(last, 'in_features'); logger.warning(f"Inferred feature_dim: {feature_dim}")
            except: raise AttributeError(f"Cannot infer feature dimension for {backbone_type}")
        num_patches = getattr(model.patch_embed, 'num_patches', -1) if hasattr(model, 'patch_embed') else -1
        patch_size = -1; grid_size = None
        if hasattr(model, 'patch_embed'):
            if hasattr(model.patch_embed, 'patch_size'): ps = model.patch_embed.patch_size; patch_size = ps[0] if isinstance(ps, (tuple, list)) else ps
            if num_patches > 0: grid_dim_f = math.sqrt(num_patches); grid_size = (int(grid_dim_f), int(grid_dim_f)) if grid_dim_f == int(grid_dim_f) else None
            elif hasattr(model.patch_embed, 'grid_size'): grid_size = model.patch_embed.grid_size
        model.forward_features_dict = lambda x: get_timm_features(model, x)
        model.forward = model.forward_features_dict
        logger.info(f"Loaded TIMM backbone: {backbone_type} | Pretrained: {pretrained} | Feature Dim: {feature_dim} | Num Patches: {num_patches} | Patch Size: {patch_size}")
        return model, feature_dim, num_patches, patch_size, grid_size
    except Exception as e: logger.error(f"Failed to load TIMM model '{backbone_type}': {e}", exc_info=True); raise


def get_dinov2_backbone(backbone_type='dinov2_vits14', pretrained=True, **kwargs):
    """Loads a DINOv2 backbone from torch.hub."""
    global _logged_dinov2_mismatch_warning # Declare intention to modify global flag

    if not pretrained: logger.warning("DINOv2 models used pretrained. Setting pretrained=True.")
    try:
        valid_dinov2 = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14']
        if backbone_type not in valid_dinov2: raise ValueError(f"Invalid DINOv2 name: {backbone_type}. Use: {valid_dinov2}")
        model = torch.hub.load('facebookresearch/dinov2', backbone_type); model.eval()
        feature_dim = getattr(model, 'embed_dim', None); assert feature_dim is not None
        patch_embed = getattr(model, 'patch_embed', None); assert patch_embed is not None
        num_patches = getattr(patch_embed, 'num_patches', -1)
        patch_size_attr = getattr(patch_embed, 'patch_size', (-1,)); patch_size = patch_size_attr[0] if isinstance(patch_size_attr, (tuple, list)) else patch_size_attr
        grid_size = None
        if num_patches > 0:
            grid_dim_f = math.sqrt(num_patches);
            if grid_dim_f == int(grid_dim_f): grid_size = (int(grid_dim_f), int(grid_dim_f)); logger.info(f"DINOv2 pre-train num_patches={num_patches}, grid={grid_size}")
            else: logger.warning(f"DINOv2 num_patches ({num_patches}) not square.")
        else: logger.warning("Could not get num_patches for DINOv2.")

        def forward_features_dict(inner_model, x): # Pass model explicitly
            global _logged_dinov2_mismatch_warning # Access global flag
            B = x.shape[0]
            features = inner_model.forward_features(x)
            patch_tokens = features.get('x_norm_patchtokens', features.get('x_prenorm'))
            if patch_tokens is None: raise ValueError(f"Failed to get patch tokens from DINOv2 forward_features: {features.keys()}")
            cls_token = patch_tokens.mean(dim=1)
            patch_feat_grid = None
            if grid_size and len(grid_size) == 2: # Use grid_size calculated at init time
                try:
                    if patch_tokens.dim() == 3 and patch_tokens.shape[1] == grid_size[0] * grid_size[1]:
                         patch_feat_grid = patch_tokens.permute(0, 2, 1).reshape(B, feature_dim, grid_size[0], grid_size[1])
                    else:
                         # --- FIX: Log warning only once ---
                         if not _logged_dinov2_mismatch_warning:
                             logger.warning(f"DINOv2: Actual patch token count ({patch_tokens.shape[1]}) "
                                            f"differs from expected grid size ({grid_size[0]}x{grid_size[1]}) "
                                            f"derived from pre-training. Cannot reshape to grid format. "
                                            f"Using sequence output (B, N, D). Downstream head should handle this. (This warning will only appear once.)")
                             _logged_dinov2_mismatch_warning = True # Set flag after logging
                         # --- End Fix ---
                except Exception as reshape_e: logger.warning(f"Error reshaping DINOv2 patch tokens: {reshape_e}")
            else: logger.debug("DINOv2 grid size unknown or invalid. Patch features returned as sequence.")
            return {'cls_token': cls_token, 'patch_features': patch_feat_grid if patch_feat_grid is not None else patch_tokens}

        model.forward_features_dict = lambda x: forward_features_dict(model, x)
        model.forward = model.forward_features_dict
        logger.info(f"Loaded DINOv2 backbone: {backbone_type} | Pretrained: True | Feature Dim: {feature_dim} | Pre-train Patches: {num_patches} | Patch Size: {patch_size}")
        return model, feature_dim, num_patches, patch_size, grid_size # Return original num_patches/grid_size
    except Exception as e: logger.error(f"Failed to initialize backbone {backbone_type}: {e}", exc_info=True); raise

# (get_backbone function remains the same)
def get_backbone(backbone_type, pretrained=True, **kwargs):
    """Main function to load various backbone types."""
    if backbone_type.startswith('dinov2_'):
        return get_dinov2_backbone(backbone_type, pretrained, **kwargs)
    else:
        return get_timm_backbone(backbone_type, pretrained, **kwargs)