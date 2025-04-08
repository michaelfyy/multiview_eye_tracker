# models/base_multiview_model.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class ConfigurableMultiViewModel(nn.Module):
    # --- FIX: Add use_separate_2d_heads flag ---
    def __init__(self, backbone, fusion_module, head_2d, head_3d, num_views=4,
                 head_2d_type='regression', use_separate_2d_heads=False):
        # (Docstring updated)
        """
        Initializes the configurable multi-view model.

        Args:
            backbone (nn.Module): Shared backbone network.
            fusion_module (nn.Module | None): Module for fusing features across views.
            head_2d (nn.Module | nn.ModuleList): Head(s) for 2D predictions.
            head_3d (nn.Module | None): Head for 3D predictions.
            num_views (int): Number of camera views.
            head_2d_type (str): 'regression' or 'heatmap'.
            use_separate_2d_heads (bool): Whether head_2d is a ModuleList of separate heads.
        """
        super().__init__()
        self.backbone = backbone
        self.fusion_module = fusion_module
        self.head_2d = head_2d # Can be single Module or ModuleList
        self.head_3d = head_3d
        self.num_views = num_views
        self.head_2d_type = head_2d_type
        self.use_separate_2d_heads = use_separate_2d_heads # Store the flag

        # Input validation for heads
        if self.use_separate_2d_heads:
            if not isinstance(self.head_2d, nn.ModuleList):
                raise TypeError("If use_separate_2d_heads is True, head_2d must be an nn.ModuleList.")
            if len(self.head_2d) != self.num_views:
                raise ValueError(f"ModuleList head_2d must have {self.num_views} heads, found {len(self.head_2d)}.")
            logger.info(f"Using SEPARATE {head_2d_type} 2D heads.")
        else:
             if not isinstance(self.head_2d, nn.Module):
                 raise TypeError("If use_separate_2d_heads is False, head_2d must be an nn.Module.")
             logger.info(f"Using SHARED {head_2d_type} 2D head.")

        fusion_name = type(fusion_module).__name__ if fusion_module else "None"
        logger.info(f"Initialized ConfigurableMultiViewModel with {head_2d_type} 2D head(s) ({'Separate' if use_separate_2d_heads else 'Shared'}) and {fusion_name} fusion.")
        # --- End Fix ---

    def forward(self, x):
        """ Forward pass through the multi-view model. """
        batch_size = x.shape[0]
        if x.dim() != 5 or x.shape[1] != self.num_views: raise ValueError(f"Input tensor shape should be (B, N, C, H, W), got {x.shape}")

        # Reshape for backbone: (B * N, C, H, W)
        x_reshaped = x.view(batch_size * self.num_views, *x.shape[2:])

        # --- Pass through shared backbone ---
        if not hasattr(self.backbone, 'forward_features_dict'): raise AttributeError("Backbone needs 'forward_features_dict'.")
        try: backbone_features = self.backbone.forward_features_dict(x_reshaped)
        except Exception as e: logger.error(f"Backbone forward error: {e}"); raise e

        patch_features = backbone_features.get('patch_features') # (B*N, N_patch, D) or (B*N, D, Hp, Wp)
        cls_tokens = backbone_features.get('cls_token') # Shape (B*N, D)

        # Check required features exist
        if cls_tokens is None: raise ValueError("Backbone missing 'cls_token'. Needed.")
        if self.head_2d_type == 'heatmap' and patch_features is None: raise ValueError("Backbone missing 'patch_features'. Needed for heatmap head.")
        if self.head_2d_type == 'regression' and self.use_separate_2d_heads and cls_tokens is None: raise ValueError("Backbone missing 'cls_token'. Needed for separate regression heads.")
        # Add check for shared regression head input? Assumes cls_tokens if not specified otherwise.

        feature_dim = cls_tokens.shape[-1]
        cls_tokens_views = cls_tokens.view(batch_size, self.num_views, feature_dim) # Shape (B, N, D) for fusion

        outputs = {}
        outputs['pred_2d_type'] = self.head_2d_type # Store type for downstream

        # --- 2D Head Logic ---
        if self.head_2d:
            # Determine input based on head type
            # Shared heatmap uses patch features, shared regression uses cls tokens by default here
            # Separate regression heads will use cls tokens reshaped below
            head_2d_input_shared = patch_features if self.head_2d_type == 'heatmap' else cls_tokens

            # --- FIX: Handle separate vs shared heads ---
            if self.use_separate_2d_heads:
                # Input features need to be (B, N, D)
                # cls_tokens_views is already (B, N, D)
                if head_2d_input_shared is None: raise ValueError("Input required for separate 2D heads is missing.")

                for i in range(self.num_views):
                    view_features = cls_tokens_views[:, i, :] # Get features for view i: (B, D)
                    pred_2d_view = self.head_2d[i](view_features) # Apply i-th head: (B, 2)
                    if pred_2d_view.dim() != 2 or pred_2d_view.shape[-1] != 2: raise ValueError(f"Separate Regression 2D head {i} output shape mismatch: expected (B, 2), got {pred_2d_view.shape}")
                    outputs[f'pupil_2d_cam{i+1}'] = pred_2d_view # Store per-view prediction
            else:
                # Use the single shared head with the appropriate input
                if head_2d_input_shared is None: raise ValueError(f"Input required for shared 2D head type '{self.head_2d_type}' is missing.")

                pred_2d_all_views = self.head_2d(head_2d_input_shared) # Shape (B*N, 2) or (B*N, 1, Hh, Ww)

                # Reshape and store per-view outputs
                if self.head_2d_type == 'heatmap':
                    if pred_2d_all_views.dim() != 4 or pred_2d_all_views.shape[1] != 1: raise ValueError(f"Shared Heatmap head output shape mismatch: expected (B*V, 1, H, W), got {pred_2d_all_views.shape}")
                    heatmap_h, heatmap_w = pred_2d_all_views.shape[-2:]
                    pred_2d_views = pred_2d_all_views.view(batch_size, self.num_views, 1, heatmap_h, heatmap_w)
                    for i in range(self.num_views): outputs[f'pupil_heatmap_cam{i+1}'] = pred_2d_views[:, i]
                else: # Shared regression
                    if pred_2d_all_views.dim() != 2 or pred_2d_all_views.shape[-1] != 2: raise ValueError(f"Shared Regression 2D head output shape mismatch: expected (B*V, 2), got {pred_2d_all_views.shape}")
                    pred_2d_views = pred_2d_all_views.view(batch_size, self.num_views, 2)
                    for i in range(self.num_views): outputs[f'pupil_2d_cam{i+1}'] = pred_2d_views[:, i]
            # --- End Fix ---

        # --- Fusion (Uses cls_tokens_views: B, N, D) ---
        if self.fusion_module:
            try: fused_features = self.fusion_module(cls_tokens_views) # Expect (B, fused_dim)
            except Exception as e: logger.error(f"Fusion module error: {e}"); raise e
        else: # Default: Concatenate CLS tokens if no fusion module
             fused_features = cls_tokens_views.reshape(batch_size, -1) # (B, N * D)

        # --- 3D Head (Input: fused_features) ---
        if self.head_3d:
             pred_3d = self.head_3d(fused_features) # Expect (B, N * 6) based on __init__
             # Check output shape consistency
             expected_3d_out_dim = self.num_views * 6
             if pred_3d.dim() != 2 or pred_3d.shape[-1] != expected_3d_out_dim: raise ValueError(f"3D head output shape mismatch: expected (B, {expected_3d_out_dim}), got {pred_3d.shape}")
             # Reshape and split into per-view pupil and gaze
             pred_3d_views = pred_3d.view(batch_size, self.num_views, 6)
             for i in range(self.num_views):
                 outputs[f'pupil_3d_cam{i+1}'] = pred_3d_views[:, i, 0:3] # (B, 3)
                 outputs[f'gaze_endpoint_3d_cam{i+1}'] = pred_3d_views[:, i, 3:6] # (B, 3)

        return outputs