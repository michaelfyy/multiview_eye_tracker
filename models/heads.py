# models/heads.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

logger = logging.getLogger(__name__)

# (RegressionHead remains the same)
class RegressionHead(nn.Module):
    """Simple MLP head for coordinate regression."""
    def __init__(self, in_features, out_features, hidden_dims=[512], activation=nn.ReLU, dropout_p=0.3):
        """
        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output coordinates/values.
            hidden_dims (list[int]): List of hidden layer dimensions.
            activation (nn.Module): Activation function class (ensure inplace=False).
            dropout_p (float): Dropout probability (set to 0.0 to disable).
        """
        super().__init__()
        layers = []
        prev_dim = in_features

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(activation())
            if dropout_p > 0: # Add dropout if prob > 0
                layers.append(nn.Dropout(p=dropout_p)) # No inplace=True
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, out_features))
        self.mlp = nn.Sequential(*layers)
        logger.info(f"Initialized RegressionHead: in={in_features}, out={out_features}, hidden={hidden_dims}")

    def forward(self, x):
        return self.mlp(x)


class HeatmapHead(nn.Module):
    """Heatmap prediction head using Transposed Convolutions."""
    def __init__(self, in_features, output_res=64, decoder_channels=[256, 128]):
        super().__init__()
        self.in_features = in_features
        self.decoder_channels = decoder_channels
        self.output_res = output_res

        self.decoder_layers = nn.ModuleList()
        current_channels = self.in_features
        for i, channels in enumerate(self.decoder_channels):
            self.decoder_layers.append(nn.ConvTranspose2d(current_channels, channels, kernel_size=4, stride=2, padding=1))
            self.decoder_layers.append(nn.BatchNorm2d(channels))
            self.decoder_layers.append(nn.ReLU(inplace=True))
            current_channels = channels

        self.final_conv = nn.Conv2d(current_channels, 1, kernel_size=3, stride=1, padding=1)
        logger.info(f"Initialized HeatmapHead: in_dim={in_features}, decoder={decoder_channels}, target_out_res={output_res}. Final size adjustment in forward.")


    def forward(self, patch_features):
        batch_views, dim_or_n, d_or_h = patch_features.shape[:3]
        grid_h, grid_w = -1, -1
        x = None

        if patch_features.dim() == 3: # Shape (B*V, N, D)
            num_p = dim_or_n
            dim = d_or_h
            if dim != self.in_features: raise ValueError(f"Input feature dimension {dim} != expected {self.in_features}")
            grid_f = math.sqrt(num_p)
            if grid_f != int(grid_f): raise ValueError(f"Num patches {num_p} is not a perfect square.")
            grid_h = grid_w = int(grid_f)
            logger.debug(f"HeatmapHead reshape (B*V, N, D) {patch_features.shape} -> (B*V, D, H, W) grid {grid_h}x{grid_w}")
            x = patch_features.permute(0, 2, 1).reshape(batch_views, dim, grid_h, grid_w)
        elif patch_features.dim() == 4: # Shape (B*V, D, H_p, W_p)
            dim, grid_h, grid_w = patch_features.shape[1:]
            if dim != self.in_features: raise ValueError(f"Input feature dimension {dim} != expected {self.in_features}")
            x = patch_features
            logger.debug(f"HeatmapHead received grid input {x.shape}")
        else: raise ValueError(f"Unsupported shape for HeatmapHead: {patch_features.shape}")

        for layer in self.decoder_layers:
            x = layer(x)

        x = self.final_conv(x) # Output: (B*V, 1, H_inter, W_inter)

        _, _, current_h, current_w = x.shape
        if current_h != self.output_res or current_w != self.output_res:
             logger.debug(f"HeatmapHead resizing from {current_h}x{current_w} to {self.output_res}x{self.output_res}")
             x = F.interpolate(x, size=(self.output_res, self.output_res), mode='bilinear', align_corners=False)

        # --- FIX: Return logits directly, remove sigmoid ---
        return x
        # --- End Fix ---