# models/fusion_modules.py
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class ConcatFusion(nn.Module):
    """Simple fusion by concatenation."""
    def forward(self, x_views):
        # x_views shape: (batch, num_views, dim)
        batch_size = x_views.shape[0]
        fused = x_views.reshape(batch_size, -1) # Concatenate features
        logger.debug(f"ConcatFusion input shape: {x_views.shape}, output shape: {fused.shape}")
        return fused

class AttentionFusion(nn.Module):
    """
    Fusion using Transformer Encoder layers for self-attention across view features.
    """
    def __init__(self, dim, depth=2, heads=6, mlp_ratio=4.0, num_views=4):
        """
        Args:
            dim (int): Input feature dimension (embedding dim of backbone).
            depth (int): Number of Transformer Encoder layers.
            heads (int): Number of attention heads.
            mlp_ratio (float): Ratio for hidden dimension in MLP block.
            num_views (int): Number of input views.
        """
        super().__init__()
        self.num_views = num_views
        self.dim = dim

        # Simple learnable positional embedding for the view dimension
        self.pos_embed = nn.Parameter(torch.zeros(1, num_views, dim))

        try:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=int(dim * mlp_ratio),
                activation='gelu', # GELU is common in Transformers
                batch_first=True   # Input format (batch, seq, feature)
            )
            self.transformer_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=depth
            )
        except Exception as e:
             logger.error(f"Error creating TransformerEncoderLayer/Encoder: {e}. Check parameters dim={dim}, heads={heads}.")
             raise

        self.norm = nn.LayerNorm(dim)
        logger.info(f"Initialized AttentionFusion: dim={dim}, depth={depth}, heads={heads}")


    def forward(self, x_views):
        """
        Args:
            x_views (torch.Tensor): Features from different views, shape (batch, num_views, dim).

        Returns:
            torch.Tensor: Fused features, shape (batch, dim).
        """
        if x_views.shape[1] != self.num_views or x_views.shape[2] != self.dim:
             raise ValueError(f"Input shape mismatch in AttentionFusion. Expected (B, {self.num_views}, {self.dim}), got {x_views.shape}")

        batch_size = x_views.shape[0]

        # Add positional embedding
        x = x_views + self.pos_embed

        # Apply transformer encoder (self-attention across views)
        x = self.transformer_encoder(x)
        x = self.norm(x)

        # Pool features across views (simple averaging)
        # Other options: use a dedicated [FUSION] token, attention pooling
        fused = x.mean(dim=1)
        logger.debug(f"AttentionFusion input shape: {x_views.shape}, output shape: {fused.shape}")
        return fused