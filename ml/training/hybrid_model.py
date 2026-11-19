"""
TerraSpectra - Hybrid 3D-CNN + Vision Transformer (ViT) Architecture
=====================================================================
Module: ml.training.hybrid_model
Domain: Hyperspectral Deep Learning & Precision Agriculture
Input: 3D Spatial-Spectral Patches (Batch, 1, Depth=16, Height=16, Width=16)
Classes:
  0: Non-crop Soil / Access Tracks
  1: Healthy Crop Canopy
  2: Pre-Symptomatic Fungal Infection (High Risk)

Architectural Flow:
  1. 3D-CNN Frontend: Joint spatial-spectral convolution capturing localized
     chlorophyll, carotenoid, and cellular structural patterns.
  2. Spectral-Spatial Tokenizer: Reshapes 3D feature volumes into sequence tokens.
  3. Vision Transformer (ViT) Encoder: Multi-Head Self-Attention modeling
     long-range cross-band correlations across the light spectrum.
  4. Classification Head: Projects latent embeddings to class probabilities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Hybrid3DCNN_ViT(nn.Module):
    """
    Hybrid Deep Learning Network combining 3D Convolutions with a Vision Transformer.
    """
    def __init__(
        self,
        in_channels: int = 1,
        spectral_depth: int = 16,
        patch_size: int = 16,
        num_classes: int = 3,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_vit_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.spectral_depth = spectral_depth
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # -------------------------------------------------------------
        # 1. 3D Convolutional Frontend (Joint Spectral-Spatial Convolutions)
        # -------------------------------------------------------------
        # Input shape: (B, 1, Depth, Height, Width)
        self.conv3d_1 = nn.Conv3d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)
        )
        self.bn1 = nn.BatchNorm3d(8)
        
        self.conv3d_2 = nn.Conv3d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3, 3),
            padding=(1, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(16)
        
        # Max pool along spatial dimensions to reduce token count
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        # Spatial dimensions after pooling: Height/2, Width/2
        pooled_h = patch_size // 2
        pooled_w = patch_size // 2
        
        # Linear projection to Transformer embedding dimension
        self.token_proj = nn.Linear(16 * pooled_h * pooled_w, embed_dim)

        # -------------------------------------------------------------
        # 2. Vision Transformer (ViT) Encoder
        # -------------------------------------------------------------
        # Sequence length = spectral_depth (each spectral slice is a token)
        self.pos_embedding = nn.Parameter(torch.randn(1, spectral_depth, embed_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 2,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )
        self.vit_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_vit_layers)

        # -------------------------------------------------------------
        # 3. Multi-Class Classification Head
        # -------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        x: (Batch, 1, Depth, Height, Width) or (Batch, Depth, Height, Width)
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Add channel dimension: (B, 1, D, H, W)
            
        B, C, D, H, W = x.shape
        
        # 1. 3D Convolution Feature Extraction
        out = F.relu(self.bn1(self.conv3d_1(x)))
        out = F.relu(self.bn2(self.conv3d_2(out)))
        out = self.pool(out)  # Shape: (B, 16, D, H//2, W//2)
        
        # 2. Tokenize across spectral depth
        # Permute to (B, D, 16, H//2, W//2) -> flatten spatial features per spectral slice
        out = out.permute(0, 2, 1, 3, 4).contiguous()
        out = out.view(B, D, -1)  # Shape: (B, Depth, 16 * H//2 * W//2)
        
        # Project to embedding space and add positional encodings
        tokens = self.token_proj(out) + self.pos_embedding  # Shape: (B, Depth, embed_dim)
        
        # 3. Vision Transformer Self-Attention
        vit_out = self.vit_encoder(tokens)  # Shape: (B, Depth, embed_dim)
        
        # Global spectral average pooling
        pooled_spectral = vit_out.mean(dim=1)  # Shape: (B, embed_dim)
        
        # 4. Classification Logits
        logits = self.classifier(pooled_spectral)
        return logits

def test_model_forward():
    """Sanity check verifying tensor shape transformations."""
    model = Hybrid3DCNN_ViT(spectral_depth=16, patch_size=16, num_classes=3)
    dummy_input = torch.randn(4, 1, 16, 16, 16)
    output = model(dummy_input)
    print(f"[*] Input Tensor Shape:  {dummy_input.shape}")
    print(f"[*] Output Logits Shape: {output.shape} (Batch=4, Classes=3)")
    assert output.shape == (4, 3), "Invalid output shape!"
    print("[+] Hybrid 3D-CNN + ViT forward pass verified successfully.")

if __name__ == "__main__":
    test_model_forward()
