"""
Vision Transformer with Debugging Capabilities
===============================================

This module implements a Vision Transformer (ViT) with comprehensive debugging
features to trace tensor shapes at every processing step.

Author: Vision Transformer Debugging Assignment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from PIL import Image
import torchvision.transforms as transforms


@dataclass
class DebugInfo:
    """Container for debugging information at each step."""
    step_name: str
    tensor_shape: tuple
    tensor_stats: dict = field(default_factory=dict)
    additional_info: dict = field(default_factory=dict)


class DebugTracker:
    """Tracks and logs debugging information throughout ViT processing."""
    
    def __init__(self):
        self.debug_log: List[DebugInfo] = []
        self.verbose = True
    
    def log_step(self, step_name: str, tensor: torch.Tensor, **kwargs):
        """Log a processing step with tensor information."""
        stats = {
            'mean': tensor.mean().item(),
            'std': tensor.std().item(),
            'min': tensor.min().item(),
            'max': tensor.max().item(),
            'dtype': str(tensor.dtype)
        }
        
        debug_info = DebugInfo(
            step_name=step_name,
            tensor_shape=tuple(tensor.shape),
            tensor_stats=stats,
            additional_info=kwargs
        )
        
        self.debug_log.append(debug_info)
        
        if self.verbose:
            print(f"🔍 {step_name}:")
            print(f"   Shape: {tuple(tensor.shape)}")
            print(f"   Stats: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
            print(f"   Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            if kwargs:
                print(f"   Info: {kwargs}")
            print()
    
    def get_summary(self) -> str:
        """Generate a summary of all debugging steps."""
        summary = "🔍 Vision Transformer Debug Summary\n" + "="*50 + "\n"
        for i, debug_info in enumerate(self.debug_log, 1):
            summary += f"{i:2d}. {debug_info.step_name}\n"
            summary += f"    Shape: {debug_info.tensor_shape}\n"
            summary += f"    Mean: {debug_info.tensor_stats['mean']:.4f}, "
            summary += f"Std: {debug_info.tensor_stats['std']:.4f}\n\n"
        return summary


class PatchEmbedding(nn.Module):
    """
    Converts input image into patches and embeds them.
    Includes detailed debugging of the patchification process.
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, embed_dim: int = 768, 
                 debug_tracker: Optional[DebugTracker] = None):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.debug_tracker = debug_tracker
        
        # Patch embedding using convolution
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with debugging.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Embedded patches of shape (batch_size, n_patches, embed_dim)
        """
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "Input Image", x, 
                image_size=(x.shape[2], x.shape[3]),
                channels=x.shape[1]
            )
        
        # Apply patch embedding (convolution)
        x = self.patch_embed(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "After Patch Embedding (Conv2d)", x,
                n_patches_h=x.shape[2],
                n_patches_w=x.shape[3],
                embed_dim=x.shape[1]
            )
        
        # Flatten patches: (B, embed_dim, H//P, W//P) -> (B, embed_dim, n_patches)
        x = x.flatten(2)
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "Flattened Patches", x,
                total_patches=x.shape[2]
            )
        
        # Transpose: (B, embed_dim, n_patches) -> (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "Transposed Patches (Final Patch Embedding)", x,
                sequence_length=x.shape[1],
                feature_dim=x.shape[2]
            )
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with detailed debugging.
    """
    
    def __init__(self, embed_dim: int = 768, n_heads: int = 12, 
                 dropout: float = 0.1, debug_tracker: Optional[DebugTracker] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.debug_tracker = debug_tracker
        
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with detailed attention debugging.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim)
        """
        B, N, C = x.shape
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "MultiHead Attention Input", x,
                batch_size=B, seq_len=N, embed_dim=C,
                n_heads=self.n_heads, head_dim=self.head_dim
            )
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # (B, N, 3*C)
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "QKV Linear Projection", qkv,
                total_dim=qkv.shape[2]
            )
        
        # Reshape and split into Q, K, V
        qkv = qkv.reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "Query (Q) Matrix", q,
                heads=q.shape[1], seq_len=q.shape[2], head_dim=q.shape[3]
            )
            self.debug_tracker.log_step(
                "Key (K) Matrix", k,
                heads=k.shape[1], seq_len=k.shape[2], head_dim=k.shape[3]
            )
            self.debug_tracker.log_step(
                "Value (V) Matrix", v,
                heads=v.shape[1], seq_len=v.shape[2], head_dim=v.shape[3]
            )
        
        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn_scores = (q @ k.transpose(-2, -1)) * scale  # (B, n_heads, N, N)
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "Attention Scores (Q@K^T)", attn_scores,
                scale_factor=scale,
                attention_matrix_size=(attn_scores.shape[2], attn_scores.shape[3])
            )
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "Attention Weights (after softmax)", attn_weights,
                sum_per_row=attn_weights.sum(dim=-1).mean().item()
            )
        
        # Apply attention to values
        out = attn_weights @ v  # (B, n_heads, N, head_dim)
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "Attention Output (Attn@V)", out
            )
        
        # Concatenate heads
        out = out.transpose(1, 2).reshape(B, N, C)  # (B, N, embed_dim)
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "Concatenated Heads", out,
                final_embed_dim=out.shape[2]
            )
        
        # Final projection
        out = self.proj(out)
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "Final Attention Projection", out
            )
        
        return out


class FeedForward(nn.Module):
    """
    Feed-forward network with debugging.
    """
    
    def __init__(self, embed_dim: int = 768, hidden_dim: int = 3072, 
                 dropout: float = 0.1, debug_tracker: Optional[DebugTracker] = None):
        super().__init__()
        self.debug_tracker = debug_tracker
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with debugging."""
        if self.debug_tracker:
            self.debug_tracker.log_step("FFN Input", x)
        
        # First linear layer
        x = self.fc1(x)
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "FFN First Linear", x,
                hidden_dim=x.shape[2]
            )
        
        # GELU activation
        x = F.gelu(x)
        
        if self.debug_tracker:
            self.debug_tracker.log_step("FFN After GELU", x)
        
        x = self.dropout(x)
        
        # Second linear layer
        x = self.fc2(x)
        
        if self.debug_tracker:
            self.debug_tracker.log_step("FFN Output", x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer encoder block with debugging.
    """
    
    def __init__(self, embed_dim: int = 768, n_heads: int = 12, 
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 debug_tracker: Optional[DebugTracker] = None):
        super().__init__()
        self.debug_tracker = debug_tracker
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, n_heads, dropout, debug_tracker)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FeedForward(embed_dim, int(embed_dim * mlp_ratio), dropout, debug_tracker)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connections and debugging."""
        if self.debug_tracker:
            self.debug_tracker.log_step("Transformer Block Input", x)
        
        # First residual connection: attention
        residual = x
        x = self.norm1(x)
        
        if self.debug_tracker:
            self.debug_tracker.log_step("After LayerNorm1", x)
        
        x = self.attn(x)
        x = x + residual
        
        if self.debug_tracker:
            self.debug_tracker.log_step("After Attention + Residual", x)
        
        # Second residual connection: feed-forward
        residual = x
        x = self.norm2(x)
        
        if self.debug_tracker:
            self.debug_tracker.log_step("After LayerNorm2", x)
        
        x = self.ffn(x)
        x = x + residual
        
        if self.debug_tracker:
            self.debug_tracker.log_step("After FFN + Residual (Block Output)", x)
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer with comprehensive debugging capabilities.
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, 
                 in_channels: int = 3, n_classes: int = 1000, 
                 embed_dim: int = 768, n_layers: int = 12, 
                 n_heads: int = 12, mlp_ratio: float = 4.0, 
                 dropout: float = 0.1, debug_mode: bool = True):
        super().__init__()
        
        self.debug_tracker = DebugTracker() if debug_mode else None
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size, patch_size, in_channels, embed_dim, self.debug_tracker
        )
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout, self.debug_tracker)
            for _ in range(n_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with comprehensive debugging.
        
        Args:
            x: Input image tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Classification logits of shape (batch_size, n_classes)
        """
        B = x.shape[0]
        
        # Patch embedding and patchification
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, n_patches+1, embed_dim)
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "After Adding CLS Token", x,
                cls_token_shape=cls_tokens.shape,
                total_sequence_length=x.shape[1]
            )
        
        # Add positional embedding
        x = x + self.pos_embed
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "After Positional Embedding", x,
                pos_embed_shape=self.pos_embed.shape
            )
        
        # Apply transformer blocks
        for i, block in enumerate(self.blocks):
            if self.debug_tracker:
                self.debug_tracker.log_step(f"Before Transformer Block {i+1}", x)
            
            x = block(x)
            
            if self.debug_tracker:
                self.debug_tracker.log_step(f"After Transformer Block {i+1}", x)
        
        # Classification head
        x = self.norm(x)
        
        if self.debug_tracker:
            self.debug_tracker.log_step("After Final LayerNorm", x)
        
        # Use only the class token for classification
        cls_token_final = x[:, 0]  # (B, embed_dim)
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "Extracted CLS Token", cls_token_final,
                cls_token_index=0
            )
        
        # Final classification
        logits = self.head(cls_token_final)  # (B, n_classes)
        
        if self.debug_tracker:
            self.debug_tracker.log_step(
                "Final Classification Logits", logits,
                n_classes=logits.shape[1]
            )
        
        return logits
    
    def get_debug_summary(self) -> str:
        """Get debugging summary."""
        if self.debug_tracker:
            return self.debug_tracker.get_summary()
        return "Debug mode is disabled."
    
    def visualize_attention(self, x: torch.Tensor, layer_idx: int = -1) -> np.ndarray:
        """
        Visualize attention patterns from a specific layer.
        
        Args:
            x: Input image tensor
            layer_idx: Which transformer layer to visualize (-1 for last layer)
            
        Returns:
            Attention weights as numpy array
        """
        # This is a simplified version for visualization
        # In practice, you'd need to modify the attention module to return weights
        with torch.no_grad():
            # Forward pass up to the desired layer
            x = self.patch_embed(x)
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            x = x + self.pos_embed
            
            target_layer = self.blocks[layer_idx] if layer_idx >= 0 else self.blocks[-1]
            x = self.norm1(x) if hasattr(target_layer, 'norm1') else x
            
            # Get attention weights (simplified)
            # Note: This would need modification to the attention module to return weights
            return np.random.rand(x.shape[1], x.shape[1])  # Placeholder


def create_sample_input(batch_size: int = 1, img_size: int = 224) -> torch.Tensor:
    """Create a sample input tensor for testing."""
    return torch.randn(batch_size, 3, img_size, img_size)


def demonstrate_vit_debugging():
    """
    Demonstrate the Vision Transformer with debugging capabilities.
    """
    print("🚀 Vision Transformer Debugging Demonstration")
    print("=" * 60)
    
    # Create a smaller ViT for demonstration
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        n_classes=10,  # Smaller for demo
        embed_dim=384,  # Smaller for demo
        n_layers=6,     # Fewer layers for demo
        n_heads=6,      # Fewer heads for demo
        debug_mode=True
    )
    
    # Create sample input
    sample_input = create_sample_input(batch_size=2, img_size=224)
    
    print(f"Created ViT model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Input shape: {sample_input.shape}")
    print("\n" + "="*60)
    print("🔍 STARTING FORWARD PASS WITH DEBUGGING")
    print("="*60 + "\n")
    
    # Forward pass with debugging
    with torch.no_grad():
        logits = model(sample_input)
    
    print("="*60)
    print("📊 FINAL RESULTS")
    print("="*60)
    print(f"Output logits shape: {logits.shape}")
    print(f"Predicted classes: {torch.argmax(logits, dim=1)}")
    
    print("\n" + "="*60)
    print("📋 DEBUGGING SUMMARY")
    print("="*60)
    print(model.get_debug_summary())
    
    return model, sample_input, logits


if __name__ == "__main__":
    # Run the demonstration
    model, input_tensor, output_logits = demonstrate_vit_debugging()