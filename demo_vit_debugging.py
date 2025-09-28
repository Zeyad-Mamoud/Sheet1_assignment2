#!/usr/bin/env python3
"""
Vision Transformer Debugging Demonstration
==========================================

This script demonstrates how to use the Vision Transformer with debugging
capabilities to trace tensor shapes at every processing step.

Usage:
    python demo_vit_debugging.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from vit_debugger import VisionTransformer, DebugTracker, create_sample_input
import os


def create_test_image(size: tuple = (224, 224)) -> torch.Tensor:
    """
    Create a test image with recognizable patterns for debugging.
    
    Args:
        size: Image size (height, width)
        
    Returns:
        Processed image tensor ready for ViT input
    """
    # Create a colorful test image with patterns
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    # Add some patterns for visual debugging
    # Red gradient in top-left
    img[:size[0]//2, :size[1]//2, 0] = np.linspace(0, 255, size[0]//2)[:, None]
    
    # Green gradient in top-right
    img[:size[0]//2, size[1]//2:, 1] = np.linspace(0, 255, size[1]//2)[None, :]
    
    # Blue gradient in bottom-left
    img[size[0]//2:, :size[1]//2, 2] = np.linspace(0, 255, size[0]//2)[:, None]
    
    # Mixed pattern in bottom-right
    x, y = np.meshgrid(np.linspace(0, 1, size[1]//2), np.linspace(0, 1, size[0]//2))
    img[size[0]//2:, size[1]//2:, :] = (np.stack([x, y, x*y]) * 255).transpose(1, 2, 0)
    
    # Convert to PIL Image and then to tensor
    pil_img = Image.fromarray(img)
    
    # Apply standard ImageNet preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(pil_img).unsqueeze(0)  # Add batch dimension


def demonstrate_patch_extraction(model, input_img):
    """Demonstrate how images are converted to patches."""
    print("\n🖼️  PATCH EXTRACTION DEMONSTRATION")
    print("="*50)
    
    patch_embed = model.patch_embed
    patch_size = patch_embed.patch_size
    
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Image size: {input_img.shape[2]}x{input_img.shape[3]}")
    print(f"Number of patches: {patch_embed.n_patches}")
    print(f"Patches per dimension: {input_img.shape[2] // patch_size}")
    
    # Extract patches step by step
    with torch.no_grad():
        # Apply conv2d to get patches
        patches_conv = patch_embed.patch_embed(input_img)
        print(f"After convolution: {patches_conv.shape}")
        
        # Flatten
        patches_flat = patches_conv.flatten(2)
        print(f"After flattening: {patches_flat.shape}")
        
        # Transpose
        patches_final = patches_flat.transpose(1, 2)
        print(f"Final patches: {patches_final.shape}")


def demonstrate_attention_patterns():
    """Demonstrate attention pattern extraction (simplified)."""
    print("\n👁️  ATTENTION PATTERNS DEMONSTRATION")
    print("="*50)
    
    # Create a simplified attention pattern for visualization
    seq_len = 197  # 196 patches + 1 CLS token for 224x224 image with 16x16 patches
    
    # Simulated attention pattern (normally extracted from model)
    attention_pattern = np.random.rand(seq_len, seq_len)
    attention_pattern = attention_pattern / attention_pattern.sum(axis=1, keepdims=True)
    
    print(f"Attention matrix shape: {attention_pattern.shape}")
    print(f"Attention from CLS token to patches (first row): {attention_pattern[0, 1:6]}")
    print(f"Self-attention within patches: {attention_pattern[1:6, 1:6].mean():.4f}")


def run_comprehensive_debugging():
    """Run a comprehensive debugging session."""
    print("🚀 COMPREHENSIVE VISION TRANSFORMER DEBUGGING")
    print("="*60)
    
    # Create model with debugging enabled
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        in_channels=3,
        n_classes=1000,
        embed_dim=768,
        n_layers=12,
        n_heads=12,
        debug_mode=True
    )
    
    print(f"✅ Created ViT model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create test input
    test_input = create_test_image()
    print(f"✅ Created test image with shape: {test_input.shape}")
    
    # Demonstrate patch extraction
    demonstrate_patch_extraction(model, test_input)
    
    # Run full forward pass with debugging
    print("\n🔄 RUNNING FULL FORWARD PASS WITH DEBUGGING")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        logits = model(test_input)
    
    print(f"✅ Forward pass completed!")
    print(f"Output shape: {logits.shape}")
    print(f"Predicted class: {torch.argmax(logits, dim=1).item()}")
    print(f"Max confidence: {torch.softmax(logits, dim=1).max().item():.4f}")
    
    # Show debugging summary
    print("\n📋 DEBUGGING SUMMARY")
    print("="*60)
    print(model.get_debug_summary())
    
    # Demonstrate attention patterns
    demonstrate_attention_patterns()
    
    return model, test_input, logits


def create_debug_visualization():
    """Create visualization of the debugging process."""
    print("\n📊 CREATING DEBUG VISUALIZATION")
    print("="*50)
    
    # This would create actual visualizations in a real implementation
    # For now, we'll just print what would be visualized
    
    visualizations = [
        "1. Input image with patch boundaries overlaid",
        "2. Individual patches as a grid",
        "3. Attention heatmaps for each head",
        "4. Feature evolution through transformer layers",
        "5. Classification token attention pattern"
    ]
    
    print("Potential visualizations:")
    for viz in visualizations:
        print(f"   {viz}")
    
    print("\n💡 To implement these visualizations:")
    print("   - Use matplotlib for plotting attention heatmaps")
    print("   - Use PIL/OpenCV for image patch visualization")
    print("   - Use seaborn for correlation matrices")
    print("   - Save intermediate activations for layer-wise analysis")


def save_debug_report(model, input_tensor, output_logits, filename="debug_report.txt"):
    """Save a detailed debug report to file."""
    print(f"\n💾 SAVING DEBUG REPORT TO {filename}")
    print("="*50)
    
    with open(filename, 'w') as f:
        f.write("VISION TRANSFORMER DEBUGGING REPORT\n")
        f.write("="*50 + "\n\n")
        
        f.write("MODEL CONFIGURATION:\n")
        f.write(f"- Image size: {input_tensor.shape[2]}x{input_tensor.shape[3]}\n")
        f.write(f"- Patch size: {model.patch_embed.patch_size}x{model.patch_embed.patch_size}\n")
        f.write(f"- Number of patches: {model.patch_embed.n_patches}\n")
        f.write(f"- Embedding dimension: {model.embed_dim}\n")
        f.write(f"- Number of layers: {len(model.blocks)}\n")
        f.write(f"- Number of classes: {model.n_classes}\n")
        f.write(f"- Total parameters: {sum(p.numel() for p in model.parameters()):,}\n\n")
        
        f.write("FORWARD PASS RESULTS:\n")
        f.write(f"- Input shape: {input_tensor.shape}\n")
        f.write(f"- Output shape: {output_logits.shape}\n")
        f.write(f"- Predicted class: {torch.argmax(output_logits, dim=1).item()}\n")
        f.write(f"- Max confidence: {torch.softmax(output_logits, dim=1).max().item():.4f}\n\n")
        
        f.write("DETAILED DEBUGGING LOG:\n")
        f.write(model.get_debug_summary())
    
    print(f"✅ Debug report saved to {filename}")


if __name__ == "__main__":
    print("🔍 Vision Transformer Debugging Assignment")
    print("This script demonstrates comprehensive debugging of Vision Transformers")
    print("showing tensor shapes at every processing step.\n")
    
    try:
        # Run the main debugging demonstration
        model, test_input, output = run_comprehensive_debugging()
        
        # Create visualization info
        create_debug_visualization()
        
        # Save debug report
        save_debug_report(model, test_input, output)
        
        print("\n✅ ALL DEBUGGING DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("\nKey debugging features demonstrated:")
        print("  ✓ Input image patchification process")
        print("  ✓ Patch embedding with shape tracking")
        print("  ✓ Positional embedding addition")
        print("  ✓ Multi-head attention computation")
        print("  ✓ Feed-forward network processing")
        print("  ✓ Transformer block residual connections")
        print("  ✓ Classification head final processing")
        print("  ✓ Complete tensor shape tracking throughout")
        
    except Exception as e:
        print(f"❌ Error during debugging demonstration: {e}")
        import traceback
        traceback.print_exc()