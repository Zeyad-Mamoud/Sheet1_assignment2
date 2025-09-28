#!/usr/bin/env python3
"""
Test Suite for Vision Transformer Debugger
==========================================

This script contains tests to validate the Vision Transformer debugging
implementation.
"""

import torch
import numpy as np
import unittest
from vit_debugger import (
    VisionTransformer, 
    DebugTracker, 
    PatchEmbedding, 
    MultiHeadSelfAttention,
    FeedForward,
    TransformerBlock,
    create_sample_input
)


class TestVisionTransformerDebugger(unittest.TestCase):
    """Test cases for the Vision Transformer debugger."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 2
        self.img_size = 224
        self.patch_size = 16
        self.embed_dim = 384
        self.n_heads = 6
        self.n_layers = 4
        self.n_classes = 10
        
        self.test_input = create_sample_input(self.batch_size, self.img_size)
    
    def test_debug_tracker(self):
        """Test the debug tracker functionality."""
        tracker = DebugTracker()
        
        # Test logging
        test_tensor = torch.randn(2, 10, 384)
        tracker.log_step("Test Step", test_tensor, extra_info="test")
        
        self.assertEqual(len(tracker.debug_log), 1)
        self.assertEqual(tracker.debug_log[0].step_name, "Test Step")
        self.assertEqual(tracker.debug_log[0].tensor_shape, (2, 10, 384))
        self.assertIn("extra_info", tracker.debug_log[0].additional_info)
        
        # Test summary generation
        summary = tracker.get_summary()
        self.assertIn("Test Step", summary)
        self.assertIn("(2, 10, 384)", summary)
    
    def test_patch_embedding(self):
        """Test patch embedding with debugging."""
        tracker = DebugTracker()
        patch_embed = PatchEmbedding(
            self.img_size, self.patch_size, 3, self.embed_dim, tracker
        )
        
        output = patch_embed(self.test_input)
        
        # Check output shape
        expected_n_patches = (self.img_size // self.patch_size) ** 2
        expected_shape = (self.batch_size, expected_n_patches, self.embed_dim)
        self.assertEqual(output.shape, expected_shape)
        
        # Check that debugging steps were logged
        self.assertGreater(len(tracker.debug_log), 0)
        step_names = [step.step_name for step in tracker.debug_log]
        self.assertIn("Input Image", step_names)
        self.assertIn("Transposed Patches (Final Patch Embedding)", step_names)
    
    def test_multihead_attention(self):
        """Test multi-head self-attention with debugging."""
        tracker = DebugTracker()
        attention = MultiHeadSelfAttention(
            self.embed_dim, self.n_heads, debug_tracker=tracker
        )
        
        seq_len = 197  # 196 patches + 1 CLS token
        test_input = torch.randn(self.batch_size, seq_len, self.embed_dim)
        
        output = attention(test_input)
        
        # Check output shape
        self.assertEqual(output.shape, test_input.shape)
        
        # Check debugging logs
        self.assertGreater(len(tracker.debug_log), 0)
        step_names = [step.step_name for step in tracker.debug_log]
        self.assertIn("MultiHead Attention Input", step_names)
        self.assertIn("Query (Q) Matrix", step_names)
        self.assertIn("Key (K) Matrix", step_names)
        self.assertIn("Value (V) Matrix", step_names)
        self.assertIn("Attention Scores (Q@K^T)", step_names)
        self.assertIn("Final Attention Projection", step_names)
    
    def test_feedforward(self):
        """Test feed-forward network with debugging."""
        tracker = DebugTracker()
        ffn = FeedForward(self.embed_dim, self.embed_dim * 4, debug_tracker=tracker)
        
        seq_len = 197
        test_input = torch.randn(self.batch_size, seq_len, self.embed_dim)
        
        output = ffn(test_input)
        
        # Check output shape
        self.assertEqual(output.shape, test_input.shape)
        
        # Check debugging logs
        step_names = [step.step_name for step in tracker.debug_log]
        self.assertIn("FFN Input", step_names)
        self.assertIn("FFN Output", step_names)
    
    def test_transformer_block(self):
        """Test transformer block with debugging."""
        tracker = DebugTracker()
        block = TransformerBlock(
            self.embed_dim, self.n_heads, debug_tracker=tracker
        )
        
        seq_len = 197
        test_input = torch.randn(self.batch_size, seq_len, self.embed_dim)
        
        output = block(test_input)
        
        # Check output shape
        self.assertEqual(output.shape, test_input.shape)
        
        # Check debugging logs
        step_names = [step.step_name for step in tracker.debug_log]
        self.assertIn("Transformer Block Input", step_names)
        self.assertIn("After FFN + Residual (Block Output)", step_names)
    
    def test_vision_transformer_shapes(self):
        """Test Vision Transformer with shape validation."""
        model = VisionTransformer(
            img_size=self.img_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_classes=self.n_classes,
            debug_mode=True
        )
        
        with torch.no_grad():
            output = model(self.test_input)
        
        # Check final output shape
        expected_shape = (self.batch_size, self.n_classes)
        self.assertEqual(output.shape, expected_shape)
        
        # Check that debugging was performed
        self.assertIsNotNone(model.debug_tracker)
        self.assertGreater(len(model.debug_tracker.debug_log), 0)
    
    def test_vision_transformer_forward_pass(self):
        """Test complete forward pass with debugging."""
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=384,
            n_layers=2,  # Smaller for faster testing
            n_heads=6,
            n_classes=10,
            debug_mode=True
        )
        
        model.eval()
        with torch.no_grad():
            logits = model(self.test_input)
        
        # Validate output
        self.assertEqual(logits.shape, (self.batch_size, 10))
        
        # Check that all major steps were logged
        debug_log = model.debug_tracker.debug_log
        step_names = [step.step_name for step in debug_log]
        
        # Essential steps that should be present
        essential_steps = [
            "Input Image",
            "After Adding CLS Token",
            "After Positional Embedding",
            "Final Classification Logits"
        ]
        
        for step in essential_steps:
            self.assertIn(step, step_names, f"Missing essential step: {step}")
    
    def test_debug_mode_toggle(self):
        """Test that debug mode can be turned on/off."""
        # Model with debug mode off
        model_no_debug = VisionTransformer(
            embed_dim=384, n_layers=1, n_heads=6, debug_mode=False
        )
        self.assertIsNone(model_no_debug.debug_tracker)
        
        # Model with debug mode on
        model_with_debug = VisionTransformer(
            embed_dim=384, n_layers=1, n_heads=6, debug_mode=True
        )
        self.assertIsNotNone(model_with_debug.debug_tracker)
    
    def test_tensor_statistics(self):
        """Test that tensor statistics are properly computed."""
        tracker = DebugTracker()
        
        # Create tensor with known statistics
        test_tensor = torch.ones(2, 3, 4) * 5.0  # Mean=5, std=0
        tracker.log_step("Test Stats", test_tensor)
        
        stats = tracker.debug_log[0].tensor_stats
        self.assertAlmostEqual(stats['mean'], 5.0, places=4)
        self.assertAlmostEqual(stats['std'], 0.0, places=4)
        self.assertAlmostEqual(stats['min'], 5.0, places=4)
        self.assertAlmostEqual(stats['max'], 5.0, places=4)
    
    def test_model_parameter_count(self):
        """Test that model has expected number of parameters."""
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=384,
            n_layers=6,
            n_heads=6,
            n_classes=1000
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        
        # Should have a reasonable number of parameters (not too small or large)
        self.assertGreater(param_count, 1_000_000)  # At least 1M parameters
        self.assertLess(param_count, 100_000_000)   # Less than 100M parameters


def run_manual_verification():
    """Run manual verification of the debugging system."""
    print("🧪 MANUAL VERIFICATION OF DEBUGGING SYSTEM")
    print("="*50)
    
    # Create a small model for quick testing
    model = VisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=192,
        n_layers=3,
        n_heads=3,
        n_classes=5,
        debug_mode=True
    )
    
    # Test input
    test_input = create_sample_input(batch_size=1, img_size=224)
    
    print(f"✅ Created test model with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"✅ Test input shape: {test_input.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(test_input)
    
    print(f"✅ Forward pass completed, output shape: {output.shape}")
    
    # Verify debugging information
    debug_log = model.debug_tracker.debug_log
    print(f"✅ Captured {len(debug_log)} debugging steps")
    
    # Show key shape transformations
    print("\n🔍 KEY SHAPE TRANSFORMATIONS:")
    key_steps = [
        "Input Image",
        "Transposed Patches (Final Patch Embedding)",
        "After Adding CLS Token",
        "After Positional Embedding",
        "Final Classification Logits"
    ]
    
    for step_name in key_steps:
        matching_steps = [s for s in debug_log if s.step_name == step_name]
        if matching_steps:
            step = matching_steps[0]
            print(f"  {step_name}: {step.tensor_shape}")
    
    print("\n✅ Manual verification completed successfully!")
    return True


if __name__ == "__main__":
    print("🧪 Testing Vision Transformer Debugger")
    print("="*50)
    
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run manual verification
    print("\n" + "="*50)
    run_manual_verification()