"""
Test ONNX pruning
"""
import pytest
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import tempfile

from src.converters.onnx_converter import convert_to_onnx
from src.optimizers.pruner import (
    ONNXPruner,
    PruningConfig,
    prune_magnitude_global,
    prune_structured_filters,
    analyze_model_sparsity
)


class TestModel(nn.Module):
    """Test model for pruning"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_magnitude_pruning_global():
    """Test global magnitude pruning"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and convert model
        model = TestModel()
        pt_path = tmpdir / 'model.pt'
        torch.save(model, pt_path)
        
        onnx_path = tmpdir / 'model.onnx'
        convert_to_onnx(pt_path, onnx_path, input_shape=(1, 3, 224, 224))
        
        # Analyze original sparsity
        original_sparsity = analyze_model_sparsity(onnx_path)
        print(f"Original sparsity: {original_sparsity['overall_sparsity']*100:.2f}%")
        
        # Prune with 50% sparsity
        pruned_path = tmpdir / 'model_pruned.onnx'
        result_path = prune_magnitude_global(onnx_path, pruned_path, sparsity=0.5)
        
        # Analyze pruned sparsity
        pruned_sparsity = analyze_model_sparsity(pruned_path)
        print(f"Pruned sparsity: {pruned_sparsity['overall_sparsity']*100:.2f}%")
        
        assert result_path.exists()
        assert pruned_sparsity['overall_sparsity'] > original_sparsity['overall_sparsity']
        
        print(f"✓ Global magnitude pruning test passed")


def test_magnitude_pruning_local():
    """Test local (per-layer) magnitude pruning"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and convert model
        model = TestModel()
        pt_path = tmpdir / 'model.pt'
        torch.save(model, pt_path)
        
        onnx_path = tmpdir / 'model.onnx'
        convert_to_onnx(pt_path, onnx_path, input_shape=(1, 3, 224, 224))
        
        # Prune with local scope
        config = PruningConfig(
            pruning_type='magnitude',
            pruning_scope='local',
            sparsity=0.6
        )
        pruner = ONNXPruner(config)
        
        pruned_path = tmpdir / 'model_pruned_local.onnx'
        pruner.prune(onnx_path, pruned_path)
        
        # Verify
        assert pruned_path.exists()
        
        pruned_sparsity = analyze_model_sparsity(pruned_path)
        print(f"Pruned sparsity (local): {pruned_sparsity['overall_sparsity']*100:.2f}%")
        
        print(f"✓ Local magnitude pruning test passed")


def test_structured_pruning():
    """Test structured filter pruning"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and convert model
        model = TestModel()
        pt_path = tmpdir / 'model.pt'
        torch.save(model, pt_path)
        
        onnx_path = tmpdir / 'model.onnx'
        convert_to_onnx(pt_path, onnx_path, input_shape=(1, 3, 224, 224))
        
        # Analyze original
        original_sparsity = analyze_model_sparsity(onnx_path)
        
        # Prune with structured pruning
        pruned_path = tmpdir / 'model_pruned_structured.onnx'
        result_path = prune_structured_filters(onnx_path, pruned_path, sparsity=0.3)
        
        # Analyze pruned
        pruned_sparsity = analyze_model_sparsity(pruned_path)
        
        assert result_path.exists()
        print(f"Original sparsity: {original_sparsity['overall_sparsity']*100:.2f}%")
        print(f"Pruned sparsity (structured): {pruned_sparsity['overall_sparsity']*100:.2f}%")
        
        print(f"✓ Structured pruning test passed")


def test_random_pruning():
    """Test random pruning (baseline)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and convert model
        model = TestModel()
        pt_path = tmpdir / 'model.pt'
        torch.save(model, pt_path)
        
        onnx_path = tmpdir / 'model.onnx'
        convert_to_onnx(pt_path, onnx_path, input_shape=(1, 3, 224, 224))
        
        # Random pruning
        config = PruningConfig(
            pruning_type='random',
            sparsity=0.5
        )
        pruner = ONNXPruner(config)
        
        pruned_path = tmpdir / 'model_pruned_random.onnx'
        pruner.prune(onnx_path, pruned_path)
        
        # Verify
        assert pruned_path.exists()
        
        pruned_sparsity = analyze_model_sparsity(pruned_path)
        print(f"Random pruned sparsity: {pruned_sparsity['overall_sparsity']*100:.2f}%")
        
        print(f"✓ Random pruning test passed")


def test_sparsity_analysis():
    """Test sparsity analysis"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and convert model
        model = TestModel()
        pt_path = tmpdir / 'model.pt'
        torch.save(model, pt_path)
        
        onnx_path = tmpdir / 'model.onnx'
        convert_to_onnx(pt_path, onnx_path, input_shape=(1, 3, 224, 224))
        
        # Analyze sparsity
        sparsity_stats = analyze_model_sparsity(onnx_path)
        
        print("\nSparsity Analysis:")
        print(f"  Overall sparsity: {sparsity_stats['overall_sparsity']*100:.2f}%")
        print(f"  Total parameters: {sparsity_stats['total_parameters']:,}")
        print(f"  Zero parameters: {sparsity_stats['zero_parameters']:,}")
        print(f"  Non-zero parameters: {sparsity_stats['non_zero_parameters']:,}")
        
        print(f"✓ Sparsity analysis test passed")


if __name__ == '__main__':
    print("Running pruning tests...\n")
    test_magnitude_pruning_global()
    print()
    test_magnitude_pruning_local()
    print()
    test_structured_pruning()
    print()
    test_random_pruning()
    print()
    test_sparsity_analysis()
    print("\n✓ All pruning tests passed!")
