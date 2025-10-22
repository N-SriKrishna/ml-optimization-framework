"""
Test model format detection and conversion
"""
import pytest
from pathlib import Path
import torch
import torch.nn as nn
import tempfile

from src.converters.format_detector import detect_format
from src.converters.onnx_converter import convert_to_onnx


class SimpleModel(nn.Module):
    """Simple test model"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_format_detection():
    """Test format detection"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create PyTorch model
        model = SimpleModel()
        pt_path = tmpdir / 'model.pt'
        torch.save(model, pt_path)
        
        # Test detection
        assert detect_format(pt_path) == 'pytorch'
        print("✓ Format detection test passed")


def test_pytorch_to_onnx_conversion():
    """Test PyTorch to ONNX conversion"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and save PyTorch model
        model = SimpleModel()
        pt_path = tmpdir / 'model.pt'
        torch.save(model, pt_path)
        
        # Convert to ONNX
        onnx_path = tmpdir / 'model.onnx'
        result_path = convert_to_onnx(
            pt_path,
            onnx_path,
            input_shape=(1, 3, 224, 224)
        )
        
        # Verify conversion
        assert result_path.exists()
        assert detect_format(result_path) == 'onnx'
        
        print("✓ PyTorch to ONNX conversion test passed")


if __name__ == '__main__':
    test_format_detection()
    test_pytorch_to_onnx_conversion()
    print("\n✓ All tests passed!")
