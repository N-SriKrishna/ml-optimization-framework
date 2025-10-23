"""
Test ONNX quantization
"""
import pytest
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import tempfile

from src.converters.onnx_converter import convert_to_onnx
from src.optimizers.quantizer import (
    ONNXQuantizer,
    QuantizationConfig,
    quantize_dynamic_int8
)


class SimpleModel(nn.Module):
    """Simple model for testing"""
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


def test_dynamic_quantization():
    """Test dynamic quantization"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and convert model
        model = SimpleModel()
        pt_path = tmpdir / 'model.pt'
        torch.save(model, pt_path)
        
        onnx_path = tmpdir / 'model.onnx'
        convert_to_onnx(pt_path, onnx_path, input_shape=(1, 3, 224, 224))
        
        # Quantize
        quantized_path = tmpdir / 'model_quantized.onnx'
        result_path = quantize_dynamic_int8(onnx_path, quantized_path)
        
        assert result_path.exists()
        
        # Check size reduction
        original_size = onnx_path.stat().st_size
        quantized_size = quantized_path.stat().st_size
        
        assert quantized_size < original_size
        print(f"✓ Dynamic quantization test passed")
        print(f"  Size reduction: {(1 - quantized_size/original_size) * 100:.1f}%")


def test_static_quantization():
    """Test static quantization"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and convert model
        model = SimpleModel()
        pt_path = tmpdir / 'model.pt'
        torch.save(model, pt_path)
        
        onnx_path = tmpdir / 'model.onnx'
        convert_to_onnx(
            pt_path,
            onnx_path,
            input_shape=(1, 3, 224, 224),
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )

        
        # Create calibration data generator
        def calibration_data_generator():
            for _ in range(50):
                yield np.random.randn(1, 3, 224, 224).astype(np.float32)
        
        # Quantize
        config = QuantizationConfig(quantization_type='static')
        quantizer = ONNXQuantizer(config)
        
        quantized_path = tmpdir / 'model_quantized.onnx'
        result_path = quantizer.quantize(
            onnx_path,
            quantized_path,
            calibration_data_generator=calibration_data_generator,
            input_name='input',
            num_calibration_samples=50
        )
        
        assert result_path.exists()
        
        print(f"✓ Static quantization test passed")


if __name__ == '__main__':
    test_dynamic_quantization()
    test_static_quantization()
    print("\n✓ All quantization tests passed!")
