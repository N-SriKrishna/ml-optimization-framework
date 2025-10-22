"""
Test ONNX model analyzer
"""
import pytest
from pathlib import Path
import torch
import torch.nn as nn
import tempfile

from src.converters.onnx_converter import convert_to_onnx
from src.analyzers.onnx_analyzer import ONNXAnalyzer


class TestModel(nn.Module):
    """Test model for analysis"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def test_model_analysis():
    """Test complete model analysis"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and convert model
        model = TestModel()
        pt_path = tmpdir / 'model.pt'
        torch.save(model, pt_path)
        
        onnx_path = tmpdir / 'model.onnx'
        convert_to_onnx(
            pt_path,
            onnx_path,
            input_shape=(1, 3, 224, 224)
        )
        
        # Analyze model
        analyzer = ONNXAnalyzer(onnx_path)
        analysis = analyzer.analyze()
        
        # Verify analysis results
        assert 'model_info' in analysis
        assert 'architecture' in analysis
        assert 'parameters' in analysis
        assert 'operations' in analysis
        assert 'memory' in analysis
        assert 'computation' in analysis
        assert 'optimization_potential' in analysis
        
        # Print summary
        analyzer.print_summary(analysis)
        
        print("✓ Model analysis test passed")


if __name__ == '__main__':
    test_model_analysis()
