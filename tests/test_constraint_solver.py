"""
Test constraint solver and variant generator
"""
import pytest
from pathlib import Path
import torch
import torch.nn as nn
import tempfile

from src.converters.onnx_converter import convert_to_onnx
from src.analyzers.onnx_analyzer import ONNXAnalyzer
from src.solvers.constraint_solver import (
    ConstraintSolver,
    OptimizationConstraints,
    HardwareConstraints,
    PerformanceConstraints
)
from src.solvers.variant_generator import VariantGenerator


class TestModel(nn.Module):
    """Test model"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_constraint_solver():
    """Test constraint solver"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and convert model
        model = TestModel()
        pt_path = tmpdir / 'model.pt'
        torch.save(model, pt_path)
        
        onnx_path = tmpdir / 'model.onnx'
        convert_to_onnx(pt_path, onnx_path, input_shape=(1, 3, 224, 224))
        
        # Analyze model
        analyzer = ONNXAnalyzer(onnx_path)
        analysis = analyzer.analyze()
        
        # Define constraints
        hardware = HardwareConstraints(
            device_name="Snapdragon 680",
            cpu_cores=4,
            ram_available_gb=2.0,
            has_gpu=False
        )
        
        performance = PerformanceConstraints(
            max_latency_ms=300,
            max_model_size_mb=20,
            min_accuracy=0.90
        )
        
        constraints = OptimizationConstraints(
            hardware=hardware,
            performance=performance,
            optimization_goal='balanced'
        )
        
        # Solve constraints
        solver = ConstraintSolver(constraints)
        strategy = solver.solve(analysis)
        
        # Verify strategy
        assert strategy is not None
        assert strategy.strategy_name is not None
        assert len(strategy.techniques) > 0
        
        print(f"✓ Constraint solver test passed")
        print(f"  Strategy: {strategy.strategy_name}")
        print(f"  Techniques: {len(strategy.techniques)}")


def test_variant_generator():
    """Test variant generator"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and convert model
        model = TestModel()
        pt_path = tmpdir / 'model.pt'
        torch.save(model, pt_path)
        
        onnx_path = tmpdir / 'model.onnx'
        convert_to_onnx(pt_path, onnx_path, input_shape=(1, 3, 224, 224))
        
        # Analyze and solve
        analyzer = ONNXAnalyzer(onnx_path)
        analysis = analyzer.analyze()
        
        constraints = OptimizationConstraints(
            optimization_goal='balanced'
        )
        
        solver = ConstraintSolver(constraints)
        strategy = solver.solve(analysis)
        
        # Generate variants
        output_dir = tmpdir / 'variants'
        generator = VariantGenerator(output_dir)
        variants = generator.generate_variants(onnx_path, strategy, num_variants=3)
        
        # Verify variants
        assert len(variants) == 3
        assert all(v.model_path.exists() for v in variants)
        
        # Print summary
        generator.print_variants_summary(variants)
        
        print(f"✓ Variant generator test passed")
        print(f"  Generated {len(variants)} variants")


if __name__ == '__main__':
    print("Running constraint solver tests...\n")
    test_constraint_solver()
    print()
    test_variant_generator()
    print("\n✓ All tests passed!")
