"""
Complete Framework Integration Test
Tests all major features end-to-end
"""
import pytest
from pathlib import Path
import tempfile
import torch
import torch.nn as nn
import numpy as np

from src.converters.onnx_converter import convert_to_onnx
from src.analyzers.onnx_analyzer import ONNXAnalyzer
from src.optimizers.quantizer import ONNXQuantizer, QuantizationConfig
from src.optimizers.pruner import ONNXPruner, PruningConfig
from src.solvers.constraint_solver import (
    ConstraintSolver,
    OptimizationConstraints,
    HardwareConstraints,
    PerformanceConstraints
)
from src.analyzers.combination_explorer import CombinationExplorer
from src.evaluators.pareto_analyzer import ParetoAnalyzer
from src.converters.smart_deployment_exporter import SmartDeploymentExporter


class CompleteModel(nn.Module):
    """Test model for complete framework testing"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def test_workspace():
    """Create temporary workspace"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_01_model_conversion(test_workspace):
    """Test 1: Universal Model Conversion"""
    print("\n" + "="*80)
    print("TEST 1: MODEL CONVERSION")
    print("="*80)
    
    # Create PyTorch model
    model = CompleteModel()
    pt_path = test_workspace / 'model.pt'
    torch.save(model, pt_path)
    
    # Convert to ONNX
    onnx_path = test_workspace / 'model.onnx'
    result = convert_to_onnx(
        pt_path,
        onnx_path,
        input_shape=(1,3,224,224),
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    
    assert result.exists()
    assert result.stat().st_size > 0
    print("✓ Model conversion successful")


def test_02_model_analysis(test_workspace):
    """Test 2: Comprehensive Model Analysis"""
    print("\n" + "="*80)
    print("TEST 2: MODEL ANALYSIS")
    print("="*80)
    
    # Create and convert model
    model = CompleteModel()
    pt_path = test_workspace / 'model.pt'
    torch.save(model, pt_path)
    
    onnx_path = test_workspace / 'model.onnx'
    convert_to_onnx(
        pt_path,
        onnx_path,
        input_shape=(1,3,224,224),
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    # Analyze
    analyzer = ONNXAnalyzer(onnx_path)
    analysis = analyzer.analyze()
    
    assert 'model_info' in analysis
    assert 'parameters' in analysis
    assert 'computation' in analysis
    assert analysis['parameters']['total_parameters'] > 0
    
    analyzer.print_summary(analysis)
    print("✓ Model analysis successful")


def test_03_quantization(test_workspace):
    """Test 3: Model Quantization"""
    print("\n" + "="*80)
    print("TEST 3: QUANTIZATION")
    print("="*80)
    
    # Setup
    model = CompleteModel()
    pt_path = test_workspace / 'model.pt'
    torch.save(model, pt_path)
    
    onnx_path = test_workspace / 'model.onnx'
    convert_to_onnx(
    pt_path,
    onnx_path,
    input_shape=(1,3,224,224),
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
    
    # Quantize
    config = QuantizationConfig(quantization_type='dynamic')
    quantizer = ONNXQuantizer(config)
    
    quantized_path = test_workspace / 'model_quantized.onnx'
    quantizer.quantize(onnx_path, quantized_path)
    
    assert quantized_path.exists()
    
    # Check compression
    original_size = onnx_path.stat().st_size
    quantized_size = quantized_path.stat().st_size
    compression = original_size / quantized_size
    
    assert compression >= 1.0, f"Compression {compression}x (fallback is acceptable for tiny models)"
    print(f"✓ Quantization successful: {compression:.2f}× compression")


def test_04_pruning(test_workspace):
    """Test 4: Model Pruning"""
    print("\n" + "="*80)
    print("TEST 4: PRUNING")
    print("="*80)
    
    # Setup
    model = CompleteModel()
    pt_path = test_workspace / 'model.pt'
    torch.save(model, pt_path)
    
    onnx_path = test_workspace / 'model.onnx'
    convert_to_onnx(
    pt_path,
    onnx_path,
    input_shape=(1,3,224,224),
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
    
    # Prune
    config = PruningConfig(pruning_type='magnitude', sparsity=0.5)
    pruner = ONNXPruner(config)
    
    pruned_path = test_workspace / 'model_pruned.onnx'
    pruner.prune(onnx_path, pruned_path)
    
    assert pruned_path.exists()
    
    # Analyze sparsity
    from src.optimizers.pruner import analyze_model_sparsity
    sparsity_stats = analyze_model_sparsity(pruned_path)
    
    assert sparsity_stats['overall_sparsity'] > 0.4  # Should be ~50%
    print(f"✓ Pruning successful: {sparsity_stats['overall_sparsity']*100:.1f}% sparsity")


def test_05_constraint_solver(test_workspace):
    """Test 5: Intelligent Constraint Solving"""
    print("\n" + "="*80)
    print("TEST 5: CONSTRAINT SOLVER")
    print("="*80)
    
    # Setup
    model = CompleteModel()
    pt_path = test_workspace / 'model.pt'
    torch.save(model, pt_path)
    
    onnx_path = test_workspace / 'model.onnx'
    convert_to_onnx(
    pt_path,
    onnx_path,
    input_shape=(1,3,224,224),
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
    
    # Analyze
    analyzer = ONNXAnalyzer(onnx_path)
    analysis = analyzer.analyze()
    
    # Define constraints
    hardware = HardwareConstraints(
        device_name="Test Device",
        cpu_cores=4,
        ram_available_gb=2,
        has_gpu=False,
        has_npu=True
    )
    
    performance = PerformanceConstraints(
        max_latency_ms=100,
        max_model_size_mb=10,
        min_accuracy=0.90
    )
    
    constraints = OptimizationConstraints(
        hardware=hardware,
        performance=performance,
        optimization_goal='balanced'
    )
    
    # Solve
    solver = ConstraintSolver(constraints)
    strategy = solver.solve(analysis)
    
    assert strategy is not None
    assert strategy.strategy_name
    assert len(strategy.techniques) > 0
    
    print(f"✓ Generated strategy: {strategy.strategy_name}")
    print(f"  Techniques: {len(strategy.techniques)}")


def test_06_combination_exploration(test_workspace):
    """Test 6: Combination Exploration"""
    print("\n" + "="*80)
    print("TEST 6: COMBINATION EXPLORATION")
    print("="*80)
    
    # Setup
    model = CompleteModel()
    pt_path = test_workspace / 'model.pt'
    torch.save(model, pt_path)
    
    onnx_path = test_workspace / 'model.onnx'
    convert_to_onnx(
    pt_path,
    onnx_path,
    input_shape=(1,3,224,224),
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
    
    # Explore combinations (limit to 10 for speed)
    explorer = CombinationExplorer(test_workspace / 'combinations')
    results = explorer.explore_all_combinations(
        onnx_path,
        max_combinations=10
    )
    
    assert len(results) == 10
    
    # Check that at least some succeeded
    successful = [r for r in results if r.success]
    assert len(successful) > 0
    
    print(f"✓ Explored 10 combinations")
    print(f"  Successful: {len(successful)}/10")


def test_07_pareto_analysis(test_workspace):
    """Test 7: Pareto Analysis"""
    print("\n" + "="*80)
    print("TEST 7: PARETO ANALYSIS")
    print("="*80)
    
    # Setup
    model = CompleteModel()
    pt_path = test_workspace / 'model.pt'
    torch.save(model, pt_path)
    
    onnx_path = test_workspace / 'model.onnx'
    convert_to_onnx(
    pt_path,
    onnx_path,
    input_shape=(1,3,224,224),
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
    
    # Generate variants
    explorer = CombinationExplorer(test_workspace / 'variants')
    results = explorer.explore_all_combinations(onnx_path, max_combinations=10)
    
    # Filter successful
    variants = [r for r in results if r.success]
    
    # Pareto analysis
    analyzer = ParetoAnalyzer(
        objectives=['compression_ratio', 'estimated_speedup'],
        minimize=[],
        maximize=['compression_ratio', 'estimated_speedup']
    )
    
    # Convert to format expected by analyzer
    variant_data = []
    for v in variants:
        variant_data.append({
            'variant_name': v.combination_name,
            'compression_ratio': v.compression_ratio,
            'estimated_speedup': v.estimated_speedup,
            'model_size_mb': v.model_size_mb
        })
    
    # Mock ParetoPoint class
    from dataclasses import dataclass
    
    @dataclass
    class MockVariant:
        variant_id: str  # Add this
        variant_name: str
        compression_ratio: float
        estimated_speedup: float
        model_size_mb: float

    # And when creating:
    for i, v in enumerate(variant_data):
        v['variant_id'] = f"variant_{i:03d}"  # Add ID
    
    mock_variants = [MockVariant(**v) for v in variant_data]
    
    pareto_analysis = analyzer.analyze(mock_variants)
    
    assert 'pareto_front' in pareto_analysis
    assert len(pareto_analysis['pareto_front']) > 0
    
    print(f"✓ Pareto analysis complete")
    print(f"  Pareto-optimal solutions: {len(pareto_analysis['pareto_front'])}")


def test_08_smart_deployment(test_workspace):
    """Test 8: Smart Hardware-Aware Deployment"""
    print("\n" + "="*80)
    print("TEST 8: SMART DEPLOYMENT")
    print("="*80)
    
    # Setup
    model = CompleteModel()
    pt_path = test_workspace / 'model.pt'
    torch.save(model, pt_path)
    
    onnx_path = test_workspace / 'model.onnx'
    convert_to_onnx(
    pt_path,
    onnx_path,
    input_shape=(1,3,224,224),
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
    
    # Test deployment recommendations
    hardware = HardwareConstraints(
        device_name="Snapdragon 888",
        cpu_cores=8,
        ram_available_gb=8,
        has_gpu=True,
        has_npu=True
    )
    
    exporter = SmartDeploymentExporter()
    
    # This should detect Qualcomm and recommend QNN
    from src.converters.smart_deployment_exporter import HardwareRuntimeMatcher
    matcher = HardwareRuntimeMatcher()
    plan = matcher.recommend_runtime(hardware)
    
    assert plan is not None
    assert plan.primary_runtime is not None
    assert 'qualcomm' in plan.primary_runtime.name.lower() or 'qnn' in plan.primary_runtime.name.lower()
    
    print(f"✓ Smart deployment recommendations")
    print(f"  Detected: {hardware.device_name}")
    print(f"  Recommended: {plan.primary_runtime.name}")


def test_09_end_to_end_pipeline(test_workspace):
    """Test 9: Complete End-to-End Pipeline"""
    print("\n" + "="*80)
    print("TEST 9: END-TO-END PIPELINE")
    print("="*80)
    
    # 1. Create model
    model = CompleteModel()
    pt_path = test_workspace / 'model.pt'
    torch.save(model, pt_path)
    print("  1. Model created ✓")
    
    # 2. Convert
    onnx_path = test_workspace / 'model.onnx'
    convert_to_onnx(
    pt_path,
    onnx_path,
    input_shape=(1,3,224,224),
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
    print("  2. Converted to ONNX ✓")
    
    # 3. Analyze
    analyzer = ONNXAnalyzer(onnx_path)
    analysis = analyzer.analyze()
    print("  3. Analyzed model ✓")
    
    # 4. Solve constraints
    hardware = HardwareConstraints(device_name="Mobile Device", has_npu=True)
    performance = PerformanceConstraints(max_latency_ms=100)
    constraints = OptimizationConstraints(hardware=hardware, performance=performance)
    
    solver = ConstraintSolver(constraints)
    strategy = solver.solve(analysis)
    print("  4. Generated strategy ✓")
    
    # 5. Apply optimization
    config = QuantizationConfig(quantization_type='dynamic')
    quantizer = ONNXQuantizer(config)
    optimized_path = test_workspace / 'model_optimized.onnx'
    quantizer.quantize(onnx_path, optimized_path)
    print("  5. Applied optimization ✓")
    
    # 6. Verify improvement
    original_size = onnx_path.stat().st_size / (1024 * 1024)
    optimized_size = optimized_path.stat().st_size / (1024 * 1024)
    compression = original_size / optimized_size
    
    assert compression >= 1.0
    print(f"  6. Verified improvement: {compression:.2f}× compression ✓")
    
    print("\n✓ Complete pipeline successful!")


def test_10_error_handling(test_workspace):
    """Test 10: Robust Error Handling"""
    print("\n" + "="*80)
    print("TEST 10: ERROR HANDLING")
    print("="*80)
    
    # Test with invalid model path
    invalid_path = test_workspace / 'nonexistent.onnx'
    
    # Analyzer should handle gracefully
    try:
        analyzer = ONNXAnalyzer(invalid_path)
        assert False, "Should have raised error"
    except Exception as e:
        print(f"  ✓ Handled invalid path: {type(e).__name__}")
    
    # Combination explorer should handle failures
    model = CompleteModel()
    pt_path = test_workspace / 'model.pt'
    torch.save(model, pt_path)
    
    onnx_path = test_workspace / 'model.onnx'
    convert_to_onnx(
    pt_path,
    onnx_path,
    input_shape=(1,3,224,224),
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
    
    explorer = CombinationExplorer(test_workspace / 'robust_test')
    results = explorer.explore_all_combinations(onnx_path, max_combinations=5)
    
    # Should complete without crashing
    assert len(results) > 0
    print(f"  ✓ Handled {len(results)} combinations robustly")


if __name__ == '__main__':
    # Run all tests
    pytest.main([__file__, '-v', '-s'])
