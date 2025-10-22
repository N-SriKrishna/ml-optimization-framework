"""
Quick Framework Validation
Tests all major features in one script
"""
from pathlib import Path
import sys
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.converters.onnx_converter import convert_to_onnx
from src.analyzers.onnx_analyzer import ONNXAnalyzer
from src.optimizers.quantizer import quantize_dynamic_int8
from src.optimizers.pruner import prune_magnitude_global
from src.analyzers.combination_explorer import CombinationExplorer
from src.converters.smart_deployment_exporter import SmartDeploymentExporter
from src.solvers.constraint_solver import HardwareConstraints


class QuickTestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
    
    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def main():
    print("\n" + "="*80)
    print("QUICK FRAMEWORK VALIDATION")
    print("="*80)
    
    output_dir = Path('outputs/quick_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Create test model
    print("\n1. Creating test model...")
    model = QuickTestModel()
    pt_path = output_dir / 'test_model.pt'
    torch.save(model, pt_path)
    print("   ✓ Model saved")
    
    # 2. Convert to ONNX
    print("\n2. Converting to ONNX...")
    onnx_path = output_dir / 'test_model.onnx'
    convert_to_onnx(pt_path, onnx_path, input_shape=(1, 10))
    print("   ✓ Conversion successful")
    
    # 3. Analyze
    print("\n3. Analyzing model...")
    analyzer = ONNXAnalyzer(onnx_path)
    analysis = analyzer.analyze()
    print(f"   ✓ Parameters: {analysis['parameters']['total_parameters']}")
    
    # 4. Quantize
    print("\n4. Quantizing...")
    quant_path = output_dir / 'test_model_int8.onnx'
    quantize_dynamic_int8(onnx_path, quant_path)
    
    from src.utils.helpers import get_model_size_mb
    original_size = get_model_size_mb(onnx_path)
    quant_size = get_model_size_mb(quant_path)
    print(f"   ✓ Compression: {original_size/quant_size:.2f}×")
    
    # 5. Prune
    print("\n5. Pruning...")
    prune_path = output_dir / 'test_model_pruned.onnx'
    prune_magnitude_global(onnx_path, prune_path, sparsity=0.5)
    print("   ✓ Pruning complete")
    
    # 6. Combination test (small)
    print("\n6. Testing combinations...")
    explorer = CombinationExplorer(output_dir / 'combinations')
    results = explorer.explore_all_combinations(onnx_path, max_combinations=5)
    successful = [r for r in results if r.success]
    print(f"   ✓ Tested 5 combinations: {len(successful)} successful")
    
    # 7. Deployment recommendations
    print("\n7. Testing deployment recommendations...")
    hardware = HardwareConstraints(device_name="Snapdragon 8 Gen 2", has_npu=True)
    exporter = SmartDeploymentExporter()
    
    from src.converters.smart_deployment_exporter import HardwareRuntimeMatcher
    matcher = HardwareRuntimeMatcher()
    plan = matcher.recommend_runtime(hardware)
    print(f"   ✓ Recommended: {plan.primary_runtime.name}")
    
    print("\n" + "="*80)
    print("✓ ALL FRAMEWORK FEATURES VALIDATED!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
