"""
Example: Complete optimization workflow
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.converters.onnx_converter import convert_to_onnx
from src.analyzers.onnx_analyzer import ONNXAnalyzer
from src.optimizers.quantizer import quantize_dynamic_int8, quantize_static_int8
from src.optimizers.pruner import prune_magnitude_global, analyze_model_sparsity
from src.utils.logger import logger


def main():
    """Complete optimization example"""
    
    print("\n" + "=" * 80)
    print("ML MODEL OPTIMIZATION FRAMEWORK - Example Workflow")
    print("=" * 80)
    
    # Example workflow (commented out - requires actual model)
    
    # Step 1: Convert to ONNX
    logger.info("\n### Step 1: Convert Model to ONNX ###")
    # model_path = Path("path/to/your/model.pt")
    # onnx_path = Path("outputs/model.onnx")
    # convert_to_onnx(model_path, onnx_path, input_shape=(1, 3, 640, 640))
    
    # Step 2: Analyze Model
    logger.info("\n### Step 2: Analyze Model ###")
    # analyzer = ONNXAnalyzer(onnx_path)
    # analysis = analyzer.analyze()
    # analyzer.print_summary(analysis)
    
    # Step 3: Apply Quantization
    logger.info("\n### Step 3: Apply Quantization ###")
    # quantized_path = Path("outputs/model_quantized.onnx")
    # quantize_dynamic_int8(onnx_path, quantized_path)
    
    # Step 4: Apply Pruning
    logger.info("\n### Step 4: Apply Pruning ###")
    # pruned_path = Path("outputs/model_pruned.onnx")
    # prune_magnitude_global(onnx_path, pruned_path, sparsity=0.5)
    
    # Step 5: Combined Optimization
    logger.info("\n### Step 5: Combined Optimization (Quantization + Pruning) ###")
    # First prune, then quantize
    # pruned_quantized_path = Path("outputs/model_pruned_quantized.onnx")
    # prune_magnitude_global(onnx_path, pruned_path, sparsity=0.5)
    # quantize_dynamic_int8(pruned_path, pruned_quantized_path)
    
    # Step 6: Analyze Results
    logger.info("\n### Step 6: Analyze Optimization Results ###")
    # sparsity = analyze_model_sparsity(pruned_quantized_path)
    # logger.info(f"Final sparsity: {sparsity['overall_sparsity']*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("To use this example:")
    print("1. Uncomment the code sections")
    print("2. Provide your model path")
    print("3. Run: python examples/optimize_model.py")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
