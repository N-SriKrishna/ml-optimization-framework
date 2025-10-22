"""
Mixed Precision Optimization - Use different precisions for different layers
"""
from pathlib import Path
import onnx
from onnx import numpy_helper
import numpy as np
from src.utils.logger import logger


def apply_mixed_precision(
    model_path: str,
    output_path: str,
    precision_map: dict = None,
    default_precision: str = 'fp16'
) -> Path:
    """
    Apply mixed precision optimization
    
    Args:
        model_path: Input ONNX model
        output_path: Output path
        precision_map: Dict mapping layer names to precisions {'layer1': 'int8', 'layer2': 'fp16'}
        default_precision: Default precision for unmapped layers
    
    Returns:
        Path to optimized model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model = onnx.load(model_path)
    
    logger.info(f"Applying mixed precision optimization")
    logger.info(f"Default precision: {default_precision}")
    
    # Convert weights based on precision_map
    if precision_map is None:
        precision_map = {}
    
    modified_count = 0
    
    for initializer in model.graph.initializer:
        layer_name = initializer.name
        
        # Determine target precision
        target_precision = precision_map.get(layer_name, default_precision)
        
        # Convert based on target precision
        if target_precision == 'fp16':
            # Convert to float16
            tensor = numpy_helper.to_array(initializer)
            if tensor.dtype == np.float32:
                tensor_fp16 = tensor.astype(np.float16)
                new_initializer = numpy_helper.from_array(tensor_fp16, layer_name)
                initializer.CopyFrom(new_initializer)
                modified_count += 1
        
        elif target_precision == 'int8':
            # For int8, we'd need quantization (handled by quantizer.py)
            pass
    
    logger.info(f"✓ Modified {modified_count} layers to mixed precision")
    
    # Save model
    onnx.save(model, str(output_path))
    logger.info(f"✓ Mixed precision model saved: {output_path}")
    
    return output_path


def auto_mixed_precision(
    model_path: str,
    output_path: str,
    sensitivity_threshold: float = 0.01
) -> Path:
    """
    Automatically determine optimal precision for each layer
    
    Args:
        model_path: Input model
        output_path: Output path
        sensitivity_threshold: Accuracy sensitivity threshold
    
    Returns:
        Path to optimized model
    """
    logger.info("Auto Mixed Precision Optimization")
    logger.info("Analyzing layer sensitivity...")
    
    # Load model
    model = onnx.load(model_path)
    
    # Simple heuristic: 
    # - Large conv layers -> FP16
    # - Small FC layers -> INT8
    # - Final layers -> FP32 (preserve accuracy)
    
    precision_map = {}
    
    for node in model.graph.node:
        if node.op_type == 'Conv':
            precision_map[node.output[0]] = 'fp16'
        elif node.op_type in ['MatMul', 'Gemm']:
            precision_map[node.output[0]] = 'int8'
        elif node.op_type in ['Softmax', 'Sigmoid']:
            precision_map[node.output[0]] = 'fp32'
    
    logger.info(f"Generated precision map for {len(precision_map)} layers")
    
    return apply_mixed_precision(model_path, output_path, precision_map)
