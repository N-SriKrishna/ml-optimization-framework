"""
Test Advanced Quantization Methods
Compare GPTQ, AWQ, and SmoothQuant vs standard quantization
"""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimizers.advanced_quantization import (
    GPTQQuantizer,
    AWQQuantizer,
    SmoothQuantOptimizer,
    AdvancedQuantConfig
)
from src.optimizers.quantizer import ONNXQuantizer, QuantizationConfig
from src.utils.logger import logger
from src.utils.helpers import get_model_size_mb


def generate_dummy_calibration_data(num_samples=100):
    """Generate dummy calibration data"""
    for _ in range(num_samples):
        yield np.random.randn(1, 3, 640, 640).astype(np.float32)


def test_gptq():
    """Test GPTQ quantization"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING GPTQ QUANTIZATION")
    logger.info("=" * 80)
    
    model_path = Path('models/yolov8n.onnx')
    if not model_path.exists():
        logger.error("Model not found. Please run YOLOv8 optimization first.")
        return
    
    # Configure GPTQ
    config = AdvancedQuantConfig(
        method='gptq',
        bits=4,  # 4-bit quantization
        group_size=128,
        act_order=True
    )
    
    # Create quantizer
    quantizer = GPTQQuantizer(config)
    
    # Generate calibration data
    calibration_data = generate_dummy_calibration_data(50)
    
    # Quantize
    output_path = Path('outputs/advanced_quant/yolov8n_gptq_4bit.onnx')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        result = quantizer.quantize(
            model_path,
            output_path,
            calibration_data=calibration_data
        )
        
        # Compare sizes
        original_size = get_model_size_mb(model_path)
        gptq_size = get_model_size_mb(result)
        
        logger.info(f"\n📊 GPTQ Results:")
        logger.info(f"  Original: {original_size:.2f} MB")
        logger.info(f"  GPTQ 4-bit: {gptq_size:.2f} MB")
        logger.info(f"  Compression: {original_size/gptq_size:.2f}×")
        
        return result
        
    except Exception as e:
        logger.error(f"GPTQ test failed: {e}")
        return None


def test_awq():
    """Test AWQ quantization"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING AWQ QUANTIZATION")
    logger.info("=" * 80)
    
    model_path = Path('models/yolov8n.onnx')
    if not model_path.exists():
        logger.error("Model not found.")
        return
    
    # Configure AWQ
    config = AdvancedQuantConfig(
        method='awq',
        bits=4,
        group_size=128
    )
    
    # Create quantizer
    quantizer = AWQQuantizer(config)
    
    # Generate calibration data (AWQ needs activation statistics)
    calibration_data = list(generate_dummy_calibration_data(50))
    
    # Quantize
    output_path = Path('outputs/advanced_quant/yolov8n_awq_4bit.onnx')
    
    try:
        result = quantizer.quantize(
            model_path,
            output_path,
            calibration_data=calibration_data
        )
        
        # Compare sizes
        original_size = get_model_size_mb(model_path)
        awq_size = get_model_size_mb(result)
        
        logger.info(f"\n📊 AWQ Results:")
        logger.info(f"  Original: {original_size:.2f} MB")
        logger.info(f"  AWQ 4-bit: {awq_size:.2f} MB")
        logger.info(f"  Compression: {original_size/awq_size:.2f}×")
        
        return result
        
    except Exception as e:
        logger.error(f"AWQ test failed: {e}")
        return None


def test_smoothquant():
    """Test SmoothQuant optimization"""
    logger.info("\n" + "=" * 80)
    logger.info("TESTING SMOOTHQUANT")
    logger.info("=" * 80)
    
    model_path = Path('models/yolov8n.onnx')
    if not model_path.exists():
        logger.error("Model not found.")
        return
    
    # Create optimizer
    optimizer = SmoothQuantOptimizer(alpha=0.5)
    
    # Optimize
    output_path = Path('outputs/advanced_quant/yolov8n_smoothquant.onnx')
    
    try:
        result = optimizer.optimize(model_path, output_path)
        
        logger.info(f"\n✓ SmoothQuant optimization complete")
        logger.info(f"  Output: {result}")
        
        return result
        
    except Exception as e:
        logger.error(f"SmoothQuant test failed: {e}")
        return None


def compare_all_methods():
    """Compare standard INT8 vs advanced methods"""
    logger.info("\n" + "=" * 80)
    logger.info("COMPARING ALL QUANTIZATION METHODS")
    logger.info("=" * 80)
    
    model_path = Path('models/yolov8n.onnx')
    if not model_path.exists():
        logger.error("Model not found. Please ensure yolov8n.onnx exists.")
        return
    
    original_size = get_model_size_mb(model_path)
    
    results = {
        'Original (FP32)': {
            'size_mb': original_size,
            'compression': 1.0,
            'notes': 'Baseline model'
        }
    }
    
    # Test standard INT8
    logger.info("\n--- Standard INT8 Dynamic ---")
    try:
        int8_config = QuantizationConfig(quantization_type='dynamic', precision='int8')
        int8_quantizer = ONNXQuantizer(int8_config)
        int8_path = Path('outputs/advanced_quant/yolov8n_int8.onnx')
        int8_path.parent.mkdir(parents=True, exist_ok=True)
        int8_quantizer.quantize(model_path, int8_path)
        
        int8_size = get_model_size_mb(int8_path)
        results['INT8 Dynamic'] = {
            'size_mb': int8_size,
            'compression': original_size / int8_size,
            'notes': 'Standard dynamic quantization'
        }
    except Exception as e:
        logger.warning(f"INT8 test failed: {e}")
    
    # Test GPTQ
    logger.info("\n--- GPTQ 4-bit ---")
    gptq_result = test_gptq()
    if gptq_result:
        gptq_size = get_model_size_mb(gptq_result)
        results['GPTQ 4-bit'] = {
            'size_mb': gptq_size,
            'compression': original_size / gptq_size,
            'notes': 'Gradient-based PTQ with 4-bit precision'
        }
    
    # Test AWQ
    logger.info("\n--- AWQ 4-bit ---")
    awq_result = test_awq()
    if awq_result:
        awq_size = get_model_size_mb(awq_result)
        results['AWQ 4-bit'] = {
            'size_mb': awq_size,
            'compression': original_size / awq_size,
            'notes': 'Activation-aware 4-bit quantization'
        }
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("QUANTIZATION METHODS COMPARISON")
    print("=" * 80)
    print(f"\n{'Method':<20} {'Size (MB)':<12} {'Compression':<12} {'Notes':<30}")
    print("-" * 80)
    
    for method, data in results.items():
        print(f"{method:<20} {data['size_mb']:<12.2f} {data['compression']:<12.2f}× {data['notes']:<30}")
    
    print("\n" + "=" * 80)
    
    # Key insights
    print("\n💡 Key Insights:")
    
    if 'GPTQ 4-bit' in results and 'INT8 Dynamic' in results:
        gptq_vs_int8 = results['GPTQ 4-bit']['compression'] / results['INT8 Dynamic']['compression']
        print(f"  • GPTQ provides {gptq_vs_int8:.2f}× more compression than INT8")
    
    if 'AWQ 4-bit' in results and 'INT8 Dynamic' in results:
        awq_vs_int8 = results['AWQ 4-bit']['compression'] / results['INT8 Dynamic']['compression']
        print(f"  • AWQ provides {awq_vs_int8:.2f}× more compression than INT8")
    
    print(f"  • Advanced methods enable sub-byte precision (2-4 bits)")
    print(f"  • Ideal for LLMs and large vision models")
    print(f"  • Trade-off: More compression but potential accuracy impact")


def main():
    """Main test function"""
    
    print("\n" + "=" * 80)
    print("ADVANCED QUANTIZATION METHODS TEST SUITE")
    print("Testing GPTQ, AWQ, and SmoothQuant")
    print("=" * 80)
    
    # Check if model exists
    if not Path('models/yolov8n.onnx').exists():
        print("\n❌ Error: yolov8n.onnx not found")
        print("\nPlease run first:")
        print("  python examples/real_world/01_download_yolov8.py")
        print("  python examples/real_world/02_optimize_yolov8.py")
        return
    
    # Run comprehensive comparison
    compare_all_methods()
    
    print("\n" + "=" * 80)
    print("✓ Advanced quantization test complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
