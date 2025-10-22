"""
Example: Convert a model to ONNX format
"""
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.converters.onnx_converter import convert_to_onnx
from src.utils.logger import logger


def main():
    """Example conversion workflow"""
    
    # Example 1: Convert PyTorch model
    logger.info("=" * 60)
    logger.info("Example 1: PyTorch Model Conversion")
    logger.info("=" * 60)
    
    # Assuming you have a PyTorch model file
    # model_path = Path("path/to/your/model.pt")
    # output_path = Path("outputs/model.onnx")
    
    # convert_to_onnx(
    #     model_path=model_path,
    #     output_path=output_path,
    #     input_shape=(1, 3, 640, 640),  # Adjust for your model
    #     input_names=['images'],
    #     output_names=['output']
    # )
    
    logger.info("\nTo use this script:")
    logger.info("1. Uncomment the code above")
    logger.info("2. Set your model_path")
    logger.info("3. Run: python examples/convert_model.py")
    
    print("\n" + "=" * 60)
    print("Conversion pipeline is ready!")
    print("=" * 60)


if __name__ == '__main__':
    main()
