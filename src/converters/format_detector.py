"""
Automatic detection of model format
"""
from pathlib import Path
from typing import Literal, Union
import struct

from src.utils.logger import logger

ModelFormat = Literal['pytorch', 'tensorflow', 'tflite', 'onnx', 'unknown']


class FormatDetector:
    """
    Automatically detect model format from file
    """
    
    @staticmethod
    def detect(model_path: Union[str, Path]) -> ModelFormat:
        """
        Detect model format from file path and content
        
        Args:
            model_path: Path to model file or directory
        
        Returns:
            ModelFormat: Detected format type
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            logger.error(f"Model path does not exist: {model_path}")
            return 'unknown'
        
        # Check by extension first
        extension = model_path.suffix.lower()
        
        if extension in ['.pt', '.pth']:
            if FormatDetector._is_pytorch(model_path):
                logger.info(f"Detected PyTorch model: {model_path.name}")
                return 'pytorch'
        
        elif extension == '.onnx':
            if FormatDetector._is_onnx(model_path):
                logger.info(f"Detected ONNX model: {model_path.name}")
                return 'onnx'
        
        elif extension == '.tflite':
            if FormatDetector._is_tflite(model_path):
                logger.info(f"Detected TFLite model: {model_path.name}")
                return 'tflite'
        
        elif extension in ['.pb', ''] and model_path.is_dir():
            if FormatDetector._is_tensorflow_savedmodel(model_path):
                logger.info(f"Detected TensorFlow SavedModel: {model_path.name}")
                return 'tensorflow'
        
        # Fallback: check by content
        logger.warning(f"Unknown extension '{extension}', checking content...")
        
        if model_path.is_file():
            # Check ONNX first (has clear signature)
            if FormatDetector._is_onnx(model_path):
                logger.info(f"Detected ONNX model by content: {model_path.name}")
                return 'onnx'
            
            # Check TFLite (has magic bytes)
            if FormatDetector._is_tflite(model_path):
                logger.info(f"Detected TFLite model by content: {model_path.name}")
                return 'tflite'
            
            # Check PyTorch last (most expensive check)
            if FormatDetector._is_pytorch(model_path):
                logger.info(f"Detected PyTorch model by content: {model_path.name}")
                return 'pytorch'
        
        elif model_path.is_dir():
            if FormatDetector._is_tensorflow_savedmodel(model_path):
                logger.info(f"Detected TensorFlow SavedModel by content: {model_path.name}")
                return 'tensorflow'
        
        logger.error(f"Could not detect format for: {model_path}")
        return 'unknown'
    
    @staticmethod
    def _is_pytorch(file_path: Path) -> bool:
        """Check if file is PyTorch model"""
        try:
            import torch
            
            # Try loading as PyTorch with weights_only=False for compatibility
            try:
                # First try with weights_only=False (PyTorch 2.0+)
                checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
                return True
            except TypeError:
                # Fallback for older PyTorch versions
                checkpoint = torch.load(file_path, map_location='cpu')
                return True
            
        except Exception as e:
            # Not a valid PyTorch file
            logger.debug(f"PyTorch check failed: {e}")
            return False
    
    @staticmethod
    def _is_onnx(file_path: Path) -> bool:
        """Check if file is ONNX model"""
        try:
            import onnx
            
            # Try loading as ONNX
            model = onnx.load(str(file_path))
            return True
            
        except Exception as e:
            logger.debug(f"ONNX check failed: {e}")
            return False
    
    @staticmethod
    def _is_tflite(file_path: Path) -> bool:
        """Check if file is TFLite model"""
        try:
            # TFLite files start with specific magic bytes
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                
                # TFLite magic number: "TFL3" (0x544C4633)
                # Actually it's a different format - check for flatbuffer signature
                # FlatBuffer files start with a specific pattern
                
                # Reset and check more thoroughly
                f.seek(0)
                header = f.read(8)
                
                # Check for common TFLite patterns
                # TFLite uses FlatBuffers format
                if len(header) >= 4:
                    # Check for FlatBuffer root table offset
                    # This is a heuristic check
                    try:
                        # Try to validate as TFLite using TensorFlow
                        import tensorflow as tf
                        interpreter = tf.lite.Interpreter(model_path=str(file_path))
                        interpreter.allocate_tensors()
                        return True
                    except:
                        pass
            
            return False
            
        except Exception as e:
            logger.debug(f"TFLite check failed: {e}")
            return False
    
    @staticmethod
    def _is_tensorflow_savedmodel(dir_path: Path) -> bool:
        """Check if directory is TensorFlow SavedModel"""
        try:
            # SavedModel must have saved_model.pb and variables directory
            pb_file = dir_path / 'saved_model.pb'
            variables_dir = dir_path / 'variables'
            assets_dir = dir_path / 'assets'
            
            # Must have saved_model.pb
            if not pb_file.exists():
                return False
            
            # Usually has variables directory (but not always for inference-only models)
            # So we check for either variables or that it's a valid protobuf
            if variables_dir.exists():
                return True
            
            # Check if saved_model.pb is a valid protobuf
            try:
                with open(pb_file, 'rb') as f:
                    # Read first few bytes to check if it's a protobuf
                    header = f.read(10)
                    if len(header) > 0:
                        # Basic check - protobufs typically start with field tags
                        return True
            except:
                pass
            
            return False
            
        except Exception as e:
            logger.debug(f"TensorFlow SavedModel check failed: {e}")
            return False


# Convenience function
def detect_format(model_path: Union[str, Path]) -> ModelFormat:
    """Detect model format (convenience wrapper)"""
    return FormatDetector.detect(model_path)
