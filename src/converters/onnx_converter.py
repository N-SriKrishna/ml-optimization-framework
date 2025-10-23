"""
Universal converter to ONNX format
Supports: PyTorch, TensorFlow SavedModel, TFLite, and ONNX
"""
from pathlib import Path
from typing import Optional, Tuple, Union, List, Dict, Any
import tempfile
import shutil
import subprocess
import sys

import onnx
from onnx import checker, helper, shape_inference
import torch
import numpy as np

from src.converters.format_detector import detect_format, ModelFormat
from src.utils.logger import logger
from src.utils.helpers import get_model_size_mb


class ONNXConverter:
    """
    Convert models from various formats to ONNX
    """
    
    def __init__(self, simplify: bool = True, opset_version: int = 14):
        """
        Initialize ONNX converter
        
        Args:
            simplify: Whether to simplify ONNX graph after conversion
            opset_version: ONNX opset version (default: 14 for broad compatibility)
        """
        self.simplify = simplify
        self.opset_version = opset_version
    
    def convert(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path],
        input_shape: Optional[Tuple[int, ...]] = None,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
        **kwargs
    ) -> Path:
        """
        Convert model to ONNX format (auto-detect source format)
        
        Args:
            model_path: Path to input model
            output_path: Path to save ONNX model
            input_shape: Input tensor shape (required for some conversions)
            input_names: Input tensor names
            output_names: Output tensor names
            dynamic_axes: Dynamic axes specification for ONNX export
            **kwargs: Additional conversion parameters
        
        Returns:
            Path: Path to converted ONNX model
        
        Raises:
            ValueError: If format detection fails or conversion is not supported
        """
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Detect format
        format_type = detect_format(model_path)
        
        if format_type == 'unknown':
            raise ValueError(f"Could not detect format for: {model_path}")
        
        logger.info(f"Converting {format_type} model to ONNX...")
        logger.info(f"Input: {model_path}")
        logger.info(f"Output: {output_path}")
        
        # Route to appropriate converter
        if format_type == 'onnx':
            logger.info("Model is already in ONNX format, copying...")
            shutil.copy(model_path, output_path)
        
        elif format_type == 'pytorch':
            self._pytorch_to_onnx(
                model_path, output_path, input_shape,
                input_names, output_names, dynamic_axes, **kwargs
            )
        
        elif format_type == 'tensorflow':
            self._tensorflow_to_onnx(
                model_path, output_path, input_names, output_names, **kwargs
            )
        
        elif format_type == 'tflite':
            self._tflite_to_onnx(model_path, output_path, **kwargs)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Simplify if requested
        if self.simplify and output_path.exists():
            logger.info("Simplifying ONNX graph...")
            self._simplify_onnx(output_path)
        
        # Validate
        self._validate_onnx(output_path)
        
        size_mb = get_model_size_mb(output_path)
        logger.info(f"✓ Conversion complete! Model size: {size_mb:.2f} MB")
        
        return output_path
    
    def _pytorch_to_onnx(
            self,
            model_path: Path,
            output_path: Path,
            input_shape: Optional[Tuple[int, ...]] = None,
            input_names: Optional[List[str]] = None,
            output_names: Optional[List[str]] = None,
            dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
            **kwargs
        ):
            """Convert PyTorch model to ONNX"""
            try:
                logger.info("Loading PyTorch model...")
                device = torch.device('cpu')
                
                checkpoint = torch.load(model_path, map_location=device, weights_only=False)
                model = self._extract_pytorch_model(checkpoint)
                model.eval()
                
                if input_shape is None:
                    input_shape = self._infer_pytorch_input_shape(model, kwargs)
                    logger.warning(f"No input shape provided, using inferred: {input_shape}")
                
                dummy_input = torch.randn(*input_shape, device=device)
                
                if input_names is None:
                    input_names = ['input']
                if output_names is None:
                    output_names = ['output']
                
                if dynamic_axes is None:
                    dynamic_axes = {
                        input_names[0]: {0: 'batch_size'},
                        output_names[0]: {0: 'batch_size'}
                    }
                
                logger.info(f"Exporting with input shape: {input_shape}")
                logger.info(f"Input names: {input_names}")
                logger.info(f"Output names: {output_names}")
                
                try:
                    if dynamic_axes is not None:
                        # Use new dynamo exporter for proper symbolic batch dimension
                        self._export_dynamic_onnx(
                            model,
                            dummy_input,
                            output_path,
                            input_names,
                            output_names,
                            self.opset_version,
                            verbose=False,
                        )
                    else:
                        torch.onnx.export(
                            model,
                            dummy_input,
                            str(output_path),
                            export_params=True,
                            opset_version=self.opset_version,
                            do_constant_folding=True,
                            input_names=input_names,
                            output_names=output_names,
                            dynamic_axes=None,
                            verbose=False,
                        )

                    logger.info("✓ PyTorch to ONNX conversion successful")
                except Exception as e:
                    logger.error(f"Failed during ONNX export: {e}")
                    raise
            except Exception as e:
                logger.error(f"Error in _pytorch_to_onnx: {e}")
                raise


    def _export_dynamic_onnx(
            self,
            model: torch.nn.Module,
            dummy_input: torch.Tensor,
            output_path: Path,
            input_names: List[str],
            output_names: List[str],
            opset_version: int,
            verbose: bool = False,
        ) -> None:
            """Export PyTorch model to ONNX with real symbolic dynamic shapes using torch.onnx.dynamo_export."""
            
            from onnx import save as onnx_save

                # Try modern dynamo exporter (PyTorch 2.4+), else fallback to torch.onnx.export
            try:
                    from torch.onnx import dynamo_export
                    HAS_DYNAMO_EXPORT = True
            except Exception:
                    HAS_DYNAMO_EXPORT = False


            if HAS_DYNAMO_EXPORT:
                    logger.info("Using torch.onnx.dynamo_export for dynamic ONNX export")
                    dynamic_shapes = [{0: "batch"}]
                    onnx_model = dynamo_export(
                        model,
                        dummy_input,
                        dynamic_shapes=dynamic_shapes,
                        export_options=torch.onnx.ExportOptions(
                            opset_version=opset_version,
                            dynamic_shapes=True,
                        ),
                    )
                    onnx_save(onnx_model.model_proto, str(output_path))
            else:
                    logger.warning("torch.onnx.dynamo_export not found — falling back to torch.onnx.export")
                    torch.onnx.export(
                        model,
                        dummy_input,
                        str(output_path),
                        export_params=True,
                        opset_version=opset_version,
                        do_constant_folding=True,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes={input_names[0]: {0: "batch_size"}, output_names[0]: {0: "batch_size"}},
                        verbose=verbose,
                    )



    def _extract_pytorch_model(self, checkpoint: Any) -> torch.nn.Module:
        """Extract PyTorch model from various checkpoint formats"""
        
        # Case 1: Checkpoint is already a model
        if isinstance(checkpoint, torch.nn.Module):
            return checkpoint
        
        # Case 2: Checkpoint is a dictionary
        if isinstance(checkpoint, dict):
            # Try common keys
            if 'model' in checkpoint:
                model = checkpoint['model']
                if isinstance(model, torch.nn.Module):
                    return model
            
            if 'model_state_dict' in checkpoint:
                raise ValueError(
                    "Checkpoint contains only state_dict. "
                    "Please provide a complete model or load state_dict into model architecture first."
                )
            
            if 'state_dict' in checkpoint:
                raise ValueError(
                    "Checkpoint contains only state_dict. "
                    "Please provide a complete model or load state_dict into model architecture first."
                )
            
            # If dictionary doesn't have known keys, assume it's the model itself
            # (some frameworks save models as dicts)
            raise ValueError(
                "Could not extract model from checkpoint dictionary. "
                "Please save the complete torch.nn.Module, not just the state_dict."
            )
        
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")
    
    def _infer_pytorch_input_shape(
        self,
        model: torch.nn.Module,
        kwargs: Dict[str, Any]
    ) -> Tuple[int, ...]:
        """Infer input shape from model or kwargs"""
        
        # Check if shape hint provided in kwargs
        if 'image_size' in kwargs:
            size = kwargs['image_size']
            if isinstance(size, int):
                return (1, 3, size, size)
            elif isinstance(size, (tuple, list)) and len(size) == 2:
                return (1, 3, size[0], size[1])
        
        # Try to infer from first layer
        try:
            first_layer = next(model.modules())
            if hasattr(first_layer, 'in_channels'):
                # Likely a Conv2d layer
                channels = first_layer.in_channels
                # Default to 640x640 for detection models
                return (1, channels, 640, 640)
        except:
            pass
        
        # Default: assume vision model with 3 channels, 640x640
        logger.warning("Could not infer input shape, using default (1, 3, 640, 640)")
        return (1, 3, 640, 640)
    
    def _tensorflow_to_onnx(
        self,
        model_path: Path,
        output_path: Path,
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
        **kwargs
    ):
        """Convert TensorFlow SavedModel to ONNX"""
        try:
            logger.info("Converting TensorFlow SavedModel to ONNX...")
            logger.info("This requires tf2onnx package")
            
            # Check if tf2onnx is available
            try:
                import tf2onnx
            except ImportError:
                raise ImportError(
                    "tf2onnx is required for TensorFlow conversion. "
                    "Install with: pip install tf2onnx"
                )
            
            # Build command for tf2onnx
            cmd = [
                sys.executable, '-m', 'tf2onnx.convert',
                '--saved-model', str(model_path),
                '--output', str(output_path),
                '--opset', str(self.opset_version)
            ]
            
            # Add input/output names if provided
            if input_names:
                cmd.extend(['--inputs', ','.join(input_names)])
            if output_names:
                cmd.extend(['--outputs', ','.join(output_names)])
            
            # Add additional arguments
            if kwargs.get('tag', None):
                cmd.extend(['--tag', kwargs['tag']])
            if kwargs.get('signature_def', None):
                cmd.extend(['--signature_def', kwargs['signature_def']])
            
            logger.info(f"Running: {' '.join(cmd)}")
            
            # Execute conversion
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"tf2onnx conversion failed: {result.stderr}")
            
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout}")
            
            logger.info("✓ TensorFlow to ONNX conversion successful")
            
        except subprocess.TimeoutExpired:
            logger.error("TensorFlow conversion timed out (>5 minutes)")
            raise
        except Exception as e:
            logger.error(f"Failed to convert TensorFlow model: {e}")
            raise
    
    def _tflite_to_onnx(self, model_path: Path, output_path: Path, **kwargs):
        """Convert TFLite model to ONNX"""
        try:
            logger.warning(
                "TFLite to ONNX conversion has limited support. "
                "Recommended: Use TensorFlow SavedModel → ONNX instead."
            )
            
            # Check if tflite2onnx is available
            try:
                import tflite2onnx
            except ImportError:
                raise ImportError(
                    "tflite2onnx is required for TFLite conversion. "
                    "Install with: pip install tflite2onnx\n"
                    "Note: This package has limited operator support."
                )
            
            logger.info("Converting TFLite to ONNX...")
            logger.info("Warning: Not all TFLite operators are supported")
            
            # Method 1: Try using tflite2onnx package
            try:
                tflite2onnx.convert(str(model_path), str(output_path))
                logger.info("✓ TFLite to ONNX conversion successful")
            except Exception as e1:
                logger.error(f"tflite2onnx failed: {e1}")
                
                # Method 2: Try alternative approach via TensorFlow
                logger.info("Attempting alternative conversion via TensorFlow...")
                try:
                    self._tflite_to_onnx_via_tf(model_path, output_path)
                    logger.info("✓ TFLite to ONNX conversion successful (via TensorFlow)")
                except Exception as e2:
                    logger.error(f"Alternative method also failed: {e2}")
                    raise RuntimeError(
                        f"TFLite conversion failed with both methods.\n"
                        f"Method 1 (tflite2onnx): {e1}\n"
                        f"Method 2 (via TF): {e2}\n"
                        f"Recommendation: Convert your model to SavedModel format first."
                    )
            
        except Exception as e:
            logger.error(f"Failed to convert TFLite model: {e}")
            raise
    
    def _tflite_to_onnx_via_tf(self, tflite_path: Path, output_path: Path):
        """Alternative TFLite conversion via TensorFlow"""
        try:
            import tensorflow as tf
            import tf2onnx
            
            logger.info("Loading TFLite model with TensorFlow interpreter...")
            
            # Load TFLite model
            interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
            interpreter.allocate_tensors()
            
            # Get input/output details
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            logger.info(f"Inputs: {input_details}")
            logger.info(f"Outputs: {output_details}")
            
            # This is a complex conversion that requires rebuilding the model
            # For now, raise an error with instructions
            raise NotImplementedError(
                "Direct TFLite to ONNX conversion via TensorFlow not yet implemented. "
                "Please convert your model using one of these workflows:\n"
                "1. Original Model → TF SavedModel → ONNX\n"
                "2. Use tflite2onnx package directly (may have limited operator support)"
            )
            
        except ImportError as e:
            raise ImportError(f"TensorFlow is required: {e}")
    
    def _simplify_onnx(self, onnx_path: Path):
        """Simplify ONNX model graph"""
        try:
            from onnxsim import simplify
            
            logger.info("Simplifying ONNX model...")
            
            # Load model
            model = onnx.load(str(onnx_path))
            
            # Simplify (updated API - removed unsupported parameters)
            model_simplified, check = simplify(model)
            
            if not check:
                logger.warning("Simplification validation failed, using original model")
                return
            
            # Calculate size reduction
            import io
            original_size = len(model.SerializeToString())
            simplified_size = len(model_simplified.SerializeToString())
            reduction_pct = (1 - simplified_size / original_size) * 100
            
            # Save simplified model
            onnx.save(model_simplified, str(onnx_path))
            logger.info(f"✓ ONNX model simplified (graph size reduced by {reduction_pct:.1f}%)")
            
        except ImportError:
            logger.warning(
                "onnx-simplifier not installed. Install with: pip install onnx-simplifier"
            )
        except Exception as e:
            logger.warning(f"Failed to simplify ONNX model: {e}")
            logger.info("Continuing with unsimplified model...")

    
    def _validate_onnx(self, onnx_path: Path):
        """Validate ONNX model"""
        try:
            logger.info("Validating ONNX model...")
            
            # Load and check model
            model = onnx.load(str(onnx_path))
            checker.check_model(model)
            
            # Try shape inference
            try:
                model_with_shapes = shape_inference.infer_shapes(model)
                logger.info("✓ Shape inference successful")
            except Exception as e:
                logger.warning(f"Shape inference failed: {e}")
            
            # Get model info
            graph = model.graph
            
            # Count operations
            op_types = {}
            for node in graph.node:
                op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
            
            logger.info(f"✓ ONNX validation passed")
            logger.info(f"  Graph nodes: {len(graph.node)}")
            logger.info(f"  Inputs: {len([i for i in graph.input if i.name not in [init.name for init in graph.initializer]])}")
            logger.info(f"  Outputs: {len(graph.output)}")
            logger.info(f"  Parameters: {len(graph.initializer)}")
            logger.info(f"  Operator types: {len(op_types)}")
            
            # Show top 5 operators
            top_ops = sorted(op_types.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"  Top operators: {', '.join([f'{op}({count})' for op, count in top_ops])}")
            
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            raise
    
    def get_model_info(self, onnx_path: Path) -> Dict[str, Any]:
        """
        Get detailed information about ONNX model
        
        Args:
            onnx_path: Path to ONNX model
        
        Returns:
            Dictionary with model information
        """
        model = onnx.load(str(onnx_path))
        graph = model.graph
        
        # Extract inputs
        inputs = []
        initializer_names = {init.name for init in graph.initializer}
        for input_tensor in graph.input:
            if input_tensor.name not in initializer_names:
                shape = [
                    dim.dim_value if dim.dim_value > 0 else -1
                    for dim in input_tensor.type.tensor_type.shape.dim
                ]
                inputs.append({
                    'name': input_tensor.name,
                    'shape': shape,
                    'dtype': onnx.TensorProto.DataType.Name(
                        input_tensor.type.tensor_type.elem_type
                    )
                })
        
        # Extract outputs
        outputs = []
        for output_tensor in graph.output:
            shape = [
                dim.dim_value if dim.dim_value > 0 else -1
                for dim in output_tensor.type.tensor_type.shape.dim
            ]
            outputs.append({
                'name': output_tensor.name,
                'shape': shape,
                'dtype': onnx.TensorProto.DataType.Name(
                    output_tensor.type.tensor_type.elem_type
                )
            })
        
        # Count operators
        op_counts = {}
        for node in graph.node:
            op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        
        info = {
            'model_path': str(onnx_path),
            'model_size_mb': get_model_size_mb(onnx_path),
            'opset_version': model.opset_import[0].version if model.opset_import else None,
            'producer': model.producer_name,
            'inputs': inputs,
            'outputs': outputs,
            'num_nodes': len(graph.node),
            'num_parameters': len(graph.initializer),
            'operator_counts': op_counts,
        }
        
        return info


# Convenience functions
def convert_to_onnx(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    **kwargs
) -> Path:
    """
    Convert model to ONNX (convenience wrapper)
    
    Args:
        model_path: Path to input model
        output_path: Path to save ONNX model
        **kwargs: Additional parameters for conversion
    
    Returns:
        Path to converted ONNX model
    
    Example:
        >>> convert_to_onnx(
        ...     'model.pt',
        ...     'model.onnx',
        ...     input_shape=(1, 3, 640, 640)
        ... )
    """
    converter = ONNXConverter()
    return converter.convert(model_path, output_path, **kwargs)


def get_onnx_info(onnx_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get ONNX model information (convenience wrapper)
    
    Args:
        onnx_path: Path to ONNX model
    
    Returns:
        Dictionary with model information
    """
    converter = ONNXConverter()
    return converter.get_model_info(Path(onnx_path))
