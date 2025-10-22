"""
ONNX Pruning Engine - Structured and Unstructured Pruning
"""
from pathlib import Path
from typing import Optional, List, Literal, Dict, Tuple, Any
from dataclasses import dataclass
import copy

import onnx
from onnx import numpy_helper
import numpy as np

from src.utils.logger import logger
from src.utils.helpers import get_model_size_mb


PruningType = Literal['magnitude', 'structured', 'gradient', 'random']
PruningScope = Literal['global', 'local']


@dataclass
class PruningConfig:
    """Configuration for pruning"""
    pruning_type: PruningType = 'magnitude'
    pruning_scope: PruningScope = 'global'
    sparsity: float = 0.5
    structured_dim: Optional[int] = None  # For structured pruning: 0=filters, 1=channels
    block_size: Optional[Tuple[int, int]] = None  # For block pruning
    preserve_output_shape: bool = True  # Whether to maintain output dimensions


class ONNXPruner:
    """
    Comprehensive ONNX pruning engine
    """
    
    def __init__(self, config: Optional[PruningConfig] = None):
        """
        Initialize pruner
        
        Args:
            config: Pruning configuration
        """
        self.config = config or PruningConfig()
        logger.info(f"Initialized pruner with {self.config.pruning_type} pruning")
        logger.info(f"Target sparsity: {self.config.sparsity * 100:.1f}%")
    
    def prune(
        self,
        model_path: Path,
        output_path: Path,
        sensitivity_data: Optional[Dict[str, float]] = None
    ) -> Path:
        """
        Prune ONNX model
        
        Args:
            model_path: Path to input ONNX model
            output_path: Path to save pruned model
            sensitivity_data: Optional layer sensitivity information
        
        Returns:
            Path to pruned model
        """
        model_path = Path(model_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting {self.config.pruning_type} pruning...")
        logger.info(f"Input model: {model_path}")
        logger.info(f"Output model: {output_path}")
        
        # Load model
        model = onnx.load(str(model_path))
        original_size = get_model_size_mb(model_path)
        logger.info(f"Original model size: {original_size:.2f} MB")
        
        # Count original parameters
        original_params = self._count_parameters(model)
        logger.info(f"Original parameters: {original_params:,}")
        
        # Apply pruning based on type
        if self.config.pruning_type == 'magnitude':
            pruned_model = self._prune_magnitude(model, sensitivity_data)
        
        elif self.config.pruning_type == 'structured':
            pruned_model = self._prune_structured(model, sensitivity_data)
        
        elif self.config.pruning_type == 'gradient':
            logger.warning("Gradient pruning requires gradient information. Using magnitude as fallback.")
            pruned_model = self._prune_magnitude(model, sensitivity_data)
        
        elif self.config.pruning_type == 'random':
            pruned_model = self._prune_random(model)
        
        else:
            raise ValueError(f"Unknown pruning type: {self.config.pruning_type}")
        
        # Save pruned model
        onnx.save(pruned_model, str(output_path))
        
        # Report results
        pruned_size = get_model_size_mb(output_path)
        pruned_params = self._count_parameters(pruned_model)
        actual_sparsity = 1 - (pruned_params / original_params) if original_params > 0 else 0
        
        logger.info(f"✓ Pruning complete!")
        logger.info(f"Pruned model size: {pruned_size:.2f} MB")
        logger.info(f"Pruned parameters: {pruned_params:,}")
        logger.info(f"Actual sparsity: {actual_sparsity * 100:.1f}%")
        logger.info(f"Size reduction: {(1 - pruned_size/original_size) * 100:.1f}%")
        
        return output_path
    
    def _count_parameters(self, model: onnx.ModelProto) -> int:
        """Count total parameters in model"""
        total = 0
        for initializer in model.graph.initializer:
            tensor = numpy_helper.to_array(initializer)
            total += tensor.size
        return total
    
    def _count_nonzero_parameters(self, model: onnx.ModelProto) -> int:
        """Count non-zero parameters in model"""
        total = 0
        for initializer in model.graph.initializer:
            tensor = numpy_helper.to_array(initializer)
            total += np.count_nonzero(tensor)
        return total
    
    def _prune_magnitude(
        self,
        model: onnx.ModelProto,
        sensitivity_data: Optional[Dict[str, float]] = None
    ) -> onnx.ModelProto:
        """
        Apply magnitude-based unstructured pruning
        
        Args:
            model: Input ONNX model
            sensitivity_data: Optional layer-wise sensitivity
        
        Returns:
            Pruned model
        """
        logger.info("Applying magnitude-based pruning...")
        
        pruned_model = copy.deepcopy(model)
        
        if self.config.pruning_scope == 'global':
            # Global pruning: threshold across all layers
            pruned_model = self._prune_magnitude_global(pruned_model, sensitivity_data)
        else:
            # Local pruning: threshold per layer
            pruned_model = self._prune_magnitude_local(pruned_model, sensitivity_data)
        
        return pruned_model
    
    def _prune_magnitude_global(
        self,
        model: onnx.ModelProto,
        sensitivity_data: Optional[Dict[str, float]] = None
    ) -> onnx.ModelProto:
        """Global magnitude pruning"""
        logger.info(f"Using global pruning scope (threshold across all layers)")
        
        # Collect all weights
        all_weights = []
        weight_info = []  # Store (name, shape, original_tensor) for reconstruction
        
        for initializer in model.graph.initializer:
            # Only prune weight tensors (skip biases, BatchNorm params, etc.)
            if self._is_prunable_tensor(initializer):
                tensor = numpy_helper.to_array(initializer)
                all_weights.append(np.abs(tensor.flatten()))
                weight_info.append((initializer.name, tensor.shape, tensor))
        
        if not all_weights:
            logger.warning("No prunable tensors found")
            return model
        
        # Concatenate all weights
        all_weights_flat = np.concatenate(all_weights)
        
        # Calculate global threshold
        threshold = np.percentile(all_weights_flat, self.config.sparsity * 100)
        logger.info(f"Global magnitude threshold: {threshold:.6f}")
        
        # Apply pruning to each tensor
        pruned_count = 0
        total_count = 0
        
        for initializer in model.graph.initializer:
            if self._is_prunable_tensor(initializer):
                tensor = numpy_helper.to_array(initializer)
                total_count += tensor.size
                
                # Create mask
                mask = np.abs(tensor) >= threshold
                pruned_tensor = tensor * mask
                
                pruned_count += np.sum(mask == 0)
                
                # Update initializer
                new_initializer = numpy_helper.from_array(
                    pruned_tensor.astype(tensor.dtype),
                    name=initializer.name
                )
                initializer.CopyFrom(new_initializer)
        
        actual_sparsity = pruned_count / total_count if total_count > 0 else 0
        logger.info(f"Pruned {pruned_count:,} / {total_count:,} parameters ({actual_sparsity*100:.2f}%)")
        
        return model
    
    def _prune_magnitude_local(
        self,
        model: onnx.ModelProto,
        sensitivity_data: Optional[Dict[str, float]] = None
    ) -> onnx.ModelProto:
        """Local magnitude pruning (per-layer)"""
        logger.info(f"Using local pruning scope (per-layer thresholds)")
        
        pruned_count = 0
        total_count = 0
        
        for initializer in model.graph.initializer:
            if self._is_prunable_tensor(initializer):
                tensor = numpy_helper.to_array(initializer)
                total_count += tensor.size
                
                # Adjust sparsity based on sensitivity
                layer_sparsity = self.config.sparsity
                if sensitivity_data and initializer.name in sensitivity_data:
                    # Reduce sparsity for sensitive layers
                    sensitivity = sensitivity_data[initializer.name]
                    layer_sparsity = self.config.sparsity * (1 - sensitivity * 0.5)
                    logger.debug(f"Layer {initializer.name}: adjusted sparsity to {layer_sparsity*100:.1f}%")
                
                # Calculate per-layer threshold
                threshold = np.percentile(np.abs(tensor), layer_sparsity * 100)
                
                # Create mask and prune
                mask = np.abs(tensor) >= threshold
                pruned_tensor = tensor * mask
                
                pruned_count += np.sum(mask == 0)
                
                # Update initializer
                new_initializer = numpy_helper.from_array(
                    pruned_tensor.astype(tensor.dtype),
                    name=initializer.name
                )
                initializer.CopyFrom(new_initializer)
        
        actual_sparsity = pruned_count / total_count if total_count > 0 else 0
        logger.info(f"Pruned {pruned_count:,} / {total_count:,} parameters ({actual_sparsity*100:.2f}%)")
        
        return model
    
    def _prune_structured(
        self,
        model: onnx.ModelProto,
        sensitivity_data: Optional[Dict[str, float]] = None
    ) -> onnx.ModelProto:
        """
        Apply structured pruning (filter/channel pruning)
        
        Args:
            model: Input ONNX model
            sensitivity_data: Optional layer-wise sensitivity
        
        Returns:
            Pruned model
        """
        logger.info("Applying structured pruning (filter/channel level)...")
        
        pruned_model = copy.deepcopy(model)
        
        # Track pruning statistics
        total_filters = 0
        pruned_filters = 0
        
        for initializer in pruned_model.graph.initializer:
            if self._is_conv_weight(initializer):
                tensor = numpy_helper.to_array(initializer)
                
                # Conv weights typically have shape: [out_channels, in_channels, kernel_h, kernel_w]
                if len(tensor.shape) == 4:
                    num_filters = tensor.shape[0]
                    total_filters += num_filters
                    
                    # Calculate filter importance (L1 norm)
                    filter_importance = np.sum(np.abs(tensor), axis=(1, 2, 3))
                    
                    # Determine number of filters to prune
                    num_to_prune = int(num_filters * self.config.sparsity)
                    
                    if num_to_prune > 0 and num_to_prune < num_filters:
                        # Get indices of least important filters
                        prune_indices = np.argsort(filter_importance)[:num_to_prune]
                        
                        # Zero out pruned filters
                        pruned_tensor = tensor.copy()
                        pruned_tensor[prune_indices] = 0
                        
                        pruned_filters += num_to_prune
                        
                        # Update initializer
                        new_initializer = numpy_helper.from_array(
                            pruned_tensor.astype(tensor.dtype),
                            name=initializer.name
                        )
                        initializer.CopyFrom(new_initializer)
                        
                        logger.debug(f"Layer {initializer.name}: pruned {num_to_prune}/{num_filters} filters")
        
        if total_filters > 0:
            actual_sparsity = pruned_filters / total_filters
            logger.info(f"Pruned {pruned_filters} / {total_filters} filters ({actual_sparsity*100:.2f}%)")
        else:
            logger.warning("No convolutional layers found for structured pruning")
        
        return pruned_model
    
    def _prune_random(self, model: onnx.ModelProto) -> onnx.ModelProto:
        """
        Apply random pruning (baseline for comparison)
        
        Args:
            model: Input ONNX model
        
        Returns:
            Pruned model
        """
        logger.info("Applying random pruning (baseline)...")
        
        pruned_model = copy.deepcopy(model)
        
        pruned_count = 0
        total_count = 0
        
        for initializer in pruned_model.graph.initializer:
            if self._is_prunable_tensor(initializer):
                tensor = numpy_helper.to_array(initializer)
                total_count += tensor.size
                
                # Create random mask
                mask = np.random.random(tensor.shape) >= self.config.sparsity
                pruned_tensor = tensor * mask
                
                pruned_count += np.sum(mask == 0)
                
                # Update initializer
                new_initializer = numpy_helper.from_array(
                    pruned_tensor.astype(tensor.dtype),
                    name=initializer.name
                )
                initializer.CopyFrom(new_initializer)
        
        actual_sparsity = pruned_count / total_count if total_count > 0 else 0
        logger.info(f"Randomly pruned {pruned_count:,} / {total_count:,} parameters ({actual_sparsity*100:.2f}%)")
        
        return model
    
    def _is_prunable_tensor(self, initializer: onnx.TensorProto) -> bool:
        """Check if tensor is prunable (weights, not biases or BatchNorm params)"""
        tensor = numpy_helper.to_array(initializer)
        
        # Skip 1D tensors (biases, BatchNorm params)
        if len(tensor.shape) == 1:
            return False
        
        # Skip small tensors
        if tensor.size < 100:
            return False
        
        return True
    
    def _is_conv_weight(self, initializer: onnx.TensorProto) -> bool:
        """Check if tensor is a convolutional weight"""
        tensor = numpy_helper.to_array(initializer)
        
        # Conv weights are typically 4D: [out_channels, in_channels, kernel_h, kernel_w]
        return len(tensor.shape) == 4
    
    def analyze_sparsity(self, model_path: Path) -> Dict[str, Any]:
        """
        Analyze sparsity of a model
        
        Args:
            model_path: Path to ONNX model
        
        Returns:
            Dictionary with sparsity statistics
        """
        model = onnx.load(str(model_path))
        
        total_params = 0
        zero_params = 0
        layer_sparsity = {}
        
        for initializer in model.graph.initializer:
            tensor = numpy_helper.to_array(initializer)
            layer_total = tensor.size
            layer_zeros = np.sum(tensor == 0)
            
            total_params += layer_total
            zero_params += layer_zeros
            
            if layer_total > 0:
                layer_sparsity[initializer.name] = {
                    'total': layer_total,
                    'zeros': layer_zeros,
                    'sparsity': layer_zeros / layer_total
                }
        
        overall_sparsity = zero_params / total_params if total_params > 0 else 0
        
        results = {
            'overall_sparsity': overall_sparsity,
            'total_parameters': total_params,
            'zero_parameters': zero_params,
            'non_zero_parameters': total_params - zero_params,
            'layer_sparsity': layer_sparsity
        }
        
        return results
    
    def calculate_sensitivity(
        self,
        model_path: Path,
        validation_func: callable,
        sparsity_levels: List[float] = [0.1, 0.3, 0.5, 0.7, 0.9]
    ) -> Dict[str, float]:
        """
        Calculate layer-wise sensitivity to pruning
        
        Args:
            model_path: Path to ONNX model
            validation_func: Function that returns accuracy for a model
            sparsity_levels: List of sparsity levels to test
        
        Returns:
            Dictionary mapping layer names to sensitivity scores
        """
        logger.info("Calculating layer-wise pruning sensitivity...")
        
        model = onnx.load(str(model_path))
        baseline_accuracy = validation_func(model)
        
        logger.info(f"Baseline accuracy: {baseline_accuracy:.4f}")
        
        sensitivity = {}
        
        # Test each layer individually
        for initializer in model.graph.initializer:
            if self._is_prunable_tensor(initializer):
                layer_name = initializer.name
                layer_sensitivity = []
                
                for sparsity in sparsity_levels:
                    # Create temporary pruned model
                    temp_model = copy.deepcopy(model)
                    
                    # Prune only this layer
                    for temp_init in temp_model.graph.initializer:
                        if temp_init.name == layer_name:
                            tensor = numpy_helper.to_array(temp_init)
                            threshold = np.percentile(np.abs(tensor), sparsity * 100)
                            mask = np.abs(tensor) >= threshold
                            pruned_tensor = tensor * mask
                            
                            new_init = numpy_helper.from_array(
                                pruned_tensor.astype(tensor.dtype),
                                name=temp_init.name
                            )
                            temp_init.CopyFrom(new_init)
                    
                    # Evaluate
                    accuracy = validation_func(temp_model)
                    accuracy_drop = baseline_accuracy - accuracy
                    layer_sensitivity.append(accuracy_drop)
                
                # Sensitivity is average accuracy drop
                avg_sensitivity = np.mean(layer_sensitivity)
                sensitivity[layer_name] = avg_sensitivity
                
                logger.debug(f"Layer {layer_name}: sensitivity = {avg_sensitivity:.4f}")
        
        logger.info(f"✓ Sensitivity analysis complete for {len(sensitivity)} layers")
        
        return sensitivity


# Convenience functions
def prune_magnitude_global(
    model_path: Path,
    output_path: Path,
    sparsity: float = 0.5
) -> Path:
    """Quick global magnitude pruning"""
    config = PruningConfig(
        pruning_type='magnitude',
        pruning_scope='global',
        sparsity=sparsity
    )
    pruner = ONNXPruner(config)
    return pruner.prune(model_path, output_path)


def prune_structured_filters(
    model_path: Path,
    output_path: Path,
    sparsity: float = 0.3
) -> Path:
    """Quick structured filter pruning"""
    config = PruningConfig(
        pruning_type='structured',
        sparsity=sparsity
    )
    pruner = ONNXPruner(config)
    return pruner.prune(model_path, output_path)


def analyze_model_sparsity(model_path: Path) -> Dict[str, Any]:
    """Analyze sparsity of a model"""
    pruner = ONNXPruner()
    return pruner.analyze_sparsity(model_path)
