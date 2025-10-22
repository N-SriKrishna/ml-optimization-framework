"""
Utilities for pruning operations
"""
from typing import Dict, List, Tuple
import numpy as np
from pathlib import Path

import onnx
from onnx import numpy_helper

from src.utils.logger import logger


class PruningSensitivityAnalyzer:
    """
    Analyze layer sensitivity to pruning
    """
    
    @staticmethod
    def compute_layer_importance(
        model_path: Path,
        method: str = 'l1_norm'
    ) -> Dict[str, float]:
        """
        Compute importance score for each layer
        
        Args:
            model_path: Path to ONNX model
            method: Importance metric ('l1_norm', 'l2_norm', 'variance')
        
        Returns:
            Dictionary mapping layer names to importance scores
        """
        model = onnx.load(str(model_path))
        importance = {}
        
        for initializer in model.graph.initializer:
            tensor = numpy_helper.to_array(initializer)
            
            if len(tensor.shape) < 2:  # Skip biases
                continue
            
            if method == 'l1_norm':
                score = np.sum(np.abs(tensor))
            elif method == 'l2_norm':
                score = np.sum(tensor ** 2)
            elif method == 'variance':
                score = np.var(tensor)
            else:
                raise ValueError(f"Unknown importance method: {method}")
            
            importance[initializer.name] = float(score)
        
        # Normalize scores
        if importance:
            max_score = max(importance.values())
            if max_score > 0:
                importance = {k: v / max_score for k, v in importance.items()}
        
        return importance
    
    @staticmethod
    def rank_layers_by_importance(
        model_path: Path,
        method: str = 'l1_norm'
    ) -> List[Tuple[str, float]]:
        """
        Rank layers by importance
        
        Returns:
            List of (layer_name, importance_score) tuples, sorted by importance
        """
        importance = PruningSensitivityAnalyzer.compute_layer_importance(
            model_path, method
        )
        
        ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        return ranked


class IterativePruner:
    """
    Iterative pruning with fine-tuning
    """
    
    @staticmethod
    def iterative_prune(
        model_path: Path,
        output_path: Path,
        target_sparsity: float,
        num_iterations: int = 5,
        fine_tune_func: callable = None
    ) -> Path:
        """
        Iteratively prune model with optional fine-tuning
        
        Args:
            model_path: Path to input model
            output_path: Path to save final pruned model
            target_sparsity: Final target sparsity
            num_iterations: Number of pruning iterations
            fine_tune_func: Optional function to fine-tune between iterations
        
        Returns:
            Path to final pruned model
        """
        from src.optimizers.pruner import ONNXPruner, PruningConfig
        
        logger.info(f"Starting iterative pruning over {num_iterations} iterations")
        logger.info(f"Target sparsity: {target_sparsity * 100:.1f}%")
        
        current_model = model_path
        sparsity_schedule = np.linspace(0, target_sparsity, num_iterations + 1)[1:]
        
        for i, sparsity in enumerate(sparsity_schedule):
            logger.info(f"\n--- Iteration {i+1}/{num_iterations} ---")
            logger.info(f"Target sparsity for this iteration: {sparsity * 100:.1f}%")
            
            # Prune
            temp_output = output_path.parent / f"iter_{i+1}_{output_path.name}"
            
            config = PruningConfig(
                pruning_type='magnitude',
                pruning_scope='global',
                sparsity=sparsity
            )
            pruner = ONNXPruner(config)
            pruner.prune(current_model, temp_output)
            
            # Optional fine-tuning
            if fine_tune_func:
                logger.info("Fine-tuning pruned model...")
                fine_tune_func(temp_output)
            
            current_model = temp_output
        
        # Copy final model to output path
        import shutil
        shutil.copy(current_model, output_path)
        
        logger.info(f"\n✓ Iterative pruning complete!")
        logger.info(f"Final model saved to: {output_path}")
        
        return output_path


class StructuredPruningOptimizer:
    """
    Advanced structured pruning with dimension optimization
    """
    
    @staticmethod
    def optimize_channel_pruning(
        model_path: Path,
        output_path: Path,
        target_reduction: float = 0.5
    ) -> Path:
        """
        Optimize channel pruning to achieve target reduction
        
        Args:
            model_path: Path to input model
            output_path: Path to save optimized model
            target_reduction: Target parameter reduction (0.5 = 50% reduction)
        
        Returns:
            Path to optimized model
        """
        logger.info("Optimizing structured channel pruning...")
        
        model = onnx.load(str(model_path))
        
        # Analyze conv layers
        conv_layers = []
        for initializer in model.graph.initializer:
            tensor = numpy_helper.to_array(initializer)
            if len(tensor.shape) == 4:  # Conv weight
                conv_layers.append({
                    'name': initializer.name,
                    'shape': tensor.shape,
                    'channels': tensor.shape[0],
                    'params': tensor.size
                })
        
        if not conv_layers:
            logger.warning("No convolutional layers found")
            return model_path
        
        # Calculate per-layer pruning to achieve target reduction
        total_params = sum(layer['params'] for layer in conv_layers)
        target_params = total_params * (1 - target_reduction)
        
        logger.info(f"Total conv params: {total_params:,}")
        logger.info(f"Target params: {target_params:,}")
        
        # Simple heuristic: prune proportionally
        for layer in conv_layers:
            layer['target_channels'] = int(layer['channels'] * (1 - target_reduction))
            logger.debug(f"Layer {layer['name']}: {layer['channels']} → {layer['target_channels']} channels")
        
        # Apply pruning (simplified - actual implementation would modify graph)
        from src.optimizers.pruner import ONNXPruner, PruningConfig
        
        config = PruningConfig(
            pruning_type='structured',
            sparsity=target_reduction
        )
        pruner = ONNXPruner(config)
        return pruner.prune(model_path, output_path)
