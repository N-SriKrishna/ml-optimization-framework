"""
Variant Generator - Generate multiple optimized model variants
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import itertools
import json

from src.solvers.constraint_solver import OptimizationStrategy
from src.optimizers.quantizer import ONNXQuantizer, QuantizationConfig
from src.optimizers.pruner import ONNXPruner, PruningConfig
from src.utils.logger import logger
from src.utils.helpers import get_model_size_mb


@dataclass
class ModelVariant:
    """Represents a single optimized model variant"""
    variant_id: str
    variant_name: str
    model_path: Path
    base_model_path: Path
    optimizations_applied: List[Dict[str, Any]]
    
    # Metrics
    model_size_mb: float
    compression_ratio: float
    estimated_speedup: float
    estimated_accuracy_impact: float
    
    # Metadata
    created_timestamp: str
    config: Dict[str, Any]


class VariantGenerator:
    """
    Generate multiple optimized model variants based on strategy
    """
    
    def __init__(self, output_dir: Path):
        """
        Initialize variant generator
        
        Args:
            output_dir: Directory to save generated variants
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized variant generator (output: {output_dir})")
    
    def generate_variants(
        self,
        model_path: Path,
        strategy: OptimizationStrategy,
        num_variants: int = 3
    ) -> List[ModelVariant]:
        """
        Generate multiple model variants following the strategy
        
        Args:
            model_path: Path to input ONNX model
            strategy: Optimization strategy
            num_variants: Number of variants to generate
        
        Returns:
            List of generated variants
        """
        logger.info(f"Generating {num_variants} model variants...")
        logger.info(f"Strategy: {strategy.strategy_name}")
        
        # Generate variant configurations
        variant_configs = self._create_variant_configs(strategy, num_variants)
        
        # Generate each variant
        variants = []
        base_size = get_model_size_mb(model_path)
        
        for i, config in enumerate(variant_configs, 1):
            logger.info(f"\n--- Generating Variant {i}/{len(variant_configs)}: {config['name']} ---")
            
            variant = self._generate_single_variant(
                model_path,
                config,
                base_size,
                i
            )
            
            variants.append(variant)
            
            logger.info(f"✓ Variant {i} complete: {variant.variant_name}")
            logger.info(f"  Size: {variant.model_size_mb:.2f} MB ({variant.compression_ratio:.2f}× compression)")
        
        # Save variant metadata
        self._save_variants_metadata(variants)
        
        logger.info(f"\n✓ Generated {len(variants)} variants successfully!")
        
        return variants
    
    def _create_variant_configs(
        self,
        strategy: OptimizationStrategy,
        num_variants: int
    ) -> List[Dict[str, Any]]:
        """Create configurations for different variants"""
        
        if num_variants == 1:
            # Single variant - use strategy as-is
            return [{
                'name': strategy.strategy_name,
                'level': 'balanced',
                'techniques': strategy.techniques
            }]
        
        elif num_variants == 3:
            # Three variants: aggressive, balanced, conservative
            return self._create_three_variants(strategy)
        
        elif num_variants == 5:
            # Five variants: very aggressive, aggressive, balanced, conservative, very conservative
            return self._create_five_variants(strategy)
        
        else:
            # Default to 3 variants
            return self._create_three_variants(strategy)
    
    def _create_three_variants(self, strategy: OptimizationStrategy) -> List[Dict[str, Any]]:
        """Create 3 variants: aggressive, balanced, conservative"""
        
        base_techniques = strategy.techniques
        
        variants = []
        
        # Variant 1: Aggressive
        aggressive_techniques = []
        for tech in base_techniques:
            tech_copy = tech.copy()
            
            if tech['name'] == 'quantization':
                # Keep as-is or make more aggressive
                if tech.get('precision') == 'int8':
                    tech_copy['precision'] = 'int8'  # Could use int4 for more aggression
            
            elif tech['name'] == 'pruning':
                # Increase sparsity
                current_sparsity = tech.get('sparsity', 0.3)
                tech_copy['sparsity'] = min(0.7, current_sparsity * 1.5)
            
            aggressive_techniques.append(tech_copy)
        
        variants.append({
            'name': 'Aggressive',
            'level': 'aggressive',
            'techniques': aggressive_techniques
        })
        
        # Variant 2: Balanced (use strategy as-is)
        variants.append({
            'name': 'Balanced',
            'level': 'balanced',
            'techniques': base_techniques
        })
        
        # Variant 3: Conservative
        conservative_techniques = []
        for tech in base_techniques:
            tech_copy = tech.copy()
            
            if tech['name'] == 'quantization':
                # Use QAT for better accuracy
                tech_copy['type'] = 'qat'
            
            elif tech['name'] == 'pruning':
                # Reduce sparsity
                current_sparsity = tech.get('sparsity', 0.3)
                tech_copy['sparsity'] = max(0.1, current_sparsity * 0.6)
            
            conservative_techniques.append(tech_copy)
        
        variants.append({
            'name': 'Conservative',
            'level': 'conservative',
            'techniques': conservative_techniques
        })
        
        return variants
    
    def _create_five_variants(self, strategy: OptimizationStrategy) -> List[Dict[str, Any]]:
        """Create 5 variants with different optimization levels"""
        
        three_variants = self._create_three_variants(strategy)
        
        # Add very aggressive and very conservative
        base_techniques = strategy.techniques
        
        # Very aggressive
        very_aggressive_techniques = []
        for tech in base_techniques:
            tech_copy = tech.copy()
            
            if tech['name'] == 'quantization':
                tech_copy['precision'] = 'int4'  # More aggressive
            
            elif tech['name'] == 'pruning':
                tech_copy['sparsity'] = 0.8  # Very high sparsity
            
            very_aggressive_techniques.append(tech_copy)
        
        very_aggressive = {
            'name': 'Very Aggressive',
            'level': 'very_aggressive',
            'techniques': very_aggressive_techniques
        }
        
        # Very conservative
        very_conservative_techniques = []
        for tech in base_techniques:
            tech_copy = tech.copy()
            
            if tech['name'] == 'quantization':
                tech_copy['precision'] = 'fp16'  # Less aggressive
            
            elif tech['name'] == 'pruning':
                tech_copy['sparsity'] = 0.1  # Very low sparsity
            
            very_conservative_techniques.append(tech_copy)
        
        very_conservative = {
            'name': 'Very Conservative',
            'level': 'very_conservative',
            'techniques': very_conservative_techniques
        }
        
        # Order: very aggressive, aggressive, balanced, conservative, very conservative
        return [
            very_aggressive,
            three_variants[0],  # aggressive
            three_variants[1],  # balanced
            three_variants[2],  # conservative
            very_conservative
        ]
    
    def _generate_single_variant(
        self,
        base_model_path: Path,
        config: Dict[str, Any],
        base_size: float,
        variant_index: int
    ) -> ModelVariant:
        """Generate a single model variant"""
        
        variant_id = f"variant_{variant_index:02d}"
        variant_name = config['name']
        
        logger.info(f"Applying optimizations for {variant_name}...")
        
        # Start with base model
        current_model = base_model_path
        optimizations_applied = []
        
        # Apply each technique in order
        for tech in sorted(config['techniques'], key=lambda x: x.get('priority', 999)):
            tech_name = tech['name']
            
            if tech_name == 'quantization':
                logger.info(f"  Applying quantization: {tech.get('precision', 'int8')}")
                current_model = self._apply_quantization(current_model, tech, variant_id)
                optimizations_applied.append(tech)
            
            elif tech_name == 'pruning':
                logger.info(f"  Applying pruning: {tech.get('sparsity', 0.3)*100:.0f}% sparsity")
                current_model = self._apply_pruning(current_model, tech, variant_id)
                optimizations_applied.append(tech)
            
            elif tech_name == 'graph_optimization':
                logger.info(f"  Applying graph optimization")
                # Graph optimization is typically done by ONNX Runtime or converters
                optimizations_applied.append(tech)
        
        # Calculate metrics
        final_size = get_model_size_mb(current_model)
        compression_ratio = base_size / final_size if final_size > 0 else 1.0
        
        # Estimate speedup and accuracy impact (heuristic)
        estimated_speedup = self._estimate_speedup(optimizations_applied)
        estimated_accuracy_impact = self._estimate_accuracy_impact(optimizations_applied)
        
        # Create variant object
        variant = ModelVariant(
            variant_id=variant_id,
            variant_name=variant_name,
            model_path=current_model,
            base_model_path=base_model_path,
            optimizations_applied=optimizations_applied,
            model_size_mb=final_size,
            compression_ratio=compression_ratio,
            estimated_speedup=estimated_speedup,
            estimated_accuracy_impact=estimated_accuracy_impact,
            created_timestamp=self._get_timestamp(),
            config=config
        )
        
        return variant
    
    def _apply_quantization(
        self,
        model_path: Path,
        tech: Dict[str, Any],
        variant_id: str
    ) -> Path:
        """Apply quantization to model"""
        
        output_path = self.output_dir / f"{variant_id}_quantized.onnx"
        
        quant_config = QuantizationConfig(
            quantization_type=tech.get('type', 'static'),
            precision=tech.get('precision', 'int8'),
            per_channel=tech.get('per_channel', True)
        )
        
        quantizer = ONNXQuantizer(quant_config)
        
        # For static quantization, we need calibration data
        # For now, use dynamic if calibration not provided
        if quant_config.quantization_type == 'static':
            # Fallback to dynamic for now
            logger.warning("Static quantization requires calibration data. Using dynamic.")
            quant_config.quantization_type = 'dynamic'
        
        try:
            quantizer.quantize(model_path, output_path)
        except Exception as e:
            logger.error(f"Quantization failed: {e}")
            # Return original model if quantization fails
            return model_path
        
        return output_path
    
    def _apply_pruning(
        self,
        model_path: Path,
        tech: Dict[str, Any],
        variant_id: str
    ) -> Path:
        """Apply pruning to model"""
        
        output_path = self.output_dir / f"{variant_id}_pruned.onnx"
        
        prune_config = PruningConfig(
            pruning_type=tech.get('type', 'magnitude'),
            pruning_scope=tech.get('scope', 'global'),
            sparsity=tech.get('sparsity', 0.3)
        )
        
        pruner = ONNXPruner(prune_config)
        
        try:
            pruner.prune(model_path, output_path)
        except Exception as e:
            logger.error(f"Pruning failed: {e}")
            # Return original model if pruning fails
            return model_path
        
        return output_path
    
    def _estimate_speedup(self, optimizations: List[Dict[str, Any]]) -> float:
        """Estimate speedup from optimizations (heuristic)"""
        speedup = 1.0
        
        for opt in optimizations:
            if opt['name'] == 'quantization':
                if opt.get('precision') == 'int8':
                    speedup *= 2.0
                elif opt.get('precision') == 'int4':
                    speedup *= 3.0
                elif opt.get('precision') == 'fp16':
                    speedup *= 1.5
            
            elif opt['name'] == 'pruning':
                sparsity = opt.get('sparsity', 0.3)
                # Structured pruning gives more speedup
                if opt.get('type') == 'structured':
                    speedup *= (1 + sparsity * 0.8)
                else:
                    speedup *= (1 + sparsity * 0.3)  # Unstructured gives less
        
        return speedup
    
    def _estimate_accuracy_impact(self, optimizations: List[Dict[str, Any]]) -> float:
        """Estimate accuracy impact from optimizations (heuristic)"""
        accuracy_drop = 0.0
        
        for opt in optimizations:
            if opt['name'] == 'quantization':
                if opt.get('type') == 'qat':
                    accuracy_drop -= 0.01  # QAT has minimal impact
                elif opt.get('precision') == 'int8':
                    accuracy_drop -= 0.02
                elif opt.get('precision') == 'int4':
                    accuracy_drop -= 0.05
            
            elif opt['name'] == 'pruning':
                sparsity = opt.get('sparsity', 0.3)
                accuracy_drop -= sparsity * 0.08  # ~8% accuracy drop per 100% sparsity
        
        return accuracy_drop
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _save_variants_metadata(self, variants: List[ModelVariant]):
        """Save metadata for all variants"""
        
        metadata = {
            'num_variants': len(variants),
            'variants': []
        }
        
        for variant in variants:
            metadata['variants'].append({
                'variant_id': variant.variant_id,
                'variant_name': variant.variant_name,
                'model_path': str(variant.model_path),
                'model_size_mb': variant.model_size_mb,
                'compression_ratio': variant.compression_ratio,
                'estimated_speedup': variant.estimated_speedup,
                'estimated_accuracy_impact': variant.estimated_accuracy_impact,
                'optimizations': variant.optimizations_applied,
                'created_timestamp': variant.created_timestamp
            })
        
        metadata_path = self.output_dir / 'variants_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved variant metadata to {metadata_path}")
    
    def print_variants_summary(self, variants: List[ModelVariant]):
        """Print summary of all variants"""
        
        print("\n" + "=" * 80)
        print("GENERATED VARIANTS SUMMARY")
        print("=" * 80)
        
        for i, variant in enumerate(variants, 1):
            print(f"\nVariant {i}: {variant.variant_name}")
            print(f"  Model: {variant.model_path.name}")
            print(f"  Size: {variant.model_size_mb:.2f} MB ({variant.compression_ratio:.2f}× compression)")
            print(f"  Est. Speedup: {variant.estimated_speedup:.2f}×")
            print(f"  Est. Accuracy Impact: {variant.estimated_accuracy_impact*100:+.1f}%")
            print(f"  Optimizations:")
            for opt in variant.optimizations_applied:
                print(f"    - {opt['name']}: {opt}")
        
        print("\n" + "=" * 80 + "\n")


# Convenience function
def generate_variants(
    model_path: Path,
    strategy: OptimizationStrategy,
    output_dir: Path,
    num_variants: int = 3
) -> List[ModelVariant]:
    """Generate model variants (convenience wrapper)"""
    generator = VariantGenerator(output_dir)
    return generator.generate_variants(model_path, strategy, num_variants)
