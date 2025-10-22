"""
Optimization Techniques Combination Explorer (Enhanced)
Systematically test all combinations with robust error handling
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from itertools import product
import json
import time
import traceback

import numpy as np
import pandas as pd

from src.optimizers.quantizer import ONNXQuantizer, QuantizationConfig
from src.optimizers.pruner import ONNXPruner, PruningConfig
from src.utils.logger import logger
from src.utils.helpers import get_model_size_mb, save_json


@dataclass
class TechniqueConfig:
    """Configuration space for a technique"""
    name: str
    enabled: bool = True
    variants: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class CombinationResult:
    """Result of testing one combination (Enhanced)"""
    combination_id: str
    combination_name: str
    techniques: Dict[str, Any]
    
    # Success tracking
    success: bool
    error_message: Optional[str] = None
    
    # Model paths
    model_path: Optional[Path] = None
    base_model_path: Optional[Path] = None
    
    # Metrics (None if failed)
    model_size_mb: Optional[float] = None
    compression_ratio: Optional[float] = None
    estimated_speedup: Optional[float] = None
    estimated_accuracy_impact: Optional[float] = None
    
    # Detailed metrics
    sparsity: Optional[float] = None
    quantization_level: Optional[str] = None
    optimization_time_seconds: float = 0.0
    
    # Ranking
    pareto_optimal: bool = False
    rank: int = 0


class CombinationExplorer:
    """
    Enhanced Combination Explorer with robust error handling
    """
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized Combination Explorer (output: {output_dir})")
    
    def explore_all_combinations(
        self,
        model_path: Path,
        technique_configs: Optional[Dict[str, TechniqueConfig]] = None,
        max_combinations: Optional[int] = None,
        validate_models: bool = False
    ) -> List[CombinationResult]:
        """
        Explore all combinations with robust error handling
        """
        logger.info("=" * 80)
        logger.info("OPTIMIZATION TECHNIQUES COMBINATION EXPLORATION")
        logger.info("=" * 80)
        
        # Get default configs if not provided
        if technique_configs is None:
            technique_configs = self._get_default_technique_configs()
        
        # Generate all combinations
        combinations = self._generate_combinations(technique_configs)
        
        if max_combinations and len(combinations) > max_combinations:
            logger.info(f"Limiting to {max_combinations} combinations (out of {len(combinations)})")
            combinations = combinations[:max_combinations]
        
        logger.info(f"\nTesting {len(combinations)} combinations...")
        
        # Test each combination
        results = []
        successful = 0
        failed = 0
        base_size = get_model_size_mb(model_path)
        
        for i, combo in enumerate(combinations, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Combination {i}/{len(combinations)}: {combo['name']}")
            logger.info(f"{'='*60}")
            
            result = self._test_combination_safe(
                model_path,
                combo,
                base_size,
                combination_id=f"combo_{i:03d}",
                validate=validate_models
            )
            
            results.append(result)
            
            if result.success:
                successful += 1
                logger.info(f"✓ Success: {result.model_size_mb:.2f} MB ({result.compression_ratio:.2f}× compression)")
            else:
                failed += 1
                logger.warning(f"✗ Failed: {result.error_message}")
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"EXPLORATION COMPLETE")
        logger.info(f"{'='*80}")
        logger.info(f"  Successful: {successful}/{len(combinations)} ({successful/len(combinations)*100:.1f}%)")
        logger.info(f"  Failed: {failed}/{len(combinations)}")
        
        # Analyze successful results
        successful_results = [r for r in results if r.success]
        if successful_results:
            analysis = self._analyze_results(successful_results, model_path)
            self._save_detailed_report(results, analysis)
        else:
            logger.error("No successful combinations! Check input model and dependencies.")
        
        return results
    
    def generate_variants(
        self,
        model_path: Path,
        strategy,
        num_variants: int = 3
    ) -> List[CombinationResult]:
        """
        Generate variants (backward compatibility with old API)
        """
        # Simplified: just run top N combinations
        results = self.explore_all_combinations(
            model_path,
            max_combinations=num_variants * 3  # Test more, return top N
        )
        
        # Return top N successful ones
        successful = [r for r in results if r.success]
        successful.sort(key=lambda x: x.compression_ratio * x.estimated_speedup, reverse=True)
        
        return successful[:num_variants]
    
    def _get_default_technique_configs(self) -> Dict[str, TechniqueConfig]:
        """Get default exploration space (Enhanced with more variants)"""
        configs = {
            'quantization': TechniqueConfig(
                name='quantization',
                variants=[
                    {'type': 'none', 'name': 'No Quantization'},
                    {'type': 'dynamic', 'precision': 'int8', 'name': 'Dynamic INT8'},
                    {'type': 'dynamic', 'precision': 'fp16', 'name': 'Dynamic FP16'},
                ]
            ),
            'pruning': TechniqueConfig(
                name='pruning',
                variants=[
                    {'type': 'none', 'name': 'No Pruning'},
                    {'type': 'magnitude', 'scope': 'global', 'sparsity': 0.2, 'name': 'Magnitude 20%'},
                    {'type': 'magnitude', 'scope': 'global', 'sparsity': 0.4, 'name': 'Magnitude 40%'},
                    {'type': 'magnitude', 'scope': 'global', 'sparsity': 0.6, 'name': 'Magnitude 60%'},
                    {'type': 'structured', 'sparsity': 0.3, 'name': 'Structured 30%'},
                    {'type': 'structured', 'sparsity': 0.5, 'name': 'Structured 50%'},
                ]
            ),
            'order': TechniqueConfig(
                name='order',
                variants=[
                    {'sequence': 'quantize_first', 'name': 'Quantize → Prune'},
                    {'sequence': 'prune_first', 'name': 'Prune → Quantize'},
                ]
            )
        }
        
        return configs
    
    def _generate_combinations(
        self,
        technique_configs: Dict[str, TechniqueConfig]
    ) -> List[Dict[str, Any]]:
        """Generate all possible combinations"""
        
        enabled_techniques = {
            name: config for name, config in technique_configs.items()
            if config.enabled
        }
        
        variant_lists = [
            [(name, variant) for variant in config.variants]
            for name, config in enabled_techniques.items()
        ]
        
        # Generate Cartesian product
        combinations = []
        for combo_tuple in product(*variant_lists):
            combo_dict = {name: variant for name, variant in combo_tuple}
            
            # Generate readable name
            parts = []
            for name, variant in combo_dict.items():
                if variant.get('type') != 'none' and name != 'order':
                    parts.append(variant.get('name', str(variant)))
            
            if combo_dict.get('order', {}).get('sequence'):
                order_name = combo_dict['order'].get('name', '')
                if order_name:
                    combo_name = f"{order_name}: {' + '.join(parts)}" if parts else "Baseline"
                else:
                    combo_name = ' + '.join(parts) if parts else "Baseline"
            else:
                combo_name = ' + '.join(parts) if parts else "Baseline"
            
            combinations.append({
                'name': combo_name,
                'config': combo_dict
            })
        
        logger.info(f"Generated {len(combinations)} unique combinations")
        return combinations
    
    def _test_combination_safe(
        self,
        base_model_path: Path,
        combination: Dict[str, Any],
        base_size: float,
        combination_id: str,
        validate: bool
    ) -> CombinationResult:
        """Test combination with comprehensive error handling"""
        
        start_time = time.time()
        current_model = base_model_path
        applied_techniques = {}
        error_message = None
        combo_config = combination['config']
        
        try:
            # Determine application order
            order = combo_config.get('order', {}).get('sequence', 'quantize_first')
            
            # Skip if both are 'none'
            quant_type = combo_config.get('quantization', {}).get('type', 'none')
            prune_type = combo_config.get('pruning', {}).get('type', 'none')
            
            if quant_type == 'none' and prune_type == 'none':
                # Baseline - just copy
                return CombinationResult(
                    combination_id=combination_id,
                    combination_name=combination['name'],
                    techniques={},
                    success=True,
                    model_path=base_model_path,
                    base_model_path=base_model_path,
                    model_size_mb=base_size,
                    compression_ratio=1.0,
                    estimated_speedup=1.0,
                    estimated_accuracy_impact=0.0,
                    sparsity=0.0,
                    quantization_level='fp32',
                    optimization_time_seconds=time.time() - start_time
                )
            
            # Apply techniques in order
            if order == 'prune_first':
                sequence = ['pruning', 'quantization']
            else:
                sequence = ['quantization', 'pruning']
            
            for technique in sequence:
                if technique in combo_config:
                    config = combo_config[technique]
                    
                    if config.get('type') != 'none':
                        output_path = self.output_dir / f"{combination_id}_{technique}.onnx"
                        
                        if technique == 'quantization':
                            current_model = self._apply_quantization_safe(
                                current_model, output_path, config
                            )
                        elif technique == 'pruning':
                            current_model = self._apply_pruning_safe(
                                current_model, output_path, config
                            )
                        
                        if current_model:
                            applied_techniques[technique] = config
                        else:
                            raise Exception(f"{technique.capitalize()} failed")
            
            # Calculate metrics
            final_size = get_model_size_mb(current_model)
            compression_ratio = base_size / final_size if final_size > 0 else 1.0
            
            # Estimate performance
            estimated_speedup = self._estimate_speedup(applied_techniques)
            estimated_accuracy_impact = self._estimate_accuracy_impact(applied_techniques)
            
            # Calculate sparsity
            from src.optimizers.pruner import analyze_model_sparsity
            sparsity_stats = analyze_model_sparsity(current_model)
            sparsity = sparsity_stats['overall_sparsity']
            
            # Determine quantization level
            quant_config = applied_techniques.get('quantization', {})
            quant_level = quant_config.get('precision', 'fp32') if quant_config else 'fp32'
            
            return CombinationResult(
                combination_id=combination_id,
                combination_name=combination['name'],
                techniques=applied_techniques,
                success=True,
                model_path=current_model,
                base_model_path=base_model_path,
                model_size_mb=final_size,
                compression_ratio=compression_ratio,
                estimated_speedup=estimated_speedup,
                estimated_accuracy_impact=estimated_accuracy_impact,
                sparsity=sparsity,
                quantization_level=quant_level,
                optimization_time_seconds=time.time() - start_time
            )
            
        except Exception as e:
            error_message = str(e)
            logger.debug(f"Error details: {traceback.format_exc()}")
            
            return CombinationResult(
                combination_id=combination_id,
                combination_name=combination['name'],
                techniques=combo_config,
                success=False,
                error_message=error_message,
                optimization_time_seconds=time.time() - start_time
            )
    
    def _apply_quantization_safe(self, model_path: Path, output_path: Path, config: Dict) -> Optional[Path]:
        """Apply quantization with error handling"""
        try:
            quant_config = QuantizationConfig(
                quantization_type=config.get('type', 'dynamic'),
                precision=config.get('precision', 'int8')
            )
            
            quantizer = ONNXQuantizer(quant_config)
            quantizer.quantize(model_path, output_path)
            
            return output_path if output_path.exists() else None
        except Exception as e:
            logger.warning(f"Quantization error: {e}")
            return None
    
    def _apply_pruning_safe(self, model_path: Path, output_path: Path, config: Dict) -> Optional[Path]:
        """Apply pruning with error handling"""
        try:
            prune_config = PruningConfig(
                pruning_type=config.get('type', 'magnitude'),
                pruning_scope=config.get('scope', 'global'),
                sparsity=config.get('sparsity', 0.3)
            )
            
            pruner = ONNXPruner(prune_config)
            pruner.prune(model_path, output_path)
            
            return output_path if output_path.exists() else None
        except Exception as e:
            logger.warning(f"Pruning error: {e}")
            return None
    
    def _estimate_speedup(self, techniques: Dict) -> float:
        """Estimate speedup from applied techniques"""
        speedup = 1.0
        
        if 'quantization' in techniques:
            quant = techniques['quantization']
            if quant.get('precision') == 'int8':
                speedup *= 2.0
            elif quant.get('precision') == 'fp16':
                speedup *= 1.5
        
        if 'pruning' in techniques:
            prune = techniques['pruning']
            sparsity = prune.get('sparsity', 0)
            if prune.get('type') == 'structured':
                speedup *= (1 + sparsity * 0.8)
            else:
                speedup *= (1 + sparsity * 0.3)
        
        return speedup
    
    def _estimate_accuracy_impact(self, techniques: Dict) -> float:
        """Estimate accuracy impact"""
        impact = 0.0
        
        if 'quantization' in techniques:
            quant = techniques['quantization']
            if quant.get('precision') == 'int8':
                impact -= 0.02
            elif quant.get('precision') == 'fp16':
                impact -= 0.01
        
        if 'pruning' in techniques:
            prune = techniques['pruning']
            sparsity = prune.get('sparsity', 0)
            impact -= sparsity * 0.08
        
        return impact
    
    def _analyze_results(self, results: List[CombinationResult], base_model_path: Path) -> Dict[str, Any]:
        """Analyze all successful results"""
        
        logger.info("\n" + "=" * 80)
        logger.info("ANALYZING RESULTS")
        logger.info("=" * 80)
        
        # Convert to DataFrame
        data = []
        for r in results:
            data.append({
                'id': r.combination_id,
                'name': r.combination_name,
                'size_mb': r.model_size_mb,
                'compression': r.compression_ratio,
                'speedup': r.estimated_speedup,
                'accuracy_impact': r.estimated_accuracy_impact,
                'sparsity': r.sparsity,
                'quantization': r.quantization_level,
                'time': r.optimization_time_seconds,
                'has_quant': 'quantization' in r.techniques,
                'has_prune': 'pruning' in r.techniques,
            })
        
        df = pd.DataFrame(data)
        
        # Calculate balanced score
        df['balanced_score'] = df['compression'] * df['speedup']
        
        # Find best combinations
        best_compression = df.nlargest(5, 'compression')
        best_speedup = df.nlargest(5, 'speedup')
        best_balanced = df.nlargest(5, 'balanced_score')
        
        # Identify Pareto front
        pareto_indices = self._compute_pareto_front(
            df[['compression', 'speedup', 'accuracy_impact']].values
        )
        
        for idx in pareto_indices:
            results[idx].pareto_optimal = True
        
        analysis = {
            'total_combinations': len(results),
            'pareto_optimal_count': len(pareto_indices),
            'best_compression': {
                'combination_id': best_compression.iloc[0]['id'],
                'name': best_compression.iloc[0]['name'],
                'compression': float(best_compression.iloc[0]['compression']),
                'speedup': float(best_compression.iloc[0]['speedup'])
            },
            'best_speedup': {
                'combination_id': best_speedup.iloc[0]['id'],
                'name': best_speedup.iloc[0]['name'],
                'speedup': float(best_speedup.iloc[0]['speedup']),
                'compression': float(best_speedup.iloc[0]['compression'])
            },
            'best_balanced': {
                'combination_id': best_balanced.iloc[0]['id'],
                'name': best_balanced.iloc[0]['name'],
                'score': float(best_balanced.iloc[0]['balanced_score']),
                'compression': float(best_balanced.iloc[0]['compression']),
                'speedup': float(best_balanced.iloc[0]['speedup'])
            },
            'statistics': {
                'avg_compression': float(df['compression'].mean()),
                'avg_speedup': float(df['speedup'].mean()),
                'avg_time': float(df['time'].mean()),
                'max_compression': float(df['compression'].max()),
                'max_speedup': float(df['speedup'].max())
            },
            'technique_impact': self._analyze_technique_impact(df)
        }
        
        self._print_analysis_summary(analysis, results)
        
        return analysis
    
    def _compute_pareto_front(self, objectives: np.ndarray) -> List[int]:
        """Compute Pareto front indices"""
        objectives[:, 2] = -objectives[:, 2]  # Flip accuracy impact
        
        is_pareto = np.ones(len(objectives), dtype=bool)
        
        for i, obj_i in enumerate(objectives):
            for j, obj_j in enumerate(objectives):
                if i != j:
                    if np.all(obj_j >= obj_i) and np.any(obj_j > obj_i):
                        is_pareto[i] = False
                        break
        
        return np.where(is_pareto)[0].tolist()
    
    def _analyze_technique_impact(self, df: pd.DataFrame) -> Dict:
        """Analyze impact of each technique"""
        
        impact = {}
        
        with_quant = df[df['has_quant'] == True]
        without_quant = df[df['has_quant'] == False]
        
        if len(without_quant) > 0:
            impact['quantization'] = {
                'avg_compression_gain': float(with_quant['compression'].mean() - without_quant['compression'].mean()),
                'avg_speedup_gain': float(with_quant['speedup'].mean() - without_quant['speedup'].mean())
            }
        
        with_prune = df[df['has_prune'] == True]
        without_prune = df[df['has_prune'] == False]
        
        if len(without_prune) > 0:
            impact['pruning'] = {
                'avg_compression_gain': float(with_prune['compression'].mean() - without_prune['compression'].mean()),
                'avg_speedup_gain': float(with_prune['speedup'].mean() - without_prune['speedup'].mean())
            }
        
        both = df[(df['has_quant'] == True) & (df['has_prune'] == True)]
        if len(both) > 0:
            impact['combined'] = {
                'avg_compression': float(both['compression'].mean()),
                'avg_speedup': float(both['speedup'].mean())
            }
        
        return impact
    
    def _print_analysis_summary(self, analysis: Dict, results: List[CombinationResult]):
        """Print analysis summary"""
        
        print("\n" + "=" * 80)
        print("ANALYSIS SUMMARY")
        print("=" * 80)
        
        print(f"\n📊 Overview:")
        print(f"  Total combinations: {analysis['total_combinations']}")
        print(f"  Pareto-optimal: {analysis['pareto_optimal_count']}")
        
        print(f"\n🏆 Best Results:")
        print(f"\n  Best Compression: {analysis['best_compression']['name']}")
        print(f"    - {analysis['best_compression']['compression']:.2f}× compression")
        print(f"    - {analysis['best_compression']['speedup']:.2f}× speedup")
        
        print(f"\n  Best Speedup: {analysis['best_speedup']['name']}")
        print(f"    - {analysis['best_speedup']['speedup']:.2f}× speedup")
        print(f"    - {analysis['best_speedup']['compression']:.2f}× compression")
        
        print(f"\n  Best Balanced: {analysis['best_balanced']['name']}")
        print(f"    - Score: {analysis['best_balanced']['score']:.2f}")
        print(f"    - {analysis['best_balanced']['compression']:.2f}× compression")
        print(f"    - {analysis['best_balanced']['speedup']:.2f}× speedup")
        
        print("\n" + "=" * 80)
    
    def _save_detailed_report(self, results: List[CombinationResult], analysis: Dict):
        """Save detailed report"""
        
        # JSON report
        report_data = {
            'summary': {
                'total': len(results),
                'successful': sum(1 for r in results if r.success),
                'failed': sum(1 for r in results if not r.success)
            },
            'analysis': analysis,
            'results': []
        }
        
        for r in results:
            report_data['results'].append({
                'id': r.combination_id,
                'name': r.combination_name,
                'success': r.success,
                'error': r.error_message,
                'metrics': {
                    'size_mb': r.model_size_mb,
                    'compression': r.compression_ratio,
                    'speedup': r.estimated_speedup,
                    'accuracy_impact': r.estimated_accuracy_impact
                } if r.success else None
            })
        
        save_json(report_data, self.output_dir / 'combination_exploration_report.json')
        
        # CSV for successful only
        successful = [r for r in results if r.success]
        if successful:
            df_data = []
            for r in successful:
                df_data.append({
                    'ID': r.combination_id,
                    'Name': r.combination_name,
                    'Size (MB)': r.model_size_mb,
                    'Compression': r.compression_ratio,
                    'Speedup': r.estimated_speedup,
                    'Acc Impact (%)': r.estimated_accuracy_impact * 100,
                    'Sparsity (%)': r.sparsity * 100,
                    'Quantization': r.quantization_level,
                    'Pareto Optimal': r.pareto_optimal,
                    'Time (s)': r.optimization_time_seconds
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(self.output_dir / 'combination_exploration_results.csv', index=False)
        
        logger.info(f"✓ Reports saved to {self.output_dir}")


# Convenience function
def generate_variants(
    model_path: Path,
    strategy,
    output_dir: Path,
    num_variants: int = 3
) -> List[CombinationResult]:
    """Generate model variants (backward compatibility)"""
    generator = CombinationExplorer(output_dir)
    return generator.generate_variants(model_path, strategy, num_variants)
