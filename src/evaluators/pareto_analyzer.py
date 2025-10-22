"""
Pareto Analysis - Multi-objective optimization and trade-off visualization
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import numpy as np

from src.solvers.variant_generator import ModelVariant
from src.utils.logger import logger


@dataclass
class ParetoPoint:
    """Represents a point in the Pareto front"""
    variant: ModelVariant
    objectives: Dict[str, float]  # objective_name -> value
    is_dominated: bool = False
    dominates_count: int = 0


class ParetoAnalyzer:
    """
    Multi-objective Pareto analysis for model variants
    """
    
    def __init__(
        self,
        objectives: List[str] = None,
        minimize: List[str] = None,
        maximize: List[str] = None
    ):
        """
        Initialize Pareto analyzer
        
        Args:
            objectives: List of objective names to consider
            minimize: List of objectives to minimize (e.g., 'latency', 'size')
            maximize: List of objectives to maximize (e.g., 'accuracy')
        """
        self.objectives = objectives or ['accuracy', 'latency', 'size']
        self.minimize = minimize or ['latency', 'size']
        self.maximize = maximize or ['accuracy']
        
        logger.info("Initialized Pareto analyzer")
        logger.info(f"  Objectives: {self.objectives}")
        logger.info(f"  Minimize: {self.minimize}")
        logger.info(f"  Maximize: {self.maximize}")
    
    def analyze(
        self,
        variants: List[ModelVariant],
        measured_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """
        Perform Pareto analysis on model variants
        
        Args:
            variants: List of model variants to analyze
            measured_metrics: Optional measured metrics for each variant
                             Format: {variant_id: {'accuracy': 0.95, 'latency': 150, ...}}
        
        Returns:
            Dictionary containing Pareto analysis results
        """
        logger.info(f"Performing Pareto analysis on {len(variants)} variants...")
        
        # Extract objectives for each variant
        pareto_points = self._extract_objectives(variants, measured_metrics)
        
        # Identify Pareto front
        pareto_front = self._compute_pareto_front(pareto_points)
        
        # Calculate additional metrics
        analysis = {
            'pareto_front': pareto_front,
            'all_points': pareto_points,
            'num_pareto_optimal': len(pareto_front),
            'num_dominated': len(pareto_points) - len(pareto_front),
            'statistics': self._compute_statistics(pareto_points),
            'rankings': self._rank_variants(pareto_points),
            'trade_offs': self._analyze_trade_offs(pareto_front),
        }
        
        logger.info(f"✓ Pareto analysis complete")
        logger.info(f"  Pareto optimal variants: {len(pareto_front)}/{len(variants)}")
        
        return analysis
    
    def _extract_objectives(
        self,
        variants: List[ModelVariant],
        measured_metrics: Optional[Dict[str, Dict[str, float]]] = None
    ) -> List[ParetoPoint]:
        """Extract objective values for each variant"""
        
        pareto_points = []
        
        for variant in variants:
            objectives = {}
            
            # Use measured metrics if available, otherwise use estimates
            if measured_metrics and variant.variant_id in measured_metrics:
                metrics = measured_metrics[variant.variant_id]
                
                # Accuracy
                if 'accuracy' in self.objectives:
                    objectives['accuracy'] = metrics.get('accuracy', 1.0 + variant.estimated_accuracy_impact)
                
                # Latency
                if 'latency' in self.objectives:
                    objectives['latency'] = metrics.get('latency_ms', 100.0)
                
                # Size
                if 'size' in self.objectives:
                    objectives['size'] = metrics.get('size_mb', variant.model_size_mb)
                
                # Memory
                if 'memory' in self.objectives:
                    objectives['memory'] = metrics.get('memory_mb', variant.model_size_mb * 1.5)
                
                # Throughput
                if 'throughput' in self.objectives:
                    objectives['throughput'] = metrics.get('throughput_fps', 10.0)
            
            else:
                # Use estimates from variant
                if 'accuracy' in self.objectives:
                    # Assume baseline accuracy of 95% for estimation
                    objectives['accuracy'] = 0.95 + variant.estimated_accuracy_impact
                
                if 'latency' in self.objectives:
                    # Estimate latency (inverse of speedup)
                    baseline_latency = 200.0  # ms
                    objectives['latency'] = baseline_latency / variant.estimated_speedup
                
                if 'size' in self.objectives:
                    objectives['size'] = variant.model_size_mb
                
                if 'memory' in self.objectives:
                    # Estimate memory as 1.5x model size
                    objectives['memory'] = variant.model_size_mb * 1.5
                
                if 'throughput' in self.objectives:
                    # Estimate throughput from speedup
                    baseline_fps = 5.0
                    objectives['throughput'] = baseline_fps * variant.estimated_speedup
            
            pareto_point = ParetoPoint(
                variant=variant,
                objectives=objectives
            )
            
            pareto_points.append(pareto_point)
        
        return pareto_points
    
    def _compute_pareto_front(self, points: List[ParetoPoint]) -> List[ParetoPoint]:
        """Compute the Pareto front (non-dominated solutions)"""
        
        logger.info("Computing Pareto front...")
        
        # Check each point for dominance
        for i, point_i in enumerate(points):
            for j, point_j in enumerate(points):
                if i == j:
                    continue
                
                if self._dominates(point_j, point_i):
                    point_i.is_dominated = True
                    point_j.dominates_count += 1
        
        # Pareto front consists of non-dominated points
        pareto_front = [p for p in points if not p.is_dominated]
        
        return pareto_front
    
    def _dominates(self, point_a: ParetoPoint, point_b: ParetoPoint) -> bool:
        """
        Check if point_a dominates point_b
        
        A dominates B if:
        - A is better or equal in all objectives
        - A is strictly better in at least one objective
        """
        better_or_equal = True
        strictly_better = False
        
        for obj in self.objectives:
            val_a = point_a.objectives.get(obj, 0)
            val_b = point_b.objectives.get(obj, 0)
            
            # Determine if we're minimizing or maximizing
            if obj in self.minimize:
                # For minimization: lower is better
                if val_a > val_b:
                    better_or_equal = False
                    break
                elif val_a < val_b:
                    strictly_better = True
            
            elif obj in self.maximize:
                # For maximization: higher is better
                if val_a < val_b:
                    better_or_equal = False
                    break
                elif val_a > val_b:
                    strictly_better = True
        
        return better_or_equal and strictly_better
    
    def _compute_statistics(self, points: List[ParetoPoint]) -> Dict[str, Any]:
        """Compute statistics for each objective"""
        
        stats = {}
        
        for obj in self.objectives:
            values = [p.objectives.get(obj, 0) for p in points]
            
            stats[obj] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'median': float(np.median(values)),
            }
        
        return stats
    
    def _rank_variants(self, points: List[ParetoPoint]) -> List[Dict[str, Any]]:
        """Rank variants by Pareto dominance"""
        
        # Sort by dominance count (descending) and dominated status
        sorted_points = sorted(
            points,
            key=lambda p: (-p.dominates_count, p.is_dominated)
        )
        
        rankings = []
        for rank, point in enumerate(sorted_points, 1):
            rankings.append({
                'rank': rank,
                'variant_id': point.variant.variant_id,
                'variant_name': point.variant.variant_name,
                'is_pareto_optimal': not point.is_dominated,
                'dominates_count': point.dominates_count,
                'objectives': point.objectives
            })
        
        return rankings
    
    def _analyze_trade_offs(self, pareto_front: List[ParetoPoint]) -> Dict[str, Any]:
        """Analyze trade-offs in the Pareto front"""
        
        if len(pareto_front) < 2:
            return {'trade_offs': []}
        
        trade_offs = []
        
        # Compare consecutive points in Pareto front
        # Sort by one objective to make comparisons meaningful
        if 'accuracy' in self.objectives:
            sorted_front = sorted(
                pareto_front,
                key=lambda p: p.objectives.get('accuracy', 0),
                reverse=True
            )
        else:
            sorted_front = pareto_front
        
        for i in range(len(sorted_front) - 1):
            point_a = sorted_front[i]
            point_b = sorted_front[i + 1]
            
            trade_off = {
                'from_variant': point_a.variant.variant_name,
                'to_variant': point_b.variant.variant_name,
                'changes': {}
            }
            
            for obj in self.objectives:
                val_a = point_a.objectives.get(obj, 0)
                val_b = point_b.objectives.get(obj, 0)
                
                if val_a != val_b:
                    change_pct = ((val_b - val_a) / val_a * 100) if val_a != 0 else 0
                    
                    trade_off['changes'][obj] = {
                        'from': val_a,
                        'to': val_b,
                        'change_pct': change_pct,
                        'improvement': (val_b > val_a) if obj in self.maximize else (val_b < val_a)
                    }
            
            trade_offs.append(trade_off)
        
        return {'trade_offs': trade_offs}
    
    def print_analysis(self, analysis: Dict[str, Any]):
        """Print Pareto analysis results"""
        
        print("\n" + "=" * 80)
        print("PARETO ANALYSIS RESULTS")
        print("=" * 80)
        
        # Statistics
        print("\n📊 Objective Statistics:")
        for obj, stats in analysis['statistics'].items():
            print(f"\n  {obj.upper()}:")
            print(f"    Min: {stats['min']:.4f}")
            print(f"    Max: {stats['max']:.4f}")
            print(f"    Mean: {stats['mean']:.4f}")
            print(f"    Std: {stats['std']:.4f}")
        
        # Pareto front
        print(f"\n🏆 Pareto Front ({analysis['num_pareto_optimal']} variants):")
        for point in analysis['pareto_front']:
            print(f"\n  • {point.variant.variant_name}")
            print(f"    Dominates: {point.dominates_count} other variants")
            print(f"    Objectives:")
            for obj, val in point.objectives.items():
                print(f"      - {obj}: {val:.4f}")
        
        # Rankings
        print(f"\n📋 Variant Rankings:")
        print(f"  {'Rank':<6} {'Variant':<20} {'Pareto Optimal':<15} {'Dominates':<10}")
        print(f"  {'-'*6} {'-'*20} {'-'*15} {'-'*10}")
        
        for ranking in analysis['rankings'][:10]:  # Top 10
            optimal_str = "✓ Yes" if ranking['is_pareto_optimal'] else "  No"
            print(f"  {ranking['rank']:<6} {ranking['variant_name']:<20} {optimal_str:<15} {ranking['dominates_count']:<10}")
        
        # Trade-offs
        if analysis['trade_offs']['trade_offs']:
            print(f"\n⚖️  Trade-off Analysis:")
            for i, trade_off in enumerate(analysis['trade_offs']['trade_offs'], 1):
                print(f"\n  Trade-off {i}: {trade_off['from_variant']} → {trade_off['to_variant']}")
                for obj, change in trade_off['changes'].items():
                    improvement = "↑" if change['improvement'] else "↓"
                    print(f"    {improvement} {obj}: {change['from']:.4f} → {change['to']:.4f} ({change['change_pct']:+.1f}%)")
        
        print("\n" + "=" * 80 + "\n")
    
    def save_analysis(self, analysis: Dict[str, Any], output_path: Path):
        """Save analysis to JSON file"""
        
        # Convert to serializable format
        serializable_analysis = {
            'num_pareto_optimal': analysis['num_pareto_optimal'],
            'num_dominated': analysis['num_dominated'],
            'statistics': analysis['statistics'],
            'rankings': analysis['rankings'],
            'trade_offs': analysis['trade_offs'],
            'pareto_front': [
                {
                    'variant_id': p.variant.variant_id,
                    'variant_name': p.variant.variant_name,
                    'objectives': p.objectives,
                    'dominates_count': p.dominates_count
                }
                for p in analysis['pareto_front']
            ]
        }
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)
        
        logger.info(f"Saved Pareto analysis to {output_path}")


# Convenience function
def analyze_pareto_front(
    variants: List[ModelVariant],
    objectives: List[str] = None,
    measured_metrics: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, Any]:
    """Analyze Pareto front (convenience wrapper)"""
    
    analyzer = ParetoAnalyzer(objectives=objectives)
    return analyzer.analyze(variants, measured_metrics)
