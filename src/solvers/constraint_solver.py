"""
Constraint Solver - Determine optimal optimization strategy based on constraints
"""
from pathlib import Path
from typing import Dict, List, Optional, Any, Literal
from dataclasses import dataclass, field
import json

from src.utils.logger import logger


# Type definitions
ConstraintPriority = Literal['critical', 'high', 'medium', 'low']
OptimizationGoal = Literal['speed', 'size', 'accuracy', 'balanced']


@dataclass
class HardwareConstraints:
    """Hardware-specific constraints"""
    device_name: str = "Unknown Device"
    cpu_cores: int = 4
    cpu_frequency_ghz: float = 2.0
    ram_total_gb: float = 4.0
    ram_available_gb: float = 2.0
    has_gpu: bool = False
    has_npu: bool = False
    storage_available_gb: float = 1.0
    thermal_class: str = "moderate"  # low, moderate, high
    power_budget: str = "mobile"  # mobile, desktop, server


@dataclass
class PerformanceConstraints:
    """Performance requirements"""
    max_latency_ms: Optional[float] = None
    min_throughput_fps: Optional[float] = None
    max_model_size_mb: Optional[float] = None
    max_memory_mb: Optional[float] = None
    min_accuracy: Optional[float] = None
    max_accuracy_drop: Optional[float] = None
    max_power_watts: Optional[float] = None


@dataclass
class OptimizationConstraints:
    """Complete constraint specification"""
    hardware: HardwareConstraints = field(default_factory=HardwareConstraints)
    performance: PerformanceConstraints = field(default_factory=PerformanceConstraints)
    optimization_goal: OptimizationGoal = 'balanced'
    allowed_techniques: List[str] = field(default_factory=lambda: ['quantization', 'pruning', 'distillation'])
    priority: ConstraintPriority = 'high'


@dataclass
class OptimizationStrategy:
    """Optimization strategy recommendation"""
    strategy_name: str
    description: str
    techniques: List[Dict[str, Any]]
    expected_compression: float
    expected_speedup: float
    expected_accuracy_impact: float
    risk_level: str  # low, medium, high
    estimated_time_minutes: float
    reasoning: List[str]


class ConstraintSolver:
    """
    Solve optimization constraints and recommend strategies
    """
    
    def __init__(self, constraints: OptimizationConstraints):
        """
        Initialize constraint solver
        
        Args:
            constraints: User-defined constraints
        """
        self.constraints = constraints
        logger.info("Initialized constraint solver")
    
    def solve(self, model_analysis: Dict[str, Any]) -> OptimizationStrategy:
        """
        Solve constraints and determine optimization strategy
        
        Args:
            model_analysis: Analysis results from ONNXAnalyzer
        
        Returns:
            Recommended optimization strategy
        """
        logger.info("Solving optimization constraints...")
        logger.info(f"Optimization goal: {self.constraints.optimization_goal}")
        
        # Analyze constraints
        constraint_analysis = self._analyze_constraints(model_analysis)
        
        # Determine primary bottleneck
        bottleneck = self._identify_bottleneck(constraint_analysis, model_analysis)
        logger.info(f"Primary bottleneck: {bottleneck}")
        
        # Generate strategy based on goal and bottleneck
        if self.constraints.optimization_goal == 'speed':
            strategy = self._strategy_for_speed(constraint_analysis, model_analysis, bottleneck)
        
        elif self.constraints.optimization_goal == 'size':
            strategy = self._strategy_for_size(constraint_analysis, model_analysis, bottleneck)
        
        elif self.constraints.optimization_goal == 'accuracy':
            strategy = self._strategy_for_accuracy(constraint_analysis, model_analysis, bottleneck)
        
        else:  # balanced
            strategy = self._strategy_balanced(constraint_analysis, model_analysis, bottleneck)
        
        logger.info(f"✓ Strategy determined: {strategy.strategy_name}")
        self._print_strategy(strategy)
        
        return strategy
    
    def _analyze_constraints(self, model_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze constraint feasibility"""
        analysis = {
            'latency_constrained': self.constraints.performance.max_latency_ms is not None,
            'size_constrained': self.constraints.performance.max_model_size_mb is not None,
            'accuracy_constrained': self.constraints.performance.min_accuracy is not None,
            'memory_constrained': self.constraints.performance.max_memory_mb is not None,
            'hardware_limited': self._is_hardware_limited(),
            'model_size_mb': model_analysis['model_info']['model_size_mb'],
            'model_params_m': model_analysis['parameters']['total_parameters_millions'],
        }
        
        # Check if constraints are tight
        if analysis['size_constrained']:
            current_size = analysis['model_size_mb']
            target_size = self.constraints.performance.max_model_size_mb
            analysis['size_pressure'] = current_size / target_size if target_size > 0 else float('inf')
        else:
            analysis['size_pressure'] = 1.0
        
        if analysis['memory_constrained']:
            estimated_memory = model_analysis['memory']['peak_memory_mb']
            target_memory = self.constraints.performance.max_memory_mb
            analysis['memory_pressure'] = estimated_memory / target_memory if target_memory > 0 else float('inf')
        else:
            analysis['memory_pressure'] = 1.0
        
        return analysis
    
    def _is_hardware_limited(self) -> bool:
        """Check if hardware is resource-constrained"""
        hw = self.constraints.hardware
        
        # Heuristic: low-end device
        if hw.cpu_cores <= 4 and hw.ram_available_gb < 2.0 and not hw.has_gpu:
            return True
        
        return False
    
    def _identify_bottleneck(
        self,
        constraint_analysis: Dict[str, Any],
        model_analysis: Dict[str, Any]
    ) -> str:
        """Identify primary bottleneck"""
        bottlenecks = []
        
        # Size bottleneck
        if constraint_analysis.get('size_pressure', 1.0) > 1.5:
            bottlenecks.append(('size', constraint_analysis['size_pressure']))
        
        # Memory bottleneck
        if constraint_analysis.get('memory_pressure', 1.0) > 1.5:
            bottlenecks.append(('memory', constraint_analysis['memory_pressure']))
        
        # Compute bottleneck (based on hardware)
        if constraint_analysis['hardware_limited']:
            bottlenecks.append(('compute', 2.0))
        
        # Latency bottleneck
        if constraint_analysis['latency_constrained']:
            bottlenecks.append(('latency', 2.0))
        
        if not bottlenecks:
            return 'none'
        
        # Return bottleneck with highest pressure
        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        return bottlenecks[0][0]
    
    def _strategy_for_speed(
        self,
        constraint_analysis: Dict[str, Any],
        model_analysis: Dict[str, Any],
        bottleneck: str
    ) -> OptimizationStrategy:
        """Generate strategy optimized for speed"""
        techniques = []
        reasoning = []
        
        reasoning.append("Goal: Maximize inference speed")
        reasoning.append(f"Primary bottleneck: {bottleneck}")
        
        # Quantization is almost always beneficial for speed
        if 'quantization' in self.constraints.allowed_techniques:
            techniques.append({
                'name': 'quantization',
                'type': 'static',
                'precision': 'int8',
                'per_channel': True,
                'priority': 1
            })
            reasoning.append("INT8 quantization for 2-4× speedup on most hardware")
        
        # Aggressive pruning for compute-bound scenarios
        if bottleneck in ['compute', 'latency']:
            if 'pruning' in self.constraints.allowed_techniques:
                techniques.append({
                    'name': 'pruning',
                    'type': 'structured',
                    'sparsity': 0.4,
                    'priority': 2
                })
                reasoning.append("Structured pruning (40%) to reduce compute without sparse libraries")
        
        # Light pruning for memory-bound scenarios
        elif bottleneck == 'memory':
            if 'pruning' in self.constraints.allowed_techniques:
                techniques.append({
                    'name': 'pruning',
                    'type': 'magnitude',
                    'scope': 'global',
                    'sparsity': 0.3,
                    'priority': 2
                })
                reasoning.append("Magnitude pruning (30%) to reduce memory footprint")
        
        # Operator fusion
        techniques.append({
            'name': 'graph_optimization',
            'type': 'operator_fusion',
            'priority': 3
        })
        reasoning.append("Operator fusion to reduce memory transfers")
        
        return OptimizationStrategy(
            strategy_name="Speed-Optimized",
            description="Maximize inference speed with quantization and pruning",
            techniques=techniques,
            expected_compression=2.5,
            expected_speedup=2.0,
            expected_accuracy_impact=-0.02,  # 2% drop
            risk_level='medium',
            estimated_time_minutes=15,
            reasoning=reasoning
        )
    
    def _strategy_for_size(
        self,
        constraint_analysis: Dict[str, Any],
        model_analysis: Dict[str, Any],
        bottleneck: str
    ) -> OptimizationStrategy:
        """Generate strategy optimized for model size"""
        techniques = []
        reasoning = []
        
        reasoning.append("Goal: Minimize model size")
        reasoning.append(f"Current size: {constraint_analysis['model_size_mb']:.2f} MB")
        
        if self.constraints.performance.max_model_size_mb:
            reasoning.append(f"Target size: {self.constraints.performance.max_model_size_mb:.2f} MB")
        
        # Aggressive quantization
        if 'quantization' in self.constraints.allowed_techniques:
            # Check if INT4 is feasible
            size_pressure = constraint_analysis.get('size_pressure', 1.0)
            
            if size_pressure > 3.0:
                # Very tight constraint - use INT4
                techniques.append({
                    'name': 'quantization',
                    'type': 'static',
                    'precision': 'int4',
                    'priority': 1
                })
                reasoning.append("INT4 quantization for ~8× size reduction (aggressive)")
            else:
                # Use INT8
                techniques.append({
                    'name': 'quantization',
                    'type': 'static',
                    'precision': 'int8',
                    'per_channel': True,
                    'priority': 1
                })
                reasoning.append("INT8 quantization for ~4× size reduction")
        
        # Aggressive pruning
        if 'pruning' in self.constraints.allowed_techniques:
            sparsity = min(0.7, constraint_analysis.get('size_pressure', 1.0) * 0.3)
            techniques.append({
                'name': 'pruning',
                'type': 'magnitude',
                'scope': 'global',
                'sparsity': sparsity,
                'priority': 2
            })
            reasoning.append(f"Magnitude pruning ({sparsity*100:.0f}%) for additional compression")
        
        # Knowledge distillation if available
        if 'distillation' in self.constraints.allowed_techniques:
            if constraint_analysis['model_params_m'] > 10:
                techniques.append({
                    'name': 'distillation',
                    'type': 'standard',
                    'temperature': 3.0,
                    'priority': 3
                })
                reasoning.append("Knowledge distillation to train smaller student model")
        
        return OptimizationStrategy(
            strategy_name="Size-Optimized",
            description="Aggressive compression with quantization and pruning",
            techniques=techniques,
            expected_compression=4.0,
            expected_speedup=1.5,
            expected_accuracy_impact=-0.05,  # 5% drop
            risk_level='high',
            estimated_time_minutes=25,
            reasoning=reasoning
        )
    
    def _strategy_for_accuracy(
        self,
        constraint_analysis: Dict[str, Any],
        model_analysis: Dict[str, Any],
        bottleneck: str
    ) -> OptimizationStrategy:
        """Generate strategy optimized for accuracy preservation"""
        techniques = []
        reasoning = []
        
        reasoning.append("Goal: Preserve accuracy while optimizing")
        
        if self.constraints.performance.min_accuracy:
            reasoning.append(f"Minimum accuracy: {self.constraints.performance.min_accuracy*100:.1f}%")
        
        # Conservative quantization
        if 'quantization' in self.constraints.allowed_techniques:
            techniques.append({
                'name': 'quantization',
                'type': 'qat',  # Quantization-aware training
                'precision': 'int8',
                'per_channel': True,
                'priority': 1
            })
            reasoning.append("QAT (Quantization-Aware Training) for minimal accuracy loss")
        
        # Light pruning
        if 'pruning' in self.constraints.allowed_techniques:
            techniques.append({
                'name': 'pruning',
                'type': 'magnitude',
                'scope': 'local',
                'sparsity': 0.2,
                'priority': 2
            })
            reasoning.append("Conservative pruning (20%) with per-layer thresholds")
        
        # Graph optimization (safe)
        techniques.append({
            'name': 'graph_optimization',
            'type': 'operator_fusion',
            'priority': 3
        })
        reasoning.append("Safe graph optimizations (operator fusion)")
        
        return OptimizationStrategy(
            strategy_name="Accuracy-Preserving",
            description="Conservative optimization prioritizing accuracy",
            techniques=techniques,
            expected_compression=2.0,
            expected_speedup=1.3,
            expected_accuracy_impact=-0.01,  # 1% drop
            risk_level='low',
            estimated_time_minutes=30,
            reasoning=reasoning
        )
    
    def _strategy_balanced(
        self,
        constraint_analysis: Dict[str, Any],
        model_analysis: Dict[str, Any],
        bottleneck: str
    ) -> OptimizationStrategy:
        """Generate balanced optimization strategy"""
        techniques = []
        reasoning = []
        
        reasoning.append("Goal: Balanced optimization across all objectives")
        reasoning.append(f"Primary bottleneck: {bottleneck}")
        
        # Quantization (moderate)
        if 'quantization' in self.constraints.allowed_techniques:
            techniques.append({
                'name': 'quantization',
                'type': 'static',
                'precision': 'int8',
                'per_channel': True,
                'priority': 1
            })
            reasoning.append("INT8 static quantization for good speed/size/accuracy balance")
        
        # Moderate pruning
        if 'pruning' in self.constraints.allowed_techniques:
            # Adjust sparsity based on bottleneck
            if bottleneck == 'size':
                sparsity = 0.5
                reasoning.append("Moderate pruning (50%) due to size constraints")
            elif bottleneck == 'compute':
                sparsity = 0.4
                reasoning.append("Moderate pruning (40%) due to compute constraints")
            else:
                sparsity = 0.3
                reasoning.append("Light pruning (30%) for balanced optimization")
            
            techniques.append({
                'name': 'pruning',
                'type': 'magnitude',
                'scope': 'global',
                'sparsity': sparsity,
                'priority': 2
            })
        
        # Graph optimization
        techniques.append({
            'name': 'graph_optimization',
            'type': 'operator_fusion',
            'priority': 3
        })
        reasoning.append("Operator fusion for efficiency")
        
        return OptimizationStrategy(
            strategy_name="Balanced",
            description="Balanced optimization strategy for typical use cases",
            techniques=techniques,
            expected_compression=3.0,
            expected_speedup=1.8,
            expected_accuracy_impact=-0.03,  # 3% drop
            risk_level='medium',
            estimated_time_minutes=20,
            reasoning=reasoning
        )
    
    def _print_strategy(self, strategy: OptimizationStrategy):
        """Print strategy summary"""
        print("\n" + "=" * 80)
        print(f"OPTIMIZATION STRATEGY: {strategy.strategy_name}")
        print("=" * 80)
        print(f"\n{strategy.description}")
        print(f"\nRisk Level: {strategy.risk_level.upper()}")
        print(f"Estimated Time: {strategy.estimated_time_minutes:.0f} minutes")
        
        print(f"\n📊 Expected Results:")
        print(f"  - Compression: {strategy.expected_compression:.1f}×")
        print(f"  - Speedup: {strategy.expected_speedup:.1f}×")
        print(f"  - Accuracy Impact: {strategy.expected_accuracy_impact*100:+.1f}%")
        
        print(f"\n🔧 Techniques (in order of priority):")
        for i, tech in enumerate(strategy.techniques, 1):
            print(f"  {i}. {tech['name'].upper()}")
            for key, value in tech.items():
                if key not in ['name', 'priority']:
                    print(f"     - {key}: {value}")
        
        print(f"\n💡 Reasoning:")
        for reason in strategy.reasoning:
            print(f"  • {reason}")
        
        print("\n" + "=" * 80 + "\n")
    
    @staticmethod
    def load_from_config(config_path: Path) -> 'ConstraintSolver':
        """Load constraints from configuration file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Parse hardware constraints
        hw_config = config.get('hardware', {})
        hardware = HardwareConstraints(**hw_config)
        
        # Parse performance constraints
        perf_config = config.get('performance', {})
        performance = PerformanceConstraints(**perf_config)
        
        # Parse optimization constraints
        opt_config = config.get('optimization', {})
        constraints = OptimizationConstraints(
            hardware=hardware,
            performance=performance,
            optimization_goal=opt_config.get('goal', 'balanced'),
            allowed_techniques=opt_config.get('techniques', ['quantization', 'pruning']),
            priority=opt_config.get('priority', 'high')
        )
        
        return ConstraintSolver(constraints)


# Convenience functions
def solve_constraints(
    model_analysis: Dict[str, Any],
    constraints: OptimizationConstraints
) -> OptimizationStrategy:
    """Solve constraints and get strategy (convenience wrapper)"""
    solver = ConstraintSolver(constraints)
    return solver.solve(model_analysis)
