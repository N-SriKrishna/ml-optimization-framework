"""
Comprehensive Combination Exploration Example
Test all possible combinations of optimization techniques
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.converters.onnx_converter import convert_to_onnx
from src.analyzers.combination_explorer import CombinationExplorer, TechniqueConfig
from src.utils.logger import logger


def main():
    """Run comprehensive combination exploration"""
    
    print("\n" + "=" * 80)
    print("OPTIMIZATION TECHNIQUES COMBINATION EXPLORATION")
    print("Systematically test all combinations to find optimal configuration")
    print("=" * 80)
    
    # Check if model exists
    model_path = Path('models/yolov8n.onnx')
    if not model_path.exists():
        logger.error("Model not found. Please run:")
        logger.info("  python examples/real_world/01_download_yolov8.py")
        logger.info("  python examples/real_world/02_optimize_yolov8.py")
        return
    
    # Define exploration space
    technique_configs = {
        'quantization': TechniqueConfig(
            name='quantization',
            variants=[
                {'type': 'none'},
                {'type': 'dynamic', 'precision': 'int8'},
                {'type': 'static', 'precision': 'int8'},
                {'type': 'dynamic', 'precision': 'fp16'},
            ]
        ),
        'pruning': TechniqueConfig(
            name='pruning',
            variants=[
                {'type': 'none'},
                {'type': 'magnitude', 'sparsity': 0.3},
                {'type': 'magnitude', 'sparsity': 0.5},
                {'type': 'structured', 'sparsity': 0.3},
            ]
        ),
        'order': TechniqueConfig(
            name='order',
            variants=[
                {'sequence': 'quantize_first'},
                {'sequence': 'prune_first'},
            ]
        )
    }
    
    # Total combinations: 4 quant × 4 prune × 2 orders = 32 combinations
    
    # Create explorer
    output_dir = Path('outputs/combination_exploration')
    explorer = CombinationExplorer(output_dir)
    
    # Run exploration
    results = explorer.explore_all_combinations(
        model_path=model_path,
        technique_configs=technique_configs,
        max_combinations=None  # Test all
    )
    
    # Results are automatically analyzed and saved
    print("\n" + "=" * 80)
    print("✓ Exploration complete!")
    print(f"  Results saved to: {output_dir}")
    print(f"  Total combinations tested: {len(results)}")
    print(f"  Pareto-optimal solutions: {sum(1 for r in results if r.pareto_optimal)}")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
