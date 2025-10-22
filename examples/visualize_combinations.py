from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluators.combination_visualizer import CombinationVisualizer

csv_path = Path('outputs/combination_exploration/combination_exploration_results.csv')
output_dir = Path('outputs/combination_exploration/visualizations')

visualizer = CombinationVisualizer()
visualizer.create_dashboard(csv_path, output_dir)

print(f"\n✓ Visualizations created in: {output_dir}")
