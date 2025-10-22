# Documentation Images

This folder contains proof of framework capabilities through visualizations.

## Required Screenshots

### 1. Pareto Front Analysis
- `pareto_accuracy_vs_latency.png` - Shows trade-offs
- `pareto_accuracy_vs_size.png` - Size vs accuracy
- `pareto_3d.png` - 3D visualization

### 2. Combination Exploration
- `combo_compression_speedup.png` - All 36 combinations
- `combo_heatmap.png` - Performance heatmap
- `combo_pareto.png` - Pareto optimal solutions

### 3. Model Analysis
- `model_architecture.png` - Layer breakdown
- `flops_distribution.png` - Computation analysis

### 4. Deployment Recommendations
- `hardware_recommendations.png` - Smart deployment suggestions

## How to Generate

Run the complete pipeline:


python examples/real_world/02_optimize_yolov8.py
This generates all visualizations in `outputs/yolov8_optimization/visualizations/`

Then copy them here:
cp outputs/yolov8_optimization/visualizations/*.png docs/images/

## Image Guidelines

- **Format**: PNG (for transparency)
- **Resolution**: 1920×1080 or higher
- **File Size**: < 500KB per image (optimize with `pngquant` if needed)
- **Naming**: Descriptive names with underscores

## Current Images

- docs/images/combo_compression_speedup.png (192K)
- docs/images/combo_heatmap.png (239K)
- docs/images/combo_order_comparison.png (133K)
- docs/images/combo_pareto.png (171K)
- docs/images/combo_technique_impact.png (327K)
- docs/images/pareto_3d.png (451K)
- docs/images/pareto_accuracy_vs_latency.png (191K)
- docs/images/pareto_accuracy_vs_size.png (163K)
- docs/images/pareto_latency_vs_size.png (169K)
- docs/images/pareto_radar.png (460K)
- docs/images/statistics_comparison.png (132K)
- docs/images/trade_off_heatmap.png (141K)
