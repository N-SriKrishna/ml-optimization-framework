# YOLOv8 Real-World Optimization Example

This example demonstrates the complete optimization pipeline on YOLOv8.

## Quick Start

### 1. Install Dependencies
pip install ultralytics


### 2. Download YOLOv8
python examples/real_world/01_download_yolov8.py


### 3. Run Complete Optimization
python examples/real_world/02_optimize_yolov8.py


## What This Does

1. **Converts** YOLOv8 to ONNX format
2. **Analyzes** model architecture, parameters, and FLOPs
3. **Defines** mobile deployment constraints
4. **Generates** 5 optimized variants (quantization + pruning)
5. **Performs** Pareto analysis
6. **Creates** visualizations and HTML report

## Expected Results

- **5 optimized models** with different compression levels
- **Pareto front** showing optimal trade-offs
- **7+ visualizations** (2D/3D Pareto plots, radar charts, etc.)
- **HTML comparison report** with all results

## Output Structure
outputs/yolov8_optimization/
├── yolov8n.onnx # Original ONNX model
├── model_analysis.json # Model analysis results
├── pareto_analysis.json # Pareto analysis results
├── optimization_report.html # Comprehensive HTML report
├── variants/ # Optimized model variants
│ ├── variant_01_.onnx
│ ├── variant_02_.onnx
│ ├── variant_03_.onnx
│ ├── variant_04_.onnx
│ ├── variant_05_*.onnx
│ └── variants_metadata.json
└── visualizations/ # All plots
├── pareto_accuracy_vs_latency.png
├── pareto_accuracy_vs_size.png
├── pareto_latency_vs_size.png
├── pareto_3d.png
├── pareto_radar.png
├── statistics_comparison.png
└── trade_off_heatmap.png


## Typical Results

### Original YOLOv8n
- Size: ~6 MB
- Parameters: ~3M
- FLOPs: ~8 GFLOPs

### After Optimization
- **Aggressive Variant**: 1.5 MB (4× compression), 2.5× speedup
- **Balanced Variant**: 3 MB (2× compression), 1.8× speedup
- **Conservative Variant**: 4.5 MB (1.3× compression), 1.3× speedup

## Time Required

- Download: ~10 seconds
- Complete optimization: ~2-3 minutes
- Total: **~3 minutes**
