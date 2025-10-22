# ML Model Optimization Framework

A comprehensive, ONNX-centric framework for optimizing deep learning models with automatic constraint solving, multi-objective optimization, and Pareto analysis.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-14%2F14%20passing-brightgreen.svg)

## 🎯 Features

- **Universal Model Support**: Convert PyTorch, TensorFlow, TFLite, and ONNX models
- **Intelligent Optimization**: Automatic constraint-based strategy generation
- **Multiple Techniques**: Quantization (INT8, INT4, FP16), Pruning (Magnitude, Structured), Knowledge Distillation
- **Multi-Objective Analysis**: Pareto front optimization for accuracy vs. latency vs. size
- **Rich Visualizations**: 7+ chart types for trade-off analysis
- **Production-Ready**: Tested on real models (YOLOv8, etc.)

## 📊 Quick Example

Optimize YOLOv8 in under 10 seconds:

from src.converters.onnx_converter import convert_to_onnx
from src.analyzers.onnx_analyzer import ONNXAnalyzer
from src.solvers.constraint_solver import ConstraintSolver, OptimizationConstraints

Convert to ONNX
convert_to_onnx('yolov8n.pt', 'yolov8n.onnx', input_shape=(1, 3, 640, 640))

Analyze
analyzer = ONNXAnalyzer('yolov8n.onnx')
analysis = analyzer.analyze()

Optimize
constraints = OptimizationConstraints(optimization_goal='balanced')
solver = ConstraintSolver(constraints)
strategy = solver.solve(analysis)

**Result**: 3.67× compression, 2.18× speedup, with full Pareto analysis!

## 🚀 Installation

Clone repository
git clone https://github.com/N-SriKrishna/ml-optimization-framework.git
cd ml-optimization-framework

Create virtual environment
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt


## 📦 Dependencies

pip install onnx onnxruntime onnx-simplifier
pip install torch torchvision tensorflow tf2onnx
pip install numpy pandas matplotlib seaborn plotly
pip install pyyaml tqdm colorama tabulate pytest


## 🎓 Usage

### **1. Simple Quantization**

from src.optimizers.quantizer import quantize_dynamic_int8

quantize_dynamic_int8('model.onnx', 'model_int8.onnx')

Result: 4× smaller, 2× faster


### **2. Pruning**

from src.optimizers.pruner import prune_magnitude_global

prune_magnitude_global('model.onnx', 'model_pruned.onnx', sparsity=0.5)

Result: 50% parameters removed


### **3. Complete Optimization Pipeline**

from src.solvers.variant_generator import VariantGenerator

generator = VariantGenerator(output_dir='outputs')
variants = generator.generate_variants(
model_path='model.onnx',
strategy=strategy,
num_variants=5
)

Result: 5 optimized models with different trade-offs


### **4. Pareto Analysis**

from src.evaluators.pareto_analyzer import ParetoAnalyzer

analyzer = ParetoAnalyzer(objectives=['accuracy', 'latency', 'size'])
analysis = analyzer.analyze(variants)

Result: Pareto-optimal models identified


## 📈 Real-World Example: YOLOv8

Download and optimize YOLOv8
python examples/real_world/01_download_yolov8.py
python examples/real_world/02_optimize_yolov8.py


**Results:**
- Original: 12.26 MB
- Optimized: 3.34 MB (3.67× compression)
- Speedup: 2.18× faster
- Generated: 5 variants, 7 visualizations, HTML report

See `examples/real_world/` for complete code.

## 🧪 Testing

Run all tests
PYTHONPATH=. pytest tests/ -v

Run specific test
PYTHONPATH=. pytest tests/test_quantization.py -v


**Test Coverage**: 14/14 tests passing ✅

## 📁 Project Structure

ml-optimization-framework/
├── src/
│ ├── converters/ # Format conversion (PyTorch, TF → ONNX)
│ ├── analyzers/ # Model analysis (FLOPs, memory, etc.)
│ ├── optimizers/ # Quantization, pruning, distillation
│ ├── solvers/ # Constraint solver, variant generator
│ ├── evaluators/ # Pareto analysis, visualization
│ └── utils/ # Logging, helpers
├── tests/ # Unit & integration tests
├── examples/ # Usage examples
│ ├── real_world/ # YOLOv8 optimization example
│ └── ...
├── configs/ # Configuration templates
├── requirements.txt # Dependencies
└── README.md # This file


## 🎯 Supported Models

- **Object Detection**: YOLO (v5, v8, v11), SSD, EfficientDet, Faster R-CNN
- **Segmentation**: PP-LiteSeg, U-Net, DeepLab
- **Classification**: ResNet, EfficientNet, MobileNet, VGG
- **Custom Models**: Any PyTorch or TensorFlow model

## 🔧 Optimization Techniques

### **Quantization**
- INT8 (Dynamic & Static)
- INT4 (for aggressive compression)
- FP16 (GPU acceleration)
- Mixed-precision
- QAT (Quantization-Aware Training)

### **Pruning**
- Magnitude-based (global & local)
- Structured (filter/channel pruning)
- Unstructured sparsity
- Iterative pruning with fine-tuning

### **Knowledge Distillation**
- Standard soft-label distillation
- Self-distillation
- Progressive distillation
- Multi-teacher distillation

### **Graph Optimization**
- Operator fusion
- Constant folding
- Dead code elimination
- Memory optimization

## 📊 Output Formats

- **ONNX** (.onnx) - Cross-platform
- **TensorFlow Lite** (.tflite) - Android/mobile
- **CoreML** (.mlmodel) - iOS/macOS
- **TensorRT** (.plan) - NVIDIA GPUs
- **OpenVINO** (.xml + .bin) - Intel hardware

## 🌟 Key Features

### **1. Constraint-Based Optimization**
Define your deployment constraints, and the framework automatically finds the best optimization strategy:

constraints = OptimizationConstraints(
hardware=HardwareConstraints(device_name="Snapdragon 680"),
performance=PerformanceConstraints(
max_latency_ms=300,
max_model_size_mb=20,
min_accuracy=0.90
)
)


### **2. Multi-Objective Pareto Analysis**
Identify optimal trade-offs between accuracy, latency, and model size:

pareto_analyzer = ParetoAnalyzer(
objectives=['accuracy', 'latency', 'size'],
minimize=['latency', 'size'],
maximize=['accuracy']
)


### **3. Rich Visualizations**
Automatically generate 7+ chart types:
- 2D Pareto fronts (Accuracy vs Latency, Accuracy vs Size)
- 3D Pareto visualization
- Radar charts for multi-variant comparison
- Trade-off heatmaps
- Statistical comparisons

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ONNX and ONNX Runtime teams
- PyTorch and TensorFlow communities
- Ultralytics for YOLOv8

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

## 🔗 Citation

If you use this framework in your research, please cite:

@software{ml_optimization_framework,
title={ML Model Optimization Framework: ONNX-Centric Universal Pipeline},
author={N-SriKrishna},
year={2025},
url={https://github.com/N-SriKrishna/ml-optimization-framework}
}


## 🚀 Roadmap

- [ ] Web-based GUI for interactive optimization
- [ ] Support for more quantization methods (GPTQ, AWQ)
- [ ] Automatic hyperparameter tuning
- [ ] Model architecture search (NAS)
- [ ] Deployment scripts for cloud platforms
- [ ] Pre-optimized model zoo

---

**Made with ❤️ for efficient AI deployment**
