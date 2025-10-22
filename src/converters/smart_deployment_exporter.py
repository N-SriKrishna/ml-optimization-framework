"""
Smart Deployment Exporter with Hardware-Aware Recommendations
Automatically recommends and exports to the best runtime for target hardware
"""
from pathlib import Path
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
import json

from src.utils.logger import logger
from src.solvers.constraint_solver import HardwareConstraints


@dataclass
class RuntimeRecommendation:
    """Recommended runtime for specific hardware"""
    name: str
    format: str
    priority: int  # 1 = best, 2 = fallback, 3 = compatible
    reason: str
    acceleration: str  # NPU, GPU, CPU, etc.
    install_cmd: str
    example_code: str
    performance_estimate: str


@dataclass
class DeploymentPlan:
    """Complete deployment plan with recommendations"""
    hardware: HardwareConstraints
    primary_runtime: RuntimeRecommendation
    fallback_runtimes: List[RuntimeRecommendation]
    export_format: str
    optimization_notes: List[str]


class HardwareRuntimeMatcher:
    """
    Match hardware to optimal runtime/framework
    
    Supports:
    - Qualcomm (Snapdragon): QNN (Qualcomm Neural Network SDK)
    - MediaTek: NeuroPilot SDK  
    - Apple: Core ML (Neural Engine)
    - NVIDIA: TensorRT
    - Intel: OpenVINO
    - ARM Mali GPU: ARM NN
    - Generic CPU: ONNX Runtime
    - Android NPU: NNAPI
    """
    
    def __init__(self):
        self._build_runtime_database()
    
    def _build_runtime_database(self):
        """Build comprehensive runtime database"""
        
        self.runtimes = {
            # Qualcomm (Snapdragon NPU/HTA)
            'qualcomm_qnn': RuntimeRecommendation(
                name='Qualcomm QNN',
                format='dlc',  # Deep Learning Container
                priority=1,
                reason='Native Qualcomm NPU (Hexagon) support with best performance',
                acceleration='NPU (Hexagon Tensor Accelerator)',
                install_cmd='# Install Qualcomm Neural Processing SDK\n'
                           '# Download from: https://developer.qualcomm.com/software/qualcomm-neural-processing-sdk',
                example_code='''
# Qualcomm QNN Example
from qti.aisw.dlc_utils import modeltools

# Load model
model = modeltools.load_dlc('model.dlc')

# Run inference on Hexagon NPU
output = model.execute(input_data, runtime='dsp')  # dsp = Hexagon
''',
                performance_estimate='2-5× faster than CPU, 40-60% power reduction'
            ),
            
            'qualcomm_snpe': RuntimeRecommendation(
                name='Qualcomm SNPE',
                format='dlc',
                priority=2,
                reason='Older Qualcomm SDK, still widely supported',
                acceleration='NPU/GPU/DSP',
                install_cmd='pip install snpe-wrapper',
                example_code='''
# SNPE Example
import snpe

network = snpe.load_network('model.dlc')
output = network.execute(input_data, runtime='dsp')
''',
                performance_estimate='1.5-3× faster than CPU'
            ),
            
            # MediaTek (APU)
            'mediatek_neuron': RuntimeRecommendation(
                name='MediaTek NeuroPilot',
                format='dla',
                priority=1,
                reason='Optimized for MediaTek APU (AI Processing Unit)',
                acceleration='APU',
                install_cmd='# Download NeuroPilot SDK from MediaTek',
                example_code='''
# MediaTek NeuroPilot Example
from neuron_sdk import NeuronNetwork

model = NeuronNetwork.load('model.dla')
output = model.run(input_data, device='apu')
''',
                performance_estimate='3-6× faster than CPU on MediaTek devices'
            ),
            
            # Apple (Neural Engine)
            'apple_coreml': RuntimeRecommendation(
                name='Core ML',
                format='mlpackage',
                priority=1,
                reason='Native Apple Neural Engine acceleration',
                acceleration='Neural Engine + GPU',
                install_cmd='pip install coremltools',
                example_code='''
# Core ML Example (Swift)
import CoreML

let model = try model_optimized(configuration: .init())
let output = try model.prediction(input: inputData)

// Automatically uses Neural Engine
''',
                performance_estimate='5-10× faster than CPU, best on A14+ or M1+'
            ),
            
            # NVIDIA (GPU)
            'nvidia_tensorrt': RuntimeRecommendation(
                name='TensorRT',
                format='plan',
                priority=1,
                reason='Optimized for NVIDIA GPUs with kernel fusion',
                acceleration='CUDA Tensor Cores',
                install_cmd='pip install tensorrt',
                example_code='''
# TensorRT Example
import tensorrt as trt

# Load engine
with open('model.plan', 'rb') as f:
    engine = runtime.deserialize_cuda_engine(f.read())

# Run inference
context = engine.create_execution_context()
context.execute_v2(bindings)
''',
                performance_estimate='10-100× faster than CPU on NVIDIA GPUs'
            ),
            
            # Intel (VPU/iGPU)
            'intel_openvino': RuntimeRecommendation(
                name='OpenVINO',
                format='xml+bin',
                priority=1,
                reason='Optimized for Intel CPUs, GPUs, and VPUs',
                acceleration='CPU/GPU/VPU',
                install_cmd='pip install openvino',
                example_code='''
# OpenVINO Example
from openvino.runtime import Core

core = Core()
model = core.read_model('model.xml')
compiled = core.compile_model(model, 'CPU')  # or 'GPU'

output = compiled([input_data])
''',
                performance_estimate='3-8× faster than baseline CPU'
            ),
            
            # ARM Mali GPU
            'arm_nn': RuntimeRecommendation(
                name='ARM NN',
                format='tflite',
                priority=1,
                reason='Optimized for ARM Mali GPUs',
                acceleration='Mali GPU',
                install_cmd='# Install ARM NN from ARM Developer',
                example_code='''
# ARM NN Example
import pyarmnn as ann

parser = ann.ITfLiteParser()
network = parser.CreateNetworkFromBinaryFile('model.tflite')
runtime = ann.IRuntime()
optimized = ann.Optimize(network)
''',
                performance_estimate='2-4× faster than CPU on Mali GPUs'
            ),
            
            # Android NNAPI (Generic NPU)
            'android_nnapi': RuntimeRecommendation(
                name='Android NNAPI',
                format='tflite',
                priority=1,
                reason='Cross-vendor NPU support on Android 8.1+',
                acceleration='NPU/GPU/DSP (vendor-specific)',
                install_cmd='# Built into TensorFlow Lite',
                example_code='''
// Android NNAPI Example (Kotlin)
val options = Interpreter.Options()
    .setUseNNAPI(true)  // Enable NPU acceleration

val interpreter = Interpreter(modelFile, options)
interpreter.run(inputBuffer, outputBuffer)
''',
                performance_estimate='1.5-5× faster than CPU (varies by device)'
            ),
            
            # Generic ONNX Runtime
            'onnx_runtime': RuntimeRecommendation(
                name='ONNX Runtime',
                format='onnx',
                priority=2,
                reason='Universal runtime with multiple execution providers',
                acceleration='CPU/GPU (via providers)',
                install_cmd='pip install onnxruntime',
                example_code='''
# ONNX Runtime Example
import onnxruntime as ort

session = ort.InferenceSession('model.onnx')
output = session.run(None, {'input': input_data})
''',
                performance_estimate='1-2× faster than baseline with optimizations'
            ),
        }
    
    def recommend_runtime(self, hardware: HardwareConstraints) -> DeploymentPlan:
        """
        Recommend optimal runtime based on hardware
        
        Args:
            hardware: Hardware specifications
        
        Returns:
            Complete deployment plan
        """
        logger.info(f"Analyzing hardware: {hardware.device_name}")
        
        primary = None
        fallbacks = []
        optimization_notes = []
        
        # Detect hardware vendor and capabilities
        device_lower = hardware.device_name.lower()
        
        # Qualcomm Snapdragon
        if 'snapdragon' in device_lower or 'qualcomm' in device_lower:
            primary = self.runtimes['qualcomm_qnn']
            fallbacks = [
                self.runtimes['qualcomm_snpe'],
                self.runtimes['android_nnapi'],
                self.runtimes['onnx_runtime']
            ]
            optimization_notes = [
                'Quantize to INT8 for Hexagon NPU',
                'Enable DSP acceleration for best performance',
                'Batch size = 1 recommended for mobile',
                'Consider dynamic input shapes for flexibility'
            ]
            
            logger.info("✓ Detected: Qualcomm Snapdragon")
            logger.info(f"  Primary: {primary.name} (Hexagon NPU)")
        
        # MediaTek
        elif 'mediatek' in device_lower or 'dimensity' in device_lower or 'helio' in device_lower:
            primary = self.runtimes['mediatek_neuron']
            fallbacks = [
                self.runtimes['android_nnapi'],
                self.runtimes['arm_nn'],
                self.runtimes['onnx_runtime']
            ]
            optimization_notes = [
                'Quantize to INT8 for APU',
                'APU works best with CNN models',
                'Test NNAPI fallback for compatibility'
            ]
            
            logger.info("✓ Detected: MediaTek SoC")
            logger.info(f"  Primary: {primary.name} (APU)")
        
        # Apple Silicon
        elif 'apple' in device_lower or 'a14' in device_lower or 'a15' in device_lower or 'm1' in device_lower or 'm2' in device_lower:
            primary = self.runtimes['apple_coreml']
            fallbacks = [
                self.runtimes['onnx_runtime']
            ]
            optimization_notes = [
                'Use FP16 for Neural Engine',
                'Neural Engine requires specific ops',
                'A14+ bionic has best performance',
                'Automatic GPU/ANE scheduling'
            ]
            
            logger.info("✓ Detected: Apple Silicon")
            logger.info(f"  Primary: {primary.name} (Neural Engine)")
        
        # NVIDIA GPU
        elif 'nvidia' in device_lower or 'cuda' in device_lower or hardware.has_gpu:
            primary = self.runtimes['nvidia_tensorrt']
            fallbacks = [
                self.runtimes['onnx_runtime']
            ]
            optimization_notes = [
                'Use FP16 on Tensor Cores for 2× speedup',
                'Enable TF32 on Ampere+ GPUs',
                'Batch processing recommended',
                'Dynamic shapes supported'
            ]
            
            logger.info("✓ Detected: NVIDIA GPU")
            logger.info(f"  Primary: {primary.name} (CUDA)")
        
        # Intel
        elif 'intel' in device_lower or 'core i' in device_lower or 'xeon' in device_lower:
            primary = self.runtimes['intel_openvino']
            fallbacks = [
                self.runtimes['onnx_runtime']
            ]
            optimization_notes = [
                'Use INT8 quantization for VPU',
                'AVX-512 acceleration on Xeon',
                'iGPU acceleration on mobile',
                'Heterogeneous execution (CPU+GPU+VPU)'
            ]
            
            logger.info("✓ Detected: Intel Hardware")
            logger.info(f"  Primary: {primary.name} (CPU/GPU/VPU)")
        
        # ARM Mali
        elif hardware.has_gpu and ('mali' in device_lower or 'arm' in device_lower):
            primary = self.runtimes['arm_nn']
            fallbacks = [
                self.runtimes['android_nnapi'],
                self.runtimes['onnx_runtime']
            ]
            optimization_notes = [
                'Mali GPU prefers FP16',
                'Enable GPU delegate in TFLite',
                'Optimize for mobile power profile'
            ]
            
            logger.info("✓ Detected: ARM Mali GPU")
            logger.info(f"  Primary: {primary.name} (Mali)")
        
        # Android (generic with NPU)
        elif hardware.has_npu:
            primary = self.runtimes['android_nnapi']
            fallbacks = [
                self.runtimes['onnx_runtime']
            ]
            optimization_notes = [
                'NNAPI auto-selects best accelerator',
                'INT8 quantization recommended',
                'Test on target device (vendor-specific)',
                'Fallback to GPU/CPU if NPU unavailable'
            ]
            
            logger.info("✓ Detected: Generic Android NPU")
            logger.info(f"  Primary: {primary.name} (NPU)")
        
        # Fallback: Generic CPU
        else:
            primary = self.runtimes['onnx_runtime']
            fallbacks = []
            optimization_notes = [
                'CPU-only mode',
                'INT8 quantization for 4× speedup',
                'Consider multi-threading',
                'Enable intra-op parallelism'
            ]
            
            logger.info("✓ Generic hardware detected")
            logger.info(f"  Primary: {primary.name} (CPU)")
        
        # Create deployment plan
        plan = DeploymentPlan(
            hardware=hardware,
            primary_runtime=primary,
            fallback_runtimes=fallbacks,
            export_format=primary.format,
            optimization_notes=optimization_notes
        )
        
        return plan


class SmartDeploymentExporter:
    """
    Smart deployment exporter with hardware-aware recommendations
    """
    
    def __init__(self):
        self.matcher = HardwareRuntimeMatcher()
        logger.info("Initialized Smart Deployment Exporter")
    
    def create_deployment_package(
        self,
        onnx_model_path: Path,
        hardware: HardwareConstraints,
        output_dir: Path
    ):
        """
        Create intelligent deployment package
        
        Args:
            onnx_model_path: Optimized ONNX model
            hardware: Target hardware specifications
            output_dir: Output directory for package
        """
        logger.info("\n" + "=" * 80)
        logger.info("CREATING SMART DEPLOYMENT PACKAGE")
        logger.info("=" * 80)
        
        # Get recommendations
        plan = self.matcher.recommend_runtime(hardware)
        
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Print recommendations
        self._print_recommendations(plan)
        
        # Export model to recommended format
        self._export_to_format(onnx_model_path, plan, output_dir)
        
        # Generate deployment guide
        self._generate_deployment_guide(plan, output_dir)
        
        # Save deployment plan
        self._save_deployment_plan(plan, output_dir)
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ DEPLOYMENT PACKAGE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"  Location: {output_dir}")
        logger.info(f"  Primary runtime: {plan.primary_runtime.name}")
        logger.info(f"  Format: {plan.export_format}")
        logger.info(f"  See: {output_dir / 'DEPLOYMENT_GUIDE.md'}")
    
    def _print_recommendations(self, plan: DeploymentPlan):
        """Print deployment recommendations"""
        
        print("\n" + "=" * 80)
        print("DEPLOYMENT RECOMMENDATIONS")
        print("=" * 80)
        
        print(f"\n🎯 Target Hardware: {plan.hardware.device_name}")
        print(f"   CPU Cores: {plan.hardware.cpu_cores}")
        print(f"   RAM: {plan.hardware.ram_available_gb} GB")
        print(f"   GPU: {'Yes' if plan.hardware.has_gpu else 'No'}")
        print(f"   NPU: {'Yes' if plan.hardware.has_npu else 'No'}")
        
        print(f"\n⭐ PRIMARY RECOMMENDATION: {plan.primary_runtime.name}")
        print(f"   Format: {plan.primary_runtime.format}")
        print(f"   Acceleration: {plan.primary_runtime.acceleration}")
        print(f"   Reason: {plan.primary_runtime.reason}")
        print(f"   Performance: {plan.primary_runtime.performance_estimate}")
        
        if plan.fallback_runtimes:
            print(f"\n🔄 FALLBACK OPTIONS:")
            for i, runtime in enumerate(plan.fallback_runtimes, 1):
                print(f"   {i}. {runtime.name} ({runtime.format})")
                print(f"      - {runtime.reason}")
        
        print(f"\n💡 OPTIMIZATION NOTES:")
        for note in plan.optimization_notes:
            print(f"   • {note}")
        
        print("\n" + "=" * 80)
    
    def _export_to_format(self, onnx_path: Path, plan: DeploymentPlan, output_dir: Path):
        """Export model to recommended format"""
        
        format_name = plan.export_format
        logger.info(f"\nExporting to {format_name} format...")
        
        # Implementation would call appropriate converter
        # For now, copy ONNX and note conversion needed
        
        import shutil
        output_model = output_dir / f'model.{format_name}'
        
        if format_name == 'onnx':
            shutil.copy(onnx_path, output_model)
            logger.info(f"✓ Copied ONNX model to {output_model}")
        else:
            # Note: Would call actual converters here
            logger.info(f"ℹ  Model conversion to {format_name} requires:")
            logger.info(f"   {plan.primary_runtime.install_cmd}")
            
            # Copy ONNX as source
            shutil.copy(onnx_path, output_dir / 'model_source.onnx')
            logger.info(f"   Source ONNX saved for conversion")
    
    def _generate_deployment_guide(self, plan: DeploymentPlan, output_dir: Path):
        """Generate comprehensive deployment guide"""
        
        guide = f"""# Deployment Guide

## Hardware Configuration
- **Device**: {plan.hardware.device_name}
- **CPU**: {plan.hardware.cpu_cores} cores @ {plan.hardware.cpu_frequency_ghz} GHz
- **RAM**: {plan.hardware.ram_available_gb} GB
- **GPU**: {'Available' if plan.hardware.has_gpu else 'Not available'}
- **NPU**: {'Available' if plan.hardware.has_npu else 'Not available'}

## Recommended Runtime

### {plan.primary_runtime.name}
**Acceleration**: {plan.primary_runtime.acceleration}

**Why this runtime?**
{plan.primary_runtime.reason}

**Expected Performance**
{plan.primary_runtime.performance_estimate}

### Installation

{plan.primary_runtime.install_cmd}


### Example Code

{plan.primary_runtime.example_code}


## Optimization Notes

"""
        
        for note in plan.optimization_notes:
            guide += f"- {note}\n"
        
        guide += f"""

## Fallback Options

"""
        
        for i, runtime in enumerate(plan.fallback_runtimes, 1):
            guide += f"""
### {i}. {runtime.name}
- **Format**: {runtime.format}
- **Reason**: {runtime.reason}
- **Install**: `{runtime.install_cmd}`

"""
        
        guide += """
## Performance Optimization Checklist

- [ ] Model quantized to INT8/FP16
- [ ] Batch size optimized for hardware
- [ ] Input preprocessing optimized
- [ ] Runtime configured for target accelerator
- [ ] Benchmarked on actual device
- [ ] Power consumption profiled
- [ ] Thermal performance validated

## Support

For issues or questions:
1. Check hardware-specific documentation
2. Verify runtime version compatibility
3. Test on actual target device
4. Profile with vendor tools
"""
        
        guide_path = output_dir / 'DEPLOYMENT_GUIDE.md'
        with open(guide_path, 'w') as f:
            f.write(guide)
        
        logger.info(f"✓ Deployment guide created: {guide_path}")
    
    def _save_deployment_plan(self, plan: DeploymentPlan, output_dir: Path):
        """Save deployment plan as JSON"""
        
        plan_dict = {
            'hardware': {
                'device_name': plan.hardware.device_name,
                'cpu_cores': plan.hardware.cpu_cores,
                'ram_gb': plan.hardware.ram_available_gb,
                'has_gpu': plan.hardware.has_gpu,
                'has_npu': plan.hardware.has_npu
            },
            'primary_runtime': {
                'name': plan.primary_runtime.name,
                'format': plan.primary_runtime.format,
                'acceleration': plan.primary_runtime.acceleration,
                'performance_estimate': plan.primary_runtime.performance_estimate
            },
            'fallback_runtimes': [
                {
                    'name': r.name,
                    'format': r.format,
                    'acceleration': r.acceleration
                }
                for r in plan.fallback_runtimes
            ],
            'optimization_notes': plan.optimization_notes
        }
        
        plan_path = output_dir / 'deployment_plan.json'
        with open(plan_path, 'w') as f:
            json.dump(plan_dict, f, indent=2)
        
        logger.info(f"✓ Deployment plan saved: {plan_path}")
