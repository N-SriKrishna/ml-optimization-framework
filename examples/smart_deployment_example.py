"""
Example: Smart Hardware-Aware Deployment
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.converters.smart_deployment_exporter import SmartDeploymentExporter
from src.solvers.constraint_solver import HardwareConstraints


def main():
    # Example 1: Snapdragon Device
    print("\n" + "=" * 80)
    print("Example 1: Snapdragon 8 Gen 2 Deployment")
    print("=" * 80)
    
    snapdragon_hw = HardwareConstraints(
        device_name="Snapdragon 8 Gen 2",
        cpu_cores=8,
        cpu_frequency_ghz=3.2,
        ram_available_gb=8,
        has_gpu=True,
        has_npu=True
    )
    
    exporter = SmartDeploymentExporter()
    exporter.create_deployment_package(
        onnx_model_path=Path('models/yolov8n.onnx'),
        hardware=snapdragon_hw,
        output_dir=Path('outputs/deployment_snapdragon')
    )
    
    # Example 2: Apple M2
    print("\n\n" + "=" * 80)
    print("Example 2: Apple M2 Deployment")
    print("=" * 80)
    
    apple_hw = HardwareConstraints(
        device_name="Apple M2",
        cpu_cores=8,
        ram_available_gb=16,
        has_gpu=True,
        has_npu=True
    )
    
    exporter.create_deployment_package(
        onnx_model_path=Path('models/yolov8n.onnx'),
        hardware=apple_hw,
        output_dir=Path('outputs/deployment_apple')
    )
    
    # Example 3: NVIDIA GPU
    print("\n\n" + "=" * 80)
    print("Example 3: NVIDIA RTX 4090 Deployment")
    print("=" * 80)
    
    nvidia_hw = HardwareConstraints(
        device_name="NVIDIA RTX 4090",
        cpu_cores=16,
        ram_available_gb=64,
        has_gpu=True,
        has_npu=False
    )
    
    exporter.create_deployment_package(
        onnx_model_path=Path('models/yolov8n.onnx'),
        hardware=nvidia_hw,
        output_dir=Path('outputs/deployment_nvidia')
    )


if __name__ == '__main__':
    main()
