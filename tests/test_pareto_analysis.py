"""
Test Pareto analysis and visualization
"""
import pytest
from pathlib import Path
import torch
import torch.nn as nn
import tempfile

from src.converters.onnx_converter import convert_to_onnx
from src.analyzers.onnx_analyzer import ONNXAnalyzer
from src.solvers.constraint_solver import (
    ConstraintSolver,
    OptimizationConstraints
)
from src.solvers.variant_generator import VariantGenerator
from src.evaluators.pareto_analyzer import ParetoAnalyzer
from src.evaluators.pareto_visualizer import ParetoVisualizer


class TestModel(nn.Module):
    """Test model"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_pareto_analysis():
    """Test complete Pareto analysis workflow"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and convert model
        model = TestModel()
        pt_path = tmpdir / 'model.pt'
        torch.save(model, pt_path)
        
        onnx_path = tmpdir / 'model.onnx'
        convert_to_onnx(pt_path, onnx_path, input_shape=(1, 3, 224, 224))
        
        # Analyze model
        analyzer = ONNXAnalyzer(onnx_path)
        analysis = analyzer.analyze()
        
        # Generate strategy
        constraints = OptimizationConstraints(optimization_goal='balanced')
        solver = ConstraintSolver(constraints)
        strategy = solver.solve(analysis)
        
        # Generate variants
        output_dir = tmpdir / 'variants'
        generator = VariantGenerator(output_dir)
        variants = generator.generate_variants(onnx_path, strategy, num_variants=5)
        
        # Pareto analysis
        pareto_analyzer = ParetoAnalyzer(
            objectives=['accuracy', 'latency', 'size'],
            minimize=['latency', 'size'],
            maximize=['accuracy']
        )
        
        pareto_analysis = pareto_analyzer.analyze(variants)
        
        # Verify results
        assert pareto_analysis is not None
        assert 'pareto_front' in pareto_analysis
        assert len(pareto_analysis['pareto_front']) > 0
        assert pareto_analysis['num_pareto_optimal'] <= len(variants)
        
        # Print analysis
        pareto_analyzer.print_analysis(pareto_analysis)
        
        # Save analysis
        analysis_path = output_dir / 'pareto_analysis.json'
        pareto_analyzer.save_analysis(pareto_analysis, analysis_path)
        assert analysis_path.exists()
        
        print(f"✓ Pareto analysis test passed")
        print(f"  Variants: {len(variants)}")
        print(f"  Pareto optimal: {pareto_analysis['num_pareto_optimal']}")


def test_pareto_visualization():
    """Test Pareto visualization"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create and convert model
        model = TestModel()
        pt_path = tmpdir / 'model.pt'
        torch.save(model, pt_path)
        
        onnx_path = tmpdir / 'model.onnx'
        convert_to_onnx(pt_path, onnx_path, input_shape=(1, 3, 224, 224))
        
        # Generate variants
        analyzer = ONNXAnalyzer(onnx_path)
        analysis = analyzer.analyze()
        
        constraints = OptimizationConstraints(optimization_goal='balanced')
        solver = ConstraintSolver(constraints)
        strategy = solver.solve(analysis)
        
        output_dir = tmpdir / 'variants'
        generator = VariantGenerator(output_dir)
        variants = generator.generate_variants(onnx_path, strategy, num_variants=5)
        
        # Pareto analysis
        pareto_analyzer = ParetoAnalyzer()
        pareto_analysis = pareto_analyzer.analyze(variants)
        
        # Visualize
        viz_dir = tmpdir / 'visualizations'
        visualizer = ParetoVisualizer()
        visualizer.visualize_all(pareto_analysis, viz_dir, show_plots=False)
        
        # Verify plots were created
        expected_plots = [
            'pareto_accuracy_vs_latency.png',
            'pareto_accuracy_vs_size.png',
            'pareto_latency_vs_size.png',
            'pareto_3d.png',
            'pareto_radar.png',
            'statistics_comparison.png',
            'trade_off_heatmap.png'
        ]
        
        for plot_name in expected_plots:
            plot_path = viz_dir / plot_name
            assert plot_path.exists(), f"Plot {plot_name} not created"
        
        print(f"✓ Pareto visualization test passed")
        print(f"  Generated {len(expected_plots)} plots")


if __name__ == '__main__':
    print("Running Pareto analysis tests...\n")
    test_pareto_analysis()
    print()
    test_pareto_visualization()
    print("\n✓ All Pareto tests passed!")
