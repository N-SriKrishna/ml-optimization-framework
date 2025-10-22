"""
Complete YOLOv8 Optimization Pipeline
Demonstrates the full framework capabilities
"""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.converters.onnx_converter import convert_to_onnx
from src.analyzers.onnx_analyzer import ONNXAnalyzer
from src.solvers.constraint_solver import (
    ConstraintSolver,
    OptimizationConstraints,
    HardwareConstraints,
    PerformanceConstraints
)
from src.solvers.variant_generator import VariantGenerator
from src.evaluators.pareto_analyzer import ParetoAnalyzer
from src.evaluators.pareto_visualizer import ParetoVisualizer
from src.utils.logger import logger


def main():
    """Complete optimization pipeline"""
    
    print("\n" + "=" * 80)
    print("YOLOV8 OPTIMIZATION PIPELINE")
    print("Demonstrating ML Model Optimization Framework")
    print("=" * 80)
    
    # Configuration
    model_path = Path('models/yolov8n.pt')
    output_base = Path('outputs/yolov8_optimization')
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Check if model exists
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Please run: python examples/real_world/01_download_yolov8.py")
        return
    
    start_time = time.time()
    
    # ============================================================================
    # STEP 1: CONVERT TO ONNX
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: CONVERTING YOLOV8 TO ONNX")
    logger.info("=" * 80)
    
    onnx_path = output_base / 'yolov8n.onnx'
    
    try:
        convert_to_onnx(
            model_path,
            onnx_path,
            input_shape=(1, 3, 640, 640),
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch'},
                'output': {0: 'batch'}
            }
        )
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        logger.info("\nTrying alternative conversion method...")
        
        # Alternative: Export using ultralytics
        try:
            from ultralytics import YOLO
            model = YOLO(str(model_path))
            model.export(format='onnx', simplify=True)
            
            # Move exported file
            exported_onnx = Path('yolov8n.onnx')
            if exported_onnx.exists():
                import shutil
                shutil.move(exported_onnx, onnx_path)
                logger.info(f"✓ Exported using ultralytics: {onnx_path}")
        except Exception as e2:
            logger.error(f"Alternative method also failed: {e2}")
            return
    
    # ============================================================================
    # STEP 2: ANALYZE MODEL
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: ANALYZING MODEL")
    logger.info("=" * 80)
    
    analyzer = ONNXAnalyzer(onnx_path)
    analysis = analyzer.analyze()
    analyzer.print_summary(analysis)
    
    # Save analysis
    from src.utils.helpers import save_json
    save_json(
        {k: v for k, v in analysis.items() if k != 'layer_details'},  # Exclude large data
        output_base / 'model_analysis.json'
    )
    
    # ============================================================================
    # STEP 3: DEFINE CONSTRAINTS (Mobile Deployment Scenario)
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: DEFINING OPTIMIZATION CONSTRAINTS")
    logger.info("=" * 80)
    
    # Scenario: Deploy on mid-range Android phone (Snapdragon 680)
    hardware = HardwareConstraints(
        device_name="Snapdragon 680",
        cpu_cores=8,
        cpu_frequency_ghz=2.4,
        ram_total_gb=4.0,
        ram_available_gb=2.5,
        has_gpu=False,
        has_npu=False,
        storage_available_gb=1.0,
        thermal_class="moderate",
        power_budget="mobile"
    )
    
    performance = PerformanceConstraints(
        max_latency_ms=300,          # Real-time requirement
        max_model_size_mb=15,        # Storage constraint
        min_accuracy=0.85,           # Acceptable accuracy
        max_accuracy_drop=0.10       # Max 10% drop
    )
    
    constraints = OptimizationConstraints(
        hardware=hardware,
        performance=performance,
        optimization_goal='balanced',
        allowed_techniques=['quantization', 'pruning'],
        priority='high'
    )
    
    logger.info(f"\n📱 Target Device: {hardware.device_name}")
    logger.info(f"⏱️  Max Latency: {performance.max_latency_ms}ms")
    logger.info(f"💾 Max Size: {performance.max_model_size_mb}MB")
    logger.info(f"🎯 Min Accuracy: {performance.min_accuracy*100:.0f}%")
    
    # ============================================================================
    # STEP 4: SOLVE CONSTRAINTS & GENERATE STRATEGY
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: SOLVING CONSTRAINTS")
    logger.info("=" * 80)
    
    solver = ConstraintSolver(constraints)
    strategy = solver.solve(analysis)
    
    # ============================================================================
    # STEP 5: GENERATE OPTIMIZED VARIANTS
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: GENERATING OPTIMIZED VARIANTS")
    logger.info("=" * 80)
    
    variants_dir = output_base / 'variants'
    generator = VariantGenerator(variants_dir)
    
    variants = generator.generate_variants(
        onnx_path,
        strategy,
        num_variants=5  # Generate 5 variants
    )
    
    generator.print_variants_summary(variants)
    
    # ============================================================================
    # STEP 6: PARETO ANALYSIS
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 6: PARETO ANALYSIS")
    logger.info("=" * 80)
    
    pareto_analyzer = ParetoAnalyzer(
        objectives=['accuracy', 'latency', 'size'],
        minimize=['latency', 'size'],
        maximize=['accuracy']
    )
    
    pareto_analysis = pareto_analyzer.analyze(variants)
    pareto_analyzer.print_analysis(pareto_analysis)
    
    # Save analysis
    pareto_analyzer.save_analysis(
        pareto_analysis,
        output_base / 'pareto_analysis.json'
    )
    
    # ============================================================================
    # STEP 7: GENERATE VISUALIZATIONS
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 7: GENERATING VISUALIZATIONS")
    logger.info("=" * 80)
    
    viz_dir = output_base / 'visualizations'
    visualizer = ParetoVisualizer()
    visualizer.visualize_all(pareto_analysis, viz_dir, show_plots=False)
    
    # ============================================================================
    # STEP 8: GENERATE COMPARISON REPORT
    # ============================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 8: GENERATING COMPARISON REPORT")
    logger.info("=" * 80)
    
    generate_html_report(
        analysis,
        strategy,
        variants,
        pareto_analysis,
        output_base / 'optimization_report.html'
    )
    
    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    elapsed_time = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("✓ OPTIMIZATION COMPLETE!")
    print("=" * 80)
    print(f"\n⏱️  Total Time: {elapsed_time:.1f} seconds")
    print(f"\n📁 Results saved to: {output_base}")
    print(f"\n📊 Generated Files:")
    print(f"  • Original ONNX: {onnx_path.name}")
    print(f"  • Optimized Variants: {len(variants)} models")
    print(f"  • Pareto Analysis: pareto_analysis.json")
    print(f"  • Visualizations: {len(list(viz_dir.glob('*.png')))} plots")
    print(f"  • HTML Report: optimization_report.html")
    
    print(f"\n🏆 Best Variant (Pareto Optimal):")
    best_variant = pareto_analysis['pareto_front'][0].variant
    print(f"  Name: {best_variant.variant_name}")
    print(f"  Size: {best_variant.model_size_mb:.2f} MB")
    print(f"  Compression: {best_variant.compression_ratio:.2f}×")
    print(f"  Est. Speedup: {best_variant.estimated_speedup:.2f}×")
    print(f"  Est. Accuracy Impact: {best_variant.estimated_accuracy_impact*100:+.1f}%")
    
    print("\n" + "=" * 80 + "\n")


def generate_html_report(analysis, strategy, variants, pareto_analysis, output_path):
    """Generate HTML comparison report"""
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLOv8 Optimization Report</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 40px;
                background: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                margin-top: 30px;
            }}
            .metric {{
                display: inline-block;
                margin: 10px 20px 10px 0;
                padding: 15px 25px;
                background: #ecf0f1;
                border-radius: 5px;
                border-left: 4px solid #3498db;
            }}
            .metric-label {{
                font-size: 12px;
                color: #7f8c8d;
                text-transform: uppercase;
            }}
            .metric-value {{
                font-size: 24px;
                font-weight: bold;
                color: #2c3e50;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background: #3498db;
                color: white;
            }}
            tr:hover {{
                background: #f5f5f5;
            }}
            .pareto-optimal {{
                background: #2ecc71;
                color: white;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 11px;
            }}
            .strategy {{
                background: #fff3cd;
                border-left: 4px solid #ffc107;
                padding: 15px;
                margin: 20px 0;
            }}
            img {{
                max-width: 100%;
                height: auto;
                margin: 20px 0;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 YOLOv8 Optimization Report</h1>
            <p>Generated by ML Model Optimization Framework</p>
            
            <h2>📊 Original Model Analysis</h2>
            <div class="metric">
                <div class="metric-label">Model Size</div>
                <div class="metric-value">{analysis['model_info']['model_size_mb']:.2f} MB</div>
            </div>
            <div class="metric">
                <div class="metric-label">Parameters</div>
                <div class="metric-value">{analysis['parameters']['total_parameters_millions']:.2f}M</div>
            </div>
            <div class="metric">
                <div class="metric-label">FLOPs</div>
                <div class="metric-value">{analysis['computation']['total_gflops']:.2f} G</div>
            </div>
            
            <h2>🔧 Optimization Strategy</h2>
            <div class="strategy">
                <h3>{strategy.strategy_name}</h3>
                <p>{strategy.description}</p>
                <ul>
    """
    
    for reason in strategy.reasoning:
        html += f"                    <li>{reason}</li>\n"
    
    html += f"""
                </ul>
            </div>
            
            <h2>📦 Generated Variants</h2>
            <table>
                <tr>
                    <th>Variant</th>
                    <th>Size (MB)</th>
                    <th>Compression</th>
                    <th>Est. Speedup</th>
                    <th>Est. Accuracy Impact</th>
                    <th>Pareto Status</th>
                </tr>
    """
    
    pareto_variants = {p.variant.variant_id for p in pareto_analysis['pareto_front']}
    
    for variant in variants:
        is_pareto = variant.variant_id in pareto_variants
        pareto_badge = '<span class="pareto-optimal">PARETO OPTIMAL</span>' if is_pareto else ''
        
        html += f"""
                <tr>
                    <td><strong>{variant.variant_name}</strong></td>
                    <td>{variant.model_size_mb:.2f}</td>
                    <td>{variant.compression_ratio:.2f}×</td>
                    <td>{variant.estimated_speedup:.2f}×</td>
                    <td>{variant.estimated_accuracy_impact*100:+.1f}%</td>
                    <td>{pareto_badge}</td>
                </tr>
        """
    
    html += """
            </table>
            
            <h2>📈 Pareto Front Visualization</h2>
            <img src="visualizations/pareto_accuracy_vs_latency.png" alt="Accuracy vs Latency">
            <img src="visualizations/pareto_accuracy_vs_size.png" alt="Accuracy vs Size">
            <img src="visualizations/pareto_radar.png" alt="Radar Comparison">
            
            <h2>🏆 Recommendation</h2>
            <p>Based on Pareto analysis, the following variant(s) provide the best trade-offs:</p>
            <ul>
    """
    
    for point in pareto_analysis['pareto_front'][:3]:
        html += f"""
                <li><strong>{point.variant.variant_name}</strong>: 
                    {point.variant.model_size_mb:.2f} MB, 
                    {point.variant.compression_ratio:.2f}× compression, 
                    {point.variant.estimated_speedup:.2f}× speedup
                </li>
        """
    
    html += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)
    
    logger.info(f"✓ HTML report generated: {output_path}")


if __name__ == '__main__':
    main()
