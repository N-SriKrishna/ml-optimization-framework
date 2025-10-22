"""
Pareto Visualization - Generate plots and charts for trade-off analysis
"""
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

from src.solvers.variant_generator import ModelVariant
from src.evaluators.pareto_analyzer import ParetoAnalyzer, ParetoPoint
from src.utils.logger import logger


class ParetoVisualizer:
    """
    Visualize Pareto analysis results with multiple chart types
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8')  # Fallback
        
        # Color palette
        self.colors = {
            'pareto': '#2ecc71',  # Green
            'dominated': '#e74c3c',  # Red
            'neutral': '#95a5a6',  # Gray
            'highlight': '#3498db',  # Blue
        }
        
        logger.info("Initialized Pareto visualizer")
    
    def visualize_all(
        self,
        analysis: Dict[str, Any],
        output_dir: Path,
        show_plots: bool = False
    ):
        """
        Generate all visualization types
        
        Args:
            analysis: Pareto analysis results
            output_dir: Directory to save plots
            show_plots: Whether to display plots interactively
        """
        logger.info("Generating Pareto visualizations...")
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 2D Pareto plots
        self.plot_2d_pareto(
            analysis,
            objectives=('accuracy', 'latency'),
            output_path=output_dir / 'pareto_accuracy_vs_latency.png',
            show=show_plots
        )
        
        self.plot_2d_pareto(
            analysis,
            objectives=('accuracy', 'size'),
            output_path=output_dir / 'pareto_accuracy_vs_size.png',
            show=show_plots
        )
        
        self.plot_2d_pareto(
            analysis,
            objectives=('latency', 'size'),
            output_path=output_dir / 'pareto_latency_vs_size.png',
            show=show_plots
        )
        
        # 3D Pareto plot
        if len(analysis['all_points'][0].objectives) >= 3:
            self.plot_3d_pareto(
                analysis,
                objectives=('accuracy', 'latency', 'size'),
                output_path=output_dir / 'pareto_3d.png',
                show=show_plots
            )
        
        # Trade-off radar chart
        self.plot_radar_chart(
            analysis,
            output_path=output_dir / 'pareto_radar.png',
            show=show_plots
        )
        
        # Statistics comparison
        self.plot_statistics_comparison(
            analysis,
            output_path=output_dir / 'statistics_comparison.png',
            show=show_plots
        )
        
        # Trade-off heatmap
        self.plot_trade_off_heatmap(
            analysis,
            output_path=output_dir / 'trade_off_heatmap.png',
            show=show_plots
        )
        
        logger.info(f"✓ Visualizations saved to {output_dir}")
    
    def plot_2d_pareto(
        self,
        analysis: Dict[str, Any],
        objectives: Tuple[str, str],
        output_path: Optional[Path] = None,
        show: bool = False
    ):
        """
        Plot 2D Pareto front
        
        Args:
            analysis: Pareto analysis results
            objectives: Tuple of (x_objective, y_objective)
            output_path: Path to save figure
            show: Whether to display plot
        """
        obj_x, obj_y = objectives
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Extract data
        pareto_points = analysis['pareto_front']
        dominated_points = [p for p in analysis['all_points'] if p.is_dominated]
        
        # Plot dominated points
        if dominated_points:
            x_dom = [p.objectives.get(obj_x, 0) for p in dominated_points]
            y_dom = [p.objectives.get(obj_y, 0) for p in dominated_points]
            ax.scatter(x_dom, y_dom, c=self.colors['dominated'], 
                      s=100, alpha=0.6, label='Dominated', marker='o')
        
        # Plot Pareto front
        x_pareto = [p.objectives.get(obj_x, 0) for p in pareto_points]
        y_pareto = [p.objectives.get(obj_y, 0) for p in pareto_points]
        ax.scatter(x_pareto, y_pareto, c=self.colors['pareto'], 
                  s=150, alpha=0.8, label='Pareto Optimal', marker='*', 
                  edgecolors='black', linewidths=1.5, zorder=5)
        
        # Connect Pareto points
        if len(pareto_points) > 1:
            # Sort by x-axis for line connection
            sorted_indices = np.argsort(x_pareto)
            x_sorted = [x_pareto[i] for i in sorted_indices]
            y_sorted = [y_pareto[i] for i in sorted_indices]
            ax.plot(x_sorted, y_sorted, 'g--', alpha=0.4, linewidth=1.5, zorder=3)
        
        # Annotate points
        for point in pareto_points:
            x = point.objectives.get(obj_x, 0)
            y = point.objectives.get(obj_y, 0)
            ax.annotate(
                point.variant.variant_name,
                (x, y),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3)
            )
        
        # Labels and title
        ax.set_xlabel(obj_x.capitalize(), fontsize=12, fontweight='bold')
        ax.set_ylabel(obj_y.capitalize(), fontsize=12, fontweight='bold')
        ax.set_title(f'Pareto Front: {obj_y.capitalize()} vs {obj_x.capitalize()}',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 2D Pareto plot to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_3d_pareto(
        self,
        analysis: Dict[str, Any],
        objectives: Tuple[str, str, str],
        output_path: Optional[Path] = None,
        show: bool = False
    ):
        """
        Plot 3D Pareto front
        
        Args:
            analysis: Pareto analysis results
            objectives: Tuple of (x_objective, y_objective, z_objective)
            output_path: Path to save figure
            show: Whether to display plot
        """
        obj_x, obj_y, obj_z = objectives
        
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract data
        pareto_points = analysis['pareto_front']
        dominated_points = [p for p in analysis['all_points'] if p.is_dominated]
        
        # Plot dominated points
        if dominated_points:
            x_dom = [p.objectives.get(obj_x, 0) for p in dominated_points]
            y_dom = [p.objectives.get(obj_y, 0) for p in dominated_points]
            z_dom = [p.objectives.get(obj_z, 0) for p in dominated_points]
            ax.scatter(x_dom, y_dom, z_dom, c=self.colors['dominated'],
                      s=80, alpha=0.5, label='Dominated', marker='o')
        
        # Plot Pareto front
        x_pareto = [p.objectives.get(obj_x, 0) for p in pareto_points]
        y_pareto = [p.objectives.get(obj_y, 0) for p in pareto_points]
        z_pareto = [p.objectives.get(obj_z, 0) for p in pareto_points]
        ax.scatter(x_pareto, y_pareto, z_pareto, c=self.colors['pareto'],
                  s=200, alpha=0.9, label='Pareto Optimal', marker='*',
                  edgecolors='black', linewidths=2, zorder=5)
        
        # Labels
        ax.set_xlabel(obj_x.capitalize(), fontsize=11, fontweight='bold')
        ax.set_ylabel(obj_y.capitalize(), fontsize=11, fontweight='bold')
        ax.set_zlabel(obj_z.capitalize(), fontsize=11, fontweight='bold')
        ax.set_title(f'3D Pareto Front', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved 3D Pareto plot to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_radar_chart(
        self,
        analysis: Dict[str, Any],
        output_path: Optional[Path] = None,
        show: bool = False,
        max_variants: int = 5
    ):
        """
        Plot radar chart comparing variants
        
        Args:
            analysis: Pareto analysis results
            output_path: Path to save figure
            show: Whether to display plot
            max_variants: Maximum number of variants to show
        """
        pareto_points = analysis['pareto_front'][:max_variants]
        
        if len(pareto_points) == 0:
            logger.warning("No Pareto points to plot")
            return
        
        # Get objectives
        objectives = list(pareto_points[0].objectives.keys())
        num_vars = len(objectives)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Normalize values to [0, 1] range for visualization
        stats = analysis['statistics']
        
        for point in pareto_points:
            values = []
            for obj in objectives:
                val = point.objectives.get(obj, 0)
                min_val = stats[obj]['min']
                max_val = stats[obj]['max']
                
                # Normalize
                if max_val != min_val:
                    normalized = (val - min_val) / (max_val - min_val)
                else:
                    normalized = 0.5
                
                values.append(normalized)
            
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=point.variant.variant_name)
            ax.fill(angles, values, alpha=0.15)
        
        # Set labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([obj.capitalize() for obj in objectives], fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.set_title('Variant Comparison (Radar Chart)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved radar chart to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_statistics_comparison(
        self,
        analysis: Dict[str, Any],
        output_path: Optional[Path] = None,
        show: bool = False
    ):
        """Plot statistics comparison across objectives"""
        
        stats = analysis['statistics']
        objectives = list(stats.keys())
        
        fig, axes = plt.subplots(1, len(objectives), figsize=(5*len(objectives), 5))
        
        if len(objectives) == 1:
            axes = [axes]
        
        for ax, obj in zip(axes, objectives):
            obj_stats = stats[obj]
            
            # Box plot data
            pareto_values = [p.objectives.get(obj, 0) for p in analysis['pareto_front']]
            dominated_values = [p.objectives.get(obj, 0) for p in analysis['all_points'] if p.is_dominated]
            
            data = []
            labels = []
            
            if pareto_values:
                data.append(pareto_values)
                labels.append('Pareto\nOptimal')
            
            if dominated_values:
                data.append(dominated_values)
                labels.append('Dominated')
            
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            
            # Color boxes
            colors = [self.colors['pareto'], self.colors['dominated']]
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            
            ax.set_title(obj.capitalize(), fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
        
        fig.suptitle('Objective Statistics: Pareto vs Dominated', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved statistics comparison to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_trade_off_heatmap(
        self,
        analysis: Dict[str, Any],
        output_path: Optional[Path] = None,
        show: bool = False
    ):
        """Plot trade-off heatmap"""
        
        all_points = analysis['all_points']
        objectives = list(all_points[0].objectives.keys())
        
        # Create matrix of objective values
        matrix = np.zeros((len(all_points), len(objectives)))
        variant_names = []
        
        # Normalize values
        stats = analysis['statistics']
        
        for i, point in enumerate(all_points):
            variant_names.append(point.variant.variant_name)
            for j, obj in enumerate(objectives):
                val = point.objectives.get(obj, 0)
                min_val = stats[obj]['min']
                max_val = stats[obj]['max']
                
                # Normalize to [0, 1]
                if max_val != min_val:
                    normalized = (val - min_val) / (max_val - min_val)
                else:
                    normalized = 0.5
                
                matrix[i, j] = normalized
        
        fig, ax = plt.subplots(figsize=(8, max(6, len(all_points) * 0.5)))
        
        # Create heatmap
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(np.arange(len(objectives)))
        ax.set_yticks(np.arange(len(all_points)))
        ax.set_xticklabels([obj.capitalize() for obj in objectives], fontsize=10)
        ax.set_yticklabels(variant_names, fontsize=9)
        
        # Rotate x labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value (0=worst, 1=best)', fontsize=10)
        
        # Mark Pareto optimal variants
        for i, point in enumerate(all_points):
            if not point.is_dominated:
                # Add star marker
                ax.text(-0.5, i, '★', fontsize=12, color='gold', 
                       ha='right', va='center', weight='bold')
        
        ax.set_title('Variant Trade-off Heatmap\n(★ = Pareto Optimal)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved trade-off heatmap to {output_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


# Convenience function
def visualize_pareto(
    analysis: Dict[str, Any],
    output_dir: Path,
    show_plots: bool = False
):
    """Visualize Pareto analysis (convenience wrapper)"""
    
    visualizer = ParetoVisualizer()
    visualizer.visualize_all(analysis, output_dir, show_plots)
