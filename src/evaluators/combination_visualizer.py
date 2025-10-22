"""
Visualization Dashboard for Combination Exploration Results
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.utils.logger import logger


class CombinationVisualizer:
    """Visualize combination exploration results"""
    
    def __init__(self, style='seaborn-v0_8-darkgrid'):
        try:
            plt.style.use(style)
        except:
            plt.style.use('seaborn-v0_8')
        
        self.colors = {
            'pareto': '#2ecc71',
            'non_pareto': '#e74c3c',
            'highlight': '#3498db'
        }
    
    def create_dashboard(self, csv_path: Path, output_dir: Path):
        """Create complete visualization dashboard"""
        
        logger.info("Creating combination visualization dashboard...")
        
        # Load data
        df = pd.read_csv(csv_path)
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        self.plot_compression_vs_speedup(df, output_dir / 'combo_compression_speedup.png')
        self.plot_heatmap_grid(df, output_dir / 'combo_heatmap.png')
        self.plot_pareto_comparison(df, output_dir / 'combo_pareto.png')
        self.plot_technique_impact(df, output_dir / 'combo_technique_impact.png')
        self.plot_order_comparison(df, output_dir / 'combo_order_comparison.png')
        
        logger.info(f"✓ Dashboard created: {output_dir}")
    
    def plot_compression_vs_speedup(self, df: pd.DataFrame, output_path: Path):
        """Plot compression vs speedup with Pareto highlighting"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Non-Pareto points
        non_pareto = df[df['Pareto Optimal'] == False]
        ax.scatter(non_pareto['Compression'], non_pareto['Speedup'],
                  c=self.colors['non_pareto'], s=100, alpha=0.6,
                  label='Non-Pareto Optimal', marker='o')
        
        # Pareto points
        pareto = df[df['Pareto Optimal'] == True]
        ax.scatter(pareto['Compression'], pareto['Speedup'],
                  c=self.colors['pareto'], s=300, alpha=0.9,
                  label='Pareto Optimal', marker='*',
                  edgecolors='black', linewidths=2, zorder=5)
        
        # Annotate Pareto points
        for _, row in pareto.iterrows():
            ax.annotate(row['ID'], 
                       (row['Compression'], row['Speedup']),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=8, bbox=dict(boxstyle='round,pad=0.3',
                                            facecolor='yellow', alpha=0.5))
        
        ax.set_xlabel('Compression Ratio (×)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Speedup (×)', fontsize=12, fontweight='bold')
        ax.set_title('Combination Exploration: Compression vs Speedup',
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {output_path.name}")
    
    def plot_heatmap_grid(self, df: pd.DataFrame, output_path: Path):
        """Plot heatmap of all combinations"""
        
        # Extract quantization and pruning info from combo IDs
        # Assuming pattern: first 8 are no quant, next 8 are INT8 quant-first, etc.
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Reshape data for heatmaps (4 quant types × 4 prune types × 2 orders)
        n_combos = len(df)
        
        # Heatmap 1: Compression
        comp_data = df['Compression'].values.reshape(4, 8)
        sns.heatmap(comp_data, annot=True, fmt='.2f', cmap='YlOrRd',
                   ax=axes[0], cbar_kws={'label': 'Compression (×)'})
        axes[0].set_title('Compression Ratio', fontweight='bold')
        axes[0].set_xlabel('Configuration')
        axes[0].set_ylabel('Quantization Type')
        
        # Heatmap 2: Speedup
        speed_data = df['Speedup'].values.reshape(4, 8)
        sns.heatmap(speed_data, annot=True, fmt='.2f', cmap='YlGnBu',
                   ax=axes[1], cbar_kws={'label': 'Speedup (×)'})
        axes[1].set_title('Estimated Speedup', fontweight='bold')
        axes[1].set_xlabel('Configuration')
        axes[1].set_ylabel('Quantization Type')
        
        # Heatmap 3: Accuracy Impact
        acc_data = df['Acc Impact (%)'].values.reshape(4, 8)
        sns.heatmap(acc_data, annot=True, fmt='.1f', cmap='RdYlGn',
                   center=0, ax=axes[2], cbar_kws={'label': 'Accuracy Impact (%)'})
        axes[2].set_title('Accuracy Impact', fontweight='bold')
        axes[2].set_xlabel('Configuration')
        axes[2].set_ylabel('Quantization Type')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {output_path.name}")
    
    def plot_pareto_comparison(self, df: pd.DataFrame, output_path: Path):
        """Compare Pareto optimal combinations"""
        
        pareto = df[df['Pareto Optimal'] == True].sort_values('Speedup')
        
        if len(pareto) == 0:
            logger.warning("No Pareto optimal solutions found")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        x = range(len(pareto))
        
        # Compression
        axes[0].bar(x, pareto['Compression'], color=self.colors['pareto'], alpha=0.7)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(pareto['ID'], rotation=45, ha='right')
        axes[0].set_ylabel('Compression Ratio (×)', fontweight='bold')
        axes[0].set_title('Compression Comparison', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Speedup
        axes[1].bar(x, pareto['Speedup'], color=self.colors['highlight'], alpha=0.7)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(pareto['ID'], rotation=45, ha='right')
        axes[1].set_ylabel('Speedup (×)', fontweight='bold')
        axes[1].set_title('Speedup Comparison', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Accuracy Impact
        colors = ['green' if x >= -2 else 'orange' if x >= -4 else 'red' 
                 for x in pareto['Acc Impact (%)']]
        axes[2].bar(x, pareto['Acc Impact (%)'], color=colors, alpha=0.7)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(pareto['ID'], rotation=45, ha='right')
        axes[2].set_ylabel('Accuracy Impact (%)', fontweight='bold')
        axes[2].set_title('Accuracy Impact Comparison', fontweight='bold')
        axes[2].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {output_path.name}")
    
    def plot_technique_impact(self, df: pd.DataFrame, output_path: Path):
        """Analyze impact of each technique"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Group by quantization type
        quant_groups = df.groupby('Quantization').agg({
            'Compression': 'mean',
            'Speedup': 'mean',
            'Acc Impact (%)': 'mean'
        }).reset_index()
        
        # Compression by Quantization
        axes[0, 0].bar(quant_groups['Quantization'], quant_groups['Compression'],
                       color=['gray', 'lightcoral', 'lightblue'])
        axes[0, 0].set_ylabel('Avg Compression (×)', fontweight='bold')
        axes[0, 0].set_title('Compression by Quantization Type', fontweight='bold')
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Speedup by Quantization
        axes[0, 1].bar(quant_groups['Quantization'], quant_groups['Speedup'],
                       color=['gray', 'lightcoral', 'lightblue'])
        axes[0, 1].set_ylabel('Avg Speedup (×)', fontweight='bold')
        axes[0, 1].set_title('Speedup by Quantization Type', fontweight='bold')
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # Sparsity distribution
        axes[1, 0].hist(df['Sparsity (%)'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        axes[1, 0].set_xlabel('Sparsity (%)', fontweight='bold')
        axes[1, 0].set_ylabel('Frequency', fontweight='bold')
        axes[1, 0].set_title('Sparsity Distribution', fontweight='bold')
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # Optimization Time
        axes[1, 1].scatter(df['Compression'], df['Time (s)'],
                          c=df['Speedup'], cmap='viridis', s=100, alpha=0.6)
        axes[1, 1].set_xlabel('Compression (×)', fontweight='bold')
        axes[1, 1].set_ylabel('Optimization Time (s)', fontweight='bold')
        axes[1, 1].set_title('Compression vs Time', fontweight='bold')
        axes[1, 1].grid(alpha=0.3)
        cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
        cbar.set_label('Speedup (×)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {output_path.name}")
    
    def plot_order_comparison(self, df: pd.DataFrame, output_path: Path):
        """Compare quantize-first vs prune-first"""
        
        # Assuming combos 1-16 are one order, 17-32 are another
        order1 = df.iloc[:16]
        order2 = df.iloc[16:]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics = ['Compression', 'Speedup', 'Acc Impact (%)']
        labels = ['Quantize First', 'Prune First']
        colors_pair = ['#3498db', '#e74c3c']
        
        for ax, metric in zip(axes, metrics):
            means = [order1[metric].mean(), order2[metric].mean()]
            stds = [order1[metric].std(), order2[metric].std()]
            
            ax.bar(labels, means, yerr=stds, color=colors_pair, alpha=0.7, capsize=10)
            ax.set_ylabel(metric, fontweight='bold')
            ax.set_title(f'{metric} by Order', fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"  Saved: {output_path.name}")


# Example usage
def visualize_results(csv_path: str, output_dir: str):
    """Visualize combination exploration results"""
    visualizer = CombinationVisualizer()
    visualizer.create_dashboard(Path(csv_path), Path(output_dir))
