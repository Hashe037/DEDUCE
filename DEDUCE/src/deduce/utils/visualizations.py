"""
Simple visualization utilities for semantic prediction results
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from collections import Counter
import json
from PIL import Image
import random
import os
class ResultsVisualizer:
    """Simple visualizer for semantic evaluation results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Set style for clean plots
        plt.style.use('default')
        
    def plot_prediction_distributions(self, 
                                    evaluation_results: Dict[str, Any],
                                    save_path: Optional[str] = None,
                                    show: bool = True) -> None:
        """
        Plot prediction distributions for each semantic descriptor
        
        Args:
            evaluation_results: Results from Evaluator.evaluate_predictions()
            save_path: Optional path to save plots
            show: Whether to display plots
        """
        # Filter out summary
        descriptors = {k: v for k, v in evaluation_results.items() if k != 'summary'}
        
        if not descriptors:
            self.logger.warning("No descriptor results to plot")
            return
        
        # Create subplots
        n_descriptors = len(descriptors)
        cols = min(3, n_descriptors)
        rows = (n_descriptors + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
        if n_descriptors == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, (desc_name, results) in enumerate(descriptors.items()):
            ax = axes[idx] if n_descriptors > 1 else axes[0]
            
            # Get distribution data
            distribution = results.get('metrics', {}).get('distribution', {})
            most_common = distribution.get('most_common_category', 'unknown')
            percentage = distribution.get('most_common_percentage', 0)
            
            # Simple bar chart showing most common prediction
            categories = [most_common, 'others']
            percentages = [percentage, 100 - percentage]
            
            bars = ax.bar(categories, percentages, color=['#2E86C1', '#E8E8E8'])
            ax.set_title(f'{desc_name.replace("_", " ").title()}')
            ax.set_ylabel('Percentage of Images')
            ax.set_ylim(0, 100)
            
            # Add percentage labels on bars
            for bar, pct in zip(bars, percentages):
                if pct > 5:  # Only show label if bar is visible
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                           f'{pct:.1f}%', ha='center', va='center', fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_descriptors, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(Path(save_path) / 'prediction_distributions.png', 
                       dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved prediction distributions to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_confidence_scores(self, 
                             evaluation_results: Dict[str, Any],
                             save_path: Optional[str] = None,
                             show: bool = True) -> None:
        """
        Plot confidence score distributions across descriptors
        """
        descriptors = {k: v for k, v in evaluation_results.items() if k != 'summary'}
        
        if not descriptors:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot 1: Confidence means with error bars
        names = []
        means = []
        stds = []
        
        for desc_name, results in descriptors.items():
            confidence = results.get('metrics', {}).get('confidence', {})
            names.append(desc_name.replace('_', ' ').title())
            means.append(confidence.get('mean', 0))
            stds.append(confidence.get('std', 0))
        
        bars = ax1.bar(range(len(names)), means, yerr=stds, 
                      capsize=5, color='#3498DB', alpha=0.7)
        ax1.set_xticks(range(len(names)))
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.set_ylabel('Confidence Score')
        ax1.set_title('Mean Confidence by Descriptor')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Coverage ratios
        coverage_ratios = []
        for desc_name, results in descriptors.items():
            coverage = results.get('metrics', {}).get('coverage', {})
            coverage_ratios.append(coverage.get('coverage_ratio', 0))
        
        bars = ax2.bar(range(len(names)), coverage_ratios, color='#E67E22', alpha=0.7)
        ax2.set_xticks(range(len(names)))
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Coverage Ratio')
        ax2.set_title('Category Coverage by Descriptor')
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, ratio in zip(bars, coverage_ratios):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{ratio:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(Path(save_path) / 'confidence_and_coverage.png', 
                       dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved confidence plots to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_summary_dashboard(self, 
                             evaluation_results: Dict[str, Any],
                             save_path: Optional[str] = None,
                             show: bool = True) -> None:
        """
        Create a simple summary dashboard with key metrics
        """
        summary = evaluation_results.get('summary', {})
        descriptors = {k: v for k, v in evaluation_results.items() if k != 'summary'}
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Overall metrics - simple text display
        ax1.text(0.5, 0.7, f"Overall Confidence\n{summary.get('mean_confidence', 0):.3f}", 
                ha='center', va='center', fontsize=20, fontweight='bold', 
                transform=ax1.transAxes)
        ax1.text(0.5, 0.3, f"Overall Coverage\n{summary.get('mean_coverage', 0):.3f}", 
                ha='center', va='center', fontsize=20, fontweight='bold',
                transform=ax1.transAxes)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title('Overall Metrics', fontsize=14, fontweight='bold')
        ax1.axis('off')
        
        # Number of images per descriptor
        desc_names = [name.replace('_', ' ').title() for name in descriptors.keys()]
        image_counts = [results.get('num_images', 0) for results in descriptors.values()]
        
        ax2.bar(range(len(desc_names)), image_counts, color='#9B59B6', alpha=0.7)
        ax2.set_xticks(range(len(desc_names)))
        ax2.set_xticklabels(desc_names, rotation=45, ha='right')
        ax2.set_ylabel('Number of Images')
        ax2.set_title('Images Processed per Descriptor')
        
        # Confidence comparison
        confidences = [results.get('metrics', {}).get('confidence', {}).get('mean', 0) 
                      for results in descriptors.values()]
        
        ax3.bar(range(len(desc_names)), confidences, color='#1ABC9C', alpha=0.7)
        ax3.set_xticks(range(len(desc_names)))
        ax3.set_xticklabels(desc_names, rotation=45, ha='right')
        ax3.set_ylabel('Mean Confidence')
        ax3.set_title('Confidence by Descriptor')
        ax3.set_ylim(0, 1)
        
        # Coverage comparison  
        coverage_ratios = [results.get('metrics', {}).get('coverage', {}).get('coverage_ratio', 0) 
                          for results in descriptors.values()]
        
        ax4.bar(range(len(desc_names)), coverage_ratios, color='#F39C12', alpha=0.7)
        ax4.set_xticks(range(len(desc_names)))
        ax4.set_xticklabels(desc_names, rotation=45, ha='right')
        ax4.set_ylabel('Coverage Ratio')
        ax4.set_title('Category Coverage by Descriptor')
        ax4.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(Path(save_path) / 'summary_dashboard.png', 
                       dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved summary dashboard to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_category_distributions(self, 
                               evaluation_results: Dict[str, Any],
                               save_path: Optional[str] = None,
                               show: bool = True) -> None:
        """
        Plot prediction distributions by category for each descriptor
        
        Args:
            evaluation_results: Results from Evaluator.evaluate_predictions()
            save_path: Optional path to save plots
            show: Whether to display plots
        """
        # Filter out summary
        descriptors = {k: v for k, v in evaluation_results.items() if k != 'summary'}
        
        if not descriptors:
            self.logger.warning("No descriptor results to plot")
            return
        
        # Create subplots - one per descriptor
        n_descriptors = len(descriptors)
        cols = min(3, n_descriptors)
        rows = (n_descriptors + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        if n_descriptors == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, (desc_name, results) in enumerate(descriptors.items()):
            ax = axes[idx]
            
            # Get distribution data from metrics
            distribution = results.get('metrics', {}).get('distribution', {})
            
            # Extract category counts
            category_counts = distribution['category_counts']
            categories = list(category_counts.keys())
            counts = list(category_counts.values())
            
            # Create bar plot
            bars = ax.bar(range(len(categories)), counts, 
                        color=plt.cm.Set3(np.linspace(0, 1, len(categories))), 
                        alpha=0.8)
            
            # Customize plot
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], 
                            rotation=45, ha='right')
            ax.set_ylabel('Number of Predictions')
            ax.set_title(f'{desc_name.replace("_", " ").title()}\nPrediction Distribution')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                if count > 0:  # Only show labels for non-zero bars
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01, 
                        str(count), ha='center', va='bottom', fontsize=9)
            
            # Highlight zero-prediction categories (missing categories)
            if min(counts) == 0:
                ax.text(0.02, 0.98, '⚠️ Some categories unused', 
                    transform=ax.transAxes, fontsize=8, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    verticalalignment='top')
        
        # Hide empty subplots
        for idx in range(n_descriptors, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(Path(save_path) / 'category_distributions.png', 
                    dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved category distribution plots to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_category_margins(self, 
                            evaluation_results: Dict[str, Any],
                            save_path: Optional[str] = None,
                            show: bool = True) -> None:
        """
        Plot average relative margins by category for each descriptor
        Shows how decisive the model is for each category
        
        Args:
            evaluation_results: Results from Evaluator.evaluate_predictions()
            save_path: Optional path to save plots
            show: Whether to display plots
        """
        # Filter out summary
        descriptors = {k: v for k, v in evaluation_results.items() if k != 'summary'}
        
        if not descriptors:
            self.logger.warning("No descriptor results to plot")
            return
        
        # Create subplots - one per descriptor
        n_descriptors = len(descriptors)
        cols = min(3, n_descriptors)
        rows = (n_descriptors + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 5*rows))
        if n_descriptors == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, (desc_name, results) in enumerate(descriptors.items()):
            ax = axes[idx]
            
            # Get category-wise margin data
            metrics = results.get('metrics', {})
            category_margins = metrics.get('category_margins', {})
            
            categories = list(category_margins.keys())
            margins = list(category_margins.values())
            
            # Create bar plot with color coding
            colors = []
            for margin in margins:
                if margin >= 0.7:
                    colors.append('#2ECC71')  # Green - very decisive
                elif margin >= 0.4:
                    colors.append('#F39C12')  # Orange - moderately decisive  
                elif margin >= 0.2:
                    colors.append('#E74C3C')  # Red - uncertain
                else:
                    colors.append('#8E44AD')  # Purple - very uncertain
            
            bars = ax.bar(range(len(categories)), margins, color=colors, alpha=0.8)
            
            # Customize plot
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], 
                            rotation=45, ha='right')
            ax.set_ylabel('Average Relative Margin')
            ax.set_title(f'{desc_name.replace("_", " ").title()}\nDecisiveness by Category')
            ax.set_ylim(0, 1)
            
            # Add horizontal reference lines
            ax.axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Very Decisive')
            ax.axhline(y=0.4, color='orange', linestyle='--', alpha=0.5, label='Moderate')
            ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Uncertain')
            
            # Add value labels on bars
            for bar, margin in zip(bars, margins):
                if margin > 0.05:  # Only show labels for meaningful bars
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                        f'{margin:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Add legend only on first subplot
            if idx == 0:
                ax.legend(loc='upper right', fontsize=8)
            
            # Add interpretation note
            avg_margin = np.mean(margins) if margins else 0
            if avg_margin >= 0.6:
                note = "Generally decisive"
                note_color = "lightgreen"
            elif avg_margin >= 0.3:
                note = "Moderately confident"
                note_color = "lightyellow"
            else:
                note = "Often uncertain"
                note_color = "lightcoral"
            
            ax.text(0.02, 0.98, note, transform=ax.transAxes, fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=note_color, alpha=0.7),
                verticalalignment='top')
        
        # Hide empty subplots
        for idx in range(n_descriptors, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(Path(save_path) / 'category_margins.png', 
                    dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved category margin plots to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_category_confidences(self, 
                            evaluation_results: Dict[str, Any],
                            save_path: Optional[str] = None,
                            show: bool = True) -> None:
        """
        Plot average confidence scores by category for each descriptor
        """
        descriptors = {k: v for k, v in evaluation_results.items() if k != 'summary'}
        
        if not descriptors:
            self.logger.warning("No descriptor results to plot")
            return
        
        n_descriptors = len(descriptors)
        cols = min(3, n_descriptors)
        rows = (n_descriptors + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
        if n_descriptors == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, (desc_name, results) in enumerate(descriptors.items()):
            ax = axes[idx]
            
            metrics = results.get('metrics', {})
            category_confidences = metrics.get('category_confidences', {})
            
            categories = list(category_confidences.keys())
            confidences = [category_confidences[cat]['mean'] for cat in categories]
            stds = [category_confidences[cat]['std'] for cat in categories]
            counts = [category_confidences[cat]['count'] for cat in categories]
            
            # Color bars based on prediction count
            colors = []
            for count in counts:
                if count == 0:
                    colors.append('#BDC3C7')  # Gray - no predictions
                elif count < 10:
                    colors.append('#E74C3C')  # Red - few predictions
                elif count < 50:
                    colors.append('#F39C12')  # Orange - some predictions
                else:
                    colors.append('#2ECC71')  # Green - many predictions
            
            bars = ax.bar(range(len(categories)), confidences, yerr=stds,
                        color=colors, alpha=0.8, capsize=3)
            
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], 
                            rotation=45, ha='right')
            ax.set_ylabel('Average Confidence')
            ax.set_title(f'{desc_name.replace("_", " ").title()}\nConfidence by Category')
            ax.set_ylim(0, max(confidences) * 1.1 if confidences else 1)
            
            # Add count labels on bars
            for bar, conf, count in zip(bars, confidences, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{conf:.2f}\n(n={count})', ha='center', va='bottom', fontsize=8)
        
        # Hide empty subplots
        for idx in range(n_descriptors, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(Path(save_path) / 'category_confidences.png', 
                    dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved category confidence plots to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_category_confidence_vs_margin(self, 
                                        evaluation_results: Dict[str, Any],
                                        save_path: Optional[str] = None,
                                        show: bool = True) -> None:
        """
        Plot category-wise confidence vs margins - shows which categories are 
        both confident and decisive vs uncertain
        """
        descriptors = {k: v for k, v in evaluation_results.items() if k != 'summary'}
        
        if not descriptors:
            self.logger.warning("No descriptor results to plot")
            return
        
        n_descriptors = len(descriptors)
        cols = min(3, n_descriptors)
        rows = (n_descriptors + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(8*cols, 6*rows))
        if n_descriptors == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, (desc_name, results) in enumerate(descriptors.items()):
            ax = axes[idx]
            
            metrics = results.get('metrics', {})
            category_confidences = metrics.get('category_confidences', {})
            category_margins = metrics.get('category_margins', {})
            
            if not category_confidences or not category_margins:
                ax.text(0.5, 0.5, 'No category data\navailable', 
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax.set_title(f'{desc_name.replace("_", " ").title()}\nConfidence vs Margin')
                continue
            
            # Extract data for plotting
            categories = []
            confidences = []
            margins = []
            counts = []
            
            for category in category_confidences.keys():
                if category_confidences[category]['count'] > 0:  # Only categories with predictions
                    categories.append(category)
                    confidences.append(category_confidences[category]['mean'])
                    margins.append(category_margins.get(category, 0))
                    counts.append(category_confidences[category]['count'])
            
            if not categories:
                ax.text(0.5, 0.5, 'No predictions\nfor any category', 
                    ha='center', va='center', transform=ax.transAxes, fontsize=12)
                ax.set_title(f'{desc_name.replace("_", " ").title()}\nConfidence vs Margin')
                continue
            
            # Create scatter plot with point sizes based on count
            sizes = [min(count * 5, 200) for count in counts]  # Scale point sizes
            
            scatter = ax.scatter(confidences, margins, s=sizes, alpha=0.7, 
                            c=range(len(categories)), cmap='Set2')
            
            # Add category labels
            for i, category in enumerate(categories):
                ax.annotate(category.replace('_', ' ').title(), 
                        (confidences[i], margins[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=9, ha='left')
            
            # Add quadrant lines and labels
            conf_mid = (max(confidences) + min(confidences)) / 2 if confidences else 0.3
            margin_mid = 0.5
            
            ax.axvline(x=conf_mid, color='gray', linestyle='--', alpha=0.5)
            ax.axhline(y=margin_mid, color='gray', linestyle='--', alpha=0.5)
            
            # Quadrant labels
            ax.text(0.02, 0.98, 'Low Conf\nHigh Margin', transform=ax.transAxes, 
                fontsize=8, ha='left', va='top', 
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.7))
            ax.text(0.98, 0.98, 'High Conf\nHigh Margin', transform=ax.transAxes, 
                fontsize=8, ha='right', va='top',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.7))
            ax.text(0.02, 0.02, 'Low Conf\nLow Margin', transform=ax.transAxes, 
                fontsize=8, ha='left', va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcoral", alpha=0.7))
            ax.text(0.98, 0.02, 'High Conf\nLow Margin', transform=ax.transAxes, 
                fontsize=8, ha='right', va='bottom',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.7))
            
            ax.set_xlabel('Average Confidence')
            ax.set_ylabel('Average Relative Margin')
            ax.set_title(f'{desc_name.replace("_", " ").title()}\nConfidence vs Decisiveness')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_descriptors, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(Path(save_path) / 'category_confidence_vs_margin.png', 
                    dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved category confidence vs margin plots to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


    
        
    def generate_all_plots(self, 
                          evaluation_results: Dict[str, Any],
                          save_path: Optional[str] = None,
                          show: bool = False) -> None:
        """
        Generate all visualization plots at once
        
        Args:
            evaluation_results: Results from Evaluator.evaluate_predictions()
            save_path: Directory to save plots (creates if doesn't exist)
            show: Whether to display plots (default False for batch processing)
        """
        if save_path:
            Path(save_path).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Generating visualizations in {save_path}")
        
        try:
            self.plot_summary_dashboard(evaluation_results, save_path, show)
            self.plot_prediction_distributions(evaluation_results, save_path, show)
            self.plot_confidence_scores(evaluation_results, save_path, show)
            self.plot_category_distributions(evaluation_results, save_path, show)
            self.plot_category_margins(evaluation_results, save_path, show) 
            self.plot_category_confidences(evaluation_results, save_path, show)
            self.plot_category_confidence_vs_margin(evaluation_results, save_path, show)


            if save_path:
                self.logger.info(f"All visualizations saved to {save_path}")
                
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            raise
    
    def quick_summary_plot(self, evaluation_results: Dict[str, Any]) -> None:
        """
        Quick single plot summary for interactive use
        """
        summary = evaluation_results.get('summary', {})
        
        # Simple single plot with key numbers
        fig, ax = plt.subplots(figsize=(8, 4))
        
        metrics = ['Confidence', 'Coverage', 'Descriptors']
        values = [
            summary.get('mean_confidence', 0),
            summary.get('mean_coverage', 0), 
            summary.get('num_descriptors', 0) / 10  # Scale for visibility
        ]
        colors = ['#3498DB', '#E67E22', '#9B59B6']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        ax.set_ylabel('Score')
        ax.set_title('Quick Evaluation Summary')
        ax.set_ylim(0, 1)
        
        # Add value labels
        for bar, value, metric in zip(bars, values, metrics):
            if metric == 'Descriptors':
                label = f'{int(value * 10)}'  # Convert back to actual count
            else:
                label = f'{value:.3f}'
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                   label, ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()



def visualize_semantic_images(json_file, num_images=12, figsize=(15, 10)):
    """Load JSON file and display random images in a grid."""
    
    # Load filenames from JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    filenames = data['filenames']
    
    # Sample random images (or all if fewer than num_images)
    sample_size = min(num_images, len(filenames))
    selected = random.sample(filenames, sample_size)
    
    # Calculate grid dimensions
    cols = 4
    rows = (sample_size + cols - 1) // cols

    if sample_size == 0:
        print(f"No images found in {json_file}")
        return None
    
    # Create figure
    # fig, axes = plt.subplots(rows, cols, figsize=figsize)
    # axes = axes.flatten() if sample_size > 1 else [axes]

    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()  # This handles all cases
    
    # Display images
    for idx, img_path in enumerate(selected):
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].axis('off')
            axes[idx].set_title(Path(img_path).name, fontsize=8)
        except Exception as e:
            axes[idx].text(0.5, 0.5, f'Error loading\n{Path(img_path).name}', 
                          ha='center', va='center')
            axes[idx].axis('off')
    
    # Hide unused subplots
    for idx in range(sample_size, len(axes)):
        axes[idx].axis('off')
    
    # Add title with metadata
    json_name = Path(json_file).stem
    fig.suptitle(f'{json_name} (showing {sample_size}/{len(filenames)} images)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure in semantic_viz subfolder
    json_dir = Path(json_file).parent
    output_dir = json_dir / 'semantic_viz'
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f'{json_name}_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return output_path


# def plot_cluster_percentages(input_file, output_dir, output_filename='cluster_percentages_chart.png'):
#     """
#     Generate a grouped bar chart showing cluster percentages and image counts for semantic descriptors.
    
#     Parameters:
#     -----------
#     input_file : str
#         Path to the JSON file containing descriptor cluster data
#     output_dir : str
#         Directory where the output chart will be saved
#     output_filename : str, optional
#         Name of the output PNG file (default: 'cluster_percentages_chart.png')
    
#     Returns:
#     --------
#     str
#         Path to the saved chart file
    
#     Example:
#     --------
#     >>> plot_cluster_percentages(
#     ...     input_file='data/descriptor_cluster_percentages.json',
#     ...     output_dir='results/charts'
#     ... )
#     """
    
#     # Create output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Load the data
#     with open(input_file, 'r') as f:
#         data = json.load(f)
    
#     # Sort by largest_cluster_percentage in descending order (include all descriptors)
#     sorted_data = sorted(data, key=lambda x: x['largest_cluster_percentage'], reverse=True)
    
#     # Extract data for plotting
#     descriptors = [item['descriptor'] for item in sorted_data]
#     percentages = [item['largest_cluster_percentage'] for item in sorted_data]
#     total_images = [item['total_images'] for item in sorted_data]
    
#     # Normalize image counts to a 0-100 scale for better visualization alongside percentages
#     max_images = max(total_images) if max(total_images) > 0 else 1
#     normalized_images = [(count / max_images) * 100 for count in total_images]
    
#     # Create the grouped bar chart
#     fig, ax = plt.subplots(figsize=(16, 7))
    
#     # Set the width of bars and positions
#     bar_width = 0.35
#     x_pos = np.arange(len(descriptors))
    
#     # Create bars
#     bars1 = ax.bar(x_pos - bar_width/2, percentages, bar_width, 
#                    label='Cluster Percentage (%)', 
#                    color='steelblue', edgecolor='black', linewidth=1.2)
#     bars2 = ax.bar(x_pos + bar_width/2, normalized_images, bar_width, 
#                    label=f'Images (scaled, max={max_images})', 
#                    color='coral', edgecolor='black', linewidth=1.2)
    
#     # Add value labels on top of each bar
#     for i, (bar, pct) in enumerate(zip(bars1, percentages)):
#         height = bar.get_height()
#         if height > 0:
#             ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
#                     f'{pct:.1f}%',
#                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    
#     for i, (bar, count) in enumerate(zip(bars2, total_images)):
#         height = bar.get_height()
#         if height > 0 or count > 0:
#             ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
#                     f'n={count}',
#                     ha='center', va='bottom', fontsize=9, fontweight='bold')
    
#     # Customize the plot
#     ax.set_xlabel('Descriptor', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Value (Percentage % / Normalized Image Count)', fontsize=12, fontweight='bold')
#     ax.set_title('Semantic Class Distribution: Cluster Percentages and Image Counts', 
#                  fontsize=14, fontweight='bold', pad=20)
    
#     # Set x-axis
#     ax.set_xticks(x_pos)
#     ax.set_xticklabels(descriptors, rotation=45, ha='right')
    
#     # Add legend
#     ax.legend(loc='upper right', fontsize=10)
    
#     # Add grid for better readability
#     ax.grid(axis='y', alpha=0.3, linestyle='--')
#     ax.set_axisbelow(True)
    
#     # Adjust y-axis
#     ax.set_ylim(0, 105)
    
#     # Tight layout to prevent label cutoff
#     plt.tight_layout()
    
#     # Save the figure
#     output_path = os.path.join(output_dir, output_filename)
#     plt.savefig(output_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     print(f"Chart saved successfully to: {output_path}")
#     print(f"Total descriptors shown: {len(descriptors)}")
#     print(f"Max images: {max_images}")
    
#     return output_path


def plot_cluster_percentages(input_file, output_dir, output_filename='cluster_percentages_overlay.png', 
                             aspect_ratio=(16, 9), 
                             axis_font_size=12, 
                             label_font_size=10):
    """
    Generate a bar chart where bar height is total images and fill level is cluster percentage.
    
    Parameters:
    -----------
    input_file : str
        Path to input JSON.
    output_dir : str
        Path to save output.
    output_filename : str
        Filename of the chart.
    aspect_ratio : tuple or float
        (width, height) or width/height float.
    axis_font_size : int
        Font size for X/Y labels and tick marks (default: 12).
    label_font_size : int
        Font size for the bar annotations (n= and %) (default: 10).
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Sort data
    sorted_data = sorted(data, key=lambda x: x['largest_cluster_percentage'], reverse=True)
    
    # Extract data
    descriptors = [item['descriptor'] for item in sorted_data]
    percentages = [item['largest_cluster_percentage'] for item in sorted_data]
    total_images = [item['total_images'] for item in sorted_data]
    clustered_counts = [count * (pct / 100.0) for count, pct in zip(total_images, percentages)]
    
    # --- ASPECT RATIO LOGIC ---
    base_width = 16
    if isinstance(aspect_ratio, (tuple, list)):
        w, h = aspect_ratio
        calc_height = base_width * (h / w)
    elif isinstance(aspect_ratio, (float, int)):
        calc_height = base_width / aspect_ratio
    else:
        calc_height = 8

    fig, ax = plt.subplots(figsize=(base_width, calc_height))
    x_pos = np.arange(len(descriptors))
    
    # --- PLOT BARS ---
    # Background Bar (Total)
    bars_total = ax.bar(x_pos, total_images, label='Total Images', 
                        color='gainsboro', edgecolor='gray', linewidth=1)

    # Foreground Bar (Cluster Match)
    bars_fill = ax.bar(x_pos, clustered_counts, label='Cluster Match', 
                       color='steelblue', edgecolor='black', linewidth=1)
    
    # --- LABELS ---
    max_height = max(total_images) if total_images else 1
    
    # 1. Total Count (n=)
    for i, (bar, count) in enumerate(zip(bars_total, total_images)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + (max_height * 0.01),
                f'n={count}',
                ha='center', va='bottom', 
                fontsize=label_font_size, 
                fontweight='bold', color='black')

    # 2. Percentage (%)
    for i, (bar, pct) in enumerate(zip(bars_fill, percentages)):
        height = bar.get_height()
        label_text = f'{pct:.1f}%'
        
        # Check if bar is tall enough for internal label
        if height > (max_height * 0.05): 
            # Inside label
            ax.text(bar.get_x() + bar.get_width()/2., height - (max_height * 0.02),
                    label_text,
                    ha='center', va='top', 
                    fontsize=label_font_size, 
                    fontweight='bold', color='white')
        else:
            # Floating label (slightly smaller usually looks better, but using param here)
            ax.text(bar.get_x() + bar.get_width()/2., height + (max_height * 0.01),
                    label_text,
                    ha='center', va='bottom', 
                    fontsize=label_font_size, 
                    fontweight='bold', color='steelblue')

    # --- CUSTOMIZATION ---
    
    # Axis Labels
    ax.set_xlabel('Descriptor', fontsize=axis_font_size, fontweight='bold')
    ax.set_ylabel('Image Count', fontsize=axis_font_size, fontweight='bold')
    
    # Title (Usually Axis Size + 2 is a good rule of thumb)
    ax.set_title('Semantic Class Distribution', 
                 fontsize=axis_font_size + 4, 
                 fontweight='bold', pad=20)
    
    # Ticks
    ax.set_xticks(x_pos)
    ax.set_xticklabels(descriptors, rotation=45, ha='right', fontsize=axis_font_size)
    ax.tick_params(axis='y', labelsize=axis_font_size)
    
    # Legend
    ax.legend(loc='upper right', fontsize=axis_font_size)
    
    # Grid and Layout
    ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Chart saved to: {output_path}")
    return output_path