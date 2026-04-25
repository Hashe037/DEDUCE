"""
Similarity Score Analysis Utilities
Analyze and plot similarity scores for individual images or groups
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Union
import torch


class SimilarityAnalyzer:
    """Analyze similarity scores from saved evaluation results"""
    
    def __init__(self, results_dir: str, descriptor_name: str):
        """
        Initialize analyzer
        
        Args:
            results_dir: Directory containing saved results
            descriptor_name: Name of semantic descriptor (e.g., 'day_night')
        """
        self.results_dir = Path(results_dir)
        self.descriptor_name = descriptor_name
        
        # Load per-image results
        csv_path = self.results_dir / f'{descriptor_name}_per_image_results.csv'
        self.df = pd.read_csv(csv_path)
        
        # Extract category names from similarity columns
        self.categories = [col.replace('sim_', '') for col in self.df.columns 
                          if col.startswith('sim_')]
        
        print(f"Loaded {len(self.df)} images")
        print(f"Categories: {self.categories}")
    
    def get_image_similarities(self, image_idx: int) -> Dict[str, float]:
        """
        Get all similarity scores for a specific image
        
        Args:
            image_idx: Index of image
            
        Returns:
            Dictionary mapping category names to similarity scores
        """
        if image_idx >= len(self.df):
            raise ValueError(f"Image index {image_idx} out of range (max: {len(self.df)-1})")
        
        row = self.df.iloc[image_idx]
        similarities = {cat: row[f'sim_{cat}'] for cat in self.categories}
        
        return similarities
    
    def get_image_info(self, image_idx: int) -> Dict:
        """
        Get complete information for a specific image
        
        Args:
            image_idx: Index of image
            
        Returns:
            Dictionary with ground truth, prediction, confidence, similarities, etc.
        """
        if image_idx >= len(self.df):
            raise ValueError(f"Image index {image_idx} out of range")
        
        row = self.df.iloc[image_idx]
        
        info = {
            'image_idx': image_idx,
            'ground_truth': row['ground_truth'],
            'predicted': row['predicted'],
            'confidence': row['confidence'],
            'correct': row['correct'],
            'margin': row['margin'],
            'similarity_spread': row['similarity_spread'],
            'similarities': self.get_image_similarities(image_idx)
        }
        
        if 'image_path' in row:
            info['image_path'] = row['image_path']
        
        return info
    
    def plot_single_image_similarities(self, image_idx: int, 
                                      save_path: Optional[str] = None,
                                      show: bool = True):
        """
        Plot similarity scores for a single image
        
        Args:
            image_idx: Index of image to plot
            save_path: Optional path to save plot
            show: Whether to display plot
        """
        info = self.get_image_info(image_idx)
        similarities = info['similarities']
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = list(similarities.keys())
        scores = list(similarities.values())
        
        # Color bars: green for ground truth, blue for predicted, gray for others
        colors = []
        for cat in categories:
            if cat == info['ground_truth']:
                colors.append('green')
            elif cat == info['predicted']:
                colors.append('blue')
            else:
                colors.append('lightgray')
        
        bars = ax.bar(range(len(categories)), scores, color=colors, alpha=0.7)
        
        # Customize plot
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories],
                          rotation=45, ha='right')
        ax.set_ylabel('Similarity Score')
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{score:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Title with info
        status = "✓ CORRECT" if info['correct'] else "✗ INCORRECT"
        title = f"Image {image_idx}: {info['ground_truth'].title()} → {info['predicted'].title()} {status}\n"
        title += f"Confidence: {info['confidence']:.4f} | Margin: {info['margin']:.4f}"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Ground Truth'),
            Patch(facecolor='blue', alpha=0.7, label='Predicted'),
            Patch(facecolor='lightgray', alpha=0.7, label='Other')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_multiple_images_similarities(self, image_indices: List[int],
                                         save_path: Optional[str] = None,
                                         show: bool = True):
        """
        Plot similarity scores for multiple images in a grid
        
        Args:
            image_indices: List of image indices to plot
            save_path: Optional path to save plot
            show: Whether to display plot
        """
        n_images = len(image_indices)
        cols = min(3, n_images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_images == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        for idx, image_idx in enumerate(image_indices):
            ax = axes[idx]
            
            info = self.get_image_info(image_idx)
            similarities = info['similarities']
            
            categories = list(similarities.keys())
            scores = list(similarities.values())
            
            # Color bars
            colors = []
            for cat in categories:
                if cat == info['ground_truth']:
                    colors.append('green')
                elif cat == info['predicted']:
                    colors.append('blue')
                else:
                    colors.append('lightgray')
            
            bars = ax.bar(range(len(categories)), scores, color=colors, alpha=0.7)
            
            ax.set_xticks(range(len(categories)))
            ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories],
                              rotation=45, ha='right')
            ax.set_ylabel('Similarity')
            ax.set_ylim(0, 1)
            ax.grid(axis='y', alpha=0.3)
            
            # Add values
            for bar, score in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=8)
            
            # Title
            status = "✓" if info['correct'] else "✗"
            ax.set_title(f"Img {image_idx}: {info['ground_truth']} → {info['predicted']} {status}\n" +
                        f"Conf: {info['confidence']:.3f}", fontsize=10)
        
        # Hide unused subplots
        for idx in range(n_images, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'{self.descriptor_name.replace("_", " ").title()} - Similarity Scores',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_category_similarities_distribution(self, category: str,
                                               save_path: Optional[str] = None,
                                               show: bool = True):
        """
        Plot distribution of similarity scores for a specific category across all images
        
        Args:
            category: Category to analyze (e.g., 'day', 'night')
            save_path: Optional path to save plot
            show: Whether to display plot
        """
        if category not in self.categories:
            raise ValueError(f"Unknown category: {category}. Available: {self.categories}")
        
        sim_col = f'sim_{category}'
        
        # Get similarities for images where this is ground truth vs not
        is_category = self.df['ground_truth'] == category
        category_sims = self.df[is_category][sim_col]
        other_sims = self.df[~is_category][sim_col]
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Histogram
        ax1.hist(category_sims, bins=30, alpha=0.7, label=f'True {category.title()}', 
                color='green', density=True)
        ax1.hist(other_sims, bins=30, alpha=0.7, label=f'Not {category.title()}',
                color='red', density=True)
        ax1.set_xlabel(f'Similarity to {category.title()}')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Distribution of Similarity to "{category.title()}"')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Add statistics
        ax1.axvline(category_sims.mean(), color='green', linestyle='--', linewidth=2,
                   label=f'Mean (True): {category_sims.mean():.3f}')
        ax1.axvline(other_sims.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean (Other): {other_sims.mean():.3f}')
        
        # Plot 2: Box plot
        data_to_plot = [category_sims.values, other_sims.values]
        box = ax2.boxplot(data_to_plot, labels=[f'True {category.title()}', f'Not {category.title()}'],
                         patch_artist=True)
        box['boxes'][0].set_facecolor('lightgreen')
        box['boxes'][1].set_facecolor('lightcoral')
        ax2.set_ylabel(f'Similarity to {category.title()}')
        ax2.set_title('Similarity Distribution by Ground Truth')
        ax2.grid(alpha=0.3, axis='y')
        
        # Add statistics
        stats_text = f"True {category.title()}:\n"
        stats_text += f"  μ = {category_sims.mean():.4f}\n"
        stats_text += f"  σ = {category_sims.std():.4f}\n\n"
        stats_text += f"Not {category.title()}:\n"
        stats_text += f"  μ = {other_sims.mean():.4f}\n"
        stats_text += f"  σ = {other_sims.std():.4f}"
        
        ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Similarity Analysis for "{category.title()}" Category',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def compare_categories_on_image(self, image_idx: int, 
                                   categories_to_compare: Optional[List[str]] = None):
        """
        Print detailed comparison of category similarities for a specific image
        
        Args:
            image_idx: Index of image
            categories_to_compare: List of categories to compare (default: all)
        """
        info = self.get_image_info(image_idx)
        similarities = info['similarities']
        
        if categories_to_compare is None:
            categories_to_compare = self.categories
        
        print(f"\n{'='*60}")
        print(f"Image {image_idx} - Similarity Analysis")
        print(f"{'='*60}")
        print(f"Ground Truth: {info['ground_truth'].title()}")
        print(f"Predicted: {info['predicted'].title()}")
        print(f"Correct: {'Yes ✓' if info['correct'] else 'No ✗'}")
        print(f"Confidence: {info['confidence']:.6f}")
        print(f"Margin (Top1-Top2): {info['margin']:.6f}")
        
        if 'image_path' in info:
            print(f"Path: {info['image_path']}")
        
        print(f"\nSimilarity Scores:")
        print(f"-" * 40)
        
        # Sort by similarity score
        sorted_sims = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (cat, score) in enumerate(sorted_sims, 1):
            if cat in categories_to_compare:
                marker = ""
                if cat == info['ground_truth']:
                    marker = " [GROUND TRUTH]"
                elif cat == info['predicted']:
                    marker = " [PREDICTED]"
                
                print(f"  {rank}. {cat.title():<15}: {score:.6f}{marker}")
        
        # Differences
        if len(categories_to_compare) == 2:
            cat1, cat2 = categories_to_compare
            diff = similarities[cat1] - similarities[cat2]
            print(f"\nDifference ({cat1} - {cat2}): {diff:.6f}")
            if diff > 0:
                print(f"  → Model prefers {cat1.title()} by {abs(diff):.6f}")
            else:
                print(f"  → Model prefers {cat2.title()} by {abs(diff):.6f}")
        
        print(f"{'='*60}\n")
    
    def find_images_by_similarity_pattern(self, 
                                         category: str,
                                         min_sim: float = 0.0,
                                         max_sim: float = 1.0,
                                         ground_truth_filter: Optional[str] = None) -> pd.DataFrame:
        """
        Find images matching a specific similarity pattern
        
        Args:
            category: Category to filter by
            min_sim: Minimum similarity score
            max_sim: Maximum similarity score
            ground_truth_filter: Optional filter for ground truth category
            
        Returns:
            DataFrame of matching images
        """
        sim_col = f'sim_{category}'
        
        mask = (self.df[sim_col] >= min_sim) & (self.df[sim_col] <= max_sim)
        
        if ground_truth_filter:
            mask = mask & (self.df['ground_truth'] == ground_truth_filter)
        
        return self.df[mask]
    
    def plot_similarity_heatmap(self, image_indices: Optional[List[int]] = None,
                               save_path: Optional[str] = None,
                               show: bool = True):
        """
        Plot heatmap of similarities for multiple images
        
        Args:
            image_indices: List of image indices (default: first 50)
            save_path: Optional path to save plot
            show: Whether to display plot
        """
        if image_indices is None:
            image_indices = list(range(min(50, len(self.df))))
        
        # Extract similarity matrix
        sim_cols = [f'sim_{cat}' for cat in self.categories]
        sim_matrix = self.df.iloc[image_indices][sim_cols].values
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, max(6, len(image_indices) * 0.2)))
        
        im = ax.imshow(sim_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set ticks
        ax.set_xticks(range(len(self.categories)))
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in self.categories],
                          rotation=45, ha='right')
        ax.set_yticks(range(len(image_indices)))
        ax.set_yticklabels([f"Img {i}" for i in image_indices])
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Similarity Score', rotation=270, labelpad=20)
        
        ax.set_title(f'Similarity Heatmap: {self.descriptor_name.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Category')
        ax.set_ylabel('Image Index')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved heatmap to {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


# Convenience functions

def plot_image_similarities(results_dir: str, descriptor: str, image_idx: int):
    """Quick plot of similarities for a single image"""
    analyzer = SimilarityAnalyzer(results_dir, descriptor)
    analyzer.plot_single_image_similarities(image_idx)


def compare_day_night_on_image(results_dir: str, image_idx: int):
    """Quick comparison of day/night similarities for a specific image"""
    analyzer = SimilarityAnalyzer(results_dir, 'day_night')
    analyzer.compare_categories_on_image(image_idx, ['day', 'night'])
    analyzer.plot_single_image_similarities(image_idx)


def analyze_category_separation(results_dir: str, descriptor: str, category: str):
    """Analyze how well a category separates from others"""
    analyzer = SimilarityAnalyzer(results_dir, descriptor)
    analyzer.plot_category_similarities_distribution(category)


# Example usage
if __name__ == "__main__":
    results_dir = "results/day_night/"
    
    # Create analyzer
    analyzer = SimilarityAnalyzer(results_dir, 'day_night')
    
    # Example 1: Analyze a specific image
    print("Example 1: Single image analysis")
    analyzer.compare_categories_on_image(image_idx=5)
    analyzer.plot_single_image_similarities(image_idx=5)
    
    # Example 2: Compare multiple images
    print("\nExample 2: Multiple images")
    analyzer.plot_multiple_images_similarities([0, 5, 10, 15])
    
    # Example 3: Analyze category separation
    print("\nExample 3: Category separation")
    analyzer.plot_category_similarities_distribution('day')
    
    # Example 4: Find images with similar patterns
    print("\nExample 4: Finding specific patterns")
    # Find day images with low day similarity (potential errors)
    suspicious = analyzer.find_images_by_similarity_pattern(
        category='day',
        max_sim=0.6,
        ground_truth_filter='day'
    )
    print(f"Found {len(suspicious)} day images with low day similarity")
    
    # Example 5: Similarity heatmap
    print("\nExample 5: Heatmap visualization")
    analyzer.plot_similarity_heatmap(image_indices=list(range(20)))