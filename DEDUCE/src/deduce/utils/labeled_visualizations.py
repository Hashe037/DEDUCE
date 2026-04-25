"""
Plotting Utilities for Labeled Evaluation Results
Easy-to-use functions for visualizing saved labeled evaluation data
"""

import json
import csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import os

class LabeledEvaluationPlotter:
    """Plot results from saved labeled evaluation data"""
    
    def __init__(self, results_dir: str):
        """
        Initialize plotter with results directory
        
        Args:
            results_dir: Directory containing saved evaluation results
        """
        self.results_dir = Path(results_dir)
        if not self.results_dir.exists():
            raise ValueError(f"Results directory not found: {results_dir}")
        self.save_subfolder = "labeled_evaluation"
        self.save_path = Path(os.path.join(self.results_dir, self.save_subfolder ))
        self.save_path.mkdir(exist_ok=True)
        
    
    def plot_confusion_matrix(self, descriptor_name: str, 
                             save_plot: bool = True, show: bool = True):
        """
        Plot confusion matrix as heatmap
        
        Args:
            descriptor_name: Name of semantic descriptor (e.g., 'day_night')
            save_plot: Whether to save the plot
            show: Whether to display the plot
        """
        # Load confusion matrix
        json_path = self.results_dir / f'{descriptor_name}_confusion_matrix.json'
        with open(json_path, 'r') as f:
            confusion_data = json.load(f)
        
        matrix = np.array(confusion_data['matrix'])
        categories = confusion_data['categories']
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=categories, yticklabels=categories,
                   cbar_kws={'label': 'Count'})
        
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.title(f'Confusion Matrix: {descriptor_name.replace("_", " ").title()}')
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.save_path / f'{descriptor_name}_confusion_matrix.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix plot to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_category_accuracy(self, descriptor_name: str,
                              save_plot: bool = True, show: bool = True):
        """
        Plot per-category accuracy with confidence bars
        
        Args:
            descriptor_name: Name of semantic descriptor
            save_plot: Whether to save the plot
            show: Whether to display the plot
        """
        # Load category accuracy data
        json_path = self.results_dir / f'{descriptor_name}_category_accuracy.json'
        with open(json_path, 'r') as f:
            category_data = json.load(f)
        
        categories = list(category_data.keys())
        accuracies = [category_data[cat]['accuracy'] for cat in categories]
        counts = [category_data[cat]['total_count'] for cat in categories]
        
        # Create bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Accuracy
        bars1 = ax1.bar(range(len(categories)), accuracies, 
                       color=['green' if acc > 0.9 else 'orange' if acc > 0.7 else 'red' 
                             for acc in accuracies],
                       alpha=0.7)
        ax1.set_xticks(range(len(categories)))
        ax1.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], 
                           rotation=45, ha='right')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.set_title(f'Per-Category Accuracy')
        ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, label='90%')
        ax1.axhline(y=0.7, color='orange', linestyle='--', alpha=0.3, label='70%')
        ax1.legend()
        
        # Add value labels
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.2%}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Sample counts
        bars2 = ax2.bar(range(len(categories)), counts, color='steelblue', alpha=0.7)
        ax2.set_xticks(range(len(categories)))
        ax2.set_xticklabels([cat.replace('_', ' ').title() for cat in categories],
                           rotation=45, ha='right')
        ax2.set_ylabel('Number of Samples')
        ax2.set_title(f'Samples per Category')
        
        # Add value labels
        for bar, count in zip(bars2, counts):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                    str(count), ha='center', va='bottom')
        
        plt.suptitle(f'{descriptor_name.replace("_", " ").title()} - Category Analysis')
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.save_path / f'{descriptor_name}_category_accuracy.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved category accuracy plot to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_confidence_by_correctness(self, descriptor_name: str,
                                      save_plot: bool = True, show: bool = True):
        """
        Plot confidence distributions for correct vs incorrect predictions
        
        Args:
            descriptor_name: Name of semantic descriptor
            save_plot: Whether to save the plot
            show: Whether to display the plot
        """
        # Load per-image results
        csv_path = self.results_dir / f'{descriptor_name}_per_image_results.csv'
        df = pd.read_csv(csv_path)
        
        # Separate correct and incorrect
        correct_df = df[df['correct'] == True]
        incorrect_df = df[df['correct'] == False]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Histogram of confidences
        ax1.hist(correct_df['confidence'], bins=30, alpha=0.7, 
                label=f'Correct (n={len(correct_df)})', color='green')
        ax1.hist(incorrect_df['confidence'], bins=30, alpha=0.7,
                label=f'Incorrect (n={len(incorrect_df)})', color='red')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Count')
        ax1.set_title('Confidence Distribution by Correctness')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Box plot
        data_to_plot = [correct_df['confidence'].values, incorrect_df['confidence'].values]
        box = ax2.boxplot(data_to_plot, labels=['Correct', 'Incorrect'],
                         patch_artist=True)
        box['boxes'][0].set_facecolor('lightgreen')
        box['boxes'][1].set_facecolor('lightcoral')
        ax2.set_ylabel('Confidence Score')
        ax2.set_title('Confidence by Correctness')
        ax2.grid(alpha=0.3, axis='y')
        
        # Add statistics
        correct_mean = correct_df['confidence'].mean()
        incorrect_mean = incorrect_df['confidence'].mean()
        ax2.text(1, correct_mean, f'μ={correct_mean:.3f}', 
                ha='right', va='center', fontweight='bold')
        ax2.text(2, incorrect_mean, f'μ={incorrect_mean:.3f}',
                ha='left', va='center', fontweight='bold')
        
        plt.suptitle(f'{descriptor_name.replace("_", " ").title()} - Confidence Analysis')
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.save_path / f'{descriptor_name}_confidence_analysis.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved confidence analysis plot to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_margin_analysis(self, descriptor_name: str,
                            save_plot: bool = True, show: bool = True):
        """
        Plot margin (decisiveness) analysis
        
        Args:
            descriptor_name: Name of semantic descriptor
            save_plot: Whether to save the plot  
            show: Whether to display the plot
        """
        # Load per-image results
        csv_path = self.results_dir / f'{descriptor_name}_per_image_results.csv'
        df = pd.read_csv(csv_path)
        
        correct_df = df[df['correct'] == True]
        incorrect_df = df[df['correct'] == False]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Margin distribution
        ax1.hist(correct_df['margin'], bins=30, alpha=0.7,
                label=f'Correct (n={len(correct_df)})', color='green')
        ax1.hist(incorrect_df['margin'], bins=30, alpha=0.7,
                label=f'Incorrect (n={len(incorrect_df)})', color='red')
        ax1.set_xlabel('Margin (Top1 - Top2)')
        ax1.set_ylabel('Count')
        ax1.set_title('Margin Distribution by Correctness')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Confidence vs Margin scatter
        ax2.scatter(correct_df['confidence'], correct_df['margin'],
                   alpha=0.5, s=20, c='green', label='Correct')
        ax2.scatter(incorrect_df['confidence'], incorrect_df['margin'],
                   alpha=0.5, s=20, c='red', label='Incorrect')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Margin')
        ax2.set_title('Confidence vs Margin')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Margin box plots
        data_to_plot = [correct_df['margin'].values, incorrect_df['margin'].values]
        box = ax3.boxplot(data_to_plot, labels=['Correct', 'Incorrect'],
                         patch_artist=True)
        box['boxes'][0].set_facecolor('lightgreen')
        box['boxes'][1].set_facecolor('lightcoral')
        ax3.set_ylabel('Margin')
        ax3.set_title('Margin by Correctness')
        ax3.grid(alpha=0.3, axis='y')
        
        # Plot 4: Low margin analysis
        low_margin_threshold = 0.1
        low_margin_df = df[df['margin'] < low_margin_threshold]
        low_margin_correct = len(low_margin_df[low_margin_df['correct'] == True])
        low_margin_incorrect = len(low_margin_df[low_margin_df['correct'] == False])
        
        ax4.bar(['Correct', 'Incorrect'], 
               [low_margin_correct, low_margin_incorrect],
               color=['green', 'red'], alpha=0.7)
        ax4.set_ylabel('Count')
        ax4.set_title(f'Low Margin Predictions (< {low_margin_threshold})')
        ax4.text(0, low_margin_correct + 1, str(low_margin_correct),
                ha='center', fontweight='bold')
        ax4.text(1, low_margin_incorrect + 1, str(low_margin_incorrect),
                ha='center', fontweight='bold')
        
        plt.suptitle(f'{descriptor_name.replace("_", " ").title()} - Margin Analysis')
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.save_path / f'{descriptor_name}_margin_analysis.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved margin analysis plot to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()

    def plot_margin_analysis_byclass(self, descriptor_name: str,
                            save_plot: bool = True, show: bool = True):
        """
        Plot margin (decisiveness) analysis by the class, not just the correct/incorrect
        
        Args:
            descriptor_name: Name of semantic descriptor
            save_plot: Whether to save the plot  
            show: Whether to display the plot
        """
        # Load per-image results
        csv_path = self.results_dir / f'{descriptor_name}_per_image_results.csv'
        df = pd.read_csv(csv_path)

        # Assume two classes
        classes = df['ground_truth'].unique()
        class1, class2 = classes
        df1 = df[df['ground_truth'] == class1]
        df2 = df[df['ground_truth'] == class2]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Margin distribution
        ax1.hist(df1['margin'], bins=30, alpha=0.7,
                label=f'{class1} (n={len(df1)})', color='green')
        ax1.hist(df2['margin'], bins=30, alpha=0.7,
                label=f'{class2} (n={len(df2)})', color='red')
        ax1.set_xlabel('Margin (Top1 - Top2)')
        ax1.set_ylabel('Count')
        ax1.set_title('Margin Distribution by Class')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Confidence vs Margin scatter
        ax2.scatter(df1['confidence'], df1['margin'],
                   alpha=0.5, s=20, c='green', label=class1)
        ax2.scatter(df2['confidence'], df2['margin'],
                   alpha=0.5, s=20, c='red', label=class2)
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Margin')
        ax2.set_title('Confidence vs Margin')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Margin box plots
        data_to_plot = [df1['margin'].values, df2['margin'].values]
        box = ax3.boxplot(data_to_plot, labels=[class1,class2],
                         patch_artist=True)
        box['boxes'][0].set_facecolor('lightgreen')
        box['boxes'][1].set_facecolor('lightcoral')
        ax3.set_ylabel('Margin')
        ax3.set_title('Margin by Class')
        ax3.grid(alpha=0.3, axis='y')
        
        if save_plot:
            plot_path = self.save_path / f'{descriptor_name}_margin_analysis_byclass.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved margin analysis plot to {plot_path}")
        
        if show:
            plt.show()
        else:
            plt.close()


    def compute_single_metrics(self, descriptor_name: str) -> Dict:
        """
        Compute single-value metrics for comparing multiple runs
        
        Metrics computed:
        - Average margin (incorrect counts as negative) per class
        - Average margin (incorrect counts as negative) overall
        - Average confidence per class
        - Average confidence overall
        - Accuracy per class
        - Accuracy overall
        
        Args:
            descriptor_name: Name of semantic descriptor
            
        Returns:
            Dictionary with all computed metrics
        """
        # Load per-image results
        csv_path = self.results_dir / f'{descriptor_name}_per_image_results.csv'
        df = pd.read_csv(csv_path)
        
        # Get unique classes
        classes = df['ground_truth'].unique()
        
        metrics = {
            'descriptor': descriptor_name,
            'total_images': len(df),
            'classes': list(classes)
        }
        
        # --- Overall Metrics ---
        
        # Overall accuracy
        metrics['overall_accuracy'] = (df['correct'] == True).sum() / len(df)
        
        # Overall average confidence
        metrics['overall_avg_confidence'] = df['confidence'].mean()
        
        # Overall average margin (negative for incorrect)
        df['signed_margin'] = df['margin'] * df['correct'].map({True: 1, False: -1})
        metrics['overall_avg_margin_signed'] = df['signed_margin'].mean()
        
        # --- Per-Class Metrics ---
        
        class_metrics = {}
        
        for cls in classes:
            class_df = df[df['ground_truth'] == cls]
            
            cls_metrics = {
                'count': len(class_df),
                'accuracy': (class_df['correct'] == True).sum() / len(class_df),
                'avg_confidence': class_df['confidence'].mean(),
                'avg_margin_signed': class_df['signed_margin'].mean()
            }
            
            class_metrics[cls] = cls_metrics
        
        metrics['per_class'] = class_metrics
        
        return metrics
    
    def save_single_metrics(self, descriptor_name: str, 
                           output_file: str = 'single_metrics.json'):
        """
        Compute and save single-value metrics to JSON file
        
        Args:
            descriptor_name: Name of semantic descriptor
            output_file: Name of output JSON file
        """
        metrics = self.compute_single_metrics(descriptor_name)
        
        output_path = self.save_path / output_file
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Saved single metrics to: {output_path}")
        return metrics
    
    def print_single_metrics(self, descriptor_name: str):
        """
        Compute and print single-value metrics in readable format
        
        Args:
            descriptor_name: Name of semantic descriptor
        """
        metrics = self.compute_single_metrics(descriptor_name)
        
        print(f"\n{'='*60}")
        print(f"Single-Value Metrics: {descriptor_name}")
        print(f"{'='*60}\n")
        
        print("Overall Metrics:")
        print(f"  • Total Images: {metrics['total_images']}")
        print(f"  • Accuracy: {metrics['overall_accuracy']:.4f}")
        print(f"  • Avg Confidence: {metrics['overall_avg_confidence']:.4f}")
        print(f"  • Avg Margin (signed): {metrics['overall_avg_margin_signed']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        for cls, cls_metrics in metrics['per_class'].items():
            print(f"\n  {cls.upper()}:")
            print(f"    • Count: {cls_metrics['count']}")
            print(f"    • Accuracy: {cls_metrics['accuracy']:.4f}")
            print(f"    • Avg Confidence: {cls_metrics['avg_confidence']:.4f}")
            print(f"    • Avg Margin (signed): {cls_metrics['avg_margin_signed']:.4f}")
        
        print(f"\n{'='*60}\n")
        
        return metrics
    
    def plot_all(self, descriptor_name: str, save_plots: bool = True, show: bool = False):
        """Generate all plots for a descriptor"""
        print(f"\n{'='*60}")
        print(f"Generating plots for: {descriptor_name}")
        print(f"{'='*60}\n")
        
        self.plot_confusion_matrix(descriptor_name, save_plots, show)
        self.plot_category_accuracy(descriptor_name, save_plots, show)
        self.plot_confidence_by_correctness(descriptor_name, save_plots, show)
        self.plot_margin_analysis(descriptor_name, save_plots, show)
        self.plot_margin_analysis_byclass(descriptor_name, save_plots, show)
        
        print(f"\n✅ All plots generated for {descriptor_name}")

        # Compute and save single metrics
        print(f"\nComputing single-value metrics...")
        self.print_single_metrics(descriptor_name)
        self.save_single_metrics(descriptor_name)



def quick_analysis(results_dir: str, descriptor_name: str):
    """
    Quick command-line analysis of results
    
    Usage:
        >>> quick_analysis('results/day_night/', 'day_night')
    """
    results_dir = Path(results_dir)
    
    print(f"\n{'='*60}")
    print(f"Quick Analysis: {descriptor_name}")
    print(f"{'='*60}\n")
    
    # Load evaluation metrics
    eval_path = results_dir / 'evaluation_metrics.json'
    if eval_path.exists():
        with open(eval_path, 'r') as f:
            evaluation = json.load(f)
        
        if descriptor_name in evaluation:
            desc_eval = evaluation[descriptor_name]
            
            print(f"Overall Performance:")
            print(f"  • Images: {desc_eval.get('num_images', 0)}")
            
            if 'accuracy' in desc_eval:
                acc = desc_eval['accuracy']
                print(f"  • Accuracy: {acc.get('accuracy', 0):.2%}")
                print(f"  • Correct: {acc.get('num_correct', 0)}/{acc.get('num_total', 0)}")
            
            metrics = desc_eval.get('metrics', {})
            if 'confidence' in metrics:
                conf = metrics['confidence']
                print(f"  • Mean Confidence: {conf.get('mean', 0):.4f} ± {conf.get('std', 0):.4f}")
    
    # Load category accuracy
    cat_acc_path = results_dir / f'{descriptor_name}_category_accuracy.json'
    if cat_acc_path.exists():
        with open(cat_acc_path, 'r') as f:
            category_data = json.load(f)
        
        print(f"\nPer-Category Performance:")
        for category, stats in category_data.items():
            print(f"  • {category.title()}:")
            print(f"      Accuracy: {stats['accuracy']:.2%} ({stats['correct_count']}/{stats['total_count']})")
            print(f"      Confidence: {stats['mean_confidence']:.4f}")
    
    # Misclassification summary
    misclass_path = results_dir / f'{descriptor_name}_misclassifications.csv'
    if misclass_path.exists():
        df = pd.read_csv(misclass_path)
        print(f"\nMisclassifications:")
        print(f"  • Total: {len(df)}")
        if len(df) > 0:
            print(f"  • Mean confidence: {df['confidence'].mean():.4f}")
            print(f"  • Mean margin: {df['margin'].mean():.4f}")
            
            # Most common misclassification pattern
            pattern_counts = df.groupby(['ground_truth', 'predicted']).size()
            if len(pattern_counts) > 0:
                most_common = pattern_counts.idxmax()
                print(f"  • Most common: {most_common[0]} → {most_common[1]} ({pattern_counts.max()} times)")
    
    print(f"\n{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Example: Plot results from a saved evaluation
    results_dir = "results/day_night/"
    descriptor = "day_night"
    
    # Quick text analysis
    quick_analysis(results_dir, descriptor)
    
    # Generate all plots
    plotter = LabeledEvaluationPlotter(results_dir)
    plotter.plot_all(descriptor, save_plots=True, show=False)