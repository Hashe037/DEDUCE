"""
Enhanced Results Saver for Labeled Evaluation
Saves comprehensive data for easy plotting and investigation
"""

import json
import csv
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging


class LabeledEvaluationSaver:
    """Saves detailed results for labeled semantic evaluation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def save_all_results(self, results: Dict[str, Any], save_path: str):
        """
        Save comprehensive results for labeled evaluation
        
        Args:
            results: Dictionary containing:
                - predictions: Dict[str, PredictionResult]
                - evaluation: Dict[str, Any]
                - ground_truth: Dict[str, List[str]] (optional)
                - dataset_size: int
                - image_paths: List[str] (optional)
        """
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Always save evaluation metrics
        self._save_evaluation_json(results.get('evaluation', {}), save_path)
        
        # Always save raw predictions (for reconstruction)
        if 'predictions' in results:
            self._save_predictions_tensor(results['predictions'], save_path)
        
        # If we have ground truth, save detailed labeled results
        if 'ground_truth' in results and results['ground_truth']:
            self._save_labeled_results(results, save_path)
        else:
            self.logger.info("No ground truth provided - skipping labeled analysis")
    
    def _save_evaluation_json(self, evaluation: Dict[str, Any], save_path: str):
        """Save evaluation metrics as JSON"""
        eval_path = Path(save_path) / 'evaluation_metrics.json'
        with open(eval_path, 'w') as f:
            json.dump(evaluation, f, indent=2)
        self.logger.info(f"Saved evaluation metrics to {eval_path}")
    
    def _save_predictions_tensor(self, predictions: Dict[str, Any], save_path: str):
        """Save raw predictions as PyTorch tensors"""
        pred_path = Path(save_path) / 'predictions.pt'
        torch.save(predictions, pred_path)
        self.logger.info(f"Saved predictions tensor to {pred_path}")
    
    def _save_labeled_results(self, results: Dict[str, Any], save_path: str):
        """Save detailed results for labeled evaluation"""
        predictions = results.get('predictions', {})
        ground_truth = results.get('ground_truth', {})
        image_paths = results.get('image_paths', None)
        
        for descriptor_name, pred_result in predictions.items():
            if descriptor_name not in ground_truth:
                self.logger.warning(f"No ground truth for {descriptor_name}, skipping")
                continue
            
            gt_labels = ground_truth[descriptor_name]
            predicted_labels = pred_result.predictions()
            confidences = pred_result.confidence()
            similarities = pred_result.similarities
            categories = pred_result.category_names
            
            # Save all detailed files for this descriptor
            self._save_per_image_csv(
                descriptor_name, gt_labels, predicted_labels, confidences, 
                similarities, categories, image_paths, save_path
            )
            self._save_confusion_matrix(
                descriptor_name, gt_labels, predicted_labels, categories, save_path
            )
            self._save_misclassifications(
                descriptor_name, gt_labels, predicted_labels, confidences, 
                similarities, categories, image_paths, save_path
            )
            self._save_category_accuracy(
                descriptor_name, gt_labels, predicted_labels, confidences,
                categories, save_path
            )
    
    def _save_per_image_csv(self, descriptor_name: str, gt_labels: List[str],
                           predicted_labels: List[str], confidences: List[float],
                           similarities: torch.Tensor, categories: List[str],
                           image_paths: List[str], save_path: str):
        """
        Save per-image results in CSV format for easy analysis
        
        CSV columns: image_idx, [image_path], ground_truth, predicted, 
                    confidence, correct, margin, similarity_spread, sim_{category1}, sim_{category2}, ...
        """
        csv_path = Path(save_path) / f'{descriptor_name}_per_image_results.csv'
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header row
            header = ['image_idx']
            if image_paths:
                header.append('image_path')
            header.extend(['ground_truth', 'predicted', 'confidence', 
                         'correct', 'margin', 'similarity_spread'])
            # Add similarity columns for each category
            header.extend([f'sim_{cat}' for cat in categories])
            writer.writerow(header)
            
            # Data rows
            for i, (gt, pred, conf) in enumerate(zip(gt_labels, predicted_labels, confidences)):
                # Calculate margin (difference between top-2 similarities)
                sorted_sims, _ = torch.sort(similarities[i], descending=True)
                margin = (sorted_sims[0] - sorted_sims[1]).item() if len(sorted_sims) > 1 else 0.0
                
                # Calculate spread (std of all similarities)
                spread = similarities[i].std().item()
                
                correct = (gt == pred)
                
                row = [i]
                if image_paths:
                    row.append(image_paths[i] if i < len(image_paths) else 'unknown')
                row.extend([gt, pred, f'{conf:.4f}', correct, f'{margin:.4f}', f'{spread:.4f}'])
                
                # Add all similarity scores
                row.extend([f'{similarities[i][j].item():.6f}' for j in range(len(categories))])
                
                writer.writerow(row)
        
        self.logger.info(f"Saved per-image results to {csv_path}")
    
    def _save_confusion_matrix(self, descriptor_name: str, gt_labels: List[str],
                              predicted_labels: List[str], categories: List[str],
                              save_path: str):
        """
        Save confusion matrix as JSON and CSV
        
        JSON format: Easy to load programmatically
        CSV format: Easy to view in spreadsheet
        """
        confusion = self._compute_confusion_matrix(gt_labels, predicted_labels, categories)
        
        # Save as JSON
        json_path = Path(save_path) / f'{descriptor_name}_confusion_matrix.json'
        confusion_data = {
            'categories': categories,
            'matrix': confusion.tolist(),
            'row_labels': 'ground_truth',
            'col_labels': 'predicted',
            'description': 'Rows = ground truth, Columns = predicted'
        }
        with open(json_path, 'w') as f:
            json.dump(confusion_data, f, indent=2)
        
        # Save as CSV (human-readable)
        csv_path = Path(save_path) / f'{descriptor_name}_confusion_matrix.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header row
            writer.writerow(['Ground Truth \\ Predicted'] + categories)
            
            # Data rows
            for i, category in enumerate(categories):
                writer.writerow([category] + confusion[i].tolist())
        
        self.logger.info(f"Saved confusion matrix to {json_path} and {csv_path}")
    
    def _save_misclassifications(self, descriptor_name: str, gt_labels: List[str],
                                predicted_labels: List[str], confidences: List[float],
                                similarities: torch.Tensor, categories: List[str],
                                image_paths: List[str], save_path: str):
        """
        Save detailed misclassification analysis
        
        CSV includes all similarity scores for each misclassified image
        """
        csv_path = Path(save_path) / f'{descriptor_name}_misclassifications.csv'
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['image_idx']
            if image_paths:
                header.append('image_path')
            header.extend(['ground_truth', 'predicted', 'confidence', 'margin'])
            # Add similarity columns for each category
            header.extend([f'sim_{cat}' for cat in categories])
            writer.writerow(header)
            
            # Only write misclassified images
            for i, (gt, pred, conf) in enumerate(zip(gt_labels, predicted_labels, confidences)):
                if gt != pred:  # Misclassification
                    sorted_sims, _ = torch.sort(similarities[i], descending=True)
                    margin = (sorted_sims[0] - sorted_sims[1]).item() if len(sorted_sims) > 1 else 0.0
                    
                    row = [i]
                    if image_paths:
                        row.append(image_paths[i] if i < len(image_paths) else 'unknown')
                    row.extend([gt, pred, f'{conf:.4f}', f'{margin:.4f}'])
                    
                    # Add all similarity scores
                    row.extend([f'{similarities[i][j].item():.4f}' 
                               for j in range(len(categories))])
                    
                    writer.writerow(row)
        
        self.logger.info(f"Saved misclassification details to {csv_path}")
    
    def _save_category_accuracy(self, descriptor_name: str, gt_labels: List[str],
                               predicted_labels: List[str], confidences: List[float],
                               categories: List[str], save_path: str):
        """
        Save per-category accuracy and confidence statistics
        
        Useful for identifying which categories are hardest to classify
        """
        category_stats = {}
        
        for category in categories:
            # Get all images with this ground truth
            category_mask = [gt == category for gt in gt_labels]
            category_count = sum(category_mask)
            
            if category_count > 0:
                # Calculate accuracy for this category
                category_correct = sum([
                    gt == pred 
                    for gt, pred, mask in zip(gt_labels, predicted_labels, category_mask) 
                    if mask
                ])
                
                # Get confidences for this category
                category_confidences = [
                    conf 
                    for conf, mask in zip(confidences, category_mask) 
                    if mask
                ]
                
                # Get confidences for correct predictions
                correct_confidences = [
                    conf 
                    for gt, pred, conf, mask in zip(gt_labels, predicted_labels, 
                                                    confidences, category_mask)
                    if mask and gt == pred
                ]
                
                # Get confidences for incorrect predictions
                incorrect_confidences = [
                    conf 
                    for gt, pred, conf, mask in zip(gt_labels, predicted_labels,
                                                    confidences, category_mask)
                    if mask and gt != pred
                ]
                
                category_stats[category] = {
                    'total_count': category_count,
                    'correct_count': category_correct,
                    'incorrect_count': category_count - category_correct,
                    'accuracy': category_correct / category_count,
                    'mean_confidence': np.mean(category_confidences),
                    'std_confidence': np.std(category_confidences),
                    'mean_confidence_correct': np.mean(correct_confidences) if correct_confidences else 0.0,
                    'mean_confidence_incorrect': np.mean(incorrect_confidences) if incorrect_confidences else 0.0
                }
        
        # Save as JSON
        json_path = Path(save_path) / f'{descriptor_name}_category_accuracy.json'
        with open(json_path, 'w') as f:
            json.dump(category_stats, f, indent=2)
        
        # Save as CSV for easy viewing
        csv_path = Path(save_path) / f'{descriptor_name}_category_accuracy.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['category', 'total', 'correct', 'incorrect', 
                           'accuracy', 'mean_conf', 'std_conf', 
                           'conf_correct', 'conf_incorrect'])
            
            for category, stats in category_stats.items():
                writer.writerow([
                    category,
                    stats['total_count'],
                    stats['correct_count'],
                    stats['incorrect_count'],
                    f"{stats['accuracy']:.4f}",
                    f"{stats['mean_confidence']:.4f}",
                    f"{stats['std_confidence']:.4f}",
                    f"{stats['mean_confidence_correct']:.4f}",
                    f"{stats['mean_confidence_incorrect']:.4f}"
                ])
        
        self.logger.info(f"Saved category accuracy to {json_path} and {csv_path}")
    
    def _compute_confusion_matrix(self, ground_truth: List[str], 
                                 predictions: List[str], 
                                 categories: List[str]) -> np.ndarray:
        """Compute confusion matrix for classification results"""
        n_categories = len(categories)
        confusion = np.zeros((n_categories, n_categories), dtype=int)
        
        category_to_idx = {cat: i for i, cat in enumerate(categories)}
        
        for gt, pred in zip(ground_truth, predictions):
            gt_idx = category_to_idx.get(gt, -1)
            pred_idx = category_to_idx.get(pred, -1)
            
            if gt_idx >= 0 and pred_idx >= 0:
                confusion[gt_idx, pred_idx] += 1
        
        return confusion


# Example usage function
def save_labeled_evaluation(predictions, evaluation, ground_truth, 
                           save_path, image_paths=None):
    """
    Convenience function to save labeled evaluation results
    
    Args:
        predictions: Dict[str, PredictionResult] - from pipeline.predict()
        evaluation: Dict[str, Any] - from pipeline.evaluate()
        ground_truth: Dict[str, List[str]] - ground truth labels per descriptor
        save_path: str - directory to save results
        image_paths: Optional[List[str]] - paths to images for traceability
    
    Example:
        >>> results = pipeline.predict(dataset)
        >>> evaluation = pipeline.evaluate(results, ground_truth=gt_dict)
        >>> save_labeled_evaluation(
        ...     predictions=results,
        ...     evaluation=evaluation,
        ...     ground_truth=gt_dict,
        ...     save_path='results/day_night/',
        ...     image_paths=dataset.image_paths
        ... )
    """
    saver = LabeledEvaluationSaver()
    
    results_package = {
        'predictions': predictions,
        'evaluation': evaluation,
        'ground_truth': ground_truth,
        'image_paths': image_paths
    }
    
    saver.save_all_results(results_package, save_path)