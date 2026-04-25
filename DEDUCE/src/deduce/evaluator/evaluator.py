
"""
Simple evaluator for semantic predictions
"""

from typing import Dict, List, Any, Optional
import torch
import logging
from .metrics import SemanticMetrics
from ..prediction.base import PredictionResult
from .base import BaseEvaluator
import json
import os
import pdb

class Evaluator(BaseEvaluator):
    """Simple evaluator for zero-shot semantic predictions"""
    
    def __init__(self):
        self.metrics = SemanticMetrics()
        self.logger = logging.getLogger(__name__)
        
    def evaluate_predictions(self, 
                        predictions: Dict[str, PredictionResult],
                        ground_truth: Optional[Dict[str, Any]] = None,
                        dataset: Optional[Any] = None) -> Dict[str, Any]:
        """
        Evaluate predictions with simple, useful metrics
        
        Args:
            predictions: Dictionary mapping descriptor names to PredictionResult objects
            ground_truth: Optional ground truth labels for accuracy computation
            dataset: Optional dataset to extract labels from
            
        Returns:
            Simple evaluation results
        """
        self.logger.info(f"Evaluating {len(predictions)} descriptors...")
        
        results = {}

        #TODO: To extract ground truth
        # if dataset is not None and hasattr(dataset, 'label_indices'):
            # Map label indices to descriptor categories if possible
            # ground_truth = self._extract_ground_truth_from_dataset(dataset, predictions)
        
        for descriptor_name, prediction_result in predictions.items():
            # Basic metrics - use methods instead of attributes
            basic_metrics = self.metrics.compute_basic_metrics(
                similarities=prediction_result.similarities,
                predicted_categories=prediction_result.predictions(),  
                confidence_scores=torch.tensor(prediction_result.confidence()), 
                category_names=prediction_result.category_names
            )
            
            # pdb.set_trace()
            descriptor_results = {
                'num_images': len(prediction_result.predictions()),  
                'num_categories': len(prediction_result.category_names),
                'metrics': basic_metrics,
                'filename_margin_data': prediction_result.get_predictions_with_margins()
            }
            
            # Add accuracy if ground truth available
            if ground_truth and descriptor_name in ground_truth:
                try:
                    accuracy_metrics = self.metrics.compute_supervised_accuracy(
                        predicted_categories=prediction_result.predictions(), 
                        ground_truth_categories=ground_truth[descriptor_name]
                    )
                    descriptor_results['accuracy'] = accuracy_metrics
                except Exception as e:
                    self.logger.warning(f"Could not compute accuracy for {descriptor_name}: {e}")
            
            results[descriptor_name] = descriptor_results
        
        # Simple overall summary
        all_confidences = [r['metrics']['confidence']['mean'] for r in results.values()]
        all_coverages = [r['metrics']['coverage']['coverage_ratio'] for r in results.values()]
        
        results['summary'] = {
            'mean_confidence': sum(all_confidences) / len(all_confidences) if all_confidences else 0,
            'mean_coverage': sum(all_coverages) / len(all_coverages) if all_coverages else 0,
            'num_descriptors': len(predictions)
        }
        
        return results

    
    def print_summary(self, evaluation_results: Dict[str, Any]) -> None:
        """Print a simple summary of results"""
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        
        summary = evaluation_results.get('summary', {})
        print(f"Overall Confidence: {summary.get('mean_confidence', 0):.3f}")
        print(f"Overall Coverage:   {summary.get('mean_coverage', 0):.3f}")
        print(f"Descriptors:        {summary.get('num_descriptors', 0)}")
        
        print(f"\nPer-Descriptor Results:")
        print("-" * 30)
        
        for desc_name, results in evaluation_results.items():
            if desc_name == 'summary':
                continue
                
            metrics = results.get('metrics', {})
            confidence = metrics.get('confidence', {})
            coverage = metrics.get('coverage', {})
            distribution = metrics.get('distribution', {})
            accuracy = results.get('accuracy', {})
            
            print(f"\n{desc_name.upper()}:")
            print(f"  Images: {results.get('num_images', 0)}")
            print(f"  Confidence: {confidence.get('mean', 0):.3f} ± {confidence.get('std', 0):.3f}")
            print(f"  Coverage: {coverage.get('used_categories', 0)}/{coverage.get('total_categories', 0)} categories")
            print(f"  Most common: {distribution.get('most_common_category', 'none')} ({distribution.get('most_common_percentage', 0):.1f}%)")
            
            if accuracy:
                print(f"  Accuracy: {accuracy.get('accuracy', 0):.3f}")
        
        print("\n" + "="*50)