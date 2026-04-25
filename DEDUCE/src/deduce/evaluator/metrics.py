"""
Simple metrics for zero-shot semantic predictions
"""

import torch
import numpy as np
from typing import Dict, List, Any
from collections import Counter


#TODO: Needs more direction/work

class SemanticMetrics:
    """Simple metrics for zero-shot prediction evaluation"""
    
    def compute_basic_metrics(self, 
                            similarities: torch.Tensor,
                            predicted_categories: List[str],
                            confidence_scores: torch.Tensor,
                            category_names: List[str]) -> Dict[str, Any]:
        """
        Compute essential metrics for zero-shot evaluation
        
        Returns:
            Dictionary with confidence stats, coverage, and diversity
        """
        # 1. Confidence statistics
        confidences = confidence_scores.numpy()
        confidence_stats = {
            'mean': float(np.mean(confidences)),
            'std': float(np.std(confidences)),
            'min': float(np.min(confidences)),
            'max': float(np.max(confidences))
        }

        # Category-wise confidence statistics
        category_confidences = {}
        for category in category_names:
            # Find images predicted as this category
            category_indices = [i for i, pred in enumerate(predicted_categories) if pred == category]
            
            if category_indices:
                category_confidence_values = [confidence_scores[i].item() for i in category_indices]
                category_confidences[category] = {
                    'mean': float(np.mean(category_confidence_values)),
                    'std': float(np.std(category_confidence_values)),
                    'count': len(category_confidence_values)
                }
            else:
                category_confidences[category] = {
                    'mean': 0.0,
                    'std': 0.0, 
                    'count': 0
                }
        
        # 2. Category coverage - how many categories are actually used
        used_categories = set(predicted_categories)
        coverage = {
            'used_categories': len(used_categories),
            'total_categories': len(category_names),
            'coverage_ratio': len(used_categories) / len(category_names),
            'unused_categories': [cat for cat in category_names if cat not in used_categories]
        }
        
        # 3. Prediction distribution - are predictions diverse or concentrated?
        category_counts = Counter(predicted_categories)

        # Ensure all categories are represented (even with 0 counts)
        full_category_counts = {cat: category_counts.get(cat, 0) for cat in category_names}
        
        most_common = category_counts.most_common(1)[0] if category_counts else ('none', 0)
        distribution = {
            'most_common_category': most_common[0],
            'most_common_percentage': (most_common[1] / len(predicted_categories)) * 100,
            'num_unique_predictions': len(category_counts),
            'category_counts': full_category_counts  # Add this line
        }

        # 4. Compute margins (separation between top choices)
        sorted_sims, _ = torch.sort(similarities, dim=1, descending=True)
        top_sims = sorted_sims[:, 0]  # Max similarities (confidence)
        second_sims = sorted_sims[:, 1]  # Second-best similarities
        
        margins = (top_sims - second_sims).numpy()
        relative_margins = ((top_sims - second_sims) / top_sims).numpy()
        
        margin_stats = {
            'mean': float(np.mean(margins)),
            'std': float(np.std(margins)),
            'min': float(np.min(margins)),
            'max': float(np.max(margins))
        }
        
        relative_margin_stats = {
            'mean': float(np.mean(relative_margins)),
            'std': float(np.std(relative_margins)),
            'min': float(np.min(relative_margins)),
            'max': float(np.max(relative_margins))
        }

        # Category-wise margin statistics
        category_margins = {}
        for category in category_names:
            # Find images predicted as this category
            category_indices = [i for i, pred in enumerate(predicted_categories) if pred == category]
            
            if category_indices:
                category_margin_values = [relative_margins[i] for i in category_indices]
                category_margins[category] = float(np.mean(category_margin_values))
            else:
                category_margins[category] = 0.0  # No predictions for this category
        
        return {
            'confidence': confidence_stats,
            'category_confidences': category_confidences,
            'margin': margin_stats,
            'relative_margin': relative_margin_stats,
            'category_margins': category_margins,
            'coverage': coverage,
            'distribution': distribution
        }
    
    def compute_supervised_accuracy(self, 
                                  predicted_categories: List[str],
                                  ground_truth_categories: List[str]) -> Dict[str, float]:
        """
        Simple accuracy when ground truth is available
        """
        if len(predicted_categories) != len(ground_truth_categories):
            raise ValueError("Predictions and ground truth must have same length")
        
        correct = sum(p == g for p, g in zip(predicted_categories, ground_truth_categories))
        accuracy = correct / len(predicted_categories)
        
        return {
            'accuracy': accuracy,
            'num_correct': correct,
            'num_total': len(predicted_categories)
        }