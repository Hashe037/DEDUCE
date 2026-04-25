from typing import Dict, List, Any
import torch
import logging
from .metrics import SemanticMetrics

#TODO: Class perhaps not needed, probably only one evaluator

class BaseEvaluator:
    """Handles evaluation of semantic predictions"""
    
    def __init__(self, metrics: List[str] = None):
        self.metrics = SemanticMetrics()
        self.logger = logging.getLogger(__name__)
        
    def evaluate_predictions(self, 
                           predictions: Dict[str, torch.Tensor],
                           ground_truth: Dict[str, Any] = None) -> Dict[str, float]:
        """Evaluate predictions against ground truth or using unsupervised metrics"""
        pass
        
    def compute_dataset_completeness(self, 
                                   dataset: Any,
                                   predictions: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Analyze dataset completeness based on predictions"""
        pass