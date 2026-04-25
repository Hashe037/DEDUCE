import torch
import torch.nn.functional as F
from dataclasses import dataclass
import tqdm
from typing import Dict, List, Tuple, Any, Optional
import pdb

import logging

from ..encoders.base import ImageEncoder, TextEncoder
from ..semantic_descriptors.base import SemanticDescriptor


# @dataclass
# class PredictionResult:
#     """Container for prediction results with metadata"""
#     similarities: torch.Tensor  # Shape: (n_images, n_categories)
#     predicted_categories: List[str]  # Most likely category per image
#     confidence_scores: torch.Tensor  # Max similarity per image
#     category_names: List[str]  # All category names for this descriptor

class PredictionResult:
    """Simple container for prediction results - like a smart list of predictions"""
    
    def __init__(self, similarities: torch.Tensor, category_names: List[str], filenames: Optional[List[str]] = None):
        """
        Args:
            similarities: How well each image matches each category (images x categories)
            category_names: Names of all possible categories ['sunny', 'rainy', etc.]
        """
        self.similarities = similarities
        self.category_names = category_names
        self.num_images = similarities.shape[0]
        self.filenames = filenames or [f"image_{i}" for i in range(self.num_images)]

    def predictions(self) -> List[str]:
        """What category was predicted for each image?"""
        best_indices = torch.argmax(self.similarities, dim=1)
        return [self.category_names[idx] for idx in best_indices]
    
    def confidence(self) -> List[float]:
        """How confident was each prediction? (0.0 to 1.0)"""
        max_scores = torch.max(self.similarities, dim=1)[0]
        return max_scores.tolist()
    
    def top_choices(self, image_idx: int, k: int = 3) -> List[Tuple[str, float]]:
        """Get top K category choices for a specific image"""
        if image_idx >= self.num_images:
            raise IndexError(f"Image {image_idx} not found (only have {self.num_images} images)")
        
        scores, indices = torch.topk(self.similarities[image_idx], k=min(k, len(self.category_names)))
        return [(self.category_names[idx], score.item()) for idx, score in zip(indices, scores)]
    
    def confident_predictions(self, min_confidence: float = 0.8) -> List[Tuple[int, str, float]]:
        """Get only the high-confidence predictions"""
        results = []
        predictions = self.predictions()
        confidences = self.confidence()
        
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            if conf >= min_confidence:
                results.append((i, pred, conf))  # (image_index, category, confidence)
        
        return results
    
    def summary(self) -> Dict[str, Any]:
        """Quick summary of all predictions"""
        predictions = self.predictions()
        confidences = self.confidence()
        
        from collections import Counter
        category_counts = Counter(predictions)
        
        return {
            'total_images': self.num_images,
            'avg_confidence': sum(confidences) / len(confidences),
            'most_common': category_counts.most_common(3),
            'categories_used': len(set(predictions)),
            'total_categories': len(self.category_names)
        }
    
    def get_predictions_with_margins(self) -> List[Dict[str, Any]]:
        """Get filename, predicted category, and margin for each image"""
        results = []
        predictions = self.predictions()
        
        for i in range(self.num_images):
            top_2 = self.top_choices(i, k=2)
            margin = top_2[0][1] - top_2[1][1] if len(top_2) >= 2 else 0.0
            
            results.append({
                'filename': self.filenames[i],
                'prediction': predictions[i],
                'confidence': top_2[0][1],
                'margin': margin,
                'second_choice': top_2[1][0] if len(top_2) >= 2 else None
            })
        
        return results


    def get(self, key: str, default=None):
        """Dictionary-like get method for backward compatibility with visualization code"""
        # Map keys to PredictionResult properties/methods
        if key == 'num_images':
            return self.num_images
        elif key == 'metrics':
            # Return a nested dictionary structure expected by visualization code
            predictions = self.predictions()
            confidences = self.confidence()
            from collections import Counter
            category_counts = Counter(predictions)
            
            # Calculate category-level metrics
            category_confidences = {}
            category_margins = {}
            
            for cat in self.category_names:
                # Get indices where this category was predicted
                cat_indices = [i for i, pred in enumerate(predictions) if pred == cat]
                if cat_indices:
                    cat_confs = [confidences[i] for i in cat_indices]
                    category_confidences[cat] = {
                        'mean': sum(cat_confs) / len(cat_confs),
                        'std': 0.0 if len(cat_confs) == 1 else torch.tensor(cat_confs).std().item(),
                        'count': len(cat_confs)
                    }
                    
                    # Calculate margins (difference between top prediction and second)
                    margins = []
                    for idx in cat_indices:
                        top_2 = self.top_choices(idx, k=2)
                        if len(top_2) >= 2:
                            margins.append(top_2[0][1] - top_2[1][1])
                    
                    # category_margins[cat] = {
                    #     'mean': sum(margins) / len(margins) if margins else 0.0,
                    #     'std': 0.0 if len(margins) <= 1 else torch.tensor(margins).std().item()
                    # }
                    category_margins[cat] = sum(margins) / len(margins) if margins else 0.0


            
            most_common_cat = category_counts.most_common(1)[0] if category_counts else ('unknown', 0)
            
            return {
                'confidence': {
                    'mean': sum(confidences) / len(confidences) if confidences else 0.0,
                    'std': torch.tensor(confidences).std().item() if len(confidences) > 1 else 0.0,
                    'min': min(confidences) if confidences else 0.0,
                    'max': max(confidences) if confidences else 0.0
                },
                'coverage': {
                    'categories_used': len(set(predictions)),
                    'total_categories': len(self.category_names),
                    'coverage_ratio': len(set(predictions)) / len(self.category_names) if self.category_names else 0.0
                },
                'distribution': {
                    'most_common_category': most_common_cat[0],
                    'most_common_count': most_common_cat[1],
                    'most_common_percentage': (most_common_cat[1] / self.num_images * 100) if self.num_images > 0 else 0.0,
                    'category_counts': dict(category_counts)
                },
                'category_confidences': category_confidences,
                'category_margins': category_margins
            }
        else:
            return default

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------



class BasePredictor:
    """
    Simple zero-shot predictor
    
    Computes similarities between image embeddings and text embeddings,
    returns the most similar category for each image.
    """
    
    def __init__(self, 
                 image_encoder: ImageEncoder,
                 text_encoder: TextEncoder,
                 semantic_descriptors: List[SemanticDescriptor],
                 device: Optional[str] = None):
        """
        Initialize the zero-shot predictor
        
        Args:
            image_encoder: Encoder for processing images
            text_encoder: Encoder for processing text descriptions
            semantic_descriptors: List of semantic descriptor instances
            device: Device to run computations on
        """
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.semantic_descriptors = semantic_descriptors
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger = logging.getLogger(__name__)
        
        # Move encoders to device and set to eval mode
        self.image_encoder.to(self.device).eval()
        self.text_encoder.to(self.device).eval()
        
        # Pre-compute text embeddings for all semantic descriptors
        self.text_embeddings = self._precompute_text_embeddings()

    def _precompute_text_embeddings(self):
        """Pre-compute text embeddings for all semantic descriptions"""
        self.logger.info("Pre-computing text embeddings...")
        
        embeddings = {}
        with torch.no_grad():
            for descriptor in self.semantic_descriptors:
                aggregated_descriptions = descriptor.get_aggregated_descriptions()
                # pdb.set_trace()
                category_names = descriptor.get_category_keys()
                
                # Compute embeddings for all variants and average per category
                category_embeddings = []
                
                for category in category_names:
                    variants = aggregated_descriptions[category]
                    
                    # Encode all variants for this category
                    variant_embeds = self.text_encoder.encode(variants)
                    variant_embeds = F.normalize(variant_embeds, p=2, dim=1)
                    
                    # Average embeddings across variants (prompt aggregation)
                    avg_embed = variant_embeds.mean(dim=0, keepdim=True)
                    avg_embed = F.normalize(avg_embed, p=2, dim=1)  # Re-normalize after averaging
                    
                    category_embeddings.append(avg_embed)
                    
                    if len(variants) > 1:
                        self.logger.debug(f"{descriptor.name}/{category}: averaged {len(variants)} prompts")
                
                # Stack into single tensor
                text_embeds = torch.cat(category_embeddings, dim=0)
                
                embeddings[descriptor.name] = {
                    'embeddings': text_embeds.to(self.device),
                    'categories': category_names
                }

        self.logger.info(f"Text embeddings computed for {len(self.semantic_descriptors)} descriptors")
        return embeddings
            
    def _encode_images(self, dataloader) -> torch.Tensor:
        """Encode all images from dataloader and collect filenames"""
        all_embeddings = []
        all_filenames = []

        
        with torch.no_grad():
            for batch in dataloader:
                # Handle different batch formats
                images = batch[0]
                labels = batch[1]  #NOT USED CURRENTLY
                metadata = batch[2]  #NOT USED CURRENTLY
                images = images.to(self.device)

                # pdb.set_trace()

                # Extract filenames from metadata
                if isinstance(metadata, dict) and 'filename' in metadata:
                    all_filenames.extend(metadata['filename'])
                elif isinstance(metadata, list):
                    all_filenames.extend([item['filename'] if isinstance(item, dict) else str(item) 
                                        for item in metadata])
                elif isinstance(metadata, tuple):
                    all_filenames.extend(metadata)
                
                # Encode and normalize
                embeddings = self.image_encoder.encode(images)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                all_embeddings.append(embeddings.cpu())
        
        return torch.cat(all_embeddings, dim=0).to(self.device), all_filenames
        
    def predict(self, dataloader: Any) -> Dict[str, PredictionResult]:
        """
        Perform zero-shot prediction on a dataset
        
        Args:
            dataloader: PyTorch DataLoader containing images
            
        Returns:
            Dictionary mapping descriptor names to PredictionResult objects
        """
        self.logger.info("Running zero-shot prediction...")
        
        # Encode all images
        image_embeddings, filenames = self._encode_images(dataloader)
        
        # Predict for each descriptor
        results = {}
        for descriptor in self.semantic_descriptors:
            text_data = self.text_embeddings[descriptor.name]
            
            # Compute similarities (cosine = dot product of normalized vectors)
            # similarities = torch.matmul(image_embeddings, text_data['embeddings'].T)
            similarities = self.compute_similarities(image_embeddings, text_data['embeddings'])

            results[descriptor.name] = PredictionResult(
                similarities=similarities.cpu(),
                category_names=text_data['categories'],
                filenames=filenames
            )
            
        return results
    
    
        
    def compute_similarities(self, 
                           image_embeddings: torch.Tensor, 
                           text_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarities between image and text embeddings
        
        Args:
            image_embeddings: Shape (n_images, embedding_dim)
            text_embeddings: Shape (n_categories, embedding_dim)
            
        Returns:
            Similarity matrix of shape (n_images, n_categories)
        """
        # Both embeddings should already be normalized
        # Cosine similarity = dot product of normalized vectors
        return torch.matmul(image_embeddings, text_embeddings.T)