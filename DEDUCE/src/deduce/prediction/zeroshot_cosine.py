import torch
import torch.nn.functional as F
from dataclasses import dataclass
import tqdm
from typing import Dict, List, Tuple, Any, Optional

from base import BasePredictor, PredictionResult

class ZeroShotPredictor(BasePredictor):
    """
    Simple zero-shot predictor
    
    Computes similarities between image embeddings and text embeddings,
    returns the most similar category for each image.
    """
        
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
        all_image_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding images"):
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                    
                images = images.to(self.device)
                
                # Encode and normalize
                image_embeddings = self.image_encoder.encode(images)
                image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
                
                all_image_embeddings.append(image_embeddings.cpu())
        
        # Combine all embeddings
        all_image_embeddings = torch.cat(all_image_embeddings, dim=0).to(self.device)
        self.logger.info(f"Encoded {all_image_embeddings.shape[0]} images")
        
        # Predict for each semantic descriptor
        results = {}
        
        for descriptor in self.semantic_descriptors:
            text_data = self.text_embeddings[descriptor.name]
            text_embeddings = text_data['embeddings']
            category_names = text_data['categories']
            
            # Compute similarities (cosine similarity = dot product of normalized vectors)
            similarities = torch.matmul(all_image_embeddings, text_embeddings.T)
            
            # Get predictions (highest similarity)
            predicted_indices = torch.argmax(similarities, dim=1)
            predicted_categories = [category_names[idx] for idx in predicted_indices]
            confidence_scores = torch.max(similarities, dim=1)[0]
            
            results[descriptor.name] = PredictionResult(
                similarities=similarities.cpu(),
                predicted_categories=predicted_categories,
                confidence_scores=confidence_scores.cpu(),
                category_names=category_names
            )
            
        self.logger.info("Prediction completed")
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