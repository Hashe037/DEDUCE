import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Dict, List, Tuple, Optional, Callable, Union, Any
import torch
import numpy as np

import pdb
class SemanticDataset(Dataset):
    """
    Simple dataset for loading images from a directory or multiple directories
    
    Supports:
        - Automatic image size detection from config
        - Multiple data paths
        - Flexible image preprocessing
        - Memory-efficient loading
        - Optional metadata
    """
    
    def __init__(self, data_path: Union[str, List[str]], config: Dict[str, Any] = None, transform=None):
        # Handle single path or multiple paths
        if isinstance(data_path, str):
            self.data_path = data_path
            self.data_paths = [data_path]
        else:
            self.data_path = data_path[0]  # Keep for backward compatibility
            self.data_paths = data_path
            
        self.config = config or {}
        self.image_paths = self._load_image_paths()

        # Set to labels as zeros but it can be overridden
        self.labels = np.zeros_like(self.image_paths)

        # Get image size from config (auto-detected by encoder registry)
        self.image_size = self.config.get('image_size', 224)
        
        # Use provided transform or default to basic PIL->tensor conversion
        if transform is not None:
            self.transform = transform
        else:
            transform_list = []

            # Resize for encoders
            transform_list.extend([
                transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ])

            # Optional: Add normalization if specified in config
            if self.config.get('normalize', False):
                # ImageNet normalization (standard for most vision models)
                transform_list.append(
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                )
            # Instantiate transform
            self.transform = transforms.Compose(transform_list)
        
    def __len__(self) -> int:
        return len(self.image_paths)
        
    def __getitem__(self, idx: int) -> Union[Image.Image, Tuple[Image.Image, Dict[str, Any]]]:
        """Load and return PIL image, optionally with metadata"""
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        
        # Get metadata if method exists (will be overridden in subclasses)
        metadata = self._get_metadata(idx)
        
        # if metadata is not None:
            # return image, metadata
        return image, 0, {'filename':self.image_paths[idx]}  #label as 0, metadata is a dictionary
        
    def _get_metadata(self, idx: int) -> Union[Dict[str, Any], None]:
        """
        Get metadata for an item. Override in subclasses to provide metadata.
        Base implementation returns None.
        """
        return None
    
    def get_label_name(self, idx: Union[int, torch.Tensor]) -> Union[str, List[str]]:
        """Convert numerical label(s) to class name(s)"""
        return "background"
    
    def get_labels(self):
        return self.labels
        
    def _load_image_paths(self) -> List[str]:
        """Find all image files in directory(ies)"""
        extensions = self.config.get('image_extensions', ['.jpg', '.png', '.jpeg'])

        # Handle case where config returns comma-separated string instead of list
        if isinstance(extensions, str):
            extensions = [ext.strip() for ext in extensions.split(',')]

        image_paths = []
        for data_path in self.data_paths:
            for root, _, files in os.walk(data_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in extensions):
                        image_paths.append(os.path.join(root, file))
        
        return sorted(image_paths)


class SemanticDatasetFL(SemanticDataset):
    """
    Dataset for loading images where subfolder names are class labels.
    Inherits image loading and preprocessing from SemanticDataset.
    
    Expected directory structure:
        data_path/
            class1/
                image1.jpg
                image2.jpg
            class2/
                image3.jpg
                image4.jpg
    """
    
    def __init__(self, data_path: Union[str, List[str]], config: Dict[str, Any] = None, transform=None):
        # Initialize parent class first
        super().__init__(data_path, config, transform)
        
        # Now override image_paths with labeled version and extract labels
        self.image_paths, self.labels = self._load_image_paths_and_labels()
        
        # Create label mappings
        self.class_names = sorted(list(set(self.labels)))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {idx: cls_name for cls_name, idx in self.class_to_idx.items()}
        
        # Convert string labels to indices
        self.label_indices = [self.class_to_idx[label] for label in self.labels]
        
    def __getitem__(self, idx: int):
        """Load and return image with its label index, optionally with metadata"""
        # Get image (and possibly metadata) from parent
        result = super().__getitem__(idx)
        
        # Check if parent returned metadata
        # if isinstance(result, tuple):
        #     image, metadata = result
        # else:
        #     image = result
        #     metadata = None
        
        label = self.label_indices[idx]
        
        # if metadata is not None:
        return result[0], label, result[2]
        # return image, label
    
    def _get_metadata(self, idx: int) -> Union[Dict[str, Any], None]:
        """
        Get metadata for an item. Override to provide custom metadata.
        Default implementation includes label information.
        """
        # Base metadata with label info
        metadata = {
            'image_path': self.image_paths[idx],
            'label_name': self.labels[idx],
            'label_idx': self.label_indices[idx]
        }
        return metadata
    
    def get_label_name(self, idx: Union[int, torch.Tensor]) -> Union[str, List[str]]:
        """Convert numerical label(s) to class name(s)"""
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        
        if isinstance(idx, list):
            return [self.idx_to_class[i] for i in idx]
        return self.idx_to_class[idx]
    
    def get_label_idx(self, name: Union[str, List[str]]) -> Union[int, List[int]]:
        """Convert class name(s) to numerical label(s)"""
        if isinstance(name, list):
            return [self.class_to_idx[n] for n in name]
        return self.class_to_idx[name]
        
    def _load_image_paths_and_labels(self) -> Tuple[List[str], List[str]]:
        """Find all image files and extract labels from folder names"""
        extensions = self.config.get('image_extensions', ['.jpg', '.png', '.jpeg'])

        if isinstance(extensions, str):
            extensions = [ext.strip() for ext in extensions.split(',')]

        image_paths = []
        labels = []
        
        for data_path in self.data_paths:
            # Only look at immediate subdirectories as class folders
            class_folders = [d for d in os.listdir(data_path) 
                           if os.path.isdir(os.path.join(data_path, d))]
            
            for class_folder in class_folders:
                class_path = os.path.join(data_path, class_folder)
                
                # Find all images in this class folder (including subdirectories)
                for root, _, files in os.walk(class_path):
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in extensions):
                            image_paths.append(os.path.join(root, file))
                            labels.append(class_folder)
        
        return image_paths, labels
    
    def get_num_classes(self) -> int:
        """Return number of classes"""
        return len(self.class_names)
    

def create_labeled_dataset(folders: Dict[str, str], config: Dict, transform=None):
    """
    Create a labeled dataset - MINIMAL VERSION
    
    Args:
        folders: {'/path/to/sunny': 'sunny', '/path/to/rainy': 'rainy'}
        config: Config dict with image_extensions
        transform: Optional transform
        
    Returns:
        SemanticDataset with .labels attribute
    """
    
    # Get image extensions
    extensions = config.get('image_extensions', ['.jpg', '.png', '.jpeg'])
    if isinstance(extensions, str):
        extensions = [ext.strip() for ext in extensions.split(',')]
    
    # Collect all image paths and their labels
    paths = []
    labels = []
    
    for folder, label in folders.items():
        for img_path in Path(folder).rglob('*'):
            if img_path.suffix.lower() in extensions:
                paths.append(str(img_path))
                labels.append(label)
    
    if not paths:
        raise ValueError(f"No images found in: {list(folders.keys())}")
    
    # Create SemanticDataset with dummy path
    dataset = SemanticDataset(
        data_path=[list(folders.keys())[0]],
        config=config,
        transform=transform
    )
    
    # Override with our paths and add labels
    dataset.image_paths = paths
    dataset.labels = labels
    
    return dataset



"""
Custom collate function for datasets returning (image, label, metadata)
"""



def semantic_collate_fn(batch: List[Tuple[Any, ...]]) -> Tuple[torch.Tensor, ...]:
    """
    Custom collate function for semantic dataset that returns (image, label, metadata)
    
    Args:
        batch: List of tuples from dataset __getitem__
        
    Returns:
        Tuple of (images, labels, metadata_list) or just (images,) if no labels/metadata
    """
    # Check what the dataset is returning
    if not batch:
        return torch.tensor([])
    
    first_item = batch[0]
    
    # Case 1: Dataset returns just images (tensor)
    if isinstance(first_item, torch.Tensor):
        return torch.stack(batch)
    
    # Case 2: Dataset returns tuples (image, label, metadata) or (image, label)
    if isinstance(first_item, (tuple, list)):
        num_elements = len(first_item)
        
        if num_elements == 1:
            # Just images wrapped in tuple
            images = torch.stack([item[0] for item in batch])
            return (images,)
        
        elif num_elements == 2:
            # (image, label)
            images = torch.stack([item[0] for item in batch])
            labels = [item[1] for item in batch]
            
            # Try to stack labels if they're tensors
            if isinstance(labels[0], torch.Tensor):
                labels = torch.stack(labels)
            
            return (images, labels)
        
        elif num_elements >= 3:
            # (image, label, metadata, ...)
            images = torch.stack([item[0] for item in batch])
            labels = [item[1] for item in batch]
            metadata = [item[2] for item in batch]
            
            # Try to stack labels if they're tensors
            if isinstance(labels[0], torch.Tensor):
                labels = torch.stack(labels)
            
            # Keep metadata as list (it's typically dicts or strings)
            return (images, labels, metadata)
    
    # Fallback: return as-is
    return batch