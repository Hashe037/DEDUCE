"""
Optimized Dataset Semantic Evaluation Pipeline
Streamlined version with reduced complexity while maintaining separation of concerns
"""
import pdb

from typing import Dict, List, Any, Optional, Union
import os
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from .core.config import ConfigManager
from .core.dataset import SemanticDataset, SemanticDatasetFL, semantic_collate_fn, create_labeled_dataset
from .encoders.registry import EncoderRegistry
from .semantic_descriptors.registry import SemanticRegistry
from .prediction.base import BasePredictor
from .evaluator.evaluator import Evaluator
from .utils.visualizations import ResultsVisualizer

class DatasetEvaluationPipeline:
    """
    Class that contains the main end-to-end pipeline components for the dataset evaluation
    
    This pipeline automatically classifies images across multiple semantic dimensions (time of day,
    weather, lighting, object pose, etc.) without requiring labeled training data. It uses pre-trained
    encoders like CLIP to compute similarities between images and semantic text descriptions.
    
    Key Features:
    - Zero-shot classification: No training data required
    - Multi-dimensional analysis: Evaluate multiple semantic aspects simultaneously
    - Configurable descriptors: Easily add custom semantic categories
    - Built-in evaluation: Confidence scores, coverage analysis, and distribution metrics
    - Automatic visualization: Generate plots and summaries of results
    
    Main Workflow:
    1. Load images from directory → SemanticDataset (load_dataset())
    2. Encode images and semantic descriptions → embeddings (predict())
    3. Compute similarity scores → predictions per semantic dimension (predict())
    4. Evaluate prediction quality → confidence and coverage metrics (evaluate())
    5. Generate visualizations → distribution plots and summaries (visualize())
    
    Complete pipeline execution available via run() method which orchestrates all steps.
    """
    
    #--------------------------------------------------------------------
    # Setup code
    #--------------------------------------------------------------------

    def __init__(self, config_path: str, device: Optional[str] = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.logger = self._setup_logging()
        
        # Initialize core components
        self.encoder_registry = EncoderRegistry()
        self.semantic_registry = SemanticRegistry()
        
        # Load components based on config
        self._setup_encoders()
        self._setup_descriptors()
        self._setup_pipeline_components()
        
    def _setup_logging(self) -> logging.Logger:
        """Minimal logging setup"""
        log_config = self.config_manager.get('LOGGING', {})
        log_level = log_config.get('level', 'INFO')
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _setup_encoders(self):
        """Initialize encoders from config"""
        encoder_config = self.config_manager.get('ENCODERS', {})
        
        # Create encoder configs with defaults
        base_config = {
            'model_name': encoder_config.get('model_name', 'ViT-B/32'),
            'pretrained': encoder_config.get('pretrained', None),
            'image_size': encoder_config.get('image_size', None),
            'device': self.device
        }
        
        self.image_encoder = self.encoder_registry.create_image_encoder(
            encoder_config.get('image_encoder', 'clip'), base_config
        )
        self.text_encoder = self.encoder_registry.create_text_encoder(
            encoder_config.get('text_encoder', 'clip'), base_config
        )
    
    def _setup_descriptors(self):
        """Initialize semantic descriptors from config"""
        semantics_config = self.config_manager.get('SEMANTICS', {})
        descriptor_names = semantics_config.get('descriptors', '').split(',')
        descriptor_names = [name.strip() for name in descriptor_names if name.strip()]
        
        # Extract global semantic context
        global_semantics = {
            'part_a_object': semantics_config.get('part_a_object'),
            'part_b_scene': semantics_config.get('part_b_scene'),
            'part_d_additional': semantics_config.get('part_d_additional')
        }
        
        self.semantic_descriptors = []
        for descriptor_name in descriptor_names:
            # Get descriptor-specific config section
            descriptor_config = self.config_manager.get(descriptor_name.upper(), {})
            
            # Create descriptor instance with global context
            descriptor = self.semantic_registry.create_descriptor(
                descriptor_name, descriptor_config, global_semantics
            )
            self.semantic_descriptors.append(descriptor)
            
        self.logger.info(f"Loaded semantic descriptors: {[d.name for d in self.semantic_descriptors]}")
        
        # Log global context if any
        if any(global_semantics.values()):
            self.logger.info(f"Global semantic context: {global_semantics}")
    
    def _setup_pipeline_components(self):
        """Initialize pipeline components"""
        self.predictor = BasePredictor(
            self.image_encoder, self.text_encoder, self.semantic_descriptors
        )
        self.evaluator = Evaluator()
        self.visualizer = ResultsVisualizer()


    #--------------------------------------------------------------------
    # Main pipeline methods
    #--------------------------------------------------------------------
    
    def load_dataset(self, data_path: Union[str, List[str]] = None, **kwargs) -> SemanticDataset:
        """Load dataset with optional config fallback"""

        # Check for labeled folders in config
        # labeled_folders = self.config_manager.get_labeled_folders()

        # if labeled_folders:
        #     # Load as labeled dataset
            
        #     loader = LabeledDataLoader(
        #         folders=labeled_folders,
        #         batch_size=self.config_manager.get_int('DATASET', 'batch_size', 32),
        #         shuffle=False
        #     )
            
        #     self._is_labeled = True
        #     self.logger.info(f"Loaded labeled dataset: {loader.info()}")
        #     return loader
        labeled_folders = self.config_manager.get_labeled_folders()

        if data_path is None:
            # Try to get from config as fallback
            data_path = self.config_manager.get_dataset_paths()
            if data_path is None and labeled_folders is None:
                raise ValueError("No dataset path provided and none found in config")
    
        if labeled_folders:
            # Labeled dataset
            self.logger.info(f"Loading labeled dataset") 
            dataset = create_labeled_dataset(
                folders=labeled_folders,
                config=self.config_manager.get('DATASET', {}),
                transform=kwargs.get('transform')
            )
            self.logger.info(f"Dataset loaded: {len(dataset)} images found")
        else:
            # Unlabeled dataset
            self.logger.info(f"Loading dataset from: {data_path}") 
            dataset = SemanticDataset(data_path or self.config_manager.get_dataset_paths(), kwargs)
            self.logger.info(f"Dataset loaded: {len(dataset)} images found")

        
        return dataset

    
    def predict(self, dataset: Union[SemanticDataset, str, None] = None) -> Dict[str, Any]:
        """Run prediction with automatic dataset loading"""
        if dataset is None:
            # Use dataset from config
            dataset = self.load_dataset()
        elif isinstance(dataset, str):
            dataset = self.load_dataset(dataset)
        
        # Check if it's already a DataLoader THIS PART COULD PROBABLY BE IMPROVED
        # if isinstance(dataset, DataLoader):
        #     # It's already a DataLoader (LabeledDataLoader) - use directly!
        #     dataloader = dataset
        # else:
        dataloader = DataLoader(
            dataset,
            batch_size=self.config_manager.get_int('DATASET', 'batch_size', 32),
            shuffle=False,
            num_workers=4,
            collate_fn=semantic_collate_fn
        )    
            
        self.predictions = self.predictor.predict(dataloader)  # Store as class variable
        return self.predictions
    
    def evaluate(self, predictions: Dict[str, Any] = None, ground_truth: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate predictions"""
        if predictions is None:
            predictions = self.predictions
        self.evaluation = self.evaluator.evaluate_predictions(predictions, ground_truth)  # Store as class variable
        return self.evaluation
    
    def visualize(self, evaluation_results: Dict[str, Any] = None, dataset: Optional[SemanticDataset] = None, 
                 save_path: Optional[str] = None) -> None:
        """Generate visualizations with auto path creation"""
        if evaluation_results is None:
            evaluation_results = self.evaluation  # Use stored evaluation

        save_path = save_path or self.config_manager.get('EVALUATION', {}).get('output_path', 'results/')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Generate all visualizations at once
        self.visualizer.generate_all_plots(
            evaluation_results,
            save_path=save_path,
            show=False
        )

        # Export distribution results as JSON with descriptor definitions
        from .utils.export import export_distribution_results
        export_distribution_results(
            evaluation_results,
            semantic_descriptors=self.semantic_descriptors,  
            save_path=Path(save_path) / 'distribution_results.json',
            include_raw_counts=False
        )

        # Export filenames and margins 
        from .utils.export import export_filename_margin_results
        export_filename_margin_results(
            evaluation_results,
            save_path=Path(save_path) / 'filename_margins_results.json'
        )
    
    def run(self, data_path: str, save_results: bool = True) -> Dict[str, Any]:
        """Complete pipeline execution in one method"""
        self.logger.info(f"Running pipeline on {data_path}")
        
        # Execute pipeline steps
        dataset = self.load_dataset(data_path)
        predictions = self.predict(dataset)
        evaluation = self.evaluate(predictions)
        
        if save_results:
            results_path = self.config_manager.get('EVALUATION', {}).get('output_path', 'results/')
            self.visualize(predictions, dataset, results_path)
            self._save_results({'predictions': predictions, 'evaluation': evaluation}, results_path)
        
        return {'predictions': predictions, 'evaluation': evaluation, 'dataset_size': len(dataset)}
    
    def _save_results(self, results: Dict[str, Any], save_path: str):
        """Simplified results saving"""
        import json
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save evaluation as JSON
        with open(Path(save_path) / 'evaluation.json', 'w') as f:
            json.dump(results['evaluation'], f, indent=2)
        
        # Save predictions as tensors
        if 'predictions' in results:
            torch.save(results['predictions'], Path(save_path) / 'predictions.pt')
    
    def info(self) -> Dict[str, Any]:
        """Get pipeline configuration summary"""
        return {
            'encoders': {
                'image': type(self.image_encoder).__name__,
                'text': type(self.text_encoder).__name__
            },
            'descriptors': [d.name for d in self.semantic_descriptors],
            'device': self.device,
            'config_sections': list(self.config.keys())
        }


# Usage examples:
if __name__ == "__main__":
    # Config file setup
    pipeline = DatasetEvaluationPipeline('config.ini')
    results = pipeline.run('path/to/images')
    
    # Print summary
    print(f"Processed {results['dataset_size']} images")
    print(f"Pipeline info: {pipeline.info()}")