Dataset Enrichment through Discovery of Underrepresented Conditions and Environments (DEDUCE)
======================================================================


This document outlines the framework architecture and design patterns for DEDUCE, a pipline for zero-shot semantic factor identification of image datasets and evaluating synthetic data embeddings.


Code Organization
=================

.. code-block::

  deduce/
  ├── core/                        # Configuration, dataset handling, embedding model base
  ├── encoders/                    # Vision-language model interfaces
  │   ├── image/                   # Image encoder implementations (openclip)
  │   └── text/                    # Text encoder implementations (openclip)
  ├── semantic_descriptors/        # Semantic category definitions
  │   └── semantic_classes/        # Built-in descriptor implementations
  ├── prediction/                  # Zero-shot classification logic
  ├── evaluator/                   # Metrics and analysis
  ├── utils/                       # Logging, visualization, export, similarity analysis, labeled eval plotting
  └── dataset_evaluation_pipeline.py  # Main orchestrator


Framework Overview
==================

The framework implements a modular pipeline for zero-shot semantic classification using multi-modal embeddings:

.. code-block::

   Dataset → Image Encoder → Embeddings → Zero-Shot Prediction → Evaluation
   Config  → Text Encoder  →            →                      →

.. code-block::

  DatasetEvaluationPipeline (Main Interface)
  ├── ConfigManager (Configuration)
  ├── EncoderRegistry (Model Management)
  │   ├── OpenCLIPImageEncoder
  │   └── OpenCLIPTextEncoder
  ├── SemanticRegistry (Descriptor Management)
  │   └── SemanticDescriptor (day_night, weather, lighting, etc.)
  ├── BasePredictor (Inference Engine)
  ├── Evaluator (Metrics & Analysis)
  └── ResultsVisualizer (Plotting)

Core Components
---------------

1. **Semantic Descriptors** (``semantic_descriptors/``)
   Base abstraction for semantic categories with natural language generation.

2. **Encoders** (``encoders/``)
   Modular vision-language model interfaces. Primary encoder is OpenCLIP, supporting a wide range of pretrained models (EVA02, ViT, SigLIP, PE-Core, etc.).

3. **Prediction Engine** (``prediction/``)
   Zero-shot classification via cosine similarity in embedding space.

4. **Evaluation System** (``evaluator/``)
   Confidence, coverage, distribution analysis, and enhanced results export.

5. **Pipeline Orchestrator** (``dataset_evaluation_pipeline.py``)
   Main interface coordinating all components.


Execution Pipeline
------------------

.. code-block::

  1. Configuration Loading
     ├── Load .ini config file
     ├── Initialize component registries
     └── Setup logging and device selection

  2. Model Initialization
     ├── Load image encoder (e.g., OpenCLIP EVA02-L-14)
     ├── Load text encoder (same model)
     └── Move models to GPU/CPU

  3. Semantic Descriptor Setup
     ├── Parse descriptor list from config
     ├── Instantiate descriptor classes
     ├── Generate natural language descriptions
     └── Pre-compute text embeddings

  4. Dataset Processing
     ├── Load image dataset
     ├── Create PyTorch DataLoader
     └── Batch processing for memory efficiency

  5. Zero-Shot Prediction
     ├── Encode images → normalized embeddings
     ├── Compute cosine similarities with text embeddings
     ├── Argmax for category assignment
     └── Extract confidence scores

  6. Evaluation & Analysis
     ├── Compute metrics (confidence, coverage, distribution)
     ├── Generate summary statistics
     └── Optional supervised accuracy (if ground truth available)

  7. Visualization & Export
     ├── Plot prediction distributions
     ├── Generate summary dashboard
     ├── Save results (JSON + PyTorch tensors)
     └── Export visualizations


Embedding Flow
--------------

.. code-block::

  Text Path:
  "Image of a car with bright lighting" → TextEncoder → [512-dim vector]

  Image Path:
  [Image Tensor] → ImageEncoder → [512-dim vector]

  Similarity:
  cosine_similarity(image_embedding, text_embedding) → confidence_score


Semantic Descriptor Template
-----------------------------

.. code-block::

  "Image [of a {part_a_object}] [in the setting of {part_b_scene}] with {part_c_semantic} [and {part_d_additional}]"

Example outputs:

* ``"Image with bright lighting"`` (minimal)
* ``"Image of a car with bright lighting"`` (with object)
* ``"Image of a car in the setting of urban street with bright lighting and dramatic shadows"`` (full)


Built-in Semantic Descriptors
------------------------------

The following descriptors are registered out of the box:

* ``day_night`` — day vs. night
* ``time_of_day`` — dawn, morning, midday, evening, night
* ``weather`` — clear, cloudy, rainy, foggy, snowy
* ``clear_rain`` — clear vs. rainy
* ``clear_fog`` — clear vs. foggy
* ``clear_snow`` — clear vs. snowy
* ``summer_winter`` — summer vs. winter
* ``lighting`` — lighting conditions
* ``sharp_blurry`` — image sharpness
* ``noflare_flare`` — lens flare presence
* ``noveg_veg`` — vegetation presence
* ``lowdense_highdense`` — scene density
* ``nodamage_damage`` — damage presence
* ``nopeople_people`` — people presence
* ``object_pose`` — object pose


Registry System
---------------

The registry pattern enables:

* **Modularity**: Add new descriptors without modifying core code
* **Configuration-driven**: Select descriptors via config files
* **Extensibility**: Easy integration of custom semantic categories
* **Validation**: Ensure descriptors implement required interface

Example — registering and using encoders:

.. code-block:: python

  encoder_registry = EncoderRegistry()

  # OpenCLIP is registered automatically; create an encoder from config
  image_encoder = encoder_registry.create_image_encoder(
      'openclip',
      {'model_name': 'EVA02-L-14', 'pretrained': 'merged2b_s4b_b131k', 'device': 'cuda'}
  )

.. code-block:: python

  semantic_registry = SemanticRegistry()

  descriptor = semantic_registry.create_descriptor(
      'day_night',
      {'part_a_object': 'car'}
  )

.. code-block:: python

  class CustomDescriptor(SemanticDescriptor):
      @property
      def name(self) -> str:
          return "custom_semantic"

      def _get_default_categories(self) -> Dict[str, str]:
          return {'category1': 'description1'}

  # Register manually
  registry.register_descriptor("custom_semantic", CustomDescriptor)


Configuration System
--------------------

Example ini structure:

.. code-block:: ini

  [DATASET]
  path = /path/to/dataset          ; or use 'paths' for multiple comma-separated dirs
  image_extensions = .jpg,.png,.jpeg
  batch_size = 32

  [ENCODERS]
  image_encoder = openclip
  text_encoder = openclip
  model_name = EVA02-L-14
  pretrained = merged2b_s4b_b131k
  image_size = 224

  [SEMANTICS]
  descriptors = day_night, clear_rain, sharp_blurry, noveg_veg
  part_a_object =
  part_b_scene =
  part_d_additional =

  [EVALUATION]
  metrics = zero_shot_accuracy,realism_score,diversity_score
  save_visualizations = true
  output_path = results/

  [CLUSTERING]
  n_clusters = 6
  model_path = /path/to/clustering_model.pth
  model_name = enet_b0
  method = kmeans
  min_margin = 0.010
  max_margin = 0.025
  confidence_levels = high,medium,low

  [SYNTHETIC_DATA]
  synthetic_data_path = /path/to/synthetic_images
  semantic_descriptor = day_night
  original_label = day
  synthetic_label = night

  [LOGGING]
  level = INFO
  file =
  directory = logs/

Config hierarchy:

* Default values in descriptor/encoder classes
* INI file overrides for specific use cases
* Runtime parameters via pipeline methods

Notes on ``[ENCODERS]``:

* Use ``model_name`` (not ``image_model_name``) — a single key shared by image and text encoders
* ``pretrained`` is optional; sensible defaults are applied automatically per model
* ``image_size`` is inferred from the model name if omitted (e.g., ``336`` for ``ViT-L-14-336``)

Notes on ``[CLUSTERING]``:

* Optional section for cluster-based margin analysis
* Omit the section entirely if clustering is not needed

Notes on ``[SYNTHETIC_DATA]``:

* Optional section for evaluating synthetic-vs-real distribution shift
* Omit the section or leave ``synthetic_data_path`` empty to skip

Notes on ``[DATASET]`` labeled mode (in progress):

* Add ``labeled_folders = /path/a=classA, /path/b=classB`` to load images with ground-truth labels
* Enables supervised accuracy via ``pipeline.evaluate(predictions, ground_truth=gt_dict)``
* ``LabeledEvaluationSaver`` (``evaluator/enhanced_results.py``) exports per-image CSV, confusion matrix, and category accuracy
* ``LabeledEvaluationPlotter`` (``utils/labeled_visualizations.py``) visualizes those results
