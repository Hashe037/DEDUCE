"""
Divergence Analysis from Pipeline Outputs

Reads JSON filename outputs from full_pipeline.py and calculates divergence metrics
between covered, uncovered, and synthetic datasets.

Configuration is loaded from the same .ini file used by full_pipeline.py,
with semantic selection hardcoded at the top of this script.
"""

import os
import json
import tempfile
from pathlib import Path

import torch
import numpy as np
from colorama import Fore, Style, init

from deduce.core.config import ConfigManager
from deduce.core.dataset import SemanticDatasetFL
from deduce.core.embedding_model import EmbeddingExtractor
from dataeval.core import coverage_adaptive
from dataeval.core import divergence_fnn
from dataeval.core import completeness
from dataeval import Embeddings
from dataeval.extractors import TorchExtractor
from dataeval.shift import DriftUnivariate, DriftMMD
from cleanfid import fid
from dataeval import config

# Initialize colorama for colored output
init(autoreset=True)

config.set_seed(45, all_generators=True)

# ============================================================================
# CONFIGURATION - Configuration is loaded from .ini file
# ============================================================================

# CCT - DayNight
# CONFIG_PATH = "../../configs/cct/daynight_1/eva_multisemantics_v3.ini"

# BDD100k - DayNight
CONFIG_PATH = "../../configs/bdd100k/daynight_1/eva_multisemantics_v2_cctsynth.ini"

# BDD100k - ClearRainy
# CONFIG_PATH = "../../configs/bdd100k/clearrainy_1/eva_multisemantics_v2.ini"

# BDD100k - ClearSnowy
# CONFIG_PATH = "../../configs/bdd100k/summerwinter_1/eva_multisemantics_v2.ini"

# ============================================================================
# Other Params
# ============================================================================

# Enable/disable duplicate dataset comparison (DATASET_D)
ENABLE_DUPLICATE_COMPARISON = False
DUPLICATE_DATA_PATH = None  # Set path if ENABLE_DUPLICATE_COMPARISON is True

# Device selection
USE_CUDA = True

# Metric calculation parameters
COVERAGE_NUM_OBS = 20
COVERAGE_PERCENT = 0.01

# Drift subsampling: run multiple trials with different seeds to average out
# sampling variance when test sets are larger than the reference set
DRIFT_NUM_TRIALS = 10
DRIFT_SEEDS = list(range(42, 42 + DRIFT_NUM_TRIALS))

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_filenames_from_json(json_path: str) -> list:
    """Load filenames from a pipeline-generated JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'filenames' in data:
        return data['filenames']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError(f"Unexpected JSON structure in {json_path}")


def get_confidence_level_files(output_path: str, class_name: str, confidence_levels: list) -> list:
    """
    Get list of JSON files for a class based on configured confidence levels.
    
    Args:
        output_path: Directory containing pipeline outputs
        class_name: Class name (e.g., "clear", "rain")
        confidence_levels: List of confidence levels to include (e.g., ["high", "medium", "low"])
    
    Returns:
        List of existing JSON file paths
    """
    files = []
    for level in confidence_levels:
        json_path = os.path.join(output_path, f"{class_name}_filenames_{level}.json")
        if os.path.exists(json_path):
            files.append(json_path)
        else:
            print(f"  Warning: {json_path} not found, skipping")
    return files


def load_combined_filenames(output_path: str, class_name: str, confidence_levels: list) -> list:
    """
    Load and combine filenames from multiple confidence level JSON files.
    
    Args:
        output_path: Directory containing pipeline outputs
        class_name: Class name (e.g., "clear", "rain")
        confidence_levels: List of confidence levels to include
    
    Returns:
        Combined list of filenames
    """
    json_files = get_confidence_level_files(output_path, class_name, confidence_levels)
    
    all_filenames = []
    for json_file in json_files:
        filenames = load_filenames_from_json(json_file)
        print(f"    Loaded {len(filenames)} files from {Path(json_file).name}")
        all_filenames.extend(filenames)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_filenames = []
    for f in all_filenames:
        if f not in seen:
            seen.add(f)
            unique_filenames.append(f)
    
    return unique_filenames

def create_temp_folder_from_filenames(filenames: list, temp_dir: str, folder_name: str) -> str:
    """
    Create a temporary folder with symlinks to images for FID calculation.
    
    Args:
        filenames: List of absolute file paths
        temp_dir: Base temporary directory
        folder_name: Name for this folder
    
    Returns:
        Path to the temporary folder containing symlinks
    """
    folder_path = os.path.join(temp_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    
    for filepath in filenames:
        if os.path.exists(filepath):
            # Create symlink with original filename
            link_name = os.path.basename(filepath)
            link_path = os.path.join(folder_path, link_name)
            if not os.path.exists(link_path):
                os.symlink(filepath, link_path)
    
    return folder_path


class FilenameListDataset(SemanticDatasetFL):
    """Dataset that loads images from a list of filenames."""
    
    def __init__(self, filenames: list, config: dict = None):
        self.filenames = filenames
        self.config = config or {'image_size': 256, 'normalize': False}
        
        # Don't call parent __init__, just set up what we need
        self.image_size = self.config.get('image_size', 256)
        self.normalize = self.config.get('normalize', False)
        self.transform = self._get_transform()
    
    def _get_transform(self):
        """Get image transforms."""
        from torchvision import transforms
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
        ]
        if self.normalize:
            transform_list.append(
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            )
        return transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        img_path = self.filenames[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Return format expected by Embeddings: (image, label, metadata)
        return image, 0, {'filename': img_path}


# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_divergence(embeddings_a, embeddings_b):
    """Calculate FNN-based divergence between two embedding sets."""
    return divergence_fnn(embeddings_a.cpu().numpy(), embeddings_b.cpu().numpy())['divergence']


def calculate_coverage_metrics(embeddings_list, num_obs=20, percent_coverage=0.01):
    """Calculate coverage metrics for multiple embedding combinations."""
    results = {}
    num_ref = embeddings_list[0][1].shape[0]
    
    for name, embs in embeddings_list:
        cov = coverage_adaptive(
            embs,
            num_observations=num_obs,
            percent=percent_coverage,
        )
        all_radii = cov['critical_value_radii'].copy()
        median_coverage = np.median(all_radii[:num_ref])
        results[name] = median_coverage
    
    return results


def calculate_completeness_metrics(embeddings_list):
    """Calculate completeness metrics for multiple embedding combinations."""
    results = {}

    for name, embs in embeddings_list:
        results[name] = completeness(embs)['completeness']

    return results


def calculate_drift_metrics(embeddings_ref, embeddings_test_list, seeds=None):
    """
    Calculate drift metrics using multiple detectors.
    
    If seeds are provided and a test set is larger than the reference set,
    run multiple trials with random subsamples of size num_ref, then
    average the distance and report how often drift was detected.
    """
    num_ref = embeddings_ref.shape[0]
    ref_cpu = embeddings_ref.cpu()

    results = {
        'KS': {},
        'CVM': {},
        'MMD': {}
    }

    detectors = {
        'KS': DriftUnivariate(method='ks').fit(ref_cpu),
        'CVM': DriftUnivariate(method='cvm').fit(ref_cpu),
        'MMD': DriftMMD().fit(ref_cpu),
    }

    for name, embs_test in embeddings_test_list:
        embs_test_cpu = embs_test.cpu()
        needs_subsampling = seeds and len(embs_test_cpu) > num_ref

        if needs_subsampling:
            # Multiple trials: subsample test set to match reference size
            trial_results = {d: [] for d in detectors}

            for seed in seeds:
                rng = np.random.default_rng(seed)
                indices = rng.choice(len(embs_test_cpu), size=num_ref, replace=False)
                subset = embs_test_cpu[indices]

                for detector_name, detector in detectors.items():
                    result = detector.predict(subset)
                    trial_results[detector_name].append({
                        'drifted': result.drifted,
                        'distance': result.distance
                    })

            # Aggregate: average distance, fraction of trials that detected drift
            for detector_name in detectors:
                trials = trial_results[detector_name]
                distances = [t['distance'] for t in trials]
                drift_votes = sum(1 for t in trials if t['drifted'])

                results[detector_name][name] = {
                    'drifted': drift_votes > len(seeds) / 2,  # majority vote
                    'distance': float(np.mean(distances)),
                    'distance_std': float(np.std(distances)),
                    'drift_fraction': drift_votes / len(seeds),
                    'num_trials': len(seeds)
                }
        else:
            # Single trial: test set is already <= reference size
            for detector_name, detector in detectors.items():
                result = detector.predict(embs_test_cpu)
                results[detector_name][name] = {
                    'drifted': result.drifted,
                    'distance': result.distance
                }

    return results

def calculate_fid_scores(folder_ref, folder_test_list, device):
    """
    Calculate FID scores between reference and test folders
    
    Args:
        folder_ref: Reference folder path
        folder_test_list: List of tuples (name, test_folder_path)
        device: Device to use for computation
    
    Returns:
        Dictionary mapping names to FID scores
    """
    results = {}
    
    for name, folder_test in folder_test_list:
        fid_score = fid.compute_fid(folder_ref, folder_test, device=device)
        results[name] = fid_score
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run divergence analysis using pipeline outputs."""
    
    print("=" * 80)
    print("DIVERGENCE ANALYSIS FROM PIPELINE OUTPUTS")
    print("=" * 80)
    
    # Load config
    config_manager = ConfigManager(CONFIG_PATH)

    # Load synthetic data settings from config
    synth_config = config_manager.get_synthetic_data_config()
    if synth_config is None:
        raise ValueError(f"[SYNTHETIC_DATA] section with a valid synthetic_data_path is required in {CONFIG_PATH}")
    SEMANTIC_DESCRIPTOR = synth_config['semantic_descriptor']
    COVERED_CLASS = synth_config['original_label']
    UNCOVERED_CLASS = synth_config['synthetic_label']
    SYNTHETIC_DATA_PATH = synth_config['synthetic_data_path']

    # Get paths from config
    output_path = config_manager.get('EVALUATION', {}).get('output_path')
    clustering_config = config_manager.get('CLUSTERING', {})
    
    # Get confidence levels from config
    confidence_levels_str = clustering_config.get('confidence_levels', 'high,medium,low')
    confidence_levels = [level.strip() for level in confidence_levels_str.split(',')]
    
    # Get model config
    model_path = clustering_config.get('model_path')
    model_name = clustering_config.get('model_name', 'resnet18')
    
    print(f"\nConfiguration:")
    print(f"  Config file: {CONFIG_PATH}")
    print(f"  Output path: {output_path}")
    print(f"  Model path: {model_path}")
    print(f"  Semantic: {SEMANTIC_DESCRIPTOR}")
    print(f"  Covered class: {COVERED_CLASS}")
    print(f"  Uncovered class: {UNCOVERED_CLASS}")
    print(f"  Confidence levels: {confidence_levels}")
    print(f"  Synthetic path: {SYNTHETIC_DATA_PATH}")
    print(f"  Duplicate comparison: {ENABLE_DUPLICATE_COMPARISON}")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    print(f"  Device: {device}")
    
    # ========================================================================
    # Load filenames from pipeline outputs
    # ========================================================================
    print("\n" + "=" * 80)
    print("LOADING DATASETS FROM PIPELINE OUTPUTS")
    print("=" * 80)
    
    print(f"\nLoading {UNCOVERED_CLASS} (uncovered) filenames:")
    uncovered_filenames = load_combined_filenames(output_path, UNCOVERED_CLASS, confidence_levels)
    print(f"  Total: {len(uncovered_filenames)} images")
    
    print(f"\nLoading {COVERED_CLASS} (covered) filenames:")
    covered_filenames = load_combined_filenames(output_path, COVERED_CLASS, confidence_levels)
    print(f"  Total: {len(covered_filenames)} images")
    
    # ========================================================================
    # Create datasets
    # ========================================================================
    print("\n" + "=" * 80)
    print("CREATING DATASETS")
    print("=" * 80)
    
    dataset_config = {'image_size': 256, 'normalize': False}
    
    # Dataset A: Uncovered class (e.g., rainy)
    dataset_a = FilenameListDataset(uncovered_filenames, dataset_config)
    print(f"  Dataset A ({UNCOVERED_CLASS}): {len(dataset_a)} images")
    
    # Dataset B: Covered class (e.g., clear)
    dataset_b = FilenameListDataset(covered_filenames, dataset_config)
    print(f"  Dataset B ({COVERED_CLASS}): {len(dataset_b)} images")
    
    # Dataset C: Synthetic data
    if os.path.exists(SYNTHETIC_DATA_PATH):
        dataset_c = SemanticDatasetFL(
            data_path=SYNTHETIC_DATA_PATH,
            config=dataset_config
        )
        print(f"  Dataset C (synthetic): {len(dataset_c)} images")
    else:
        print(f"  Warning: Synthetic path not found: {SYNTHETIC_DATA_PATH}")
        dataset_c = None
    
    # Dataset D: Duplicate (optional)
    dataset_d = None
    if ENABLE_DUPLICATE_COMPARISON and DUPLICATE_DATA_PATH:
        if os.path.exists(DUPLICATE_DATA_PATH):
            dataset_d = SemanticDatasetFL(
                data_path=DUPLICATE_DATA_PATH,
                config=dataset_config
            )
            print(f"  Dataset D (duplicate): {len(dataset_d)} images")
        else:
            print(f"  Warning: Duplicate path not found: {DUPLICATE_DATA_PATH}")
    
    # ========================================================================
    # Extract embeddings
    # ========================================================================
    print("\n" + "=" * 80)
    print("EXTRACTING EMBEDDINGS")
    print("=" * 80)
    
    print(f"\nLoading model: {model_name}")
    model = EmbeddingExtractor(
        model_path=model_path,
        model_name=model_name,
        device=device
    )
    
    embeddings = {}
    datasets = {'a': dataset_a, 'b': dataset_b}
    if dataset_c:
        datasets['c'] = dataset_c
    if dataset_d:
        datasets['d'] = dataset_d
    
    labels = {
        'a': UNCOVERED_CLASS,
        'b': COVERED_CLASS,
        'c': 'synthetic',
        'd': f'{UNCOVERED_CLASS} (duplicate)'
    }
    
    extractor = TorchExtractor(
        model=model,
        transforms=model.get_transforms_np(),
        device=device,
    )

    for key, dataset in datasets.items():
        print(f"  Extracting embeddings for {labels[key]}...")
        embs = torch.from_numpy(np.asarray(Embeddings(
            dataset=dataset,
            extractor=extractor,
            batch_size=64,
        )))

        # Normalize
        embeddings[key] = (embs - embs.min()) / (embs.max() - embs.min())
        print(f"    Shape: {embeddings[key].shape}")
    
    # ========================================================================
    # METRIC CALCULATIONS
    # ========================================================================
    
    # Colors for output
    COLOR_UNCOVERED = Fore.CYAN
    COLOR_COVERED = Fore.YELLOW
    COLOR_SYNTH = Fore.MAGENTA
    COLOR_DUP = Fore.GREEN
    
    print("\n" + "=" * 80)
    print("DIVERGENCE METRICS")
    print("=" * 80)
    
    # Calculate divergences
    div_ab = calculate_divergence(embeddings['a'], embeddings['b'])
    print(f"Divergence {COLOR_UNCOVERED}{UNCOVERED_CLASS}{Style.RESET_ALL} → {COLOR_COVERED}{COVERED_CLASS}{Style.RESET_ALL}: {div_ab:.4f}")
    
    if 'c' in embeddings:
        div_ac = calculate_divergence(embeddings['a'], embeddings['c'])
        div_bc = calculate_divergence(embeddings['b'], embeddings['c'])
        print(f"Divergence {COLOR_UNCOVERED}{UNCOVERED_CLASS}{Style.RESET_ALL} → {COLOR_SYNTH}synthetic{Style.RESET_ALL}: {div_ac:.4f}")
        print(f"Divergence {COLOR_COVERED}{COVERED_CLASS}{Style.RESET_ALL} → {COLOR_SYNTH}synthetic{Style.RESET_ALL}: {div_bc:.4f}")
    
    if 'd' in embeddings:
        div_ad = calculate_divergence(embeddings['a'], embeddings['d'])
        print(f"Divergence {COLOR_UNCOVERED}{UNCOVERED_CLASS}{Style.RESET_ALL} → {COLOR_DUP}duplicate{Style.RESET_ALL}: {div_ad:.4f}")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("COVERAGE METRICS")
    print("=" * 80)
    
    # Prepare embedding combinations
    emb_a_np = embeddings['a'].cpu().numpy()
    emb_b_np = embeddings['b'].cpu().numpy()
    
    coverage_inputs = [
        (f'just_{UNCOVERED_CLASS}', emb_a_np),
        (f'{UNCOVERED_CLASS}_with_{COVERED_CLASS}', np.vstack((emb_a_np, emb_b_np))),
    ]
    
    if 'c' in embeddings:
        emb_c_np = embeddings['c'].cpu().numpy()
        coverage_inputs.append((f'{UNCOVERED_CLASS}_with_synthetic', np.vstack((emb_a_np, emb_c_np))))
        coverage_inputs.append((f'{UNCOVERED_CLASS}_with_{COVERED_CLASS}_and_synthetic', 
                               np.vstack((emb_a_np, emb_b_np, emb_c_np))))
    
    if 'd' in embeddings:
        emb_d_np = embeddings['d'].cpu().numpy()
        coverage_inputs.append((f'{UNCOVERED_CLASS}_with_duplicate', np.vstack((emb_a_np, emb_d_np))))
    
    coverage_results = calculate_coverage_metrics(
        coverage_inputs,
        num_obs=COVERAGE_NUM_OBS,
        percent_coverage=COVERAGE_PERCENT
    )
    
    for name, value in coverage_results.items():
        print(f"Coverage {name}: {value:.6f}")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPLETENESS METRICS")
    print("=" * 80)
    
    completeness_results = calculate_completeness_metrics(
        coverage_inputs,
    )
    
    for name, value in completeness_results.items():
        print(f"Completeness {name}: {value:.4f}")
    
    # ========================================================================
    print("\n" + "=" * 80)
    print("DRIFT DETECTION METRICS")
    print("=" * 80)
    
    # Pass full embeddings — subsampling is handled inside the function
    drift_test_inputs = [
        (COVERED_CLASS, embeddings['b']),
    ]

    if 'c' in embeddings:
        drift_test_inputs.append(('synthetic', embeddings['c']))

    if 'd' in embeddings:
        drift_test_inputs.append(('duplicate', embeddings['d']))

    drift_results = calculate_drift_metrics(
        embeddings['a'], drift_test_inputs, seeds=DRIFT_SEEDS
    )

    for detector_name, results in drift_results.items():
        print(f"\n{detector_name} Detector:")
        for test_name, result in results.items():
            if 'num_trials' in result:
                # Multi-trial result
                print(f"  {test_name}: drifted={result['drifted']}, "
                      f"distance={result['distance']:.4f} ± {result['distance_std']:.4f}, "
                      f"drift_fraction={result['drift_fraction']:.0%} "
                      f"({result['num_trials']} trials)")
            else:
                # Single trial (test set was already small enough)
                print(f"  {test_name}: drifted={result['drifted']}, "
                      f"distance={result['distance']:.4f}")

    # ========================================================================
    print("\n" + "=" * 80)
    print("FID SCORES")
    print("=" * 80)

    # Create temporary directory for symlinked folders
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"  Creating temporary folders for FID calculation...")
        
        # Create folder for uncovered class (reference)
        folder_a = create_temp_folder_from_filenames(uncovered_filenames, temp_dir, UNCOVERED_CLASS)
        print(f"    {UNCOVERED_CLASS}: {len(os.listdir(folder_a))} images")
        
        # Create folder for covered class
        folder_b = create_temp_folder_from_filenames(covered_filenames, temp_dir, COVERED_CLASS)
        print(f"    {COVERED_CLASS}: {len(os.listdir(folder_b))} images")
        
        # Build FID test list
        fid_test_folders = [
            (COVERED_CLASS, folder_b),
        ]
        
        # Add synthetic folder if available
        if dataset_c and os.path.exists(SYNTHETIC_DATA_PATH):
            fid_test_folders.append(('synthetic', SYNTHETIC_DATA_PATH))
        
        # Add duplicate folder if available
        if dataset_d and DUPLICATE_DATA_PATH and os.path.exists(DUPLICATE_DATA_PATH):
            fid_test_folders.append(('duplicate', DUPLICATE_DATA_PATH))
        
        # Calculate FID scores
        fid_results = calculate_fid_scores(folder_a, fid_test_folders, device)
        
        for test_name, score in fid_results.items():
            if score is not None:
                print(f"FID ({UNCOVERED_CLASS} → {test_name}): {score:.4f}")
            else:
                print(f"FID ({UNCOVERED_CLASS} → {test_name}): N/A")

    # ========================================================================
    # Save results
    # ========================================================================
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    results_output = {
        'config': {
            'semantic_descriptor': SEMANTIC_DESCRIPTOR,
            'covered_class': COVERED_CLASS,
            'uncovered_class': UNCOVERED_CLASS,
            'confidence_levels': confidence_levels,
            'synthetic_path': SYNTHETIC_DATA_PATH,
            'enable_duplicate': ENABLE_DUPLICATE_COMPARISON,
        },
        'dataset_sizes': {
            'uncovered': len(dataset_a),
            'covered': len(dataset_b),
            'synthetic': len(dataset_c) if dataset_c else 0,
            'duplicate': len(dataset_d) if dataset_d else 0,
        },
        'divergence': {
            f'{UNCOVERED_CLASS}_to_{COVERED_CLASS}': div_ab,
        },
        'coverage': coverage_results,
        'completeness': completeness_results,
        'drift': drift_results,
        'fid': fid_results,
    }
    
    if 'c' in embeddings:
        results_output['divergence'][f'{UNCOVERED_CLASS}_to_synthetic'] = div_ac
        results_output['divergence'][f'{COVERED_CLASS}_to_synthetic'] = div_bc
    
    if 'd' in embeddings:
        results_output['divergence'][f'{UNCOVERED_CLASS}_to_duplicate'] = div_ad
    
    results_file = os.path.join(output_path, 'evaluation_offlabels.json')
    with open(results_file, 'w') as f:
        json.dump(results_output, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")
    
    print("\n" + "=" * 80)
    print("DIVERGENCE ANALYSIS COMPLETE")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()