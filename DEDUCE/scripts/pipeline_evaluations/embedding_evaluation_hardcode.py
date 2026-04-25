"""
Divergence Test Script

Calculates various metrics to compare dataset distributions:
- Divergence (FNN-based)
- Coverage (adaptive radius)
- Completeness (PCA-based)
- Drift detection (KS, CVM, MMD)
- FID score

This script is separate from the main pipeline with easy-to-configure
variables at the top of the file.
"""

import os
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from colorama import Fore, Style, init

from deduce.core.dataset import SemanticDatasetFL
from deduce.core.embedding_model import EmbeddingExtractor
from dataeval.core import coverage_adaptive
from dataeval.core import divergence_fnn
from dataeval.core import completeness
from dataeval import Embeddings
from dataeval.extractors import TorchExtractor
from dataeval import config
from dataeval.shift import DriftUnivariate, DriftMMD
from cleanfid import fid

# Initialize colorama for colored output
init(autoreset=True)

config.set_seed(45, all_generators=True)



# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

# CCT DayNight
# DATASET_A = '/data2/CDAO/DENSE_public/datasets/cct/daynight/subsets/100night'
# DATASET_D = '/data2/CDAO/DENSE_public/datasets/cct/daynight/subsets/100night'
# DATASET_B = '/data2/CDAO/DENSE_public/datasets/cct/daynight/subsets/all_day'
# DATASET_C = '/data2/CDAO/DENSE_public/datasets/cct/daynight/subsets/day_transformed_daynight1_cistesttrained'
# MODEL_PATH = '/data2/CDAO/DENSE_public/models/cct/daynight/daynight_model_2_clustering.pth'
# MODEL_NAME = 'enet_b0'
# LABEL_A = 'night'
# LABEL_D = 'night (duplicate)'
# LABEL_B = 'day'
# LABEL_C = 'synthetic night'

# BDD100k DayNight
DATASET_A = '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/100_night'
DATASET_D = '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/100_night'
DATASET_B = '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/all_day'
DATASET_C = '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/day_transformed_daynight1_highrez'
# MODEL_PATH = '/data2/CDAO/DENSE_public/models/bdd100k/resnet18_4x_weather_tag_bdd100k.pth'  #more distinct clusters but more specific
MODEL_NAME = 'resnet18'
LABEL_A = 'night'
LABEL_D = 'night (duplicate)'
LABEL_B = 'day'
LABEL_C = 'synthetic night'



# BDD100k ClearRainy
# DATASET_A = 'bdd100k_toy/clearrainy_1/subsets/100_rainy'
# DATASET_D = 'bdd100k_toy/clearrainy_1/subsets/100_rainy'  # Duplicate/similar to A
# DATASET_B = 'bdd100k_toy/clearrainy_1/subsets/all_clear'
# DATASET_C = 'bdd100k_toy/clearrainy_1/subsets/clear_transformed_clearrainy1'
# # DATASET_C = 'bdd100k_toy/clearrainy_1/subsets/clear_transformed_clearrainy1_existingmodel'
# # DATASET_C = '/data2/CDAO/bdd100k_toy/clearrainy_1/subsets/clear_transformed_clearrainy1_highrez'
# LABEL_A = 'rainy'
# LABEL_D = 'rainy (duplicate)'
# LABEL_B = 'clear'
# LABEL_C = 'synthetic rainy'
# # MODEL_PATH = None #'/data2/CDAO/BDD-100k/models/resnet18_4x_weather_tag_bdd100k.pth'
# MODEL_PATH = '/data2/CDAO/BDD-100k/models/resnet18_4x_weather_tag_bdd100k.pth'
# MODEL_NAME = 'resnet18'  # Options: 'resnet18', 'enet_b0', 'dinov2', etc.

# # BDD100K Summer/Winter
# DATASET_A = 'bdd100k_toy/summerwinter_1/subsets/100_snowy'
# DATASET_D = 'bdd100k_toy/summerwinter_1/subsets/100_snowy'
# DATASET_B = 'bdd100k_toy/summerwinter_1/subsets/all_clear'
# DATASET_C = 'bdd100k_toy/summerwinter_1/subsets/toy_summerwinter1_transformed_clearsnowyday_1'
# # MODEL_PATH = None #'/data2/CDAO/BDD-100k/models/resnet18_4x_weather_tag_bdd100k.pth'
# MODEL_PATH = '/data2/CDAO/BDD-100k/models/resnet18_4x_weather_tag_bdd100k.pth'
# MODEL_NAME = 'resnet18'
# LABEL_A = 'snowy'
# LABEL_D = 'snowy (duplicate)'
# LABEL_B = 'clear'
# LABEL_C = 'synthetic snowy'




# ============================================================================
# CONFIGURATION VARIABLES - Set your paths and parameters here
# ============================================================================


# Base path for all datasets
BASE_DATA_PATH = '/data2/CDAO'

# Metric calculation parameters
COVERAGE_NUM_OBS = 20
COVERAGE_PERCENT = 0.01

# Output directory for saved plots
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'embedding_viz')

# Device selection (will auto-detect GPU if available)
USE_CUDA = True  # Set to False to force CPU

# Drift subsampling: run multiple trials with different seeds to average out
# sampling variance when test sets are larger than the reference set
DRIFT_NUM_TRIALS = 10
DRIFT_SEEDS = list(range(42, 42 + DRIFT_NUM_TRIALS))



# ============================================================================
# COLOR DEFINITIONS (for output display)
# ============================================================================
COLOR_NIGHT = Fore.CYAN
COLOR_DOUBLE_NIGHT = Fore.GREEN
COLOR_DAY_NIGHT = Fore.YELLOW
COLOR_SYNTH_NIGHT = Fore.MAGENTA


# ============================================================================
# METRIC CALCULATION FUNCTIONS
# ============================================================================

def calculate_divergence(embeddings_a, embeddings_b):
    """Calculate FNN-based divergence between two embedding sets"""
    return divergence_fnn(embeddings_a.cpu().numpy(), embeddings_b.cpu().numpy())['divergence']


def calculate_coverage_metrics(embeddings_list, num_obs=20, percent_coverage=0.01):
    """Calculate coverage metrics for multiple embedding combinations."""
    results = {}
    num_ref = embeddings_list[0][1].shape[0]  # Assume first is reference

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
        folder_test_list: List of tuples (name, folder_path)
        device: PyTorch device
    
    Returns:
        Dictionary mapping names to FID scores
    """
    results = {}
    
    for name, folder_test in folder_test_list:
        fid_score = fid.compute_fid(folder_ref, folder_test, device=device)
        results[name] = fid_score
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def _scatter_datasets(ax, embs_2d_by_key, labels_dict, colors, markers):
    for key, pts in embs_2d_by_key.items():
        ax.scatter(
            pts[:, 0], pts[:, 1],
            c=colors[key], marker=markers[key],
            label=labels_dict[key], alpha=0.6, s=20, linewidths=0,
        )


def plot_tsne_embeddings(embeddings_dict, labels_dict, output_dir):
    """
    Compute t-SNE once and save two plots:
      1. Standard scatter colored by dataset.
      2. Same scatter plus arrows from each DATASET_B point to its
         same-index DATASET_C point (domain-shift visualization).
    """
    colors = {'a': '#1f77b4', 'b': '#ff7f0e', 'c': '#9467bd', 'd': '#2ca02c'}
    markers = {'a': 'o', 'b': 's', 'c': '^', 'd': 'D'}

    # Stack in fixed order; exclude duplicates (key 'd') from visualization
    ordered_keys = [k for k in embeddings_dict.keys() if k != 'd']
    arrays = [embeddings_dict[k].cpu().numpy() for k in ordered_keys]
    slices = {}
    cursor = 0
    for key, arr in zip(ordered_keys, arrays):
        slices[key] = slice(cursor, cursor + len(arr))
        cursor += len(arr)

    all_embs = np.vstack(arrays)

    print("\nComputing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    embs_2d = tsne.fit_transform(all_embs)

    embs_2d_by_key = {k: embs_2d[slices[k]] for k in ordered_keys}

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Plot 1: standard scatter ---
    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_datasets(ax, embs_2d_by_key, labels_dict, colors, markers)
    ax.legend(loc='best', fontsize=10)
    ax.set_title('t-SNE Embedding Visualization', fontsize=14)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    path1 = os.path.join(output_dir, 'tsne_embeddings.png')
    fig.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  t-SNE plot saved to: {path1}")

    # --- Plot 2: scatter + B→C domain-shift arrows ---
    fig, ax = plt.subplots(figsize=(10, 8))
    _scatter_datasets(ax, embs_2d_by_key, labels_dict, colors, markers)

    pts_b = embs_2d_by_key['b']
    pts_c = embs_2d_by_key['c']
    n_arrows = min(len(pts_b), len(pts_c))
    for i in range(n_arrows):
        ax.annotate(
            '', xy=(pts_c[i, 0], pts_c[i, 1]), xytext=(pts_b[i, 0], pts_b[i, 1]),
            arrowprops=dict(arrowstyle='->', color='gray', alpha=0.15, lw=0.6),
        )

    ax.legend(loc='best', fontsize=10)
    ax.set_title(
        f't-SNE Embedding Visualization\n'
        f'(arrows: {labels_dict["b"]} → {labels_dict["c"]})',
        fontsize=14,
    )
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    path2 = os.path.join(output_dir, 'tsne_embeddings_domain_shift.png')
    fig.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  t-SNE domain-shift plot saved to: {path2}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Run divergence test using the configuration variables set at the top of the file
    """
    
    print("="*80)
    print("DIVERGENCE TEST")
    print("="*80)
    print(f"\nDatasets:")
    print(f"  A: {LABEL_A} ({DATASET_A})")
    print(f"  D: {LABEL_D} ({DATASET_D})")
    print(f"  B: {LABEL_B} ({DATASET_B})")
    print(f"  C: {LABEL_C} ({DATASET_C})")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
    print(f"\nDevice: {device}")
    
    # Initialize model
    print(f"\nLoading model: {MODEL_NAME}")
    print(f"  Path: {MODEL_PATH}")
    model = EmbeddingExtractor(
        model_path=MODEL_PATH, 
        model_name=MODEL_NAME, 
        device=device
    )
    
    # Load datasets
    print("\nLoading datasets...")
    
    datasets = {
        'a': SemanticDatasetFL(
            data_path=os.path.join(BASE_DATA_PATH, DATASET_A),
            config={'image_size': 256, 'normalize': False}
        ),
        'b': SemanticDatasetFL(
            data_path=os.path.join(BASE_DATA_PATH, DATASET_B),
            config={'image_size': 256, 'normalize': False}
        ),
        'c': SemanticDatasetFL(
            data_path=os.path.join(BASE_DATA_PATH, DATASET_C),
            config={'image_size': 256, 'normalize': False}
        ),
        'd': SemanticDatasetFL(
            data_path=os.path.join(BASE_DATA_PATH, DATASET_D),
            config={'image_size': 256, 'normalize': False}
        ),
    }
    
    labels = {
        'a': LABEL_A,
        'b': LABEL_B,
        'c': LABEL_C,
        'd': LABEL_D
    }
    
    for key, dataset in datasets.items():
        print(f"  Dataset {key} ({labels[key]}): {len(dataset)} images")
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    extractor = TorchExtractor(
        model=model,
        transforms=model.get_transforms_np(),
        device=device,
    )
    embeddings = {}
    for key, dataset in datasets.items():
        embs = torch.from_numpy(np.asarray(Embeddings(
            dataset=dataset,
            extractor=extractor,
            batch_size=64,
        )))

        # Normalize
        embeddings[key] = (embs - embs.min()) / (embs.max() - embs.min())
        print(f"  Embeddings {key}: shape {embeddings[key].shape}")
    
    # ========================================================================
    # METRIC CALCULATIONS
    # ========================================================================
    
    print("\n" + "="*80)
    print("DIVERGENCE METRICS")
    print("="*80)
    
    # Calculate divergences
    div_ad = calculate_divergence(embeddings['a'], embeddings['d'])
    div_ab = calculate_divergence(embeddings['a'], embeddings['b'])
    div_ac = calculate_divergence(embeddings['a'], embeddings['c'])
    div_bc = calculate_divergence(embeddings['b'], embeddings['c'])
    
    print(f"Divergence from {COLOR_NIGHT}{LABEL_A}{Style.RESET_ALL} to {COLOR_NIGHT}{LABEL_D}{Style.RESET_ALL}: {div_ad:.4f}")
    print(f"Divergence from {COLOR_NIGHT}{LABEL_A}{Style.RESET_ALL} to {COLOR_DAY_NIGHT}{LABEL_B}{Style.RESET_ALL}: {div_ab:.4f}")
    print(f"Divergence from {COLOR_NIGHT}{LABEL_A}{Style.RESET_ALL} to {COLOR_SYNTH_NIGHT}{LABEL_C}{Style.RESET_ALL}: {div_ac:.4f}")
    print(f"Divergence from {COLOR_DAY_NIGHT}{LABEL_B}{Style.RESET_ALL} to {COLOR_SYNTH_NIGHT}{LABEL_C}{Style.RESET_ALL}: {div_bc:.4f}")
    
    # ========================================================================
    print("\n" + "="*80)
    print("COVERAGE METRICS")
    print("="*80)
    
    # Prepare embedding combinations
    embeddings_aa = np.vstack((embeddings['a'].cpu().numpy(), embeddings['a'].cpu().numpy()))
    embeddings_ab = np.vstack((embeddings['a'].cpu().numpy(), embeddings['b'].cpu().numpy()))
    embeddings_ac = np.vstack((embeddings['a'].cpu().numpy(), embeddings['c'].cpu().numpy()))
    embeddings_ad = np.vstack((embeddings['a'].cpu().numpy(), embeddings['d'].cpu().numpy()))
    embeddings_abc = np.vstack((embeddings_ab, embeddings['c'].cpu().numpy()))
    
    coverage_inputs = [
        ('just_a', embeddings['a'].cpu().numpy()),
        ('double_a', embeddings_aa),
        ('a_with_d', embeddings_ad),
        ('a_with_b', embeddings_ab),
        ('a_with_c', embeddings_ac),
        ('a_with_b_and_c', embeddings_abc),
    ]
    
    coverage_results = calculate_coverage_metrics(
        coverage_inputs, 
        num_obs=COVERAGE_NUM_OBS, 
        percent_coverage=COVERAGE_PERCENT
    )
    
    print(f"Coverage from just {COLOR_NIGHT}{LABEL_A}{Style.RESET_ALL}: {coverage_results['just_a']:.6f}")
    print(f"Coverage from {COLOR_DOUBLE_NIGHT}double {LABEL_A}{Style.RESET_ALL}: {coverage_results['double_a']:.6f}")
    print(f"Coverage from {COLOR_DAY_NIGHT}{LABEL_A} with {LABEL_D}{Style.RESET_ALL}: {coverage_results['a_with_d']:.6f}")
    print(f"Coverage from {COLOR_DAY_NIGHT}{LABEL_A} with {LABEL_B}{Style.RESET_ALL}: {coverage_results['a_with_b']:.6f}")
    print(f"Coverage from {COLOR_SYNTH_NIGHT}{LABEL_A} with {LABEL_C}{Style.RESET_ALL}: {coverage_results['a_with_c']:.6f}")
    print(f"Coverage from {COLOR_DAY_NIGHT}{LABEL_A} with {LABEL_B}{Style.RESET_ALL} and {COLOR_SYNTH_NIGHT}{LABEL_C}{Style.RESET_ALL}: {coverage_results['a_with_b_and_c']:.6f}")
    
    # ========================================================================
    print("\n" + "="*80)
    print("COMPLETENESS METRICS")
    print("="*80)
    
    completeness_inputs = [
        ('just_a', embeddings['a'].cpu().numpy()),
        ('double_a', embeddings_aa),
        ('a_with_d', embeddings_ad),
        ('a_with_b', embeddings_ab),
        ('a_with_c', embeddings_ac),
        ('a_with_b_and_c', embeddings_abc),
    ]
    
    completeness_results = calculate_completeness_metrics(
        completeness_inputs,
    )
    
    print(f"Completeness from just {COLOR_NIGHT}{LABEL_A}{Style.RESET_ALL}: {completeness_results['just_a']:.4f}")
    print(f"Completeness from {COLOR_DOUBLE_NIGHT}double {LABEL_A}{Style.RESET_ALL}: {completeness_results['double_a']:.4f}")
    print(f"Completeness from {COLOR_DAY_NIGHT}{LABEL_A} with {LABEL_D}{Style.RESET_ALL}: {completeness_results['a_with_d']:.4f}")
    print(f"Completeness from {COLOR_DAY_NIGHT}{LABEL_A} with {LABEL_B}{Style.RESET_ALL}: {completeness_results['a_with_b']:.4f}")
    print(f"Completeness from {COLOR_SYNTH_NIGHT}{LABEL_A} with {LABEL_C}{Style.RESET_ALL}: {completeness_results['a_with_c']:.4f}")
    print(f"Completeness from {COLOR_DAY_NIGHT}{LABEL_A} with {LABEL_B}{Style.RESET_ALL} and {COLOR_SYNTH_NIGHT}{LABEL_C}{Style.RESET_ALL}: {completeness_results['a_with_b_and_c']:.4f}")
    
    # ========================================================================
    print("\n" + "="*80)
    print("DRIFT DETECTION METRICS")
    print("="*80)
    
    # Pass full embeddings — subsampling is handled inside the function
    drift_test_inputs = [
        ('duplicate', embeddings['d']),
        ('different_condition', embeddings['b']),
        ('synthetic', embeddings['c']),
    ]

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
    print("\n" + "="*80)
    print("FID SCORES")
    print("="*80)
    
    folder_ref = os.path.join(BASE_DATA_PATH, DATASET_A)
    fid_test_folders = [
        ('duplicate', os.path.join(BASE_DATA_PATH, DATASET_D)),
        ('different_condition', os.path.join(BASE_DATA_PATH, DATASET_B)),
        ('synthetic', os.path.join(BASE_DATA_PATH, DATASET_C)),
    ]
    
    fid_results = calculate_fid_scores(folder_ref, fid_test_folders, device)
    
    for test_name, score in fid_results.items():
        print(f"FID ({test_name}): {score:.4f}")
    
    # ========================================================================
    print("\n" + "="*80)
    print("T-SNE VISUALIZATION")
    print("="*80)

    plot_tsne_embeddings(embeddings, labels, OUTPUT_DIR)

    print("\n" + "="*80)
    print("DIVERGENCE TEST COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()