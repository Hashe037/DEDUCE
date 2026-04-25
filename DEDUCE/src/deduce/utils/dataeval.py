

import os
import json
import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.cluster import KMeans

from deduce.core.embedding_model import EmbeddingExtractor
# import pdb
# pdb.set_trace()
from dataeval import Embeddings
from dataeval.extractors import TorchExtractor
from dataeval.core import cluster


def cluster_and_save(dataset, model_path, n_clusters, output_path, 
                     batch_size=64, normalize=True, model_name='resnet18', clustering_method='default'):
    """
    Run clustering on image embeddings and save results to CSV.
    
    Parameters:
    -----------
    dataset : Dataset
        Dataset that returns (image, label, metadata) where metadata['filename'] 
        contains the filename
    model_path: str
        Model to extract embeddings
    n_clusters : int
        Expected number of clusters
    output_path : str
        Path to save the CSV file (e.g., 'results/clusters.csv')
    batch_size : int, optional
        Batch size for embedding extraction (default: 64)
    normalize : bool, optional
        Whether to normalize embeddings (default: True)
    clustering_method : str, optional
        Clustering algorithm to use: 'default' or 'kmeans' (default: 'default')
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'clusters': cluster assignments for each sample
        - 'filenames': list of filenames
        - 'embeddings': normalized embeddings (numpy array)
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load in model
    model = EmbeddingExtractor(model_path=model_path, model_name=model_name)
    
    # Extract embeddings
    print('Extracting embeddings...')
    extractor = TorchExtractor(
        model=model,
        transforms=model.get_transforms_np(),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    embeddings = np.asarray(Embeddings(
        dataset=dataset,
        extractor=extractor,
        batch_size=batch_size,
    ))
    print(f'Embeddings extracted: shape {embeddings.shape}')
    
    # Normalize embeddings if requested
    if normalize:
        normalized_embs = (embeddings - embeddings.min()) / (embeddings.max() - embeddings.min())
    else:
        normalized_embs = embeddings
    
    # Convert to numpy
    embs_np = normalized_embs.cpu().numpy() if torch.is_tensor(normalized_embs) else normalized_embs
    
    # Perform clustering based on method
    print(f'Clustering into {n_clusters} clusters using {clustering_method} method...')
    
    if clustering_method == 'kmeans':
        # Use K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embs_np)
        cluster_result = {
            'clusters': cluster_labels,
            'cluster_centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_
        }
    elif clustering_method == 'default':
        # Use the original clustering method from dataeval.core
        cluster_result = cluster(embs_np, n_expected_clusters=n_clusters)
        cluster_labels = cluster_result['clusters']
    else:
        raise ValueError(f"Unknown clustering method: {clustering_method}. Use 'default' or 'kmeans'.")
    
    # Extract filenames from dataset metadata
    print('Extracting filenames...')
    filenames = []
    for i in range(len(dataset)):
        _, _, metadata = dataset[i]
        filenames.append(metadata['filename'])
    
    # Save to CSV
    print(f'Saving results to {output_path}...')
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'cluster'])
        for filename, cluster_id in zip(filenames, cluster_labels):
            writer.writerow([filename, cluster_id])
    
    # Print summary statistics
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    print('\nClustering Summary:')
    print(f'Total samples: {len(cluster_labels)}')
    print(f'Number of clusters found: {len(unique_clusters)}')
    for cluster_id, count in zip(unique_clusters, counts):
        if cluster_id == -1:
            print(f'  Noise points: {count}')
        else:
            print(f'  Cluster {cluster_id}: {count} samples')
    
    return {
        'clusters': cluster_labels,
        'filenames': filenames,
        'embeddings': embs_np,
        'cluster_result': cluster_result
    }




def analyze_cluster_overlap(json_file, clustering_csv, output_path=None):
    """
    Analyze how images from a JSON file list are distributed across clusters.
    
    Parameters:
    -----------
    json_file : str
        Path to JSON file containing a list of filenames (like night_filenames_high.json)
    clustering_csv : str
        Path to CSV file with clustering results (filename, cluster columns)
    output_path : str, optional
        Path to save analysis results as JSON. If None, only returns the dict.
    
    Returns:
    --------
    dict
        Dictionary containing:
        - 'total_images': Total number of images in JSON
        - 'matched_images': Number of images found in clustering results
        - 'cluster_distribution': Dict mapping cluster_id -> count
        - 'largest_cluster': The cluster with most images
        - 'largest_cluster_size': Number of images in largest cluster
        - 'largest_cluster_percentage': Percentage of matched images in largest cluster
        - 'unmatched_images': List of filenames not found in clustering results
    """
    
    # Load JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract filenames from JSON (handle different JSON structures)
    if isinstance(data, dict) and 'filenames' in data:
        json_filenames = data['filenames']
    elif isinstance(data, list):
        json_filenames = data
    else:
        raise ValueError("JSON must contain 'filenames' key or be a list of filenames")
    
    # Normalize paths to just filenames for matching
    json_basenames = {Path(f).name: f for f in json_filenames}
    
    # Load clustering results
    cluster_map = {}  # Maps filename -> cluster_id
    with open(clustering_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            cluster_id = int(row['cluster'])
            # Try exact match first, then basename match
            cluster_map[filename] = cluster_id
            cluster_map[Path(filename).name] = cluster_id
    
    # Find matches and count cluster distribution
    matched_clusters = []
    unmatched = []
    
    for basename, full_path in json_basenames.items():
        # Try both full path and basename
        if full_path in cluster_map:
            matched_clusters.append(cluster_map[full_path])
        elif basename in cluster_map:
            matched_clusters.append(cluster_map[basename])
        else:
            unmatched.append(full_path)
    
    # Analyze cluster distribution
    cluster_counts = Counter(matched_clusters)
    
    if cluster_counts:
        largest_cluster = cluster_counts.most_common(1)[0][0]
        largest_cluster_size = cluster_counts[largest_cluster]
        largest_cluster_percentage = (largest_cluster_size / len(matched_clusters)) * 100
    else:
        largest_cluster = None
        largest_cluster_size = 0
        largest_cluster_percentage = 0.0
    
    # Create result dictionary
    result = {
        'json_file': json_file,
        'clustering_csv': clustering_csv,
        'total_images': len(json_filenames),
        'matched_images': len(matched_clusters),
        'unmatched_images_count': len(unmatched),
        'cluster_distribution': dict(cluster_counts),
        'largest_cluster': largest_cluster,
        'largest_cluster_size': largest_cluster_size,
        'largest_cluster_percentage': round(largest_cluster_percentage, 2),
        'unmatched_images': unmatched
    }
    
    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Analysis saved to: {output_path}")
    
    return result


def analyze_cluster_overlap_multiplejsons(json_files, clustering_csv, output_path=None):
    """
    Analyze cluster overlap for multiple JSON files COMBINED together.
    
    This combines all filenames from all JSON files and analyzes them as a single group.
    
    Parameters:
    -----------
    json_files : list of str
        List of paths to JSON files to analyze
    clustering_csv : str
        Path to CSV file with clustering results
    output_path : str, optional
        Path to save combined analysis results. If None, doesn't save.
    
    Returns:
    --------
    dict
        Dictionary containing combined analysis results
    """
    
    print(f"\n{'='*60}")
    print(f"Combining {len(json_files)} JSON files for analysis...")
    print(f"{'='*60}")
    
    # Combine all filenames from all JSON files
    all_filenames = []
    json_sources = {}  # Track which JSON each filename came from
    
    for json_file in json_files:
        json_name = Path(json_file).stem
        
        # Load JSON file
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Extract filenames
        if isinstance(data, dict) and 'filenames' in data:
            filenames = data['filenames']
        elif isinstance(data, list):
            filenames = data
        else:
            raise ValueError(f"JSON {json_file} must contain 'filenames' key or be a list of filenames")
        
        print(f"  {json_name}: {len(filenames)} images")
        
        # Add to combined list
        all_filenames.extend(filenames)
        
        # Track source
        for fname in filenames:
            json_sources[fname] = json_name
    
    print(f"\nTotal combined images: {len(all_filenames)}")
    
    # Normalize paths to just filenames for matching
    json_basenames = {Path(f).name: f for f in all_filenames}
    
    # Load clustering results
    cluster_map = {}  # Maps filename -> cluster_id
    with open(clustering_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            cluster_id = int(row['cluster'])
            # Try exact match first, then basename match
            cluster_map[filename] = cluster_id
            cluster_map[Path(filename).name] = cluster_id
    
    # Find matches and count cluster distribution
    matched_clusters = []
    unmatched = []
    matched_filenames = []
    
    for basename, full_path in json_basenames.items():
        # Try both full path and basename
        if full_path in cluster_map:
            matched_clusters.append(cluster_map[full_path])
            matched_filenames.append(full_path)
        elif basename in cluster_map:
            matched_clusters.append(cluster_map[basename])
            matched_filenames.append(full_path)
        else:
            unmatched.append(full_path)
    
    # Analyze cluster distribution
    cluster_counts = Counter(matched_clusters)
    
    if cluster_counts:
        largest_cluster = cluster_counts.most_common(1)[0][0]
        largest_cluster_size = cluster_counts[largest_cluster]
        largest_cluster_percentage = (largest_cluster_size / len(matched_clusters)) * 100
        if largest_cluster==-1:
            try:
                largest_cluster = cluster_counts.most_common(2)[1][0]
                largest_cluster_size = cluster_counts[largest_cluster]
                largest_cluster_percentage = (largest_cluster_size / len(matched_clusters)) * 100
            except:  #only noise events
                largest_cluster = -1
                largest_cluster_size = 0
                largest_cluster_percentage = 0
    else:
        largest_cluster = None
        largest_cluster_size = 0
        largest_cluster_percentage = 0.0
    
    # Create result dictionary
    result = {
        'json_files': json_files,
        'clustering_csv': clustering_csv,
        'total_images': len(all_filenames),
        'matched_images': len(matched_clusters),
        'unmatched_images_count': len(unmatched),
        'cluster_distribution': dict(cluster_counts),
        'largest_cluster': largest_cluster,
        'largest_cluster_size': largest_cluster_size,
        'largest_cluster_percentage': round(largest_cluster_percentage, 2),
        'unmatched_images': unmatched
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMBINED ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"Total images: {result['total_images']}")
    print(f"Matched images: {result['matched_images']}")
    print(f"Unmatched images: {result['unmatched_images_count']}")
    print(f"\nCluster distribution:")
    for cluster_id, count in sorted(result['cluster_distribution'].items()):
        percentage = (count / result['matched_images']) * 100 if result['matched_images'] > 0 else 0
        print(f"  Cluster {cluster_id}: {count} images ({percentage:.1f}%)")
    print(f"\nLargest cluster: {result['largest_cluster']}")
    print(f"Largest cluster size: {result['largest_cluster_size']}")
    print(f"Largest cluster percentage: {result['largest_cluster_percentage']}%")
    
    # Save to file if requested
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\nCombined analysis saved to: {output_path}")
    
    return result
