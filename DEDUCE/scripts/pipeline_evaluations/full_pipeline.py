"""
Unified Semantic Evaluation Pipeline

A single script that orchestrates:
1. Semantic ID prediction (zero-shot classification)
2. Clustering analysis
3. Importance metric calculation
4. Full evaluation with visualizations

Each component can be enabled/disabled via run flags in this script.
Configuration is loaded from an .ini file.
"""

import os
import json
from pathlib import Path
from deduce.dataset_evaluation_pipeline import DatasetEvaluationPipeline
from deduce.utils.export import filter_predictions
from deduce.utils.visualizations import plot_cluster_percentages
from deduce.utils.dataeval import cluster_and_save, analyze_cluster_overlap_multiplejsons


# ============================================================================
# RUN FLAGS - Control which pipeline components to execute
# ============================================================================
RUN_SEMANTIC_ID = True          # Run semantic ID predictions
RUN_CLUSTERING = True           # Run clustering analysis
RUN_IMPORTANCE_METRIC = True     # Calculate importance metrics
RUN_FULL_EVALUATION = False      # Run complete evaluation with visualizations


# ============================================================================
# CONFIGURATION
# ============================================================================
# Path to your config file

# CCT - DayNight
# CONFIG_PATH = "../../configs/cct/daynight_1/eva_multisemantics_v3_bddsynth.ini"

# BDD100k - DayNight
CONFIG_PATH = "../../configs/bdd100k/daynight_1/eva_multisemantics_v2_cctsynth.ini"

# BDD100k - ClearRainy
# CONFIG_PATH = "../../configs/bdd100k/clearrainy_1/eva_multisemantics_v2.ini"

# BDD100k - ClearSnowy
# CONFIG_PATH = "../../configs/bdd100k/summerwinter_1/eva_multisemantics_v2.ini"

# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    print("="*80)
    print("UNIFIED SEMANTIC EVALUATION PIPELINE")
    print("="*80)
    print(f"\nConfiguration: {CONFIG_PATH}")
    print(f"\nRun Flags:")
    print(f"  - Semantic ID:        {RUN_SEMANTIC_ID}")
    print(f"  - Clustering:         {RUN_CLUSTERING}")
    print(f"  - Importance Metric:  {RUN_IMPORTANCE_METRIC}")
    print(f"  - Full Evaluation:    {RUN_FULL_EVALUATION}")
    print("="*80 + "\n")
    
    # Initialize pipeline
    pipeline = DatasetEvaluationPipeline(CONFIG_PATH)
    
    # Get configuration values
    save_path = pipeline.config_manager.get('EVALUATION', {}).get('output_path')
    clustering_config = pipeline.config_manager.get('CLUSTERING', {})
    descriptor_list_str = pipeline.config_manager.get('SEMANTICS', {}).get('descriptors', '')
    descriptor_list = [d.strip() for d in descriptor_list_str.split(',') if d.strip()]
    
    # Create output directory
    Path(save_path).mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: SEMANTIC ID PREDICTION
    # ========================================================================
    if RUN_SEMANTIC_ID:
        print("\n" + "="*80)
        print("STEP 1: SEMANTIC ID PREDICTION")
        print("="*80)
        
        # Load dataset
        dataset = pipeline.load_dataset()
        print(f"Loaded dataset: {len(dataset)} images")
        
        # Run zero-shot prediction
        results = pipeline.predict(dataset)
        print("Predictions completed")
        
        # Evaluate results (useful if we have ground truth labels)
        evaluation = pipeline.evaluate(results)
        print("Evaluation completed")
        
        # Generate visualizations
        pipeline.visualize(results, dataset)
        print(f"Visualizations saved to: {save_path}")
        
        # Export filtered predictions by confidence levels
        print("\nExporting filtered predictions by confidence...")
        min_margin = float(clustering_config.get('min_margin', 0.010))
        max_margin = float(clustering_config.get('max_margin', 0.025))
        
        for descriptor in descriptor_list:
            each_desc = descriptor.split('_')
            
            # Filter for first category
            filter_predictions(
                os.path.join(save_path, 'filename_margins_results.json'),
                descriptor=descriptor,
                prediction=each_desc[0],
                min_margin=min_margin,
                max_margin=max_margin,
                save_path=os.path.join(save_path, f"{each_desc[0]}_filenames.json")
            )
            
            # Filter for second category
            filter_predictions(
                os.path.join(save_path, 'filename_margins_results.json'),
                descriptor=descriptor,
                prediction=each_desc[1],
                min_margin=min_margin,
                max_margin=max_margin,
                save_path=os.path.join(save_path, f"{each_desc[1]}_filenames.json")
            )
    
    # ========================================================================
    # STEP 2: CLUSTERING ANALYSIS
    # ========================================================================
    if RUN_CLUSTERING:
        print("\n" + "="*80)
        print("STEP 2: CLUSTERING ANALYSIS")
        print("="*80)
        
        # Reload dataset if not already loaded
        if not RUN_SEMANTIC_ID:
            dataset = pipeline.load_dataset()
        
        # Get clustering parameters from config
        n_clusters = int(clustering_config.get('n_clusters', 6))
        model_path = clustering_config.get('model_path', None)
        model_name = clustering_config.get('model_name', 'resnet18')
        clustering_method = clustering_config.get('method', 'kmeans')
        
        print(f"Clustering parameters:")
        print(f"  - n_clusters: {n_clusters}")
        print(f"  - model: {model_name}")
        print(f"  - method: {clustering_method}")
        
        # Run clustering
        cluster_and_save(
            output_path=os.path.join(save_path, 'filename_clusters.csv'),
            n_clusters=n_clusters,
            dataset=dataset,
            model_path=model_path,
            model_name=model_name,
            clustering_method=clustering_method,
        )
        
        print("✓ Clustering analysis complete\n")
    
    # ========================================================================
    # STEP 3: IMPORTANCE METRIC CALCULATION
    # ========================================================================
    if RUN_IMPORTANCE_METRIC:
        print("\n" + "="*80)
        print("STEP 3: IMPORTANCE METRIC CALCULATION")
        print("="*80)
        
        # Get confidence levels to use from config
        confidence_levels_str = clustering_config.get('confidence_levels', 'high,medium,low')
        confidence_levels = [level.strip() for level in confidence_levels_str.split(',')]
        
        print(f"Using confidence levels: {confidence_levels}")
        
        largest_cluster_percentages = {}
        total_images = {}
        
        print("Analyzing cluster overlap for each descriptor...")
        
        for descriptor in descriptor_list:
            each_desc = descriptor.split('_')
            
            # Build file lists based on requested confidence levels
            desc0_json_files = [
                os.path.join(save_path, f"{each_desc[0]}_filenames_{level}.json")
                for level in confidence_levels
            ]
            
            desc1_json_files = [
                os.path.join(save_path, f"{each_desc[1]}_filenames_{level}.json")
                for level in confidence_levels
            ]
            
            # Analyze cluster overlap
            desc0_result = analyze_cluster_overlap_multiplejsons(
                json_files=desc0_json_files,
                clustering_csv=os.path.join(save_path, 'filename_clusters.csv'),
                output_path=os.path.join(save_path, f"{each_desc[0]}_cluster_analysis.json")
            )
            
            desc1_result = analyze_cluster_overlap_multiplejsons(
                json_files=desc1_json_files,
                clustering_csv=os.path.join(save_path, 'filename_clusters.csv'),
                output_path=os.path.join(save_path, f"{each_desc[1]}_cluster_analysis.json")
            )
            
            # Store results
            total_images[each_desc[0]] = desc0_result['total_images']
            total_images[each_desc[1]] = desc1_result['total_images']
            largest_cluster_percentages[each_desc[0]] = desc0_result['largest_cluster_percentage']
            largest_cluster_percentages[each_desc[1]] = desc1_result['largest_cluster_percentage']
        
        # Sort by largest cluster percentage (importance metric)
        sorted_items = sorted(largest_cluster_percentages.items(), 
                            key=lambda x: x[1], reverse=True)
        
        print("\n" + "-"*80)
        print("IMPORTANCE RANKING (by cluster concentration):")
        print("-"*80)
        for key, value in sorted_items:
            print(f"  {key:20s}: {value:6.2f}% (n={total_images[key]:4d} images)")
        print("-"*80)
        
        # Create structured data for export
        sorted_data = [
            {
                "descriptor": key,
                "largest_cluster_percentage": value,
                "total_images": total_images[key]
            }
            for key, value in sorted_items
        ]
        
        # Save results
        output_file = os.path.join(save_path, 'descriptor_cluster_percentages.json')
        with open(output_file, 'w') as f:
            json.dump(sorted_data, f, indent=2)
        
        print(f"\n✓ Importance metrics saved to: {output_file}")
        
        # Generate visualization
        plot_cluster_percentages(
            input_file=output_file,
            output_dir=os.path.join(save_path, 'cluster_viz'),
            aspect_ratio=(24, 9),
            axis_font_size = 20,
            label_font_size= 13
        )
        
        print("✓ Importance metric calculation complete\n")
    
    # ========================================================================
    # STEP 4: FULL EVALUATION (if not already done in Step 1)
    # ========================================================================
    if RUN_FULL_EVALUATION and not RUN_SEMANTIC_ID:
        print("\n" + "="*80)
        print("STEP 4: FULL EVALUATION")
        print("="*80)
        
        dataset = pipeline.load_dataset()
        results = pipeline.predict(dataset)
        evaluation = pipeline.evaluate(results)
        pipeline.visualize(results, dataset)
        
        print("✓ Full evaluation complete\n")
    
    # ========================================================================
    # PIPELINE COMPLETE
    # ========================================================================
    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nAll results saved to: {save_path}")
    print("\nKey outputs:")
    print(f"  - Predictions:              filename_margins_results.json")
    print(f"  - Distribution analysis:    distribution_results.json")
    print(f"  - Clustering results:       filename_clusters.csv")
    print(f"  - Importance ranking:       descriptor_cluster_percentages.json")
    print(f"  - Visualizations:           *.png files")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()