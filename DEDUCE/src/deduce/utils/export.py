"""
Simple export utility for semantic distribution results.

This is for further analysis or input to an LLM
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import pdb

def export_distribution_results(evaluation_results: Dict[str, Any], 
                                save_path: str,
                                semantic_descriptors: Optional[List] = None,
                                include_raw_counts: bool = False) -> None:
    """
    Export semantic distribution results to JSON
    
    Args:
        evaluation_results: Results from Evaluator.evaluate_predictions()
        save_path: Path to save JSON file
        include_raw_counts: Whether to include raw counts (default: False, percentages only)
    """
    # Filter out summary
    descriptors = {k: v for k, v in evaluation_results.items() if k != 'summary'}
    
    output = {
        'metadata': {
            'total_images': evaluation_results.get('summary', {}).get('num_descriptors', 0),
            'num_descriptors': len(descriptors)
        },
        'distributions': {}
    }

    # Build descriptor definitions if semantic_descriptors provided
    if semantic_descriptors:
        descriptor_defs = {}
        for desc in semantic_descriptors:
            descriptor_defs[desc.name] = {
                'category': desc.name,
                'keywords': desc.categories
            }
        output['descriptor_definitions'] = descriptor_defs
    
    for desc_name, results in descriptors.items():
        distribution = results.get('metrics', {}).get('distribution', {})
        category_counts = distribution.get('category_counts', {})
        
        # Calculate total predictions for this descriptor
        total = sum(category_counts.values())
        
        # Calculate percentages
        percentages = {
            category: (count / total * 100) if total > 0 else 0.0
            for category, count in category_counts.items()
        }
        
        descriptor_output = {
            'categories': list(category_counts.keys()),
            'distribution_percentages': percentages
        }
        
        if include_raw_counts:
            descriptor_output['raw_counts'] = category_counts
            descriptor_output['total_predictions'] = total
        
        output['distributions'][desc_name] = descriptor_output
    
    # Save to JSON
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Distribution results saved to {save_path}")



def export_filename_margin_results(evaluation_results: Dict[str, Any],
                         save_path: str,
                         min_margin: Optional[float] = None) -> None:
    """
    Export prediction margins and filenames to JSON
    
    Args:
        evaluation_results: Results from Evaluator.evaluate_predictions()
        save_path: Path to save JSON file
        min_margin: Optional minimum margin threshold to filter results
    """
    from pathlib import Path
    import json
    
    output = {}
    
    for desc_name, results in evaluation_results.items():
        if desc_name == 'summary':
            continue
            
        # margin_data = results.get('filename_margin_data', [])
        margin_data = results.get_predictions_with_margins()

        # Filter by minimum margin if specified
        if min_margin is not None:
            margin_data = [item for item in margin_data if item['margin'] >= min_margin]
        
        output[desc_name] = margin_data
    
    # Save to JSON
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"Margin results saved to {save_path}")

def filter_predictions(json_path: str, 
                      descriptor: str, 
                      prediction: str, 
                      min_margin: float,
                      max_margin: float,
                      save_path: Optional[str] = None) -> Dict[str, List[str]]:
    """
    Filter filenames by prediction and margin ranges into confidence levels
    
    Args:
        json_path: Path to margin results JSON
        descriptor: Descriptor name (e.g., "day_night")
        prediction: Predicted category to filter for (e.g., "night")
        min_margin: Minimum margin threshold (low/medium boundary)
        max_margin: Maximum margin threshold (medium/high boundary)
        save_path: Optional base path to save filtered filenames as JSON
                  (will create _low, _medium, _high variants)
    
    Returns:
        Dictionary with 'low', 'medium', 'high' keys containing filename lists
    """
    import json
    from pathlib import Path
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data.get(descriptor, [])
    
    # Separate into three confidence levels
    low_confidence = []
    medium_confidence = []
    high_confidence = []
    
    for item in results:
        if item['prediction'] == prediction:
            margin = item['margin']
            if margin < min_margin:
                low_confidence.append(item['filename'])
            elif margin < max_margin:
                medium_confidence.append(item['filename'])
            else:
                high_confidence.append(item['filename'])
    
    confidence_levels = {
        'low': low_confidence,
        'medium': medium_confidence,
        'high': high_confidence
    }
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save separate files for each confidence level
        for level, filenames in confidence_levels.items():
            level_path = save_path.parent / f"{save_path.stem}_{level}{save_path.suffix}"
            
            output = {
                'prediction': prediction,
                'confidence_level': level,
                'margin_range': {
                    'low': f'< {min_margin}',
                    'medium': f'{min_margin} - {max_margin}',
                    'high': f'> {max_margin}'
                }[level],
                'count': len(filenames),
                'filenames': filenames
            }
            
            with open(level_path, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"Saved {len(filenames)} {level} confidence filenames to {level_path}")
    
    return confidence_levels


"""
Compare two semantic distribution JSON files
"""

def compare_distribution_results(json_path_1: str, 
                                 json_path_2: str,
                                 output_path: str = None,
                                 min_change_threshold: float = 25.0) -> Dict[str, Any]:
    """
    Compare two distribution JSON files and highlight significant differences
    
    Args:
        json_path_1: Path to first JSON file (baseline)
        json_path_2: Path to second JSON file (comparison)
        output_path: Optional path to save comparison JSON
        min_change_threshold: Minimum percentage point change to report (default: 25.0)
        
    Returns:
        Dictionary with comparison results
    """
    # Load both JSON files
    with open(json_path_1, 'r') as f:
        data1 = json.load(f)
    
    with open(json_path_2, 'r') as f:
        data2 = json.load(f)
    
    # Check descriptors match
    desc1 = set(data1['distributions'].keys())
    desc2 = set(data2['distributions'].keys())
    
    if desc1 != desc2:
        print(f"⚠️  Warning: Descriptors don't match!")
        print(f"   File 1: {desc1}")
        print(f"   File 2: {desc2}")
        print(f"   Only comparing common descriptors: {desc1 & desc2}")
    
    common_descriptors = desc1 & desc2
    
    # Build comparison output
    output = {
        'metadata': {
            'baseline': str(json_path_1),
            'comparison': str(json_path_2),
            'baseline_images': data1['metadata'].get('total_images', 'unknown'),
            'comparison_images': data2['metadata'].get('total_images', 'unknown'),
            'min_change_threshold': min_change_threshold
        },
        'descriptor_definitions': data1.get('descriptor_definitions', {}),
        'significant_changes': {},
        'all_changes': {}
    }
    
    # Compare each descriptor
    for desc_name in sorted(common_descriptors):
        dist1 = data1['distributions'][desc_name]['distribution_percentages']
        dist2 = data2['distributions'][desc_name]['distribution_percentages']
        
        # Check categories match
        cats1 = set(dist1.keys())
        cats2 = set(dist2.keys())
        
        if cats1 != cats2:
            print(f"⚠️  Warning: Categories don't match for {desc_name}")
            print(f"   Using common categories: {cats1 & cats2}")
        
        common_categories = cats1 & cats2
        
        # Calculate changes for each category
        changes = []
        for category in common_categories:
            baseline_pct = dist1[category]
            comparison_pct = dist2[category]
            change = comparison_pct - baseline_pct
            
            changes.append({
                'category': category,
                'baseline_percentage': round(baseline_pct, 2),
                'comparison_percentage': round(comparison_pct, 2),
                'absolute_change': round(change, 2),
                'relative_change': round((change / baseline_pct * 100) if baseline_pct > 0 else 0, 2)
            })
        
        # Sort by absolute change magnitude
        changes.sort(key=lambda x: abs(x['absolute_change']), reverse=True)
        
        # Filter significant changes (above threshold)
        significant = [c for c in changes if abs(c['absolute_change']) >= min_change_threshold]
        
        # Store results
        output['all_changes'][desc_name] = {
            'changes': changes,
            'max_change': changes[0] if changes else None
        }
        
        if significant:
            output['significant_changes'][desc_name] = {
                'num_significant': len(significant),
                'changes': significant,
                'summary': _generate_change_summary(significant)
            }
    
    # Add overall summary
    output['summary'] = {
        'total_descriptors_compared': len(common_descriptors),
        'descriptors_with_significant_changes': len(output['significant_changes']),
        'top_changes': _get_top_changes(output['all_changes'], top_n=5)
    }
    
    # Save to file if output_path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"✅ Comparison results saved to {output_path}")
    
    return output


def _generate_change_summary(changes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a human-readable summary of changes"""
    increases = [c for c in changes if c['absolute_change'] > 0]
    decreases = [c for c in changes if c['absolute_change'] < 0]
    
    summary = {
        'num_increases': len(increases),
        'num_decreases': len(decreases),
        'largest_increase': increases[0] if increases else None,
        'largest_decrease': decreases[0] if decreases else None
    }
    
    return summary


def _get_top_changes(all_changes: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
    """Get top N changes across all descriptors, avoiding complementary duplicates"""
    all_category_changes = []
    
    for desc_name, data in all_changes.items():
        changes = data['changes']
        
        # Check if this descriptor has complementary changes (binary categories)
        # If all changes sum to ~0, it's a binary/complementary system
        total_change = sum(c['absolute_change'] for c in changes)
        is_complementary = abs(total_change) < 0.1  # Allow small rounding errors
        
        if is_complementary:
            # Only take the change with largest absolute value
            max_change = max(changes, key=lambda x: abs(x['absolute_change']))
            all_category_changes.append({
                'descriptor': desc_name,
                'category': max_change['category'],
                'absolute_change': max_change['absolute_change'],
                'baseline_percentage': max_change['baseline_percentage'],
                'comparison_percentage': max_change['comparison_percentage'],
                'is_complementary': True
            })
        else:
            # Non-complementary: include all changes
            for change in changes:
                all_category_changes.append({
                    'descriptor': desc_name,
                    'category': change['category'],
                    'absolute_change': change['absolute_change'],
                    'baseline_percentage': change['baseline_percentage'],
                    'comparison_percentage': change['comparison_percentage'],
                    'is_complementary': False
                })
    
    # Sort by absolute change magnitude
    all_category_changes.sort(key=lambda x: abs(x['absolute_change']), reverse=True)
    
    return all_category_changes[:top_n]


def print_comparison_summary(comparison_results: Dict[str, Any]) -> None:
    """Print a readable summary of comparison results"""
    print("\n" + "="*60)
    print("DISTRIBUTION COMPARISON SUMMARY")
    print("="*60)
    
    summary = comparison_results['summary']
    metadata = comparison_results['metadata']
    
    print(f"\nBaseline:   {metadata['baseline']}")
    print(f"            ({metadata['baseline_images']} images)")
    print(f"Comparison: {metadata['comparison']}")
    print(f"            ({metadata['comparison_images']} images)")
    print(f"\nThreshold: ±{metadata['min_change_threshold']}% change")
    
    print(f"\n📊 Results:")
    print(f"   • {summary['total_descriptors_compared']} descriptors compared")
    print(f"   • {summary['descriptors_with_significant_changes']} with significant changes")
    
    print(f"\n🔝 Top 5 Changes:")
    for i, change in enumerate(summary['top_changes'], 1):
        direction = "↑" if change['absolute_change'] > 0 else "↓"
        print(f"   {i}. {change['descriptor']} - {change['category']}: "
              f"{change['baseline_percentage']:.1f}% → {change['comparison_percentage']:.1f}% "
              f"({direction} {abs(change['absolute_change']):.1f}%)")
    
    # Print details for each descriptor with significant changes
    if comparison_results['significant_changes']:
        print(f"\n📋 Significant Changes by Descriptor:")
        for desc_name, data in comparison_results['significant_changes'].items():
            print(f"\n   {desc_name.upper().replace('_', ' ')}:")
            print(f"   {data['num_significant']} significant changes detected")
            
            for change in data['changes'][:3]:  # Top 3
                direction = "↑" if change['absolute_change'] > 0 else "↓"
                print(f"      • {change['category']}: "
                      f"{change['baseline_percentage']:.1f}% → {change['comparison_percentage']:.1f}% "
                      f"({direction} {abs(change['absolute_change']):.1f}%)")
    
    print("\n" + "="*60 + "\n")


