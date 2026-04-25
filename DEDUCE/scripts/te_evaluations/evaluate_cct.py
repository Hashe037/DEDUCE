import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold
from tabulate import tabulate
import time
from datetime import datetime

class FolderDataset(Dataset):
    def __init__(self, root_folder, transform=None):
        """ 
        Args:
            root_folder: Path to folder containing 'empty' and 'nonempty' subfolders
            transform: Image transforms to apply
        """
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Map folder names to labels
        label_map = {'empty': 1, 'nonempty': 0}
        
        # Traverse subfolders
        for folder_name, label in label_map.items():
            folder_path = os.path.join(root_folder, folder_name)
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} not found, skipping...")
                continue
            
            # Walk through the directory and all subdirectories
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for img_name in filenames:
                    if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
                        img_path = os.path.join(dirpath, img_name)
                        self.images.append(img_path)
                        self.labels.append(label)
        
        print(f"Loaded {len(self.images)} images")
        print(f"  Empty (1): {sum(self.labels)}")
        print(f"  Non-empty (0): {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path  # Return path for debugging

# Evaluation function for a single fold
def evaluate_fold(model_path, test_loader, device, verbose=False, class_weight=None):
    """
    Evaluate the model on a single fold.
    
    Args:
        model_path: Path to the saved model weights
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        verbose: Whether to print detailed evaluation results
        class_weight: Class weighting scheme. Options:
                     - None: No weighting (default)
                     - 'balanced': Automatic weighting inversely proportional to class frequencies
                     - dict: Custom weights, e.g., {0: 1.0, 1: 2.0}
    
    Returns:
        Dictionary containing evaluation metrics
    """
    # Load model
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 2)  # 2 classes: empty (1) and not empty (0)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Evaluation
    all_predictions = []
    all_labels = []
    all_probabilities = []
    misclassified = []
    
    with torch.no_grad():
        for images, labels, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Track misclassified images
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified.append({
                        'path': paths[i],
                        'true_label': labels[i].item(),
                        'predicted_label': predicted[i].item(),
                        'confidence': probabilities[i][predicted[i]].item()
                    })
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics with optional class weighting
    if class_weight is not None:
        # Calculate sample weights based on class_weight parameter
        if class_weight == 'balanced':
            # Automatic balancing: weight inversely proportional to class frequency
            class_counts = np.bincount(all_labels)
            class_weights = len(all_labels) / (len(class_counts) * class_counts)
        else:
            # Assume class_weight is a dict: {0: weight_for_class_0, 1: weight_for_class_1}
            class_weights = np.array([class_weight[0], class_weight[1]])
        
        sample_weights = class_weights[all_labels]
    else:
        sample_weights = None
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions, sample_weight=sample_weights)
    precision = precision_score(all_labels, all_predictions, average='binary', 
                                sample_weight=sample_weights, zero_division=0)
    recall = recall_score(all_labels, all_predictions, average='binary', 
                          sample_weight=sample_weights, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average='binary', 
                  sample_weight=sample_weights, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    if verbose:
        # Print results
        print(f"    Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall:    {recall:.4f}")
        print(f"    F1 Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'misclassified': len(misclassified),
        'num_samples': len(all_labels)
    }

def evaluate_model_kfold(model_path, test_folder, k_folds=10, batch_size=32, verbose=True, random_seed=42, class_weight=None):
    """
    Evaluate the model using k-fold cross-validation on a test dataset.
    
    Args:
        model_path: Path to the saved model weights
        test_folder: Path to folder containing 'empty' and 'nonempty' subfolders
        k_folds: Number of folds for cross-validation (default: 10)
        batch_size: Batch size for evaluation
        verbose: Whether to print detailed evaluation results
        random_seed: Random seed for reproducibility
        class_weight: Class weighting scheme. Options:
                     - None: No weighting (default)
                     - 'balanced': Automatic weighting inversely proportional to class frequencies
                     - dict: Custom weights, e.g., {0: 1.0, 1: 2.0}
    
    Returns:
        Dictionary containing evaluation metrics with means and standard deviations
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if verbose:
        print(f"Using device: {device}")
    
    # Transforms (same as training)
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    test_dataset = FolderDataset(test_folder, transform=test_transform)
    
    # Initialize k-fold cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    
    # Store results for each fold
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'confusion_matrix': [],
        'misclassified': [],
        'num_samples': []
    }
    
    if verbose:
        print(f"\nPerforming {k_folds}-fold cross-validation...")
        print(f"Total samples: {len(test_dataset)}")
        print(f"Samples per fold (approx): {len(test_dataset) // k_folds}\n")
    
    # Perform k-fold cross-validation
    for fold, (train_indices, test_indices) in enumerate(kfold.split(range(len(test_dataset)))):
        if verbose:
            print(f"Evaluating Fold {fold + 1}/{k_folds}...")
        
        # Create data loader for this fold
        test_subset = Subset(test_dataset, test_indices)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        
        # Evaluate on this fold
        fold_result = evaluate_fold(model_path, test_loader, device, verbose=verbose, class_weight=class_weight)
        
        # Store results
        for key in fold_results.keys():
            fold_results[key].append(fold_result[key])
    
    # Calculate mean and standard deviation for each metric
    results_summary = {}
    for key in ['accuracy', 'precision', 'recall', 'f1']:
        values = np.array(fold_results[key])
        results_summary[f'{key}_mean'] = np.mean(values)
        results_summary[f'{key}_std'] = np.std(values)
        results_summary[f'{key}_values'] = values
    
    # Calculate average confusion matrix
    if 'confusion_matrix' in fold_results and len(fold_results['confusion_matrix']) > 0:
        cm_sum = np.sum(fold_results['confusion_matrix'], axis=0)
        results_summary['confusion_matrix_avg'] = cm_sum / k_folds
    else:
        results_summary['confusion_matrix_avg'] = None
    
    # Total misclassified and samples
    results_summary['total_misclassified'] = sum(fold_results['misclassified'])
    results_summary['total_samples'] = sum(fold_results['num_samples'])
    
    if verbose:
        # Print summary results
        print("\n" + "="*60)
        print(f"{k_folds}-FOLD CROSS-VALIDATION RESULTS")
        print("="*60)
        print(f"\nOverall Metrics (Mean ± Std):")
        print(f"  Accuracy:  {results_summary['accuracy_mean']:.4f} ± {results_summary['accuracy_std']:.4f}")
        print(f"  Precision: {results_summary['precision_mean']:.4f} ± {results_summary['precision_std']:.4f}")
        print(f"  Recall:    {results_summary['recall_mean']:.4f} ± {results_summary['recall_std']:.4f}")
        print(f"  F1 Score:  {results_summary['f1_mean']:.4f} ± {results_summary['f1_std']:.4f}")
        
        print(f"\nConfidence Intervals (95%):")
        print(f"  Accuracy:  {results_summary['accuracy_mean']:.4f} ± {1.96 * results_summary['accuracy_std']:.4f}")
        print(f"  Precision: {results_summary['precision_mean']:.4f} ± {1.96 * results_summary['precision_std']:.4f}")
        print(f"  Recall:    {results_summary['recall_mean']:.4f} ± {1.96 * results_summary['recall_std']:.4f}")
        print(f"  F1 Score:  {results_summary['f1_mean']:.4f} ± {1.96 * results_summary['f1_std']:.4f}")
        
        if results_summary['confusion_matrix_avg'] is not None:
            print(f"\nAverage Confusion Matrix:")
            cm = results_summary['confusion_matrix_avg']
            print(f"                Predicted")
            print(f"              Non-empty  Empty")
            print(f"True Non-empty  {cm[0][0]:6.1f}  {cm[0][1]:6.1f}")
            print(f"True Empty      {cm[1][0]:6.1f}  {cm[1][1]:6.1f}")
        
        print(f"\nTotal misclassified: {results_summary['total_misclassified']}/{results_summary['total_samples']}")
        print("="*60 + "\n")
    
    return results_summary

def evaluate_multiple_kfold(model_list, dataset_list, k_folds=10, batch_size=32, verbose=True, random_seed=42, class_weight=None):
    """
    Evaluate multiple models on multiple datasets using k-fold cross-validation.
    
    Args:
        model_list: Dictionary mapping model names to model paths
        dataset_list: Dictionary mapping dataset names to dataset paths
        k_folds: Number of folds for cross-validation (default: 10)
        batch_size: Batch size for evaluation
        verbose: Whether to print detailed output
        random_seed: Random seed for reproducibility
        class_weight: Class weighting scheme. Options:
                     - None: No weighting (default)
                     - 'balanced': Automatic weighting inversely proportional to class frequencies
                     - dict: Custom weights, e.g., {0: 1.0, 1: 2.0}
    
    Returns:
        Dictionary containing all evaluation results
    """
    print("="*80)
    print(f"STARTING K-FOLD CROSS-VALIDATION EVALUATION")
    print(f"K-folds: {k_folds}")
    if class_weight is not None:
        if class_weight == 'balanced':
            print("Class weighting: BALANCED (automatic)")
        else:
            print(f"Class weighting: CUSTOM {class_weight}")
    else:
        print("Class weighting: NONE")
    print("="*80)
    print(f"Models to evaluate: {len(model_list)}")
    print(f"Datasets to evaluate: {len(dataset_list)}")
    print(f"Total evaluations: {len(model_list) * len(dataset_list)}")
    print("="*80 + "\n")
    
    # Initialize results storage
    results = {
        'model_name': [],
        'dataset_name': [],
        'accuracy_mean': [],
        'accuracy_std': [],
        'precision_mean': [],
        'precision_std': [],
        'recall_mean': [],
        'recall_std': [],
        'f1_mean': [],
        'f1_std': [],
        'total_misclassified': [],
        'total_samples': []
    }
    
    # Store detailed fold results
    detailed_results = {}
    
    # Track progress
    total_evaluations = len(model_list) * len(dataset_list)
    completed_evaluations = 0
    start_time = time.time()
    
    # Evaluate each model on each dataset
    for model_name, model_path in model_list.items():
        for dataset_name, dataset_path in dataset_list.items():
            try:
                print(f"\n{'='*80}")
                print(f"Evaluating: {model_name} on {dataset_name}")
                print(f"Progress: {completed_evaluations + 1}/{total_evaluations}")
                print(f"{'='*80}")
                
                # Run k-fold evaluation
                eval_results = evaluate_model_kfold(
                    model_path=model_path,
                    test_folder=dataset_path,
                    k_folds=k_folds,
                    batch_size=batch_size,
                    verbose=verbose,
                    random_seed=random_seed,
                    class_weight=class_weight
                )
                
                # Store results
                results['model_name'].append(model_name)
                results['dataset_name'].append(dataset_name)
                results['accuracy_mean'].append(eval_results['accuracy_mean'])
                results['accuracy_std'].append(eval_results['accuracy_std'])
                results['precision_mean'].append(eval_results['precision_mean'])
                results['precision_std'].append(eval_results['precision_std'])
                results['recall_mean'].append(eval_results['recall_mean'])
                results['recall_std'].append(eval_results['recall_std'])
                results['f1_mean'].append(eval_results['f1_mean'])
                results['f1_std'].append(eval_results['f1_std'])
                results['total_misclassified'].append(eval_results['total_misclassified'])
                results['total_samples'].append(eval_results['total_samples'])
                
                # Store detailed results for later analysis
                detailed_results[f"{model_name}_{dataset_name}"] = eval_results
                
                completed_evaluations += 1
                
                # Print brief summary
                print(f"Summary: F1 = {eval_results['f1_mean']:.4f} ± {eval_results['f1_std']:.4f}, "
                      f"Acc = {eval_results['accuracy_mean']:.4f} ± {eval_results['accuracy_std']:.4f}")
            
            except Exception as e:
                print(f"Error evaluating {model_name} on {dataset_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Calculate total evaluation time
    total_time = time.time() - start_time
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create pivot tables for different metrics (showing mean ± std)
    def create_metric_table(df, metric_name):
        """Create a formatted table showing mean ± std for a metric"""
        mean_table = df.pivot(index='model_name', columns='dataset_name', values=f'{metric_name}_mean')
        std_table = df.pivot(index='model_name', columns='dataset_name', values=f'{metric_name}_std')
        
        # Create formatted table
        formatted_table = mean_table.copy()
        for col in formatted_table.columns:
            for idx in formatted_table.index:
                mean_val = mean_table.loc[idx, col]
                std_val = std_table.loc[idx, col]
                formatted_table.loc[idx, col] = f"{mean_val:.4f} ± {std_val:.4f}"
        
        return formatted_table, mean_table, std_table
    
    # Create tables for all metrics
    f1_formatted, f1_mean, f1_std = create_metric_table(results_df, 'f1')
    accuracy_formatted, accuracy_mean, accuracy_std = create_metric_table(results_df, 'accuracy')
    precision_formatted, precision_mean, precision_std = create_metric_table(results_df, 'precision')
    recall_formatted, recall_mean, recall_std = create_metric_table(results_df, 'recall')
    
    # Print F1 Score Table (primary metric)
    print("\n" + "="*80)
    print(f"F1 SCORE RESULTS ({k_folds}-Fold CV: Mean ± Std)")
    print("="*80)
    print(tabulate(f1_formatted, headers='keys', tablefmt='grid'))
    
    # Print Accuracy Table
    print("\n" + "="*80)
    print(f"ACCURACY RESULTS ({k_folds}-Fold CV: Mean ± Std)")
    print("="*80)
    print(tabulate(accuracy_formatted, headers='keys', tablefmt='grid'))
    
    # Print Precision Table
    print("\n" + "="*80)
    print(f"PRECISION RESULTS ({k_folds}-Fold CV: Mean ± Std)")
    print("="*80)
    print(tabulate(precision_formatted, headers='keys', tablefmt='grid'))
    
    # Print Recall Table
    print("\n" + "="*80)
    print(f"RECALL RESULTS ({k_folds}-Fold CV: Mean ± Std)")
    print("="*80)
    print(tabulate(recall_formatted, headers='keys', tablefmt='grid'))
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"K-folds: {k_folds}")
    print(f"Total models evaluated: {len(model_list)}")
    print(f"Total datasets evaluated: {len(dataset_list)}")
    print(f"Total evaluations completed: {completed_evaluations}/{total_evaluations}")
    print(f"Total evaluation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("="*80)
    
    return {
        'raw_results': results_df,
        'detailed_results': detailed_results,
        'f1_formatted': f1_formatted,
        'f1_mean': f1_mean,
        'f1_std': f1_std,
        'accuracy_formatted': accuracy_formatted,
        'accuracy_mean': accuracy_mean,
        'accuracy_std': accuracy_std,
        'precision_formatted': precision_formatted,
        'precision_mean': precision_mean,
        'precision_std': precision_std,
        'recall_formatted': recall_formatted,
        'recall_mean': recall_mean,
        'recall_std': recall_std,
        'k_folds': k_folds
    }

def save_results_kfold(results, output_dir='./results_kfold'):
    """
    Save k-fold cross-validation results to files.
    
    Args:
        results: Results dictionary from evaluate_multiple_kfold
        output_dir: Directory to save results to
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    k_folds = results['k_folds']
    
    # Save formatted tables as text files (exactly as displayed)
    print(f"\nSaving results to {output_dir}...")
    
    # Save F1 Score table
    f1_txt_path = os.path.join(output_dir, f'eval_kfold_f1_table_{timestamp}.txt')
    with open(f1_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"F1 SCORE RESULTS ({k_folds}-Fold CV: Mean ± Std)\n")
        f.write("="*80 + "\n")
        f.write(tabulate(results['f1_formatted'], headers='keys', tablefmt='grid'))
        f.write("\n")
    print(f"F1 score table saved to {f1_txt_path}")
    
    # Save Accuracy table
    acc_txt_path = os.path.join(output_dir, f'eval_kfold_accuracy_table_{timestamp}.txt')
    with open(acc_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"ACCURACY RESULTS ({k_folds}-Fold CV: Mean ± Std)\n")
        f.write("="*80 + "\n")
        f.write(tabulate(results['accuracy_formatted'], headers='keys', tablefmt='grid'))
        f.write("\n")
    print(f"Accuracy table saved to {acc_txt_path}")
    
    # Save Precision table
    precision_txt_path = os.path.join(output_dir, f'eval_kfold_precision_table_{timestamp}.txt')
    with open(precision_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"PRECISION RESULTS ({k_folds}-Fold CV: Mean ± Std)\n")
        f.write("="*80 + "\n")
        f.write(tabulate(results['precision_formatted'], headers='keys', tablefmt='grid'))
        f.write("\n")
    print(f"Precision table saved to {precision_txt_path}")
    
    # Save Recall table
    recall_txt_path = os.path.join(output_dir, f'eval_kfold_recall_table_{timestamp}.txt')
    with open(recall_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"RECALL RESULTS ({k_folds}-Fold CV: Mean ± Std)\n")
        f.write("="*80 + "\n")
        f.write(tabulate(results['recall_formatted'], headers='keys', tablefmt='grid'))
        f.write("\n")
    print(f"Recall table saved to {recall_txt_path}")
    
    # Save combined summary
    summary_txt_path = os.path.join(output_dir, f'eval_kfold_summary_{timestamp}.txt')
    with open(summary_txt_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"K-FOLD CROSS-VALIDATION RESULTS SUMMARY\n")
        f.write(f"K-folds: {k_folds}\n")
        f.write("="*80 + "\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"F1 SCORE RESULTS ({k_folds}-Fold CV: Mean ± Std)\n")
        f.write("="*80 + "\n")
        f.write(tabulate(results['f1_formatted'], headers='keys', tablefmt='grid'))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"ACCURACY RESULTS ({k_folds}-Fold CV: Mean ± Std)\n")
        f.write("="*80 + "\n")
        f.write(tabulate(results['accuracy_formatted'], headers='keys', tablefmt='grid'))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"PRECISION RESULTS ({k_folds}-Fold CV: Mean ± Std)\n")
        f.write("="*80 + "\n")
        f.write(tabulate(results['precision_formatted'], headers='keys', tablefmt='grid'))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write(f"RECALL RESULTS ({k_folds}-Fold CV: Mean ± Std)\n")
        f.write("="*80 + "\n")
        f.write(tabulate(results['recall_formatted'], headers='keys', tablefmt='grid'))
        f.write("\n")
    print(f"Combined summary saved to {summary_txt_path}")
    
    # Save raw results to CSV (for further analysis if needed)
    raw_csv_path = os.path.join(output_dir, f'eval_kfold_raw_results_{timestamp}.csv')
    results['raw_results'].to_csv(raw_csv_path, index=False)
    print(f"Raw results (CSV) saved to {raw_csv_path}")
    
    # Save all tables to a single Excel file with different sheets
    excel_path = os.path.join(output_dir, f'eval_kfold_{k_folds}fold_results_{timestamp}.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        results['f1_formatted'].to_excel(writer, sheet_name='F1 Scores (Mean±Std)')
        results['f1_mean'].to_excel(writer, sheet_name='F1 Mean')
        results['f1_std'].to_excel(writer, sheet_name='F1 Std')
        results['accuracy_formatted'].to_excel(writer, sheet_name='Accuracy (Mean±Std)')
        results['accuracy_mean'].to_excel(writer, sheet_name='Accuracy Mean')
        results['accuracy_std'].to_excel(writer, sheet_name='Accuracy Std')
        results['precision_formatted'].to_excel(writer, sheet_name='Precision (Mean±Std)')
        results['recall_formatted'].to_excel(writer, sheet_name='Recall (Mean±Std)')
        results['raw_results'].to_excel(writer, sheet_name='Raw Results', index=False)
    print(f"All results saved to Excel file: {excel_path}")
    
    return {
        'f1_txt': f1_txt_path,
        'accuracy_txt': acc_txt_path,
        'precision_txt': precision_txt_path,
        'recall_txt': recall_txt_path,
        'summary_txt': summary_txt_path,
        'raw_csv': raw_csv_path,
        'excel': excel_path
    }

# Usage example
if __name__ == "__main__":
    # Example usage (uncomment and modify as needed)
    
    # Define model paths and names
    # model_list = {
    #     'day_epoch_38': "./run_6_test_synth/day_1/model_epoch_38.pth",
    #     # 'night_epoch_38': "./run_6_test_synth/night_1/model_epoch_38.pth",
    #     'daynight_epoch_18': "./run_6_test_synth/daynight_1/model_epoch_18.pth",
    #     'daynight_withsynth_epoch_18': "./run_6_test_synth/daynight_synth_1/model_epoch_18.pth",
    # }

    # model_list = {
    #     'day_epoch_38': "./run_6_testandval_synth/day_1/model_epoch_38.pth",
    #     # 'night_epoch_38': "./run_6_testandval_synth/night_1/model_epoch_38.pth",
    #     'daynight_epoch_18': "./run_6_testandval_synth/daynight_1/model_epoch_18.pth",
    #     'daynight_withsynth_epoch_18': "./run_6_testandval_synth/daynight_synth_1/model_epoch_18.pth",
    # }
    
    # model_list = {
    #     'day_epoch_38': "./run_7_testandval_synth_enet/day_1/model_epoch_36.pth",
    #     'night_epoch_38': "./run_7_testandval_synth_enet/night_1/model_epoch_36.pth",
    #     'daynight_epoch_18': "./run_7_testandval_synth_enet/daynight_1/model_epoch_16.pth",
    #     'dayreal_nightsynth_epoch_18': "./run_7_testandval_synth_enet/day_real_night_synth_1/model_epoch_16.pth",
    #     'daynight_withsynth_epoch_18': "./run_7_testandval_synth_enet/daynight_synth_1/model_epoch_16.pth",
    # }

    # model_list = {
    #     'day_bestmodel': "./trans_test_train_results/run_2_synth_enet/day_1/best_model_loss.pth",
    #     'night_bestmodel': "./trans_test_train_results/run_2_synth_enet/night_1/best_model_loss.pth",
    #     'daynight_bestmodel': "./trans_test_train_results/run_2_synth_enet/daynight_1/best_model_loss.pth",
    #     'dayreal_nightsynth_bestmodel': "./trans_test_train_results/run_2_synth_enet/day_real_night_synth_1/best_model_loss.pth",
    #     'daynight_withsynth_bestmodel': "./trans_test_train_results/run_2_synth_enet/daynight_withsynth_1/best_model_loss.pth",
    #     'daynight_withsynth_bdd_bestmodel': "./trans_test_train_results/run_2_synth_enet/day_real_night_synth_bddaligned_1/best_model_loss.pth",
    # }

    # model_list = {
    #     'day_bestmodel': "./trans_test_train_results/run_2_synth_enet/day_1/best_model_loss.pth",
    #     'day_model1': "./trans_test_train_results/run_6_enet_autoclassweights_nocis_epochsaving/day_1/model_epoch_10.pth",
    #     'day_model2': "./trans_test_train_results/run_6_enet_autoclassweights_nocis_epochsaving/day_1/model_epoch_20.pth",
    #     'day_model3': "./trans_test_train_results/run_6_enet_autoclassweights_nocis_epochsaving/day_1/best_model_loss.pth",
    #     'day_model4': "./trans_test_train_results/run_6_enet_autoclassweights_nocis_epochsaving/day_1/best_model_acc.pth",
    #     'daynight_bestmodel': "./trans_test_train_results/run_2_synth_enet/daynight_1/best_model_loss.pth",
    #     'daynight_model1': "./trans_test_train_results/run_6_enet_autoclassweights_nocis_epochsaving/daynight_1/model_epoch_10.pth",
    #     'daynight_model2': "./trans_test_train_results/run_6_enet_autoclassweights_nocis_epochsaving/daynight_1/model_epoch_20.pth",
    #     'daynight_model3': "./trans_test_train_results/run_6_enet_autoclassweights_nocis_epochsaving/daynight_1/best_model_loss.pth",
    #     'daynight_model4': "./trans_test_train_results/run_6_enet_autoclassweights_nocis_epochsaving/daynight_1/best_model_acc.pth",
    # }

    model_list = {
        # 'day_bestmodel_acc': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving/day_1/best_model_acc.pth",  #PAPER RESULTS
        # 'day_bestmodel': "./trans_test_train_results/run_2_synth_enet/day_1/best_model_loss.pth",
        # 'day_bestmodel2': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving/day_1/best_model_acc.pth",
        # 'day_bestmodel_loss': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2/day_1/best_model_loss.pth",
        # 'day_bestmodel_dayval_loss': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2_dayval/day_1/best_model_loss.pth",
        # 'day_finalmodel': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving/day_1/final_model.pth",
        # 'daynight_bestmodel': "./trans_test_train_results/run_2_synth_enet/daynight_1/best_model_loss.pth",
        # 'daynight_bestmodel2': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving/daynight_1/best_model_loss.pth",
        # 'daynight_bestmodel2': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2/daynight_1/best_model_loss.pth",
        # 'daynight_bestmodel_acc': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving/daynight_1/best_model_acc.pth",
        # 'daynight_finalmodel': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving/daynight_1/final_model.pth",

        # Paper results, not super sure how trained
        'day_model_1': "/data2/CDAO/DENSE_public/models/cct/daynight/day_model_1.pth",
        'daynight_model_2': "/data2/CDAO/DENSE_public/models/cct/daynight/daynight_model_2.pth",


        # All of these had the cis_val be validation split
        # 'day_bestmodel_acc': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2_actualactual/day_1/best_model_acc.pth", #dayval
        # 'day_bestmodel_loss': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2_actualactual/day_1/best_model_loss.pth", #dayval
        # 'daynight_bestmodel2': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2_actualactual/daynight_1/best_model_loss.pth",
        # 'day_real_night_synth_1': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2_actualactual/day_real_night_synth_1/best_model_loss.pth",
        # 'daynight_withsynth_1': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2_actualactual/daynight_withsynth_1/best_model_loss.pth",
        # 'day_real_night_bddsynth_1': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2_actualactual/day_real_night_bddsynth_1/best_model_loss.pth",
        # 'daynight_withbddsynth_1': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2_actualactual/daynight_withbddsynth_1/best_model_loss.pth",

        # All of these had the cis_test be the validation split, not cisval (in appendix)
        # 'day_bestmodel': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2/day_1/best_model_loss.pth", 
        # 'daynight_bestmodel': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2/daynight_1/best_model_loss.pth",
        # 'day_real_night_synth_1': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2/day_real_night_synth_1/best_model_loss.pth",
        # 'daynight_withsynth_1': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2/daynight_withsynth_1/best_model_loss.pth",
        # 'day_real_night_bddsynth_1': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2/day_real_night_bddsynth_1/best_model_loss.pth",
        # 'daynight_withbddsynth_1': "./trans_test_train_results/run_6_enet_autoclassweights_cisval_epochsaving_v2/daynight_withbddsynth_1/best_model_loss.pth",
    }

    # Define test dataset paths and names
    # datasets = {
    #     'day': "/data2/CDAO/toy_eccv18/dn_1_ablations/evalsets/synth_valeval_1/sets/day_real",
    #     'day_night': "/data2/CDAO/toy_eccv18/dn_1_ablations/evalsets/synth_valeval_1/sets/day_night_real",
    #     'day_synthnight_img2img': "/data2/CDAO/toy_eccv18/dn_1_ablations/evalsets/synth_valeval_1/sets/day_real_night_synth",
    # }
    # datasets = {
    #     'day': "/data2/CDAO/toy_eccv18/dn_1_ablations/evalsets/synth_valeval_1/subsets/valdata_subset_day_1",
    #     'night': "/data2/CDAO/toy_eccv18/dn_1_ablations/evalsets/synth_valeval_1/subsets/valdata_subset_night_1",
    #     'synthnight_img2img': "/data2/CDAO/toy_eccv18/dn_1_ablations/evalsets/synth_valeval_1/subsets/valdata_subset_day_img2img_cistest1",
    # }
    # datasets = {
    #     'exposed': "/data2/CDAO/toy_eccv18/dn_1_ablations/evalsets/synth_valeval_big_1/sets/day_100night",
    #     'ideal': "/data2/CDAO/toy_eccv18/dn_1_ablations/evalsets/synth_valeval_big_1/sets/day_night_matchedsize",
    # #     # 'perturbed': "/data2/CDAO/toy_eccv18/dn_1_ablations/evalsets/synth_valeval_big_1/sets/day_100night_synthnight",
    #     'perturbed': "/data2/CDAO/toy_eccv18/dn_1_ablations/evalsets/synth_valeval_big_1/sets/day_100night_synthnight_matchedsize",
    # #     # 'exposed_rep': "/data2/CDAO/toy_eccv18/dn_1_ablations/evalsets/synth_valeval_big_1/sets/day_100night_repeatednight",
    # }

    # For subsets:
    # output_dir = './eval_results/cct/daynight/subsets'
    # datasets = {
    #     'day_subset': "/data2/CDAO/DENSE_public/datasets/cct/daynight/subsets/all_day",
    #     'night_subsets': "/data2/CDAO/DENSE_public/datasets/cct/daynight/subsets/all_night",
    #     'synthetic_night_subset': "/data2/CDAO/DENSE_public/datasets/cct/daynight/subsets/day_transformed_daynight1_cistesttrained",
    # }

    output_dir = './eval_results/cct/daynight/subsets_bddsynth'
    datasets = {
        'day_subset': "/data2/CDAO/DENSE_public/datasets/cct/daynight/subsets/all_day",
        'night_subsets': "/data2/CDAO/DENSE_public/datasets/cct/daynight/subsets/all_night",
        'synthetic_night_subset': "/data2/CDAO/DENSE_public/datasets/cct/daynight/subsets/day_transformed_daynight1_bdd",
    }

    # For sets:
    # output_dir = './eval_results/cct/daynight/sets'
    # datasets = {
    #     'day_100night': "/data2/CDAO/DENSE_public/datasets/cct/daynight/sets/day_100night",
    #     'day_298night': "/data2/CDAO/DENSE_public/datasets/cct/daynight/sets/day_298night",
    #     'day_100night_198synth': "/data2/CDAO/DENSE_public/datasets/cct/daynight/sets/day_100night_198synth",
    # }
    
    # Run k-fold evaluation
    results = evaluate_multiple_kfold(
        model_list=model_list,
        dataset_list=datasets,
        k_folds=10,  # Number of folds
        batch_size=32,
        verbose=True,  # Set to True for detailed output for each fold
        random_seed=42,  # For reproducibility
        class_weight='balanced'  # Use 'balanced' for automatic weighting, None for no weighting, or dict for custom
        # class_weight={0: 0.34, 1:0.66 }
    )
    
    # Save results
    save_results_kfold(results, output_dir=output_dir)




