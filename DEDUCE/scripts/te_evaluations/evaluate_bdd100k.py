#!/usr/bin/env python3
"""
Evaluate Faster R-CNN models on BDD100K test datasets.

Converts BDD100K annotation folders to COCO format on demand (cached), then
runs inference and computes mAP. Eliminates the separate convert_multiple.py step.

Usage:
    python evaluate.py

Dataset entries point directly to BDD100K label/image folders. COCO JSON files
are written to `cache_dir` on first run and reused on subsequent runs.
Pass force_convert=True to regenerate cached JSONs.
"""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent))
from dataset import BDD100KDetectionDataset, MultiRootDetectionDataset, get_val_transforms, collate_fn
from detection_metrics import evaluate_detections, get_model

# ---------------------------------------------------------------------------
# BDD100K → COCO conversion (inline so this file is self-contained)
# ---------------------------------------------------------------------------

BOX2D_CATEGORIES = [
    'car', 'truck', 'bus', 'bike', 'motor', 'train', 'rider',
    'traffic sign', 'traffic light', 'person',
]

COCO_CATEGORIES = [
    {'id': i + 1, 'name': cat, 'supercategory': 'object'}
    for i, cat in enumerate(BOX2D_CATEGORIES)
]


def _convert_bdd100k_dir(labels_dir: Path, images_dir: Path,
                          start_image_id: int = 1,
                          start_ann_id: int = 1) -> tuple:
    """Convert one BDD100K label/image directory pair to a COCO dict."""
    cat_to_id = {cat: i + 1 for i, cat in enumerate(BOX2D_CATEGORIES)}
    coco = {'images': [], 'annotations': [], 'categories': COCO_CATEGORIES}
    image_id, ann_id = start_image_id, start_ann_id
    stats = defaultdict(int)

    label_files = [f for f in labels_dir.glob('*.json')
                   if f.name != 'dataset_summary.json']
    print(f"  Found {len(label_files)} label files in {labels_dir.name}")

    for label_file in tqdm(label_files, desc=f"  Converting {labels_dir.parent.name}"):
        try:
            with open(label_file) as f:
                bdd_data = json.load(f)
        except json.JSONDecodeError:
            stats['json_errors'] += 1
            continue

        # Locate the image
        image_name = label_file.stem + '.jpg'
        image_path = images_dir / image_name
        if not image_path.exists():
            image_name = label_file.stem + '.png'
            image_path = images_dir / image_name
            if not image_path.exists():
                stats['missing_images'] += 1
                continue

        try:
            img = Image.open(image_path)
            width, height = img.size
        except Exception:
            stats['image_errors'] += 1
            continue

        coco['images'].append({
            'id': image_id,
            'file_name': image_name,
            'width': width,
            'height': height,
            'source_dir': str(images_dir),
        })

        for frame in bdd_data.get('frames', []):
            for obj in frame.get('objects', []):
                if 'box2d' not in obj:
                    continue
                category = obj['category']
                if category not in cat_to_id:
                    stats['unknown_categories'] += 1
                    continue
                box = obj['box2d']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']
                if x2 <= x1 or y2 <= y1:
                    stats['invalid_boxes'] += 1
                    continue
                x1, y1 = max(0, min(x1, width)), max(0, min(y1, height))
                x2, y2 = max(0, min(x2, width)), max(0, min(y2, height))
                w, h = x2 - x1, y2 - y1
                if w < 5 or h < 5:
                    stats['tiny_boxes'] += 1
                    continue
                coco['annotations'].append({
                    'id': ann_id,
                    'image_id': image_id,
                    'category_id': cat_to_id[category],
                    'bbox': [x1, y1, w, h],
                    'area': w * h,
                    'iscrowd': 0,
                })
                ann_id += 1

        image_id += 1

    if any(v > 0 for v in stats.values()):
        print(f"  Skipped: {dict(stats)}")
    return coco, image_id, ann_id


def _merge_coco_dicts(coco_list: List[Dict]) -> Dict:
    combined = {'images': [], 'annotations': [], 'categories': coco_list[0]['categories']}
    for c in coco_list:
        combined['images'].extend(c['images'])
        combined['annotations'].extend(c['annotations'])
    return combined


def convert_to_coco_cached(name: str, info: Dict, cache_dir: Path,
                            force: bool = False) -> Path:
    """
    Convert a dataset entry to a COCO test JSON, caching to `cache_dir/name/test.json`.

    `info` must contain either:
      - 'labels' and 'images'  (single source)
      - 'sources': [{'labels': ..., 'images': ...}, ...]  (multiple sources)

    Returns the path to the cached test.json.
    """
    out_path = cache_dir / name / 'test.json'
    if out_path.exists() and not force:
        print(f"  [cache hit] {out_path}")
        return out_path

    print(f"\n{'='*60}")
    print(f"Converting: {name}")
    print(f"{'='*60}")

    sources = info.get('sources') or [{'labels': info['labels'], 'images': info['images']}]

    coco_list, image_id, ann_id = [], 1, 1
    for src in sources:
        labels_dir, images_dir = Path(src['labels']), Path(src['images'])
        if not labels_dir.exists():
            print(f"  WARNING: Labels not found: {labels_dir}")
            continue
        if not images_dir.exists():
            print(f"  WARNING: Images not found: {images_dir}")
            continue
        coco, image_id, ann_id = _convert_bdd100k_dir(
            labels_dir, images_dir, image_id, ann_id)
        coco_list.append(coco)

    if not coco_list:
        raise RuntimeError(f"No valid sources for dataset '{name}'")

    coco = _merge_coco_dicts(coco_list) if len(coco_list) > 1 else coco_list[0]
    print(f"  {len(coco['images'])} images, {len(coco['annotations'])} annotations")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(coco, f)
    print(f"  Saved: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Annotation filtering and distortion helpers
# ---------------------------------------------------------------------------

def filter_annotations(boxes, labels, areas, iscrowd,
                        min_area=100,
                        min_area_ratio=0.0001,
                        max_aspect_ratio=10.0,
                        filter_truncated=True,
                        image_size=(512, 512)):
    if len(boxes) == 0:
        return boxes, labels, areas, iscrowd

    keep = torch.ones(len(boxes), dtype=torch.bool)
    keep &= areas >= min_area
    keep &= areas >= min_area_ratio * image_size[0] * image_size[1]

    widths  = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    keep &= torch.maximum(widths / (heights + 1e-6),
                          heights / (widths + 1e-6)) <= max_aspect_ratio
    keep &= iscrowd == 0
    keep &= (widths > 0) & (heights > 0)

    if filter_truncated:
        m = 2
        keep &= (boxes[:, 0] > m) & (boxes[:, 1] > m) & \
                (boxes[:, 2] < image_size[0] - m) & \
                (boxes[:, 3] < image_size[1] - m)

    return boxes[keep], labels[keep], areas[keep], iscrowd[keep]


def apply_resize_distortion(image, intermediate_size=(512, 512)):
    """Simulate the resize-cycle artifact that synthetic images undergo."""
    pil_img = F.to_pil_image(image)
    original_size = pil_img.size
    pil_img = pil_img.resize(intermediate_size, Image.LANCZOS)
    pil_img = pil_img.resize(original_size, Image.LANCZOS)
    return F.to_tensor(pil_img)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

DEFAULT_FILTER_CONFIG = {
    'min_area': 100,
    'min_area_ratio': 0.0002,
    'max_aspect_ratio': 8.0,
    'filter_truncated': True,
    'score_threshold': 0.05,
    'max_detections': 100,
}


def run_inference_with_resize(model, data_loader, device,
                               target_size=None, original_size=None,
                               apply_distortion=False,
                               distortion_size=(512, 512),
                               filter_config=None):
    """
    Run model inference over a dataloader, optionally resizing images and
    applying a distortion cycle to real (non-synthetic) images.
    """
    cfg = filter_config or DEFAULT_FILTER_CONFIG
    scale_x = (target_size[0] / original_size[0]) if (target_size and original_size) else 1.0
    scale_y = (target_size[1] / original_size[1]) if (target_size and original_size) else 1.0

    predictions, targets_out = [], []

    for images, batch_targets in data_loader:
        if apply_distortion:
            images = [
                img if 'synth' in t.get('filename', '').lower()
                else apply_resize_distortion(img, distortion_size)
                for img, t in zip(images, batch_targets)
            ]

        if target_size is not None:
            images = [F.resize(img, size=(target_size[1], target_size[0])) for img in images]

        images = [img.to(device) for img in images]

        with torch.no_grad():
            outputs = model(images)

        for output, target in zip(outputs, batch_targets):
            scaled_boxes = target['boxes'].clone()
            if scale_x != 1.0 or scale_y != 1.0:
                scaled_boxes[:, [0, 2]] *= scale_x
                scaled_boxes[:, [1, 3]] *= scale_y

            scaled_areas = (
                (scaled_boxes[:, 2] - scaled_boxes[:, 0]) *
                (scaled_boxes[:, 3] - scaled_boxes[:, 1])
            )

            scaled_boxes, filt_labels, filt_areas, filt_iscrowd = filter_annotations(
                scaled_boxes, target['labels'], scaled_areas, target['iscrowd'],
                min_area=cfg['min_area'],
                min_area_ratio=cfg['min_area_ratio'],
                max_aspect_ratio=cfg['max_aspect_ratio'],
                filter_truncated=cfg['filter_truncated'],
                image_size=target_size or (1280, 720),
            )

            score_mask = output['scores'] >= cfg['score_threshold']
            pred_boxes  = output['boxes'][score_mask].cpu()
            pred_labels = output['labels'][score_mask].cpu()
            pred_scores = output['scores'][score_mask].cpu()

            if len(pred_scores) > cfg['max_detections']:
                topk = pred_scores.topk(cfg['max_detections']).indices
                pred_boxes, pred_labels, pred_scores = (
                    pred_boxes[topk], pred_labels[topk], pred_scores[topk])

            if len(pred_boxes) > 0:
                pw = pred_boxes[:, 2] - pred_boxes[:, 0]
                ph = pred_boxes[:, 3] - pred_boxes[:, 1]
                size_ok = (pw * ph) >= cfg['min_area']
                pred_boxes, pred_labels, pred_scores = (
                    pred_boxes[size_ok], pred_labels[size_ok], pred_scores[size_ok])

            predictions.append({'boxes': pred_boxes, 'labels': pred_labels, 'scores': pred_scores})
            targets_out.append({'boxes': scaled_boxes, 'labels': filt_labels})

    return predictions, targets_out


# ---------------------------------------------------------------------------
# Single and multi-model evaluation
# ---------------------------------------------------------------------------

def evaluate_single(model_path: str, coco_json: Path, images,
                    device: torch.device, batch_size: int = 4,
                    target_size: tuple = None, original_size: tuple = None,
                    target_classes: list = None,
                    apply_distortion: bool = False,
                    distortion_size: tuple = (512, 512),
                    filter_config: dict = None) -> Dict:
    """
    Evaluate one model on one COCO test JSON.

    Args:
        coco_json: Path to COCO test JSON (produced by convert_to_coco_cached).
        images: str/Path (single root) or list of str/Path (multiple roots).
    """
    with open(coco_json) as f:
        coco_data = json.load(f)

    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}
    if target_classes is not None:
        categories = {k: v for k, v in categories.items() if v in target_classes}
        print(f"  Classes: {list(categories.values())}")

    num_classes = len(coco_data['categories']) + 1
    model = get_model(num_classes, model_path, device)

    if isinstance(images, (list, tuple)):
        dataset = MultiRootDetectionDataset(
            coco_json=str(coco_json), images_roots=images,
            transforms=get_val_transforms())
    else:
        dataset = BDD100KDetectionDataset(
            coco_json=str(coco_json), images_root=str(images),
            transforms=get_val_transforms())

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=4, pin_memory=True)

    preds, targets = run_inference_with_resize(
        model, loader, device,
        target_size=target_size,
        original_size=original_size,
        apply_distortion=apply_distortion,
        distortion_size=distortion_size,
        filter_config=filter_config,
    )

    results = evaluate_detections(preds, targets, categories, iou_thresholds=[0.5])
    return {
        'mAP@0.5':      results['summary']['mAP@0.5'],
        'mAP@0.5:0.95': results['summary']['mAP@0.5:0.95'],
        'n_images':     len(dataset),
        'per_class':    results['per_class'],
    }


def evaluate_multiple(model_list: Dict[str, str],
                      dataset_list: Dict[str, Dict],
                      cache_dir: str = './cache/bdd100k_coco',
                      force_convert: bool = False,
                      batch_size: int = 4,
                      device: str = 'cuda',
                      target_classes: list = None,
                      filter_config: dict = None):
    """
    Convert (if needed) and evaluate multiple models on multiple datasets.

    Dataset entries use folder paths, not pre-built COCO JSONs:

        dataset_list = {
            'my_test_set': {
                'labels': '/path/to/bdd100k/labels',   # BDD100K JSON labels dir
                'images': '/path/to/bdd100k/images',   # Matching images dir
                'target_size':   (512, 512),            # Resize for inference
                'original_size': (1280, 720),           # Native image size
                'is_real': False,                       # True → apply distortion cycle
            },
            # Multiple sources merged into one test set:
            'combined_test': {
                'sources': [
                    {'labels': '/path/to/set_a/labels', 'images': '/path/to/set_a/images'},
                    {'labels': '/path/to/set_b/labels', 'images': '/path/to/set_b/images'},
                ],
                'target_size': (512, 512),
                'original_size': (1280, 720),
                'is_real': True,
            },
        }

    Args:
        model_list:     {name: path_to_.pth}
        dataset_list:   See above.
        cache_dir:      Where to write/read cached COCO JSONs.
        force_convert:  Re-convert even if cached JSON exists.
        batch_size:     Inference batch size.
        device:         'cuda' or 'cpu'.
        target_classes: Subset of class names to evaluate (None = all).
        filter_config:  Override DEFAULT_FILTER_CONFIG keys.

    Returns:
        (df, matrix, per_class_df)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    cache_root = Path(cache_dir)
    print(f"Device: {device}  |  Models: {len(model_list)}  |  Datasets: {len(dataset_list)}")

    # --- Step 1: Convert / load cached COCO JSONs ---
    coco_paths: Dict[str, Path] = {}
    for name, info in dataset_list.items():
        coco_paths[name] = convert_to_coco_cached(name, info, cache_root, force=force_convert)

    # --- Step 2: Evaluate ---
    results, per_class_results = [], []

    for model_name, model_path in model_list.items():
        for dataset_name, dataset_info in dataset_list.items():
            print(f"\n{'='*60}")
            print(f"Evaluating: {model_name}  on  {dataset_name}")
            if dataset_info.get('is_real'):
                print("  [distortion cycle enabled for real images]")
            print(f"{'='*60}")

            images = dataset_info.get('images') or [s['images'] for s in dataset_info['sources']]

            try:
                result = evaluate_single(
                    model_path=model_path,
                    coco_json=coco_paths[dataset_name],
                    images=images,
                    device=device,
                    batch_size=batch_size,
                    target_size=dataset_info.get('target_size'),
                    original_size=dataset_info.get('original_size'),
                    target_classes=target_classes,
                    apply_distortion=dataset_info.get('is_real', False),
                    distortion_size=dataset_info.get('distortion_size', (512, 512)),
                    filter_config=filter_config,
                )
                results.append({
                    'model': model_name, 'dataset': dataset_name,
                    'mAP@0.5': result['mAP@0.5'],
                    'mAP@0.5:0.95': result['mAP@0.5:0.95'],
                    'n_images': result['n_images'],
                })
                for cls_name, cls_metrics in result['per_class'].items():
                    per_class_results.append({
                        'model': model_name, 'dataset': dataset_name,
                        'class': cls_name,
                        'AP@0.5': cls_metrics.get('AP@0.5', 0.0),
                        'n_gt': cls_metrics.get('n_gt', 0),
                        'n_det': cls_metrics.get('n_det', 0),
                    })
                print(f"  mAP@0.5:      {result['mAP@0.5']:.4f}")
                print(f"  mAP@0.5:0.95: {result['mAP@0.5:0.95']:.4f}")

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({
                    'model': model_name, 'dataset': dataset_name,
                    'mAP@0.5': np.nan, 'mAP@0.5:0.95': np.nan, 'n_images': 0,
                })

    df = pd.DataFrame(results)
    per_class_df = pd.DataFrame(per_class_results)
    matrix = df.pivot(index='model', columns='dataset', values='mAP@0.5')

    print("\n" + "="*70)
    print("  RESULTS MATRIX (mAP@0.5)")
    print("="*70)
    _print_table(matrix.reset_index().rename(columns={'model': 'Model'}))

    for ds in sorted(per_class_df['dataset'].unique()) if not per_class_df.empty else []:
        sub = per_class_df[per_class_df['dataset'] == ds]
        pivot = sub.pivot(index='model', columns='class', values='AP@0.5')
        pivot['mAP'] = pivot.mean(axis=1)
        cols = sorted(c for c in pivot.columns if c != 'mAP')
        pivot = pivot[cols + ['mAP']]
        gt_str = ', '.join(
            f"{r['class']}={int(r['n_gt'])}"
            for _, r in sub.groupby('class')['n_gt'].first().reset_index().iterrows()
        )
        print(f"\n{'='*70}")
        print(f"  PER-CLASS AP@0.5 — {ds}")
        print(f"  GT counts: {gt_str}")
        print("="*70)
        _print_table(pivot.reset_index().rename(columns={'model': 'Model'}))

    return df, matrix, per_class_df


# ---------------------------------------------------------------------------
# Helpers: print, plot, save
# ---------------------------------------------------------------------------

def _print_table(df: pd.DataFrame):
    fmt = df.copy()
    for col in fmt.columns:
        if fmt[col].dtype in ('float64', 'float32'):
            fmt[col] = fmt[col].map(lambda v: f'{v:.4f}' if pd.notna(v) else '  —')
    widths = {c: max(len(str(c)), fmt[c].astype(str).str.len().max()) for c in fmt.columns}
    header = '  '.join(str(c).rjust(widths[c]) for c in fmt.columns)
    sep    = '  '.join('-' * widths[c] for c in fmt.columns)
    print(f"\n  {header}\n  {sep}")
    for _, row in fmt.iterrows():
        print('  ' + '  '.join(str(row[c]).rjust(widths[c]) for c in fmt.columns))
    print()


def plot_per_class_comparison(per_class_df: pd.DataFrame, output_dir: str = '.'):
    """Bar charts comparing model per-class AP for each dataset."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if per_class_df.empty:
        return

    for dataset_name in per_class_df['dataset'].unique():
        sub = per_class_df[per_class_df['dataset'] == dataset_name]
        models  = sub['model'].unique()
        classes = sorted(sub['class'].unique())

        x_labels = classes + ['mAP']
        bar_width = 0.8 / max(len(models), 1)
        x = np.arange(len(x_labels))
        fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 1.5 + 2), 5))

        for i, model_name in enumerate(models):
            ms = sub[sub['model'] == model_name]
            per_class_vals = [
                ms[ms['class'] == cls]['AP@0.5'].values[0]
                if len(ms[ms['class'] == cls]) > 0 else 0.0
                for cls in classes
            ]
            overall_map = float(np.mean(per_class_vals)) if per_class_vals else 0.0
            ap_vals = per_class_vals + [overall_map]
            offset = (i - len(models) / 2 + 0.5) * bar_width
            colors = [f'C{i}'] * len(classes) + [f'C{i}']
            bars = ax.bar(x + offset, ap_vals, bar_width, label=model_name, alpha=0.85,
                          color=colors)
            for bar, val in zip(bars, ap_vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        # Vertical separator before the mAP bar
        ax.axvline(x=len(classes) - 0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

        ax.set_xlabel('Class')
        ax.set_ylabel('AP @ IoU=0.5')
        ax.set_title(f'Per-Class AP — {dataset_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=30 if len(x_labels) > 5 else 0, ha='right')
        ax.set_ylim(0, min(1.05, ax.get_ylim()[1] + 0.1))
        ax.legend(loc='best', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        save_path = output_dir / f'per_class_ap_{dataset_name}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def save_results(df: pd.DataFrame, per_class_df: pd.DataFrame,
                 matrix: pd.DataFrame, output_dir: str = '.'):
    """Save summary, matrix, and per-class CSVs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'eval_results.csv', index=False)
    matrix.to_csv(output_dir / 'eval_matrix.csv')
    print(f"Saved: {output_dir / 'eval_results.csv'}")
    print(f"Saved: {output_dir / 'eval_matrix.csv'}")
    if not per_class_df.empty:
        per_class_df.to_csv(output_dir / 'eval_per_class.csv', index=False)
        print(f"Saved: {output_dir / 'eval_per_class.csv'}")


# ---------------------------------------------------------------------------
# Entry point — edit the config sections below then run `python evaluate.py`
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    # === MODELS ===
    # Map a short name to the .pth file for each trained model.
    model_list = {
        'day_model_1': '/data2/CDAO/DENSE_public/models/bdd100k/daynight/day_model_1.pth',
        'daynight_model_2': '/data2/CDAO/DENSE_public/models/bdd100k/daynight/daynight_model_2.pth',
    }
    # model_list = {
    #     'clear_model_1': '/home/local/KHQ/connor.hashemi/Projects/DENSE/DENSE/src/scripts/bdd100k_od_coverage/training/models/daytime_clear_train_2532/daytime_clear_train_512x512_2532size_best.pth',
    #     'clearrainy_model_2': '/home/local/KHQ/connor.hashemi/Projects/DENSE/DENSE/src/scripts/bdd100k_od_coverage/training/models/daytime_clear_rainy_train_samesize/daytime_clear_rainy_train_512x512_samesize_2532images_best.pth',
    # }
    # model_list = {
    #     'clear_model_1': '/home/local/KHQ/connor.hashemi/Projects/DENSE/DENSE/src/scripts/bdd100k_od_coverage/training/models/daytime_clear_train_2532/daytime_clear_train_512x512_2532size_best.pth',
    #     'clearsnowy_model_2': '/home/local/KHQ/connor.hashemi/Projects/DENSE/DENSE/src/scripts/bdd100k_od_coverage/training/models/daytime_clear_snowy_train_samesize/daytime_clear_snowy_train_512x512_samesize_2873images_best.pth',
    # }


    # === DATASETS ===
    # Point to BDD100K label/image folders. COCO JSONs are auto-generated and
    # cached under cache_dir on first run.
    #
    # Required keys per entry:
    #   labels / images   — BDD100K folder pair  (OR use 'sources' for multiple)
    #
    # Optional keys:
    #   target_size       — (W, H) to resize images at inference time
    #   original_size     — (W, H) native resolution (needed when target_size is set)
    #   is_real           — apply resize distortion cycle to real images (default False)
    #   distortion_size   — intermediate size for distortion cycle (default (512, 512))
    #   sources           — list of {labels, images} dicts to merge into one test set

    # For processing subsets:
    # output_dir = './eval_results/bdd100k/daynight/subsets'
    # dataset_list = {
    #     'day_subset': {
    #         'labels': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/all_day/labels',
    #         'images': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/all_day/images',
    #         'target_size':   (512, 512),  #resize all done to 512x512 (matches output of synthetic pipeline)
    #         'original_size': (1280, 720),
    #         'is_real': True,    # apply distortion cycle to match synthetic processing
    #     },
    #     'night_subset': {
    #         'labels': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/all_night/labels',
    #         'images': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/all_night/images',
    #         'target_size':   (512, 512), #resize all done to 512x512 (matches output of synthetic pipeline)
    #         'original_size': (1280, 720),
    #         'is_real': True,    # apply distortion cycle to match synthetic processing
    #     },
    #     'synthetic_night_subset': {
    #         'labels': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/day_transformed_daynight1_highrez/labels',
    #         'images': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/day_transformed_daynight1_highrez/images',
    #         'target_size':   (512, 512), 
    #         'original_size': (1280, 720), #synthetic night images are generated at 512x512, resized back to 1280x720 for evaluation to match real image processing
    #         'is_real': False,    # don't apply distortion cycle to match synthetic processing
    #     },
    # }

    # For processing total sets:
    # output_dir = './eval_results/bdd100k/daynight/sets'
    # dataset_list = {
    #     'day_100night': {
    #         'labels': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/sets/day_100night/labels',
    #         'images': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/sets/day_100night/images',
    #         'target_size':   (512, 512),  #resize all done to 512x512 (matches output of synthetic pipeline)
    #         'original_size': (1280, 720),
    #         'is_real': True,    # apply distortion cycle to match synthetic processing
    #     },
    #     'day_1000night': {
    #         'labels': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/sets/day_1000night/labels',
    #         'images': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/sets/day_1000night/images',
    #         'target_size':   (512, 512), #resize all done to 512x512 (matches output of synthetic pipeline)
    #         'original_size': (1280, 720),
    #         'is_real': True,    # apply distortion cycle to match synthetic processing
    #     },
    #     'day_100night_900synth': {
    #         'labels': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/sets/day_100night_900synthhighrez/labels',
    #         'images': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/sets/day_100night_900synthhighrez/images',
    #         'target_size':   (512, 512), 
    #         'original_size': (1280, 720), #synthetic night images are generated at 512x512, resized back to 1280x720 for evaluation to match real image processing
    #         'is_real': True,   
    #     },
    # }

    # For processing subsets:
    # output_dir = './eval_results/bdd100k/clearrainy/subsets'
    # dataset_list = {
    #     'clear_subset': {
    #         'labels': '/data2/CDAO/DENSE_public/datasets/bdd100k/clearrainy_1/subsets/all_clear/labels',
    #         'images': '/data2/CDAO/DENSE_public/datasets/bdd100k/clearrainy_1/subsets/all_clear/images',
    #         'target_size':   (512, 512),  #resize all done to 512x512 (matches output of synthetic pipeline)
    #         'original_size': (1280, 720),
    #         'is_real': True,    # apply distortion cycle to match synthetic processing
    #     },
    #     'rain_subset': {
    #         'labels': '/data2/CDAO/DENSE_public/datasets/bdd100k/clearrainy_1/subsets/all_rainy/labels',
    #         'images': '/data2/CDAO/DENSE_public/datasets/bdd100k/clearrainy_1/subsets/all_rainy/images',
    #         'target_size':   (512, 512), #resize all done to 512x512 (matches output of synthetic pipeline)
    #         'original_size': (1280, 720),
    #         'is_real': True,    # apply distortion cycle to match synthetic processing
    #     },
    #     'synthetic_rain_subset': {
    #         'labels': '/data2/CDAO/bdd100k_toy/clearrainy_1/subsets/clear_transformed_clearrainy1_highrez/labels',
    #         'images': '/data2/CDAO/bdd100k_toy/clearrainy_1/subsets/clear_transformed_clearrainy1_highrez/images',
    #         'target_size':   (512, 512), 
    #         'original_size': (1280, 720), #synthetic night images are generated at 512x512, resized back to 1280x720 for evaluation to match real image processing
    #         'is_real': False,    # don't apply distortion cycle to match synthetic processing
    #     },
    # }

    # For processing subsets:
    # output_dir = './eval_results/bdd100k/clearsnowy/subsets'
    # dataset_list = {
    #     'clear_subset': {
    #         'labels': '/data2/CDAO/DENSE_public/datasets/bdd100k/clearrainy_1/subsets/all_clear/labels',
    #         'images': '/data2/CDAO/DENSE_public/datasets/bdd100k/clearrainy_1/subsets/all_clear/images',
    #         'target_size':   (512, 512),  #resize all done to 512x512 (matches output of synthetic pipeline)
    #         'original_size': (1280, 720),
    #         'is_real': True,    # apply distortion cycle to match synthetic processing
    #     },
    #     'snowy_subset': {
    #         'labels': '/data2/CDAO/DENSE_public/datasets/bdd100k/summerwinter_1/subsets/all_snowy/labels',
    #         'images': '/data2/CDAO/DENSE_public/datasets/bdd100k/summerwinter_1/subsets/all_snowy/images',
    #         'target_size':   (512, 512), #resize all done to 512x512 (matches output of synthetic pipeline)
    #         'original_size': (1280, 720),
    #         'is_real': True,    # apply distortion cycle to match synthetic processing
    #     },
    #     'synthetic_snowy_subset': {
    #         'labels': '/data2/CDAO/bdd100k_toy/summerwinter_1/subsets/clear_transformed_summerwinter1_highrez/labels',
    #         'images': '/data2/CDAO/bdd100k_toy/summerwinter_1/subsets/clear_transformed_summerwinter1_highrez/images',
    #         'target_size':   (512, 512), 
    #         'original_size': (1280, 720), #synthetic night images are generated at 512x512, resized back to 1280x720 for evaluation to match real image processing
    #         'is_real': False,    # don't apply distortion cycle to match synthetic processing
    #     },
    # }

    # For processing subsets:
    output_dir = './eval_results/bdd100k/daynight/subsets_cctsynth'
    dataset_list = {
        'day_subset': {
            'labels': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/all_day/labels',
            'images': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/all_day/images',
            'target_size':   (512, 512),  #resize all done to 512x512 (matches output of synthetic pipeline)
            'original_size': (1280, 720),
            'is_real': True,    # apply distortion cycle to match synthetic processing
        },
        'night_subset': {
            'labels': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/all_night/labels',
            'images': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/all_night/images',
            'target_size':   (512, 512), #resize all done to 512x512 (matches output of synthetic pipeline)
            'original_size': (1280, 720),
            'is_real': True,    # apply distortion cycle to match synthetic processing
        },
        'synthetic_night_subset': {
            'labels': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/day_transformed_daynight1_cctmodel/labels',
            'images': '/data2/CDAO/DENSE_public/datasets/bdd100k/daynight_1/subsets/day_transformed_daynight1_cctmodel/images',
            'target_size':   (512, 512), 
            'original_size': (1280, 720), #synthetic night images are generated at 512x512, resized back to 1280x720 for evaluation to match real image processing
            'is_real': False,    # don't apply distortion cycle to match synthetic processing
        },
    }


    # === CLASSES TO EVALUATE (None = all 10 BDD100K classes) ===
    target_classes = None # ['person', 'car']

    # === DETECTION FILTER CONFIG ===
    filter_config = {
        'min_area':        100,    # min px² after resize
        'min_area_ratio':  0.0002, # min fraction of image area
        'max_aspect_ratio': 8.0,
        'filter_truncated': True,
        'score_threshold':  0.05,
        'max_detections':   100,
    }

    # === RUN ===
    df, matrix, per_class_df = evaluate_multiple(
        model_list=model_list,
        dataset_list=dataset_list,
        cache_dir='./cache/bdd100k_coco',   # cached COCO JSONs live here
        force_convert=True,                 # set True to regenerate
        batch_size=32,
        device='cuda',
        target_classes=target_classes,
        filter_config=filter_config,
    )

    save_results(df, per_class_df, matrix, output_dir=output_dir)
    plot_per_class_comparison(per_class_df, output_dir=output_dir)
