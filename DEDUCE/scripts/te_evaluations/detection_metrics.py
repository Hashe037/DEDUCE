#!/usr/bin/env python3
"""
Faster R-CNN model loading and mAP evaluation for BDD100K object detection.
"""

from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou


def get_model(num_classes: int, weights_path: str, device: torch.device):
    """Load Faster R-CNN (ResNet-50 FPN v2) with trained weights."""
    model = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    state_dict = torch.load(weights_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """All-points interpolated Average Precision (COCO style)."""
    if len(recalls) == 0:
        return 0.0
    r = np.concatenate([[0], recalls, [1]])
    p = np.concatenate([[0], precisions, [0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


def _class_ap_at_threshold(detections, n_gt, iou_thresh):
    """AP for one class at one IoU threshold."""
    if n_gt == 0 or not detections:
        return 0.0, np.array([]), np.array([])

    matched = defaultdict(set)
    tp = fp = 0
    precs, recs = [], []

    for det in detections:
        hit = (det['gt_idx'] >= 0
               and det['iou'] >= iou_thresh
               and det['gt_idx'] not in matched[det['img_idx']])
        if hit:
            tp += 1
            matched[det['img_idx']].add(det['gt_idx'])
        else:
            fp += 1
        precs.append(tp / (tp + fp))
        recs.append(tp / n_gt)

    return _compute_ap(np.array(recs), np.array(precs)), np.array(precs), np.array(recs)


def evaluate_detections(predictions: List[Dict],
                        targets: List[Dict],
                        categories: Dict[int, str],
                        iou_thresholds: List[float] = (0.5,),
                        score_threshold: float = 0.0) -> Dict:
    """
    Compute mAP metrics from detection predictions and ground-truth targets.

    Args:
        predictions:    List of dicts with 'boxes', 'labels', 'scores'.
        targets:        List of dicts with 'boxes', 'labels'.
        categories:     {category_id: category_name}.
        iou_thresholds: IoU thresholds for per-threshold mAP.
        score_threshold: Ignore predictions below this score.

    Returns:
        Dict with keys 'per_class', 'per_iou', 'summary'.
    """
    # Collect detections and GT counts per category
    all_dets = defaultdict(list)
    gt_counts = defaultdict(int)

    for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
        mask = pred['scores'] >= score_threshold
        p_boxes  = pred['boxes'][mask]
        p_labels = pred['labels'][mask]
        p_scores = pred['scores'][mask]

        for lbl in target['labels']:
            gt_counts[lbl.item()] += 1

        for i in torch.argsort(p_scores, descending=True):
            pb, pl = p_boxes[i], p_labels[i].item()
            best_iou, best_gt = 0.0, -1
            for gi, (gb, gl) in enumerate(zip(target['boxes'], target['labels'])):
                if gl.item() != pl:
                    continue
                iou = box_iou(pb.unsqueeze(0), gb.unsqueeze(0)).item()
                if iou > best_iou:
                    best_iou, best_gt = iou, gi
            all_dets[pl].append({
                'score': p_scores[i].item(),
                'iou': best_iou,
                'img_idx': img_idx,
                'gt_idx': best_gt,
            })

    results = {'per_class': {}, 'per_iou': {}, 'summary': {}}

    for iou_thresh in iou_thresholds:
        aps = []
        for cat_id in sorted(categories):
            name = categories[cat_id]
            dets = sorted(all_dets[cat_id], key=lambda x: x['score'], reverse=True)
            n_gt = gt_counts[cat_id]

            ap, precs, recs = _class_ap_at_threshold(dets, n_gt, iou_thresh)

            entry = results['per_class'].setdefault(name, {})
            entry[f'AP@{iou_thresh}'] = ap
            entry['n_gt']  = n_gt
            entry['n_det'] = len(dets)

            if n_gt > 0:
                aps.append(ap)
                if len(precs):
                    entry[f'max_precision@{iou_thresh}'] = float(np.max(precs))
                    entry[f'max_recall@{iou_thresh}']    = float(np.max(recs))

        results['per_iou'][f'mAP@{iou_thresh}'] = float(np.mean(aps)) if aps else 0.0

    # mAP@0.5:0.95 (COCO style)
    coco_aps = []
    for t in np.arange(0.5, 1.0, 0.05):
        aps = []
        for cat_id in sorted(categories):
            dets = sorted(all_dets[cat_id], key=lambda x: x['score'], reverse=True)
            if gt_counts[cat_id] == 0:
                continue
            ap, _, _ = _class_ap_at_threshold(dets, gt_counts[cat_id], t)
            aps.append(ap)
        if aps:
            coco_aps.append(np.mean(aps))

    results['summary']['mAP@0.5']      = results['per_iou'].get('mAP@0.5', 0.0)
    results['summary']['mAP@0.5:0.95'] = float(np.mean(coco_aps)) if coco_aps else 0.0

    return results
