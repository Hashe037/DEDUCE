#!/usr/bin/env python3
"""
Minimal PyTorch dataset for COCO-format BDD100K object detection evaluation.
"""

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BDD100KDetectionDataset(Dataset):
    """
    Load COCO-format BDD100K annotations for detection evaluation.

    Args:
        coco_json:   Path to COCO JSON produced by evaluate.py's conversion step.
        images_root: Directory containing the images (or a parent directory whose
                     immediate sub-folders contain the images).
        min_area:    Minimum bounding-box area (pixels²) to include.
    """

    def __init__(self, coco_json, images_root, transforms=None, min_area=100,
                 max_images=None, seed=42):
        self.images_root = Path(images_root)
        self.min_area = min_area
        # transforms kept for API compatibility but unused at eval time

        with open(coco_json) as f:
            coco = json.load(f)

        self.images = {img['id']: img for img in coco['images']}
        self.image_ids = list(self.images.keys())

        if max_images is not None and max_images < len(self.image_ids):
            rng = np.random.RandomState(seed)
            rng.shuffle(self.image_ids)
            self.image_ids = self.image_ids[:max_images]

        self.img_to_anns: dict = {}
        for ann in coco['annotations']:
            self.img_to_anns.setdefault(ann['image_id'], []).append(ann)

        self.categories = {cat['id']: cat['name'] for cat in coco['categories']}
        self.num_classes = len(self.categories) + 1  # +1 for background

        print(f"Loaded {len(self.image_ids)} images, "
              f"{len(coco['annotations'])} annotations, "
              f"{len(self.categories)} classes")

    def __len__(self):
        return len(self.image_ids)

    def _find_image(self, filename):
        """Locate an image under images_root, checking one subdirectory level."""
        p = self.images_root / filename
        if p.exists():
            return p
        for subdir in self.images_root.iterdir():
            if subdir.is_dir():
                candidate = subdir / filename
                if candidate.exists():
                    return candidate
        return None

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]

        img_path = self._find_image(img_info['file_name'])
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {img_info['file_name']}")

        image = Image.open(img_path).convert('RGB')
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        boxes, labels, areas, iscrowd = [], [], [], []
        for ann in self.img_to_anns.get(img_id, []):
            if ann['area'] < self.min_area:
                continue
            x1, y1, w, h = ann['bbox']
            boxes.append([x1, y1, x1 + w, y1 + h])
            labels.append(ann['category_id'])
            areas.append(ann['area'])
            iscrowd.append(ann.get('iscrowd', 0))

        if boxes:
            boxes   = torch.as_tensor(boxes,   dtype=torch.float32)
            labels  = torch.as_tensor(labels,  dtype=torch.int64)
            areas   = torch.as_tensor(areas,   dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            boxes   = torch.zeros((0, 4), dtype=torch.float32)
            labels  = torch.zeros((0,),   dtype=torch.int64)
            areas   = torch.zeros((0,),   dtype=torch.float32)
            iscrowd = torch.zeros((0,),   dtype=torch.int64)

        return image_tensor, {
            'boxes':    boxes,
            'labels':   labels,
            'image_id': torch.tensor([img_id]),
            'area':     areas,
            'iscrowd':  iscrowd,
            'filename': img_info['file_name'],
        }


class MultiRootDetectionDataset(BDD100KDetectionDataset):
    """Like BDD100KDetectionDataset but searches multiple image root directories."""

    def __init__(self, coco_json, images_roots, transforms=None, min_area=100,
                 max_images=None, seed=42):
        self.images_roots = [Path(r) for r in images_roots]
        super().__init__(coco_json, images_roots[0], transforms, min_area, max_images, seed)

    def _find_image(self, filename):
        for root in self.images_roots:
            p = root / filename
            if p.exists():
                return p
            p = root / 'images' / filename
            if p.exists():
                return p
        return None


def get_val_transforms():
    return None


def collate_fn(batch):
    return tuple(zip(*batch))
