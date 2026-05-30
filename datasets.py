import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image, ImageFile
from utils import transform

# Allow PIL to load images that are slightly truncated (non-standard JPEG encoding).
# These images are visually fine — PIL is just strict about format compliance by default.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# COCO category ID -> sequential label (1-50); defined here to avoid circular import
# Matches the order in utils.py _lvis_categories
_COCO_ID_TO_LABEL = {
    3: 1, 36: 2, 45: 3, 58: 4, 60: 5, 77: 6, 80: 7, 90: 8, 99: 9, 118: 10,
    139: 11, 154: 12, 173: 13, 225: 14, 230: 15, 271: 16, 296: 17, 378: 18,
    387: 19, 390: 20, 422: 21, 430: 22, 445: 23, 469: 24, 474: 25, 496: 26,
    569: 27, 611: 28, 703: 29, 716: 30, 816: 31, 818: 32, 943: 33, 962: 34,
    967: 35, 976: 36, 982: 37, 995: 38, 1026: 39, 1037: 40, 1071: 41,
    1077: 42, 1079: 43, 1097: 44, 1112: 45, 1115: 46, 1133: 47, 1139: 48,
    1186: 49, 1202: 50,
}


class LVISDataset(Dataset):
    """
    PyTorch Dataset for the LVIS Greedy-50 splits.
    Reads COCO-format annotations.json produced by data_split.py.

    split_dir should point to one of:
        .../splits/split_005/train   or   .../splits/split_005/val
    """

    def __init__(self, split_dir, split):
        """
        :param split_dir: path containing images/ folder and annotations.json
        :param split: 'train' or 'val'
        """
        self.split = split.upper()
        assert self.split in {'TRAIN', 'VAL'}

        self.images_dir = os.path.join(split_dir, 'images')

        with open(os.path.join(split_dir, 'annotations.json'), 'r') as f:
            data = json.load(f)

        ann_by_image = {}
        for ann in data['annotations']:
            ann_by_image.setdefault(ann['image_id'], []).append(ann)

        self.image_paths = []
        self.objects = []

        for img in data['images']:
            img_id = img['id']
            anns = ann_by_image.get(img_id, [])
            if not anns:
                continue

            fname = os.path.basename(img.get('coco_url', '')) or f"{img_id:012d}.jpg"
            fpath = os.path.join(self.images_dir, fname)
            if not os.path.exists(fpath):
                fpath = os.path.join(self.images_dir, f"{img_id:012d}.jpg")
            if not os.path.exists(fpath):
                continue

            boxes, labels = [], []
            for ann in anns:
                coco_cat = ann['category_id']
                if coco_cat not in _COCO_ID_TO_LABEL:
                    continue
                x, y, w, h = ann['bbox']
                # COCO bbox is [x, y, w, h] — convert to [xmin, ymin, xmax, ymax]
                boxes.append([x, y, x + w, y + h])
                labels.append(_COCO_ID_TO_LABEL[coco_cat])

            if not boxes:
                continue

            self.image_paths.append(fpath)
            self.objects.append({'boxes': boxes, 'labels': labels})

    def __getitem__(self, i):
        image = Image.open(self.image_paths[i], mode='r').convert('RGB')

        obj = self.objects[i]
        boxes = torch.FloatTensor(obj['boxes'])
        labels = torch.LongTensor(obj['labels'])
        # LVIS has no 'difficult' flag — treat all as non-difficult
        difficulties = torch.zeros(len(labels), dtype=torch.uint8)

        # VAL uses no augmentation (same as TEST in the original transform function)
        aug_split = 'TRAIN' if self.split == 'TRAIN' else 'TEST'
        image, boxes, labels, difficulties = transform(image, boxes, labels, difficulties, split=aug_split)

        return image, boxes, labels, difficulties

    def __len__(self):
        return len(self.image_paths)

    def collate_fn(self, batch):
        images, boxes, labels, difficulties = [], [], [], []
        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            difficulties.append(b[3])
        images = torch.stack(images, dim=0)
        return images, boxes, labels, difficulties
