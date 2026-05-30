# SSD-LVIS Convergence Study

A convergence study of the **SSD300 object detection model** trained from scratch on subsets of the LVIS dataset. The goal is to analyze how much data is needed for the model to converge, by training on 5 progressively larger splits and comparing training/validation loss curves and mAP.

---

## Project Overview

| Property | Detail |
|---|---|
| Model | SSD300 (Single Shot MultiBox Detector) |
| Backbone | ResNet50 (trained from scratch, no pretrained weights) |
| Dataset | LVIS Greedy-50 (50 selected classes, COCO format) |
| Metric | mAP@0.50, train/val loss curves |
| Framework | PyTorch |

---

## Dataset Splits

The LVIS Greedy-50 dataset is split into 5 subsets using stratified sampling with disjoint primary class assignment:

| Split | Train images | Val images |
|---|---|---|
| split_005 (5%) | 2,869 | 692 |
| split_025 (25%) | 14,341 | 3,561 |
| split_050 (50%) | 28,686 | 7,143 |
| split_075 (75%) | 43,016 | 10,729 |
| split_100 (100%) | 57,370 | 14,314 |

---

## How the Data Was Split

### The Problem with Random Splitting

A naive random split would pick images randomly across the entire dataset. This creates a serious problem for a convergence study — some classes might end up with very few images in smaller splits, or even disappear entirely. If split_005 has only 5% of the data but no images of `giraffe` at all, the model hasn't actually been given a chance to learn that class. You can't tell if poor performance is due to lack of data or lack of class coverage.

### Our Solution — Class-Based Stratified Split

We ensure **every class is represented in every split**, no matter how small. Here is the exact logic:

**Step 1 — Primary class assignment**

Each image is assigned to exactly one "primary class" — the class that appears most in that image. For example, an image with 5 bananas and 1 cat is assigned to `banana`.

```
Image 1: [banana×5, cat×1]  →  primary class = banana
Image 2: [dog×3, bowl×2]    →  primary class = dog
Image 3: [banana×2]         →  primary class = banana
```

This creates a clean group of images per class with no overlap in ownership.

**Step 2 — Shuffle once with a fixed seed**

Images within each class group are shuffled once using a fixed random seed (42). This makes the splits fully reproducible — running the script twice always produces identical splits.

**Step 3 — Per-class slicing**

For each split percentage, we take the first N% of images **from each class separately**. This is the key decision:

```
banana group:  [img_A, img_B, img_C, img_D, img_E, img_F, img_G, img_H, img_I, img_J]
                └── 5%  ──┘
                └────── 25% ──────┘
                └──────────── 50% ──────────────┘
```

Because smaller splits are always a prefix of larger splits, the 5% split is a strict subset of the 25% split, which is a subset of the 50% split, and so on. This means:
- Every split has all 50 classes represented
- Adding more data never removes images already seen — only adds new ones
- Convergence curves are directly comparable across splits

**Step 4 — 80/20 train/val within each class**

Within each class group at each percentage, 80% goes to training and 20% to validation (minimum 1 image per class in val). This guarantees the val set also has all 50 classes for every split.

**Step 5 — Sanity check**

After building all splits, the script verifies that all 50 classes appear in both train and val for every split. If any class is missing, the split is flagged as invalid.

### Why This Matters for the Convergence Study

The entire point of this study is to answer: *"how much data does the model need to converge?"* For that question to be meaningful, the **only variable** between splits must be the amount of data — not which classes are present. By guaranteeing all 50 classes appear in every split, we isolate dataset size as the single controlled variable. Any difference in convergence between splits is caused by data quantity, not data composition.

---

## Model Architecture

- **Backbone:** ResNet50 — replaces the original VGG16 backbone
  - `layer2` output → `(N, 512, 38, 38)` — replaces conv4_3
  - `layer3` output → `(N, 1024, 19, 19)` — replaces conv7
- **Auxiliary convolutions:** 4 additional feature maps (10×10, 5×5, 3×3, 1×1)
- **Prediction heads:** 8732 prior boxes across 6 feature maps
- **Loss:** MultiBox Loss (SmoothL1 localization + Cross-Entropy confidence with Hard Negative Mining)
- **No pretrained weights** — all weights randomly initialized

---

## Training

All splits are trained for **120,000 iterations** (same compute budget):

```python
# In train.py — change this one line to switch splits
SPLIT = 'split_005'  # options: split_005, split_025, split_050, split_075, split_100
```

Key training settings:
- Batch size: 8
- Optimizer: SGD (lr=1e-3, momentum=0.9, weight_decay=5e-4)
- LR warmup: linear ramp over first 500 iterations
- LR decay: ×0.1 at 80k and 100k iterations
- Augmentation: photometric distort, random expand, random crop, horizontal flip

**Run training:**
```bash
python train.py
```

Checkpoints saved to `checkpoints/<split_name>/`:
- `checkpoint_latest.pth.tar` — always latest epoch (crash recovery)
- `checkpoint_best.pth.tar` — lowest validation loss (for evaluation)
- `checkpoint_epoch_XXXX.pth.tar` — snapshot every 50 epochs (convergence study)

Loss logged to `checkpoints/<split_name>/loss_log.csv` after every epoch.

---

## Evaluation

```python
# In eval.py — set split and checkpoint to evaluate
SPLIT = 'split_005'
EPOCH = None  # None = best checkpoint, or set epoch number for a periodic snapshot
```

```bash
python eval.py
```

Outputs per-class AP and overall mAP.

---

## Detection

```python
# In detect.py — set split and checkpoint
SPLIT = 'split_005'
EPOCH = None
```

```bash
python detect.py
```

---

## Project Structure

```
├── train.py          # Training loop with validation, warmup, CSV logging
├── eval.py           # mAP evaluation
├── detect.py         # Single image detection and visualization
├── model.py          # SSD300 with ResNet50 backbone
├── datasets.py       # LVISDataset — reads COCO-format annotations
├── utils.py          # Label map, loss, transforms, helpers
└── checkpoints/
    └── split_005/
        ├── checkpoint_latest.pth.tar
        ├── checkpoint_best.pth.tar
        ├── checkpoint_epoch_0000.pth.tar
        └── loss_log.csv
```

---

## Classes

50 LVIS categories: `airplane`, `suitcase`, `banana`, `baseball_bat`, `baseball_glove`, `bed`, `cow`, `bench`, `bird`, `boat`, `bowl`, `broccoli`, `bus`, `cat`, `cellular_telephone`, `clock`, `computer_keyboard`, `dog`, `doughnut`, `drawer`, `elephant`, `faucet`, `fireplug`, `fork`, `frisbee`, `giraffe`, `horse`, `kite`, `motorcycle`, `necktie`, `pizza`, `plate`, `sheep`, `skateboard`, `ski_pole`, `snowboard`, `sofa`, `spectacles`, `street_sign`, `surfboard`, `teddy_bear`, `television_set`, `tennis_racket`, `toilet`, `traffic_light`, `train`, `umbrella`, `vase`, `windshield_wiper`, `zebra`

---

## Based On

Original SSD300 tutorial by [sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection), adapted for LVIS dataset with ResNet50 backbone and convergence study framework.
