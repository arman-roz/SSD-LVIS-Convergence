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

### Weight Initialization

Since the model is trained entirely from scratch, initialization plays an important role in how fast and stably training begins. We use **Kaiming uniform initialization** (also called He initialization) consistently across the entire network:

| Component | Initialization | Why |
|---|---|---|
| ResNet50 backbone | Kaiming uniform (PyTorch default) | Built into PyTorch's ResNet implementation |
| Auxiliary convolutions | Kaiming uniform (`nonlinearity='relu'`) | Changed from Xavier — ReLU layers need Kaiming |
| Prediction convolutions | Kaiming uniform (`nonlinearity='relu'`) | Changed from Xavier — consistency across network |
| All biases | 0 | Standard practice |

**Why Kaiming over Xavier:**
Xavier initialization was designed for sigmoid/tanh activations and assumes symmetric activations around zero. ReLU kills negative values entirely, which changes the variance of activations flowing through the network. Kaiming initialization accounts for this by scaling the initial weights differently — it keeps activation variance stable across layers in ReLU networks. Since we use ReLU throughout (backbone, auxiliary convolutions) and are training from scratch where initialization matters most, Kaiming is the theoretically correct and consistent choice.

---

## Training

```python
# In train.py — change this one line to switch splits
SPLIT = 'split_005'  # options: split_005, split_025, split_050, split_075, split_100
```

**Run training:**
```bash
python train.py
```

### Training Settings

| Parameter | Value |
|---|---|
| Batch size | 8 |
| Optimizer | SGD |
| Learning rate | 1e-3 |
| Momentum | 0.9 |
| Weight decay | 5e-4 |
| Max epochs | 500 (safety cap) |
| LR warmup | Linear ramp over first 500 iterations |
| Augmentation | Photometric distort, random expand, random crop, horizontal flip |

### Early Stopping

Instead of training for a fixed number of iterations, we use **early stopping** — training runs until the model stops improving and then automatically stops. This lets each split find its natural convergence point rather than being cut off at an arbitrary iteration count.

| Parameter | Value | Meaning |
|---|---|---|
| Metric | Validation loss | Cheap to compute every epoch — no need to run full mAP |
| Patience | 15 epochs | Stop if no improvement for 15 consecutive epochs |
| Min delta | 0.001 | Improvements smaller than this don't count |
| Max epochs | 500 | Safety ceiling — early stopping triggers well before this |

**Why validation loss and not mAP?**
Computing mAP requires running the full detection pipeline (NMS, IoU matching) on the entire val set — this takes 30-60 minutes per epoch on large splits. Validation loss is already computed every epoch in seconds and is a reliable proxy for model quality during training. Final mAP is computed once after training on `checkpoint_best`.

**Why patience = 15?**
With only 692 validation images in split_005, the val loss curve is naturally noisy — it can fluctuate for 10+ epochs even mid-training without the model truly plateauing. Patience of 15 gives enough room to distinguish real plateaus from noise, while still stopping well before 500 epochs.

### Learning Rate Schedule — ReduceLROnPlateau

Instead of hardcoded LR decay at fixed iteration counts, we use `ReduceLROnPlateau` — the learning rate is automatically reduced whenever val loss stops improving:

```
No improvement for 5 epochs  →  lr multiplied by 0.1  (model gets another chance)
No improvement for 15 epochs →  training stops (early stopping)
```

This pairs naturally with early stopping because both react to the same signal — val loss. The lr decays when training slows, giving the model a chance to fine-tune. If that still doesn't help after 15 epochs total, training ends. The lr is never reduced below `1e-6`.

### LR Warmup

For the first 500 iterations, the learning rate is linearly ramped from ~0 to the target lr (1e-3). This prevents unstable gradient updates at the start of training when weights are random and gradients are large. After 500 iterations, the lr reaches its full value and normal training begins.

```
iter 1:   lr = 0.000002  (nearly zero)
iter 250: lr = 0.000500  (halfway)
iter 500: lr = 0.001000  (full lr — warmup complete)
iter 501+: ReduceLROnPlateau takes over
```

### Checkpoints

Saved to `checkpoints/<split_name>/`:

| File | When saved | Purpose |
|---|---|---|
| `checkpoint_latest.pth.tar` | Every epoch | Crash recovery — always resumable |
| `checkpoint_best.pth.tar` | When val loss improves | Best model — used for final mAP evaluation |
| `checkpoint_epoch_XXXX.pth.tar` | Every 50 epochs | Intermediate snapshots for convergence study |

### Loss Log

`checkpoints/<split_name>/loss_log.csv` — written after every epoch:
```
epoch, train_loss, val_loss, lr
0, 48.3421, 51.2341, 0.001000
1, 32.1234, 35.4521, 0.001000
...
```
Includes current lr so you can see exactly when ReduceLROnPlateau fired during training.

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
├── train.py          # Training loop — early stopping, ReduceLROnPlateau, warmup, CSV logging
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

## Architecture Decisions

A summary of all design decisions made for this project:

| Decision | Choice | Reason |
|---|---|---|
| Initialization | Kaiming uniform throughout | Consistent across all ReLU layers — correct for from-scratch training |
| Optimization | SGD + momentum + weight decay | Standard SSD training setup |
| BatchNorm in aux heads | No — input already normalized | ResNet backbone's last layer already outputs BatchNorm-normalized features |
| BatchNorm in pred heads | No — destabilizes outputs | Detection heads must not have BatchNorm — interferes with box offsets and class scores |
| Early stopping | patience=10, min_delta=0.001 | Lets each split find its natural convergence point |
| LR schedule | ReduceLROnPlateau | Automatically decays lr when val loss plateaus — pairs naturally with early stopping |

---

## Based On

Original SSD300 tutorial by [sgrvinod](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection), adapted for LVIS dataset with ResNet50 backbone and convergence study framework.
