import os
import csv
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import LVISDataset
from utils import *

# ── Split to train on — change this one line to switch splits ──────────
SPLIT = 'split_005'  # options: split_005, split_025, split_050, split_075, split_100
# ────────────────────────────────────────────────────────────────────────

WARMUP_ITERS = 500  # linearly ramp lr from 0 → lr over first 500 iterations

# Data parameters
data_folder = os.path.join(os.path.dirname(__file__), '..', 'Dataset', SPLIT, 'train')
val_folder  = os.path.join(os.path.dirname(__file__), '..', 'Dataset', SPLIT, 'val')

# Model parameters
n_classes = len(label_map)  # 51 (50 LVIS classes + background)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint = None  # path to checkpoint to resume from, None to start fresh
batch_size = 8  # batch size
iterations = 120000  # number of iterations to train
workers = 4  # number of workers for loading data in the DataLoader
print_freq = 200  # print training status every __ batches
lr = 1e-3  # learning rate
decay_lr_at = [80000, 100000]  # decay learning rate after these many iterations
decay_lr_to = 0.1  # decay learning rate to this fraction of the existing learning rate
momentum = 0.9  # momentum
weight_decay = 5e-4  # weight decay
grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialized here so it's available whether starting fresh or resuming
    best_val_loss = float('inf')

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                    lr=lr, momentum=momentum, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        # Restore best_val_loss so checkpoint_best is never overwritten by a worse model after resume
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = LVISDataset(data_folder, split='train')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers,
                                               pin_memory=True)

    # Validation dataset — same class as training but points to the val/ folder.
    # split='val' tells LVISDataset to use TEST-mode transforms (no random flipping/cropping),
    # so we measure loss on clean, unaugmented images — a fair reflection of true model quality.
    val_dataset = LVISDataset(val_folder, split='val')
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             collate_fn=val_dataset.collate_fn, num_workers=workers,
                                             pin_memory=True)

    # Calculate total number of epochs to train and the epochs to decay learning rate at (i.e. convert iterations to epochs)
    # To convert iterations to epochs, divide iterations by the number of iterations per epoch
    # The paper trains for 120,000 iterations with a batch size of 32, decays after 80,000 and 100,000 iterations
    epochs = iterations // (len(train_dataset) // batch_size)
    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]

    # ── CSV loss log setup ─────────────────────────────────────────────────
    # We create one CSV file per split inside its checkpoint folder.
    # Each row written after an epoch: epoch number, average train loss, average val loss.
    # This lets you plot convergence curves later in Python/Excel without digging through terminal logs.
    log_dir = os.path.join('checkpoints', SPLIT)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, 'loss_log.csv')

    # If we are resuming from a checkpoint (start_epoch > 0), we open in append mode ('a')
    # so previous epochs are not lost. If starting fresh, we open in write mode ('w')
    # and write the header row first.
    csv_mode = 'a' if start_epoch > 0 else 'w'
    log_file = open(log_path, csv_mode, newline='')
    csv_writer = csv.writer(log_file)
    if start_epoch == 0:
        csv_writer.writerow(['epoch', 'train_loss', 'val_loss'])
    # ──────────────────────────────────────────────────────────────────────

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        # One epoch's training — returns the average loss over all batches this epoch
        train_loss = train(train_loader=train_loader,
                           model=model,
                           criterion=criterion,
                           optimizer=optimizer,
                           epoch=epoch)

        # One epoch's validation — returns the average loss on the val set
        val_loss = validate(val_loader=val_loader,
                            model=model,
                            criterion=criterion,
                            epoch=epoch)

        # Write this epoch's losses to the CSV immediately after computing them.
        # flush() forces Python to write to disk right away — if training crashes next epoch,
        # we still have this row saved.
        csv_writer.writerow([epoch, f'{train_loss:.4f}', f'{val_loss:.4f}'])
        log_file.flush()

        # ── Checkpoint saving strategy ─────────────────────────────────────
        # 1. Always overwrite latest — one file, always resumable after a crash
        #    best_val_loss is included so resume correctly restores it
        save_checkpoint(epoch, model, optimizer, SPLIT, 'checkpoint_latest.pth.tar', best_val_loss)

        # 2. Overwrite best only when val loss improves — used by eval.py for final mAP
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(epoch, model, optimizer, SPLIT, 'checkpoint_best.pth.tar', best_val_loss)

        # 3. Periodic snapshot every 50 epochs — lets you evaluate mAP at intermediate
        #    points for your convergence study without saving every epoch
        if epoch % 50 == 0:
            save_checkpoint(epoch, model, optimizer, SPLIT, f'checkpoint_epoch_{epoch:04d}.pth.tar', best_val_loss)
        # ──────────────────────────────────────────────────────────────────

    log_file.close()


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: MultiBox loss
    :param optimizer: optimizer
    :param epoch: epoch number
    :return: average training loss for this epoch
    """
    model.train()  # training mode enables dropout

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Linear LR warmup — ramps lr from ~0 to the target lr over the first
        # WARMUP_ITERS iterations. global_iter is computed from epoch and batch
        # index so it correctly skips warmup when resuming from a checkpoint.
        global_iter = epoch * len(train_loader) + i
        if global_iter < WARMUP_ITERS:
            warmup_lr = lr * (global_iter + 1) / WARMUP_ITERS
            for idx, pg in enumerate(optimizer.param_groups):
                pg['lr'] = warmup_lr * (2 if idx == 0 else 1)  # idx=0 is biases (always 2x)

        # Move to default device
        images = images.to(device)  # (batch_size (N), 3, 300, 300)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]

        # Forward prop.
        predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

        # Loss
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        losses.update(loss.item(), images.size(0))
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored

    return losses.avg  # return average loss so main() can log it to CSV


def validate(val_loader, model, criterion, epoch):
    """
    One epoch's validation.

    During validation we do NOT update the model weights — we only do a forward pass
    to measure how well the model performs on data it has never been trained on.
    This tells us whether the model is truly learning or just memorising the training set.

    Key differences from train():
      - model.eval()     : disables dropout and switches BatchNorm to use running statistics
                           instead of batch statistics — makes predictions deterministic
      - torch.no_grad()  : tells PyTorch not to build a computation graph, which saves GPU
                           memory and speeds up the forward pass (no gradients needed here)
      - No optimizer.step() — we never touch the weights

    :param val_loader: DataLoader for validation data
    :param model: model
    :param criterion: MultiBox loss
    :param epoch: epoch number
    :return: average validation loss for this epoch
    """
    model.eval()  # switch to eval mode — disables dropout, uses running BN stats

    losses = AverageMeter()

    # torch.no_grad() is a context manager that disables gradient tracking for everything
    # inside the 'with' block. PyTorch normally stores all intermediate activations so it
    # can compute gradients during backward — here we don't need that, so we skip it.
    with torch.no_grad():
        for _, (images, boxes, labels, _) in enumerate(val_loader):

            # Move to device — same as training
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward pass only — no backward, no optimizer step
            predicted_locs, predicted_scores = model(images)

            # Compute loss exactly the same way as training.
            # The MultiBoxLoss still computes localization + confidence loss,
            # which gives us a comparable number to the training loss.
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)

            losses.update(loss.item(), images.size(0))

    print('Validation — Epoch: [{0}]\t'
          'Loss {loss.avg:.4f}\n'.format(epoch, loss=losses))

    return losses.avg  # return average val loss so main() can log it to CSV


if __name__ == '__main__':
    main()
