"""
ResNet-18 fine-tuned on PlantVillage
Best practices applied:
  - Transfer learning (ImageNet pretrained weights)
  - Stratified train/val/test split (70/15/15)
  - Weighted sampling to handle class imbalance
  - Progressive unfreezing: head only → full fine-tune
  - Strong augmentation (train) / deterministic (val/test)
  - Cosine annealing LR with warm restarts
  - Mixed precision (skipped on CPU, auto-enabled on CUDA)
  - Early stopping + best-model checkpoint
  - Per-class metrics and confusion matrix saved to disk
"""

import os
import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns  # optional – falls back gracefully if missing

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "PlantVillage")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Hyper-parameters ───────────────────────────────────────────────────────────
IMG_SIZE       = 224
BATCH_SIZE     = 64
NUM_WORKERS    = 8          # parallel CPU workers feeding the GPU
EPOCHS_HEAD    = 5          # train only the classification head
EPOCHS_FULL    = 25         # fine-tune entire network
LR_HEAD        = 1e-3
LR_FULL        = 3e-4
WEIGHT_DECAY   = 1e-4
PATIENCE       = 7          # early stopping patience (full fine-tune phase)

# ── Device ─────────────────────────────────────────────────────────────────────
def _select_device():
    if torch.cuda.is_available():
        return torch.device("cuda"), True, "CUDA"
    try:
        import torch_directml
        return torch_directml.device(), False, f"DirectML ({torch_directml.device_name(0)})"
    except Exception as e:
        print(f"DirectML init failed ({type(e).__name__}: {e}); falling back to CPU")
        return torch.device("cpu"), False, "CPU"

# Globals populated by main(); referenced by run_epoch / train_phase below.
# They sit at module level so the existing function bodies can read them
# without changing every signature.
device       = None
use_amp      = False
criterion    = None
scaler       = None
train_loader = None
val_loader   = None
history      = None

# ── Transforms ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_tf = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# Wrap subsets with different transforms
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset   = dataset
        self.indices   = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, label = self.dataset[self.indices[i]]
        return self.transform(img), label

# ── Model ──────────────────────────────────────────────────────────────────────
def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes),
    )
    return model

# ── Helpers ────────────────────────────────────────────────────────────────────
def freeze_backbone(model):
    for name, param in model.named_parameters():
        param.requires_grad = "fc" in name

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            autocast_device = "cuda" if use_amp else "cpu"
            with torch.autocast(device_type=autocast_device, enabled=use_amp):
                logits = model(imgs)
                loss   = criterion(logits, lbls)
            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * imgs.size(0)
            correct    += (logits.argmax(1) == lbls).sum().item()
            total      += imgs.size(0)

    return total_loss / total, correct / total

# ── DirectML-safe AdamW ────────────────────────────────────────────────────────
# PyTorch's stock AdamW (both foreach and single-tensor paths) calls
# `exp_avg.lerp_(grad, 1 - beta1)`, and DirectML has no kernel for `lerp`,
# so every optimizer step falls back to CPU. This drop-in re-implementation
# uses mul_/add_/addcmul_/addcdiv_ — all natively supported — preserving
# AdamW's exact math.
class AdamWDML(optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                p.mul_(1 - lr * wd)  # decoupled weight decay
                state["exp_avg"].mul_(b1).add_(grad, alpha=1 - b1)
                state["exp_avg_sq"].mul_(b2).addcmul_(grad, grad, value=1 - b2)
                bc1 = 1 - b1 ** state["step"]
                bc2 = 1 - b2 ** state["step"]
                step_size = lr / bc1
                denom = (state["exp_avg_sq"].sqrt() / (bc2 ** 0.5)).add_(eps)
                p.addcdiv_(state["exp_avg"], denom, value=-step_size)
        return loss

def train_phase(model, epochs, lr, phase_name):
    optimizer = AdamWDML(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(epochs // 2, 1), T_mult=1, eta_min=1e-6
    )
    best_val_acc  = 0.0
    best_weights  = copy.deepcopy(model.state_dict())
    no_improve    = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        print(f"[{phase_name}] Epoch {epoch:03d}/{epochs}  "
              f"train_loss={tr_loss:.4f}  train_acc={tr_acc:.4f}  "
              f"val_loss={vl_loss:.4f}  val_acc={vl_acc:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve   = 0
            torch.save(best_weights, os.path.join(OUTPUT_DIR, "best_model.pth"))
        else:
            no_improve += 1
            if phase_name == "FULL" and no_improve >= PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    model.load_state_dict(best_weights)
    print(f"[{phase_name}] Best val acc: {best_val_acc:.4f}\n")
    return model

# ── Plots ──────────────────────────────────────────────────────────────────────
def plot_history(history, out_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=150)
    plt.close()

def plot_confusion(labels, preds, class_names, out_dir):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(14, 12))
    try:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
    except Exception:
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names, rotation=90)
        ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
        plt.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

# ── Entrypoint ─────────────────────────────────────────────────────────────────
# The `if __name__ == "__main__"` guard is required on Windows: torch's
# DataLoader uses `spawn` to create worker processes, and each worker
# re-imports this module. Without the guard, that re-import would re-run
# the training code in every worker, which crashes / loops.
def main():
    global device, use_amp, criterion, scaler, train_loader, val_loader, history

    device, use_amp, _device_label = _select_device()
    print(f"Device: {_device_label}  |  AMP: {use_amp}")

    # Dataset & stratified split
    full_dataset = datasets.ImageFolder(DATA_DIR)
    labels       = np.array([s[1] for s in full_dataset.samples])
    classes      = full_dataset.classes
    num_classes  = len(classes)
    print(f"Classes ({num_classes}): {classes}")

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    trainval_idx, test_idx = next(sss1.split(np.zeros(len(labels)), labels))
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.15/0.85, random_state=SEED)
    train_idx, val_idx = next(sss2.split(np.zeros(len(trainval_idx)), labels[trainval_idx]))
    train_idx = trainval_idx[train_idx]
    val_idx   = trainval_idx[val_idx]
    print(f"Split → train: {len(train_idx)}  val: {len(val_idx)}  test: {len(test_idx)}")

    train_ds = TransformSubset(full_dataset, train_idx, train_tf)
    val_ds   = TransformSubset(full_dataset, val_idx,   eval_tf)
    test_ds  = TransformSubset(full_dataset, test_idx,  eval_tf)

    # Weighted sampler (handle class imbalance)
    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    class_weights = 1.0 / np.where(class_counts > 0, class_counts, 1.0)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_ds),
        replacement=True,
    )

    _pin     = (device.type == "cuda")
    _persist = NUM_WORKERS > 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=_pin,
                              persistent_workers=_persist)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=_pin,
                              persistent_workers=_persist)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=_pin,
                              persistent_workers=_persist)

    model = build_model(num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    history   = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # Phase 1: train head only
    print(f"\n{'='*60}\nPhase 1 — Head only ({EPOCHS_HEAD} epochs)\n{'='*60}")
    freeze_backbone(model)
    print(f"Trainable params: {count_trainable(model):,}")
    model = train_phase(model, EPOCHS_HEAD, LR_HEAD, "HEAD")

    # Phase 2: fine-tune everything
    print(f"\n{'='*60}\nPhase 2 — Full fine-tune (up to {EPOCHS_FULL} epochs)\n{'='*60}")
    unfreeze_all(model)
    print(f"Trainable params: {count_trainable(model):,}")
    model = train_phase(model, EPOCHS_FULL, LR_FULL, "FULL")

    # Evaluation on test set
    print("\nEvaluating on test set …")
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(lbls.numpy())

    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print(report)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    plot_history(history, OUTPUT_DIR)
    plot_confusion(all_labels, all_preds, classes, OUTPUT_DIR)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
