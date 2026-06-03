"""
Knowledge Distillation: ResNet-18 (teacher) → MobileNetV3-Small (student)

Reuses the data pipeline, splits, sampler and transforms from train.py so the
only variable being changed is model capacity + the loss function.

Loss = alpha * KD_term + (1 - alpha) * CE_term
  KD_term = T^2 * KL( log_softmax(student/T) || softmax(teacher/T) )
  CE_term = CrossEntropy(student_logits, hard_labels)

The T^2 factor restores the gradient magnitude scaled down by softening with T.
"""

import os
import sys
import time
import copy
import subprocess
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models


# Skip subdirectories that are not real classes (e.g. a nested duplicate
# `PlantVillage/PlantVillage/` left over from re-extracting the dataset).
# The teacher was trained with 15 classes; this filter restores that view
# without deleting anything from disk.
_EXCLUDED_DIRS = {"PlantVillage"}


class FilteredImageFolder(datasets.ImageFolder):
    def find_classes(self, directory):
        classes = sorted(
            d.name for d in os.scandir(directory)
            if d.is_dir() and d.name not in _EXCLUDED_DIRS
        )
        if not classes:
            raise FileNotFoundError(f"No valid class folders in {directory}")
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        return classes, class_to_idx
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report

from train import (
    SEED,
    BASE_DIR, DATA_DIR, OUTPUT_DIR,
    IMG_SIZE, BATCH_SIZE, NUM_WORKERS,
    train_tf, eval_tf,
    TransformSubset,
    AdamWDML,
    build_model as build_teacher,
    _select_device,
)

# ── Distillation hyper-parameters ──────────────────────────────────────────────
TEACHER_WEIGHTS = os.path.join(OUTPUT_DIR, "best_model.pth")
STUDENTS_DIR    = os.path.join(OUTPUT_DIR, "students")
os.makedirs(STUDENTS_DIR, exist_ok=True)

EPOCHS         = 20
LR             = 1e-3
WEIGHT_DECAY   = 1e-4
TEMPERATURE    = 4.0     # softening factor for soft targets (Hinton: 3–5 typical)
ALPHA          = 0.7     # weight on KD term; (1-α) on hard-label CE
PATIENCE       = 6


def _replace_linear_head(model, attr, num_classes):
    """Resize the final Linear of a classifier head. Handles both Sequential
    classifiers (mobilenet, mnasnet) and bare Linear heads (shufflenet)."""
    layer = getattr(model, attr)
    if isinstance(layer, nn.Sequential):
        in_features = layer[-1].in_features
        layer[-1] = nn.Linear(in_features, num_classes)
    else:
        in_features = layer.in_features
        setattr(model, attr, nn.Linear(in_features, num_classes))
    return model


def build_mobilenet_v3_small(num_classes):
    m = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    return _replace_linear_head(m, "classifier", num_classes)


def build_shufflenet_v2_x1_0(num_classes):
    m = models.shufflenet_v2_x1_0(
        weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
    return _replace_linear_head(m, "fc", num_classes)


def build_mnasnet0_5(num_classes):
    m = models.mnasnet0_5(weights=models.MNASNet0_5_Weights.IMAGENET1K_V1)
    return _replace_linear_head(m, "classifier", num_classes)


def build_shufflenet_v2_x0_5(num_classes):
    m = models.shufflenet_v2_x0_5(
        weights=models.ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1)
    return _replace_linear_head(m, "fc", num_classes)


def build_squeezenet1_1(num_classes):
    # SqueezeNet's classifier is a 1×1 Conv2d, not a Linear — needs a custom path.
    m = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    m.num_classes = num_classes
    return m


# Ordered roughly from larger to smaller capacity. Each student is distilled
# independently from the same teacher; per-model artifacts let us compare
# accuracy/size trade-offs after the full sweep.
STUDENTS = [
    ("mobilenet_v3_small", build_mobilenet_v3_small),
    ("shufflenet_v2_x1_0", build_shufflenet_v2_x1_0),
    ("mnasnet0_5",         build_mnasnet0_5),
    ("shufflenet_v2_x0_5", build_shufflenet_v2_x0_5),
    ("squeezenet1_1",      build_squeezenet1_1),
]


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    Hinton KD loss.
      • KL on softened distributions captures the teacher's "dark knowledge"
        (relative confidences across non-target classes).
      • The T^2 factor compensates for gradient shrinkage caused by softening.
      • CE on hard labels keeps the student grounded in ground truth.
    """
    log_p_student = F.log_softmax(student_logits / T, dim=1)
    p_teacher     = F.softmax(teacher_logits / T,    dim=1)
    kd = F.kl_div(log_p_student, p_teacher, reduction="batchmean") * (T * T)
    ce = F.cross_entropy(student_logits, labels)
    return alpha * kd + (1.0 - alpha) * ce, kd.item(), ce.item()


def run_train_epoch(student, teacher, loader, optimizer, device, T, alpha):
    student.train()
    total_loss = total_kd = total_ce = 0.0
    correct = total = 0

    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)

        # Teacher inference: no grad, no BN update.
        with torch.no_grad():
            teacher_logits = teacher(imgs)

        student_logits = student(imgs)
        loss, kd_v, ce_v = distillation_loss(
            student_logits, teacher_logits, lbls, T, alpha
        )

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_kd   += kd_v       * bs
        total_ce   += ce_v       * bs
        correct    += (student_logits.argmax(1) == lbls).sum().item()
        total      += bs

    return total_loss / total, total_kd / total, total_ce / total, correct / total


@torch.no_grad()
def run_eval_epoch(model, loader, device):
    model.eval()
    correct = total = 0
    loss_sum = 0.0
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        loss_sum += F.cross_entropy(logits, lbls, reduction="sum").item()
        correct  += (logits.argmax(1) == lbls).sum().item()
        total    += imgs.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate_test(model, loader, device, classes, tag):
    model.eval()
    all_preds, all_labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        all_preds.extend(model(imgs).argmax(1).cpu().numpy())
        all_labels.extend(lbls.numpy())
    report = classification_report(
        all_labels, all_preds, target_names=classes, digits=4
    )
    print(f"\n=== {tag} — Test classification report ===\n{report}")
    out_path = os.path.join(OUTPUT_DIR, f"classification_report_{tag}.txt")
    with open(out_path, "w") as f:
        f.write(report)
    return all_preds, all_labels


def distill_one_student(name, builder, teacher, train_loader, val_loader,
                        test_loader, device, num_classes, classes):
    """Distill a single student model against the shared (frozen) teacher.
    Saves best weights, training history, and per-student classification
    report. Returns a metrics dict for the final comparison table."""
    print(f"\n{'='*72}\n>>> Distilling student: {name}\n{'='*72}")
    student = builder(num_classes).to(device)
    n_params = count_params(student)
    print(f"Student params: {n_params:,}")

    optimizer = AdamWDML(student.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(EPOCHS // 2, 1), T_mult=1, eta_min=1e-6
    )

    best_val_acc = 0.0
    best_weights = copy.deepcopy(student.state_dict())
    best_epoch   = 0
    no_improve   = 0
    history = {"train_loss": [], "kd": [], "ce": [], "train_acc": [],
               "val_loss": [], "val_acc": []}

    weights_path = os.path.join(STUDENTS_DIR, f"student_{name}.pth")
    t_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, kd_v, ce_v, tr_acc = run_train_epoch(
            student, teacher, train_loader, optimizer, device, TEMPERATURE, ALPHA
        )
        vl_loss, vl_acc = run_eval_epoch(student, val_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss); history["kd"].append(kd_v)
        history["ce"].append(ce_v);            history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss);   history["val_acc"].append(vl_acc)

        elapsed = time.time() - t0
        print(f"[KD:{name}] Epoch {epoch:03d}/{EPOCHS}  "
              f"loss={tr_loss:.4f} (kd={kd_v:.4f} ce={ce_v:.4f})  "
              f"train_acc={tr_acc:.4f}  val_acc={vl_acc:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  {elapsed:.0f}s")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            best_weights = copy.deepcopy(student.state_dict())
            best_epoch   = epoch
            no_improve   = 0
            torch.save(best_weights, weights_path)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping after {epoch} epochs.")
                break

    student.load_state_dict(best_weights)
    print(f"Best val acc for {name}: {best_val_acc:.4f} (epoch {best_epoch})  "
          f"({time.time() - t_start:.0f}s)")
    return weights_path


def main():
    device, _, device_label = _select_device()
    print(f"Device: {device_label}")

    # ── Data: identical pipeline to train.py, set up once and shared across runs
    full_dataset = FilteredImageFolder(DATA_DIR)
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

    train_labels = labels[train_idx]
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    class_weights = 1.0 / np.where(class_counts > 0, class_counts, 1.0)
    sample_weights = class_weights[train_labels]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(train_ds),
        replacement=True,
    )

    pin = (device.type == "cuda")
    persist = NUM_WORKERS > 0
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=persist)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=persist)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=persist)

    # ── Teacher: load once, freeze, share across all student runs ─────────────
    teacher = build_teacher(num_classes).to(device)
    # weights_only=False: checkpoint contains numpy scalars from training;
    # safe because we trust our own file.
    state = torch.load(TEACHER_WEIGHTS, map_location=device, weights_only=False)
    teacher.load_state_dict(state)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    teacher_params = count_params(teacher)
    print(f"Teacher  (ResNet-18)         params: {teacher_params:>12,}")

    # Sanity: confirm teacher accuracy before distilling. Garbage in → garbage out.
    _, teacher_val_acc = run_eval_epoch(teacher, val_loader, device)
    print(f"Teacher val acc (sanity): {teacher_val_acc:.4f}")

    # ── Sweep: distill each candidate student independently ───────────────────
    trained = []
    for name, builder in STUDENTS:
        weights_path = distill_one_student(
            name, builder, teacher,
            train_loader, val_loader, test_loader,
            device, num_classes, classes,
        )
        trained.append((name, weights_path))

    # Free GPU memory before launching the evaluator subprocesses.
    del teacher, train_loader, val_loader, test_loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Per-student evaluation via evaluate_model.py (teacher = baseline) ─────
    eval_script = os.path.join(BASE_DIR, "evaluate_model.py")
    for name, weights_path in trained:
        report_path = os.path.join(STUDENTS_DIR, f"eval_{name}.json")
        print(f"\n{'#'*72}\n# Running evaluate_model.py for {name}\n{'#'*72}")
        subprocess.run(
            [
                sys.executable, eval_script,
                "--weights", weights_path,
                "--arch", name,
                "--baseline-weights", TEACHER_WEIGHTS,
                "--baseline-arch", "resnet18",
                "--save-json", report_path,
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
