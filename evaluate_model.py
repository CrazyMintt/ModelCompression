"""
evaluate_model.py — Generic evaluation for PlantVillage models.

Reports the metrics used in model-compression research:
  • Parameter counts (total / trainable / buffers)
  • Storage footprint (file on disk + theoretical FP32/FP16/INT8 sizes)
  • Computational cost (MACs and FLOPs per forward pass)
  • Inference latency with percentiles (mean, std, P50, P95, P99) and throughput
  • Test-set accuracy (Top-1, macro F1, weighted F1, full per-class report)
  • Optional comparison vs. a baseline:
       compression ratios (params, MACs, storage)
       speedup at each batch size
       Δ accuracy / Δ F1
       prediction agreement (how often the two models predict the same class)

Examples
--------
Evaluate the teacher:
    python evaluate_model.py --weights output/best_model.pth --arch resnet18

Evaluate the distilled student and compare against the teacher as baseline:
    python evaluate_model.py \
        --weights output/student_distilled.pth --arch mobilenet_v3_small \
        --baseline-weights output/best_model.pth --baseline-arch resnet18
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, f1_score

from train import (
    SEED, DATA_DIR, OUTPUT_DIR, IMG_SIZE,
    eval_tf, TransformSubset, _select_device,
)
from distill import FilteredImageFolder


# ── Architecture registry ─────────────────────────────────────────────────────
# Each builder must produce a model whose state_dict shape exactly matches the
# checkpoint produced by the corresponding training script.
def _build_resnet18(num_classes):
    m = models.resnet18(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_f, num_classes))
    return m


def _build_mobilenet_v3_small(num_classes):
    m = models.mobilenet_v3_small(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m


def _build_shufflenet_v2_x1_0(num_classes):
    m = models.shufflenet_v2_x1_0(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _build_shufflenet_v2_x0_5(num_classes):
    m = models.shufflenet_v2_x0_5(weights=None)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def _build_mnasnet0_5(num_classes):
    m = models.mnasnet0_5(weights=None)
    in_f = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_f, num_classes)
    return m


def _build_squeezenet1_1(num_classes):
    # SqueezeNet uses a 1×1 Conv as its final classifier.
    m = models.squeezenet1_1(weights=None)
    m.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=1)
    m.num_classes = num_classes
    return m


ARCHS = {
    "resnet18":           _build_resnet18,
    "mobilenet_v3_small": _build_mobilenet_v3_small,
    "shufflenet_v2_x1_0": _build_shufflenet_v2_x1_0,
    "shufflenet_v2_x0_5": _build_shufflenet_v2_x0_5,
    "mnasnet0_5":         _build_mnasnet0_5,
    "squeezenet1_1":      _build_squeezenet1_1,
}


# ── Loading ───────────────────────────────────────────────────────────────────
def load_model(arch, weights_path, num_classes, device):
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Available: {list(ARCHS)}")
    model = ARCHS[arch](num_classes).to(device)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ── Capacity & storage ────────────────────────────────────────────────────────
def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers   = sum(b.numel() for b in model.buffers())
    return total, trainable, buffers


def file_size_mb(path):
    return os.path.getsize(path) / 1024**2


def theoretical_size_mb(num_params, bytes_per_param):
    return num_params * bytes_per_param / 1024**2


# ── MACs / FLOPs ──────────────────────────────────────────────────────────────
# Counts multiply-accumulates for the two layer types that dominate CNN cost
# (Conv2d, Linear). Activations, BN, pooling are ignored — they're <5 % of the
# total in standard CNNs and following common research convention.
#   Conv2d MACs = (in_channels / groups) * k_h * k_w * out_channels * H_out * W_out
#   Linear MACs = in_features * out_features
# FLOPs ≈ 2 × MACs (multiply + accumulate counted as 2 ops; some papers report
# 1 op — always state your convention).
def compute_macs(model, input_shape, device):
    macs = [0]

    def conv_hook(m, _inp, out):
        _, _, h, w = out.shape
        kops = m.kernel_size[0] * m.kernel_size[1]
        macs[0] += (m.in_channels // m.groups) * kops * m.out_channels * h * w

    def linear_hook(m, _inp, _out):
        macs[0] += m.in_features * m.out_features

    handles = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook))

    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, *input_shape, device=device)
        model(x)

    for h in handles:
        h.remove()
    return macs[0]


# ── Latency ───────────────────────────────────────────────────────────────────
# Measured by forcing a CPU read of the output (`float(y[0, 0])`) which acts as
# a sync point on CUDA, DirectML, and CPU alike. Warmup runs are critical:
# the first few forwards include kernel compilation / cache priming and are
# 5–50× slower than steady-state, which would skew the mean.
def benchmark_latency(model, device, batch_size, n_runs=50, warmup=10):
    model.eval()
    x = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE, device=device)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            y = model(x); float(y.flatten()[0])
        for _ in range(n_runs):
            t0 = time.perf_counter()
            y = model(x); float(y.flatten()[0])
            times.append((time.perf_counter() - t0) * 1000.0)
    times = np.array(times)
    return {
        "mean_ms":         float(times.mean()),
        "std_ms":          float(times.std()),
        "p50_ms":          float(np.percentile(times, 50)),
        "p95_ms":          float(np.percentile(times, 95)),
        "p99_ms":          float(np.percentile(times, 99)),
        "throughput_imgs": float(batch_size * 1000.0 / times.mean()),
    }


# ── Accuracy ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    preds, labels = [], []
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        preds.extend(model(imgs).argmax(1).cpu().numpy().tolist())
        labels.extend(lbls.numpy().tolist())
    return np.array(preds), np.array(labels)


# ── Full evaluation pipeline for a single model ───────────────────────────────
def evaluate(arch, weights_path, num_classes, classes, device, loader,
             batch_sizes, n_runs):
    model = load_model(arch, weights_path, num_classes, device)
    total, trainable, buffers = count_params(model)
    macs = compute_macs(model, (3, IMG_SIZE, IMG_SIZE), device)

    latencies = {
        bs: benchmark_latency(model, device, bs, n_runs=n_runs)
        for bs in batch_sizes
    }

    preds, labels = get_predictions(model, loader, device)
    top1 = float((preds == labels).mean())
    f1_macro    = float(f1_score(labels, preds, average="macro"))
    f1_weighted = float(f1_score(labels, preds, average="weighted"))
    per_class   = classification_report(labels, preds, target_names=classes, digits=4)

    return {
        "arch": arch,
        "weights": os.path.relpath(weights_path),
        "params_total":     total,
        "params_trainable": trainable,
        "buffers":          buffers,
        "macs":             macs,
        "flops":            macs * 2,
        "file_size_mb":     file_size_mb(weights_path),
        "fp32_size_mb":     theoretical_size_mb(total, 4),
        "fp16_size_mb":     theoretical_size_mb(total, 2),
        "int8_size_mb":     theoretical_size_mb(total, 1),
        "latencies":        latencies,
        "top1":             top1,
        "f1_macro":         f1_macro,
        "f1_weighted":      f1_weighted,
        "per_class_report": per_class,
        "preds":            preds,
        "labels":           labels,
    }


# ── Pretty printing ───────────────────────────────────────────────────────────
def _section(title, width=72):
    print(f"\n── {title} " + "─" * max(0, width - len(title) - 4))


def print_report(r):
    _section(f"Model: {r['arch']}  —  {r['weights']}")

    _section("Capacity & Storage")
    print(f"  Total parameters:        {r['params_total']:>15,}")
    print(f"  Trainable parameters:    {r['params_trainable']:>15,}")
    print(f"  Buffer elements:         {r['buffers']:>15,}")
    print(f"  Checkpoint file size:    {r['file_size_mb']:>13.2f} MB")
    print(f"  Theoretical FP32 size:   {r['fp32_size_mb']:>13.2f} MB")
    print(f"  Theoretical FP16 size:   {r['fp16_size_mb']:>13.2f} MB  (post-conversion)")
    print(f"  Theoretical INT8 size:   {r['int8_size_mb']:>13.2f} MB  (post-quantization)")

    _section(f"Computational Cost  (single forward pass, 1×3×{IMG_SIZE}×{IMG_SIZE})")
    print(f"  MACs:                    {r['macs']/1e9:>13.3f} G")
    print(f"  FLOPs (≈ 2 × MACs):      {r['flops']/1e9:>13.3f} G")

    _section("Inference Latency")
    for bs, lat in r["latencies"].items():
        print(f"  Batch size {bs}:")
        print(f"     Mean ± std:           {lat['mean_ms']:>7.2f} ± {lat['std_ms']:.2f} ms")
        print(f"     P50 / P95 / P99:      {lat['p50_ms']:>7.2f} / "
              f"{lat['p95_ms']:.2f} / {lat['p99_ms']:.2f} ms")
        print(f"     Throughput:           {lat['throughput_imgs']:>7.1f} images/s")

    _section("Test Accuracy")
    print(f"  Top-1 accuracy:          {r['top1']:>13.4f}")
    print(f"  Macro F1:                {r['f1_macro']:>13.4f}")
    print(f"  Weighted F1:             {r['f1_weighted']:>13.4f}")


def print_comparison(target, baseline):
    """target = the smaller / compressed model;  baseline = the reference."""
    _section("Compression  (target vs baseline)", width=72)
    print(f"  Target:    {target['arch']}  ({target['weights']})")
    print(f"  Baseline:  {baseline['arch']}  ({baseline['weights']})")

    pr = baseline["params_total"] / target["params_total"]
    mr = baseline["macs"]         / target["macs"]
    sr = baseline["file_size_mb"] / target["file_size_mb"]

    print(f"  Param compression:        {pr:>7.2f}×  "
          f"({baseline['params_total']:,} → {target['params_total']:,})")
    print(f"  MACs compression:         {mr:>7.2f}×  "
          f"({baseline['macs']/1e9:.2f} G → {target['macs']/1e9:.2f} G)")
    print(f"  Storage compression:      {sr:>7.2f}×  "
          f"({baseline['file_size_mb']:.2f} MB → {target['file_size_mb']:.2f} MB)")

    for bs in target["latencies"]:
        if bs in baseline["latencies"]:
            speedup = baseline["latencies"][bs]["mean_ms"] / target["latencies"][bs]["mean_ms"]
            print(f"  Speedup at B={bs}:           {speedup:>7.2f}×")

    acc_delta = (target["top1"]     - baseline["top1"])     * 100
    f1_delta  = (target["f1_macro"] - baseline["f1_macro"]) * 100
    print(f"  Δ Top-1 accuracy:         {acc_delta:>+7.2f} pp")
    print(f"  Δ Macro F1:               {f1_delta:>+7.2f} pp")

    if target["preds"].shape == baseline["preds"].shape:
        agree = float((target["preds"] == baseline["preds"]).mean()) * 100
        print(f"  Prediction agreement:     {agree:>7.2f}%   "
              f"(how often both models pick the same class)")

    if pr > 1 and acc_delta < 0:
        # Pareto-style summary line for compression studies.
        eff = -acc_delta / pr  # accuracy lost per × of compression
        print(f"  Accuracy lost per ×:      {eff:>+7.3f} pp/×    (lower is better)")


# ── Persisting ────────────────────────────────────────────────────────────────
def _strip_for_json(r):
    """Drop large arrays before JSON dumps."""
    out = {k: v for k, v in r.items() if k not in ("preds", "labels")}
    return out


def save_report(path, target, baseline=None):
    payload = {"target": _strip_for_json(target)}
    if baseline is not None:
        payload["baseline"] = _strip_for_json(baseline)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--weights", required=True,
                   help="Path to the model checkpoint (.pth) to evaluate.")
    p.add_argument("--arch", required=True, choices=list(ARCHS),
                   help="Architecture name (must match the checkpoint).")
    p.add_argument("--baseline-weights", default=None,
                   help="Optional reference checkpoint to compare against.")
    p.add_argument("--baseline-arch", default=None, choices=list(ARCHS),
                   help="Architecture for the baseline checkpoint.")
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32],
                   help="Batch sizes for latency benchmarking.")
    p.add_argument("--runs", type=int, default=50,
                   help="Timed runs per batch size (after warmup).")
    p.add_argument("--save-json", default=None,
                   help="Optional path to save a JSON report.")
    return p.parse_args()


def main():
    args = parse_args()
    if (args.baseline_weights is None) ^ (args.baseline_arch is None):
        raise SystemExit("--baseline-weights and --baseline-arch must be used together.")

    device, _, label = _select_device()
    print(f"Device: {label}")

    full_dataset = FilteredImageFolder(DATA_DIR)
    classes      = full_dataset.classes
    num_classes  = len(classes)
    print(f"Dataset: {len(full_dataset):,} images, {num_classes} classes")

    # Same test split as train.py / distill.py (deterministic via SEED).
    sample_labels = np.array([s[1] for s in full_dataset.samples])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
    _, test_idx = next(sss.split(np.zeros(len(sample_labels)), sample_labels))
    test_ds = TransformSubset(full_dataset, test_idx, eval_tf)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)
    print(f"Test set: {len(test_ds):,} images\n")

    target = evaluate(args.arch, args.weights, num_classes, classes, device,
                      test_loader, args.batch_sizes, args.runs)
    print_report(target)
    print("\n[Per-class report]\n" + target["per_class_report"])

    baseline = None
    if args.baseline_weights:
        baseline = evaluate(args.baseline_arch, args.baseline_weights,
                            num_classes, classes, device,
                            test_loader, args.batch_sizes, args.runs)
        print_report(baseline)
        print_comparison(target, baseline)

    if args.save_json:
        save_report(args.save_json, target, baseline)
        print(f"\nReport saved to {args.save_json}")


if __name__ == "__main__":
    main()
