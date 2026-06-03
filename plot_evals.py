"""
Plota figuras para o paper a partir de JSONs gerados por evaluate_model.

Uso:
    python plot_evals.py path1.json [path2.json ...] [--out DIR]

Gera 4 figuras + 1 tabela em CSV, todas em estilo acadêmico sóbrio:

    1. retention.png      — retenção de qualidade vs custos (em % do baseline)
    2. compression_vs_loss.png — limítrofe da destilação (compressão × gap top-1)
    3. pareto.png         — fronteira top-1 × custo, com gap aceitável sombreado
    4. per_class_delta.png — onde a destilação perde, por classe
    5. summary.csv        — tabela mestre para inclusão direta no paper
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


# ── estilo acadêmico ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":        True,
    "grid.linestyle":   ":",
    "grid.alpha":       0.4,
    "legend.frameon":   False,
    "figure.dpi":       140,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
})

# Paleta categórica neutra (ColorBrewer "Set2", segura para impressão B&W aproximada)
PALETTE = ["#4C72B0", "#DD8452", "#55A467", "#C44E52", "#8172B2",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]


# ── parsing ───────────────────────────────────────────────────────────────────
def load_eval(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


_ROW_RE = re.compile(
    r"^\s*(\S.*?)\s+([01]\.\d{4})\s+([01]\.\d{4})\s+([01]\.\d{4})\s+(\d+)\s*$"
)

def parse_per_class_report(report: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in report.splitlines():
        if not line.strip() or line.lstrip().startswith(("accuracy", "macro", "weighted")):
            continue
        m = _ROW_RE.match(line)
        if not m:
            continue
        name, prec, rec, f1, sup = m.groups()
        out[name.strip()] = {"precision": float(prec), "recall": float(rec),
                             "f1": float(f1), "support": int(sup)}
    return out


# ── figura 1: retenção / redução ──────────────────────────────────────────────
def fig_retention(targets: dict[str, dict], baseline: dict, out: Path):
    """Para cada aluno, % do baseline em cada eixo. Verde retém qualidade;
    barras de custo (vermelho) devem ser baixas."""
    names = list(targets.keys())
    metrics = [
        ("Top-1",      lambda t: t["top1"]                           / baseline["top1"],            "quality"),
        ("F1 macro",   lambda t: t["f1_macro"]                       / baseline["f1_macro"],        "quality"),
        ("Tamanho",    lambda t: t["fp32_size_mb"]                   / baseline["fp32_size_mb"],    "cost"),
        ("MACs",       lambda t: t["macs"]                           / baseline["macs"],            "cost"),
        ("Latência b=1",  lambda t: t["latencies"]["1"]["mean_ms"]   / baseline["latencies"]["1"]["mean_ms"],  "cost"),
        ("Latência b=32", lambda t: t["latencies"]["32"]["mean_ms"]  / baseline["latencies"]["32"]["mean_ms"], "cost"),
    ]

    fig, axes = plt.subplots(1, len(names), figsize=(2.4 * len(names) + 1, 3.8),
                             sharey=True)
    if len(names) == 1:
        axes = [axes]

    for ax, name in zip(axes, names):
        t = targets[name]
        vals  = [100 * f(t) for _, f, _ in metrics]
        kinds = [k for _, _, k in metrics]
        colors = ["#2A7A3E" if k == "quality" else "#B23A48" for k in kinds]
        ypos = np.arange(len(metrics))[::-1]
        ax.barh(ypos, vals, color=colors, edgecolor="black", linewidth=0.4)
        ax.axvline(100, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
        for y, v in zip(ypos, vals):
            ax.text(v + 2, y, f"{v:.1f}%", va="center", fontsize=8)
        ax.set_yticks(ypos)
        ax.set_yticklabels([m for m, _, _ in metrics])
        ax.set_xlim(0, max(110, max(vals) * 1.15))
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("% do baseline")

    fig.suptitle("Retenção de qualidade e redução de custo (relativo ao baseline)",
                 fontsize=11, y=1.02)
    fig.savefig(out / "retention.png")
    plt.close(fig)


# ── figura 2: limítrofe da destilação ─────────────────────────────────────────
def fig_compression_vs_loss(targets: dict[str, dict], baseline: dict, out: Path,
                            tolerance_pp: float = 1.0):
    """Eixo X = fator de compressão (log); Y = gap top-1 em pontos percentuais.
    Quanto mais à direita e mais baixo, melhor. Zona aceitável sombreada."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    xs, ys, labels = [], [], []
    for name, t in targets.items():
        compression = baseline["fp32_size_mb"] / t["fp32_size_mb"]
        gap_pp      = 100 * (baseline["top1"] - t["top1"])
        xs.append(compression); ys.append(gap_pp); labels.append(name)

    ax.axhspan(-1, tolerance_pp, color="#2A7A3E", alpha=0.10,
               label=f"perda ≤ {tolerance_pp:.1f} pp")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)

    for x, y, lab, c in zip(xs, ys, labels, PALETTE):
        ax.scatter(x, y, s=70, color=c, edgecolor="black", linewidth=0.5, zorder=3)
        ax.annotate(lab, (x, y), xytext=(6, 4), textcoords="offset points",
                    fontsize=8.5)

    ax.set_xscale("log")
    ax.set_xlabel("Fator de compressão (tamanho fp32 do baseline / aluno)")
    ax.set_ylabel("Gap de top-1 vs baseline (pontos percentuais)")
    ax.set_title("Limítrofe da destilação: até onde dá para comprimir")
    ax.invert_yaxis()  # gap positivo = perda; inverter coloca "melhor" em cima
    ax.legend(loc="best", fontsize=8)
    fig.savefig(out / "compression_vs_loss.png")
    plt.close(fig)


# ── figura 3: Pareto top-1 × custo, com gap tolerável ─────────────────────────
def _pareto_front(xs: list[float], ys: list[float]) -> list[int]:
    """Índices não-dominados, assumindo x menor = melhor e y maior = melhor."""
    order = sorted(range(len(xs)), key=lambda i: (xs[i], -ys[i]))
    front, best_y = [], -float("inf")
    for i in order:
        if ys[i] > best_y:
            front.append(i); best_y = ys[i]
    return front


def fig_pareto(targets: dict[str, dict], baseline: dict, out: Path,
               tolerance_pp: float = 1.0):
    cost_specs = [
        ("Tamanho fp32 (MB)",    lambda t: t["fp32_size_mb"],            False),
        ("MACs",                 lambda t: t["macs"],                    True),
        ("Latência batch=1 (ms)", lambda t: t["latencies"]["1"]["mean_ms"], False),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    names = list(targets.keys())

    for ax, (xlabel, getter, logx) in zip(axes, cost_specs):
        xs = [getter(targets[n]) for n in names]
        ys = [targets[n]["top1"] for n in names]
        bx, by = getter(baseline), baseline["top1"]

        # ylim ajustado aos dados (top-1 sempre ≤ 1, sem espaço morto)
        ymin = min(min(ys), by) - 0.01
        ymax = min(1.005, max(max(ys), by) + 0.005)
        ax.set_ylim(ymin, ymax)

        # zona de tolerância (limitada ao ylim real)
        ax.axhspan(by - tolerance_pp / 100, ymax, color="#2A7A3E", alpha=0.10,
                   label=f"gap ≤ {tolerance_pp:.1f} pp")

        # fronteira de Pareto entre os alunos (desenhada antes para ficar atrás)
        front = _pareto_front(xs, ys)
        if len(front) >= 2:
            fx = [xs[i] for i in front]; fy = [ys[i] for i in front]
            order = np.argsort(fx)
            ax.plot(np.array(fx)[order], np.array(fy)[order],
                    color="black", linewidth=0.8, alpha=0.5, zorder=2,
                    label="fronteira de Pareto")

        # pontos com offsets de label alternados para reduzir sobreposição
        for i, (x, y, lab, c) in enumerate(zip(xs, ys, names, PALETTE)):
            ax.scatter(x, y, s=55, color=c, edgecolor="black", linewidth=0.4, zorder=3)
            dy = 6 if i % 2 == 0 else -12
            ax.annotate(lab, (x, y), xytext=(5, dy), textcoords="offset points",
                        fontsize=8)

        # baseline (estrela)
        ax.scatter(bx, by, marker="*", s=180, c="black", zorder=4,
                   label="baseline")
        ax.annotate("baseline", (bx, by), xytext=(7, -12),
                    textcoords="offset points", fontsize=8, fontweight="bold")

        ax.set_xlabel(xlabel); ax.set_ylabel("Top-1")
        if logx: ax.set_xscale("log")
        ax.legend(loc="lower right", fontsize=7.5)

    fig.suptitle(f"Fronteira qualidade × custo (zona verde = gap ≤ {tolerance_pp:.1f} pp)",
                 fontsize=11, y=1.02)
    fig.savefig(out / "pareto.png")
    plt.close(fig)


# ── figura 4: F1 delta por classe ─────────────────────────────────────────────
def fig_per_class_delta(targets: dict[str, dict], baseline: dict, out: Path,
                        top_k: int = 10):
    """Mostra apenas as classes onde a destilação mais perde (pior delta médio)."""
    base_by_class = parse_per_class_report(baseline["per_class_report"])
    parsed_t = {n: parse_per_class_report(t["per_class_report"])
                for n, t in targets.items()}
    classes = list(base_by_class.keys())

    # delta = aluno − baseline (negativo = pior)
    deltas = {n: {c: parsed_t[n][c]["f1"] - base_by_class[c]["f1"] for c in classes}
              for n in parsed_t}

    # ordenar por pior média entre alunos (mais negativo primeiro)
    mean_delta = {c: np.mean([deltas[n][c] for n in deltas]) for c in classes}
    classes_sorted = sorted(classes, key=lambda c: mean_delta[c])[:top_k]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    n_models = len(targets); width = 0.8 / max(n_models, 1)
    y = np.arange(len(classes_sorted))[::-1]
    for i, (name, by) in enumerate(deltas.items()):
        offset = (i - (n_models - 1) / 2) * width
        vals = [by[c] for c in classes_sorted]
        ax.barh(y + offset, vals, width, color=PALETTE[i % len(PALETTE)],
                edgecolor="black", linewidth=0.3, label=name)

    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in classes_sorted], fontsize=8)
    ax.set_xlabel("Δ F1 (aluno − baseline)")
    ax.set_title(f"Onde a destilação perde: top {top_k} classes com maior queda média")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    fig.savefig(out / "per_class_delta.png")
    plt.close(fig)


# ── tabela mestre ─────────────────────────────────────────────────────────────
def write_summary_csv(targets: dict[str, dict], baseline: dict, out: Path):
    fields = ["model", "params_M", "macs_M", "fp32_MB",
              "latency_b1_ms", "throughput_b32",
              "top1", "f1_macro", "gap_top1_pp", "compression_x"]
    rows = []
    for name in [baseline.get("arch", "baseline")] + list(targets.keys()):
        t = baseline if name == baseline.get("arch", "baseline") else targets[name]
        rows.append({
            "model": name,
            "params_M":      round(t["params_total"] / 1e6, 3),
            "macs_M":        round(t["macs"]         / 1e6, 2),
            "fp32_MB":       round(t["fp32_size_mb"], 3),
            "latency_b1_ms": round(t["latencies"]["1"]["mean_ms"], 3),
            "throughput_b32": round(t["latencies"]["32"]["throughput_imgs"], 1),
            "top1":          round(t["top1"], 4),
            "f1_macro":      round(t["f1_macro"], 4),
            "gap_top1_pp":   round(100 * (baseline["top1"] - t["top1"]), 3),
            "compression_x": round(baseline["fp32_size_mb"] / t["fp32_size_mb"], 2),
        })
    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


# ── orquestração ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--out", type=Path, default=Path("output/plots"))
    p.add_argument("--tolerance-pp", type=float, default=1.0,
                   help="gap aceitável (pp) usado nas zonas verdes")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    targets: dict[str, dict] = {}
    baselines: list[dict] = []
    for path in args.paths:
        data = load_eval(path)
        name = data["target"]["arch"]
        if name in targets:
            name = f"{name} ({path.stem})"
        targets[name] = data["target"]
        baselines.append(data["baseline"])

    # baseline comum é pré-requisito para gráficos relativos
    same = all(b.get("arch") == baselines[0].get("arch") and
               b.get("weights") == baselines[0].get("weights") for b in baselines)
    if not same:
        raise SystemExit("Os JSONs não compartilham o mesmo baseline — "
                         "comparação relativa não é válida.")
    baseline = baselines[0]

    fig_retention            (targets, baseline, args.out)
    fig_compression_vs_loss  (targets, baseline, args.out, args.tolerance_pp)
    fig_pareto               (targets, baseline, args.out, args.tolerance_pp)
    fig_per_class_delta      (targets, baseline, args.out)
    write_summary_csv        (targets, baseline, args.out)

    print(f"[ok] {len(targets)} aluno(s) plotado(s) em {args.out.resolve()}")


if __name__ == "__main__":
    main()
