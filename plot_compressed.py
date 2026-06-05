"""
plot_compressed.py — Plota figuras para o paper a partir de JSONs gerados por
evaluate_compressed.py (pruning e quantização).

Uso:
    python plot_compressed.py results/ptq_static.json results/pruned_90.json \
        --baseline results/baseline.json --out output/plots_compressed

    # Se cada JSON já inclui o baseline dentro dele:
    python plot_compressed.py results/*.json --out output/plots_compressed

Gera 5 figuras + 1 tabela em CSV, em estilo acadêmico sóbrio:

    1. sparsity_vs_accuracy.png   — Pareto esparsidade × top-1 (pruning)
    2. size_vs_accuracy.png       — tamanho em disco × top-1 (todos os modelos)
    3. latency_comparison.png     — latência e throughput agrupados por método
    4. per_class_delta.png        — delta de F1 por classe vs baseline
    5. compression_radar.png      — radar multidimensional: qualidade vs custo
    6. summary.csv                — tabela mestre para inclusão no paper
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── estilo acadêmico ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.linestyle":    ":",
    "grid.alpha":        0.4,
    "legend.frameon":    False,
    "figure.dpi":        140,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
})

# Paleta qualitativa com cores bem distintas (daltonismo-friendly)
# Pruning: vermelho → laranja → amarelo-verde → verde-azulado → roxo → rosa → marrom
PALETTE_PRUNING  = ["#E41A1C", "#FF7F00", "#FFFF33", "#4DAF4A",
                    "#984EA3", "#F781BF", "#A65628", "#999999"]
# Quant: azuis e verdes distintos
PALETTE_QUANT    = ["#377EB8", "#17BECF", "#2CA02C", "#8C564B",
                    "#BCBD22", "#7F7F7F", "#AEC7E8", "#98DF8A"]
PALETTE_BASELINE = "#4C72B0"

# Marcadores distintos por categoria
MARKERS = {"baseline": "*", "pruning": "o", "quantization": "s", "unknown": "^"}


# ── parsing & classificação ───────────────────────────────────────────────────
_ROW_RE = re.compile(
    r"^\s*(\S.*?)\s+([01]\.\d{4})\s+([01]\.\d{4})\s+([01]\.\d{4})\s+(\d+)\s*$"
)


def classify_model(name: str) -> str:
    n = name.lower()
    if "baseline" in n or "best_model" in n:
        return "baseline"
    if any(k in n for k in ("prun", "sparse", "unstructured", "structured", "magn", "l1")):
        return "pruning"
    if any(k in n for k in ("quant", "ptq", "qat", "int8", "dynamic", "static")):
        return "quantization"
    return "unknown"


def load_eval(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_per_class_report(report: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for line in report.splitlines():
        if not line.strip() or line.lstrip().startswith(("accuracy", "macro", "weighted")):
            continue
        m = _ROW_RE.match(line)
        if not m:
            continue
        name, prec, rec, f1, sup = m.groups()
        out[name.strip()] = {
            "precision": float(prec), "recall": float(rec),
            "f1": float(f1), "support": int(sup),
        }
    return out


def sparsity_pct(t: dict) -> float:
    total = t.get("params_total", 0)
    nz    = t.get("nonzero_params", total)
    if total == 0:
        return 0.0
    return 100.0 * (1.0 - nz / total)


def _color_for(name: str, cat: str,
               pruning_idx: dict[str, int],
               quant_idx:   dict[str, int]) -> str:
    if cat == "baseline":
        return PALETTE_BASELINE
    if cat == "pruning":
        return PALETTE_PRUNING[pruning_idx.get(name, 0) % len(PALETTE_PRUNING)]
    if cat == "quantization":
        return PALETTE_QUANT[quant_idx.get(name, 0) % len(PALETTE_QUANT)]
    return "#8C8C8C"


def _build_idx(models: dict[str, dict]) -> tuple[dict, dict]:
    pi, qi = 0, 0
    pruning_i, quant_i = {}, {}
    for n in models:
        cat = classify_model(n)
        if cat == "pruning":
            pruning_i[n] = pi; pi += 1
        elif cat == "quantization":
            quant_i[n] = qi; qi += 1
    return pruning_i, quant_i


def _tight_ylim(ys: list[float], pad_frac: float = 0.25) -> tuple[float, float]:
    """
    Calcula ylim centrado nos dados, com padding proporcional.
    Trava o limite máximo (hi) em 1.0 para que a acurácia não ultrapasse 100%.
    """
    lo, hi = min(ys), max(ys)
    span   = max(hi - lo, 0.01)
    pad    = span * pad_frac
    
    # max(..., 0.0) impede de ficar negativo
    # min(..., 1.0) garante que nunca apareça 1.1 no eixo Y
    return max(0.0, lo - pad), min(1.0, hi + pad)


# ── figura 1: esparsidade × top-1 (pruning) ──────────────────────────────────
def _repel_labels(ax, points: list[tuple[float, float, str]],
                  data_to_fig_x, data_to_fig_y,
                  fontsize: float = 8.5,
                  pad_pts: float = 6.0,
                  max_iter: int = 60,
                  direction: str = "up") -> list[plt.Annotation]:
    """
    Posiciona anotações evitando colisões entre si e com os pontos de dados.
    Usa um algoritmo simples de força repulsiva em coordenadas de figura (pontos).

    Parameters
    ----------
    ax              : eixo matplotlib
    points          : lista de (x_data, y_data, label)
    data_to_fig_x/y : funções que convertem coordenada de dados → pontos de figura
    fontsize        : tamanho da fonte
    pad_pts         : distância mínima inicial do ponto ao rótulo (em pontos)
    max_iter        : iterações do loop de repulsão
    """
    fig = ax.get_figure()
    renderer = fig.canvas.get_renderer()

    dy_init = +(pad_pts + 4) if direction == "up" else -(pad_pts + 4)
    va_init = "bottom" if direction == "up" else "top"

    annotations = []
    for (xd, yd, label) in points:
        ann = ax.annotate(
            label, (xd, yd),
            xytext=(0, dy_init),
            textcoords="offset points",
            fontsize=fontsize,
            ha="center", va=va_init,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#CCCCCC",
                      alpha=0.85, lw=0.6),
            arrowprops=dict(arrowstyle="-", color="#888888",
                            lw=0.7, shrinkA=0, shrinkB=3),
            zorder=6,
        )
        annotations.append(ann)

    # ── loop de repulsão ────────────────────────────────────────────────────
    # Converte coordenadas de dado para "pontos de display" para poder calcular
    # distâncias em unidades homogêneas independentes de escala.
    def _ann_center(ann):
        bb = ann.get_window_extent(renderer=renderer)
        return np.array([(bb.x0 + bb.x1) / 2, (bb.y0 + bb.y1) / 2])

    def _point_display(xd, yd):
        return np.array(ax.transData.transform((xd, yd)))

    step = 2.0  # deslocamento por iteração (pontos display)
    min_dist = 22.0  # distância mínima entre centros de caixas (pontos display)

    for _ in range(max_iter):
        moved = False
        centers = [_ann_center(a) for a in annotations]
        anchors = [_point_display(xd, yd) for (xd, yd, _) in points]

        for i, ann_i in enumerate(annotations):
            dx_total, dy_total = 0.0, 0.0

            # Repulsão de outros rótulos
            for j, ann_j in enumerate(annotations):
                if i == j:
                    continue
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff) + 1e-9
                if dist < min_dist:
                    push = (min_dist - dist) / dist
                    dx_total += diff[0] * push
                    dy_total += diff[1] * push
                    moved = True

            # Repulsão do ponto de dado (âncora)
            diff_a = centers[i] - anchors[i]
            dist_a = np.linalg.norm(diff_a) + 1e-9
            if dist_a < pad_pts * 1.5:
                push = (pad_pts * 1.5 - dist_a) / dist_a
                dx_total += diff_a[0] * push * 0.5
                dy_total += diff_a[1] * push * 0.5
                moved = True

            if dx_total or dy_total:
                # Obtém offset atual em pontos (xytext em "offset points")
                ox, oy = ann_i.xyann
                norm = np.hypot(dx_total, dy_total) + 1e-9
                ann_i.xyann = (ox + step * dx_total / norm,
                               oy + step * dy_total / norm)

        if not moved:
            break

    return annotations


def fig_sparsity_vs_accuracy(models: dict[str, dict], baseline: dict,
                             out: Path, tolerance_pp: float = 1.0):
    """
    Scatter tamanho em disco (MB) × Acurácia Top-1 — apenas modelos de pruning.

    Eixo X invertido: baseline fica à esquerda (maior tamanho) e os modelos
    comprimidos avançam para a direita (menor tamanho), tornando o trade-off
    imediatamente legível: quanto mais à direita, mais comprimido.

    Bolhas proporcionais à esparsidade (% de parâmetros zerados).
    Fronteira de Pareto conecta os pontos ótimos (maior acurácia dado tamanho).
    Rótulos posicionados com algoritmo de repulsão para evitar sobreposição.
    """
    pruning_models = {n: t for n, t in models.items()
                      if classify_model(n) == "pruning"}
    if not pruning_models:
        print("[aviso] Nenhum modelo de pruning encontrado — fig_sparsity_vs_accuracy ignorada.")
        return

    fig, ax = plt.subplots(figsize=(10.0, 6.5))
    fig.canvas.draw()

    b_top1   = baseline["top1"]
    b_size   = baseline.get("file_size_mb", 0)

    xs    = [t.get("file_size_mb", 0) for t in pruning_models.values()]
    ys    = [t["top1"]                for t in pruning_models.values()]
    names = list(pruning_models.keys())

    # ── ylim com padding para acomodar rótulos ───────────────────────────────
    ylo, yhi = _tight_ylim(ys + [b_top1], pad_frac=0.35)
    ax.set_ylim(ylo, yhi)

    # ── eixo X invertido: baseline (maior) à esquerda ────────────────────────
    all_sizes = xs + [b_size]
    x_min, x_max = min(all_sizes), max(all_sizes)
    x_pad = max((x_max - x_min) * 0.15, 1.0)
    ax.set_xlim(x_max + x_pad, max(0, x_min - x_pad))   # invertido

    # ── zona de tolerância de acurácia ───────────────────────────────────────
    tol_lo = max(b_top1 - tolerance_pp / 100, ylo)
    ax.axhspan(tol_lo, yhi, color="#2CA02C", alpha=0.08, zorder=0,
               label=f"Gap aceitável (≤ {tolerance_pp:.1f} pp)")

    # ── linha do baseline ────────────────────────────────────────────────────
    ax.axhline(b_top1, color=PALETTE_BASELINE, lw=1.5,
               linestyle="--", alpha=0.9, zorder=1, label="Baseline Top-1")

    # ── fronteira de Pareto ───────────────────────────────────────────────────
    # No eixo invertido, "menor tamanho" = maior compressão = mais à direita.
    # A fronteira conecta, para cada tamanho, o modelo de maior acurácia.
    order   = np.argsort(xs)[::-1]   # do maior para o menor tamanho
    front_x, front_y = [], []
    best_y  = -np.inf
    for i in order:
        if ys[i] >= best_y:
            front_x.append(xs[i]); front_y.append(ys[i]); best_y = ys[i]
    if len(front_x) >= 2:
        ax.plot(front_x, front_y, color="#333333", lw=1.4,
                linestyle=":", alpha=0.7, label="Fronteira de Pareto", zorder=2)

    for i, (n, t) in enumerate(pruning_models.items()):
        x, y = t.get("file_size_mb", 0), t["top1"]
        c = PALETTE_PRUNING[i % len(PALETTE_PRUNING)]
        ax.scatter(x, y, s=90, color=c, edgecolor="white", lw=1.2,
                   marker=MARKERS["pruning"], zorder=4, alpha=0.9, label=n)

    # ── baseline (estrela) ───────────────────────────────────────────────────
    ax.scatter(b_size, b_top1, s=300, c=PALETTE_BASELINE,
               edgecolor="white", lw=1.5, marker="*", zorder=5, label="Baseline")

    # ── seta indicando direção de compressão ─────────────────────────────────
    ax.annotate(
        "",
        xy=(0.97, 0.04), xycoords="axes fraction",
        fontsize=8, color="#555555", style="italic", ha="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=3),
    )

    ax.set_xlabel("Tamanho em disco (MB)",
                  fontweight="bold")
    ax.set_ylabel("Acurácia Top-1", fontweight="bold")
    ax.set_title("Compromisso: Tamanho vs Acurácia (pruning)", pad=15)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, frameon=True, framealpha=0.95, edgecolor="#CCCCCC")
    fig.tight_layout()
    fig.savefig(out / "sparsity_vs_accuracy.png")
    plt.close(fig)

# ── figura 2: tamanho em disco × top-1 (todos) ───────────────────────────────
def fig_size_vs_accuracy(models: dict[str, dict], baseline: dict,
                         out: Path, tolerance_pp: float = 1.0):
    """
    Scatter tamanho × top-1. Eixo Y ajustado com padding para maximizar
    separação entre pontos próximos.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    b_top1 = baseline["top1"]
    b_size = baseline["file_size_mb"]

    all_ys = [t["top1"] for t in models.values()] + [b_top1]
    ylo, yhi = _tight_ylim(all_ys)
    ax.set_ylim(ylo, yhi)

    tol_lo = max(b_top1 - tolerance_pp / 100, ylo)
    ax.axhspan(tol_lo, yhi, color="#2A7A3E", alpha=0.09,
               label=f"gap ≤ {tolerance_pp:.1f} pp")
    ax.axhline(b_top1, color=PALETTE_BASELINE, lw=1.2,
               linestyle="--", alpha=0.8, label="baseline top-1")

    pruning_i, quant_i = _build_idx(models)

    legend_handles: list = []
    for i, (name, t) in enumerate(models.items()):
        cat = classify_model(name)
        c   = _color_for(name, cat, pruning_i, quant_i)
        mk  = MARKERS.get(cat, "^")
        ax.scatter(t["file_size_mb"], t["top1"],
                   s=90, color=c, edgecolor="white", lw=0.8,
                   marker=mk, zorder=3, alpha=0.92)
        legend_handles.append(mpatches.Patch(color=c, label=f"{name}  [{cat}]"))

    ax.scatter(b_size, b_top1, s=220, c=PALETTE_BASELINE,
               marker="*", zorder=4, edgecolor="white", lw=1.0, label="baseline")

    legend_handles.append(mpatches.Patch(color=PALETTE_BASELINE, label="baseline  [baseline]"))
    ax.legend(handles=legend_handles, fontsize=8.5,
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              borderaxespad=0, frameon=True, framealpha=0.95, edgecolor="#CCCCCC")
    ax.set_xlabel("Tamanho em disco (MB)")
    ax.set_ylabel("Top-1")
    ax.set_title("Tamanho × acurácia — pruning e quantização")
    fig.tight_layout()
    fig.savefig(out / "size_vs_accuracy.png")
    plt.close(fig)


# ── figura 3: latência e throughput ──────────────────────────────────────────
def fig_latency_comparison(models: dict[str, dict], baseline: dict,
                           out: Path, batch_sizes: list[str] | None = None):
    """
    Gráfico de barras horizontais por batch size.
    Labels fora das barras, sem sobreposição.
    Speedup vs baseline anotado em cada barra.
    """
    all_models = {"baseline": baseline, **models}
    if batch_sizes is None:
        batch_sizes = sorted(
            set(k for t in all_models.values()
                for k in t.get("latencies", {}).keys()),
            key=lambda x: int(x),
        )
    if not batch_sizes:
        print("[aviso] Nenhuma latência encontrada — fig_latency_comparison ignorada.")
        return

    names    = list(all_models.keys())
    n_models = len(names)
    n_bs     = len(batch_sizes)

    pruning_i, quant_i = _build_idx(models)

    fig, axes = plt.subplots(1, n_bs, figsize=(6.5 * n_bs, 0.9 * n_models + 2.5),
                             sharey=False)
    if n_bs == 1:
        axes = [axes]

    bar_h = 0.55  # altura de cada barra

    for ax, bs in zip(axes, batch_sizes):
        b_lat = baseline.get("latencies", {}).get(bs, {}).get("mean_ms", None)
        ypos  = np.arange(n_models)
        max_lat = max(
            (all_models[n].get("latencies", {}).get(bs, {}).get("mean_ms", 0)
             for n in names), default=1
        )

        for i, name in enumerate(names):
            t   = all_models[name]
            lat = t.get("latencies", {}).get(bs, {}).get("mean_ms", None)
            cat = classify_model(name)
            c   = _color_for(name, cat, pruning_i, quant_i) \
                  if name != "baseline" else PALETTE_BASELINE

            if lat is not None:
                ax.barh(ypos[i], lat, height=bar_h,
                        color=c, edgecolor="black", lw=0.5, alpha=0.88)

                # Label à direita da barra (fora)
                label_x = lat + max_lat * 0.015
                label_parts = [f"{lat:.2f} ms"]
                if b_lat and name != "baseline":
                    speedup = b_lat / lat
                    label_parts.append(f"({speedup:.2f}×)")
                ax.text(label_x, ypos[i], "  ".join(label_parts),
                        va="center", fontsize=8.5, color="black")

        if b_lat is not None:
            ax.axvline(b_lat, color=PALETTE_BASELINE, lw=1.2,
                       linestyle="--", alpha=0.75, label="baseline")

        ax.set_yticks(ypos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlim(0, max_lat * 1.35)   # margem à direita para labels
        ax.set_xlabel("Latência média (ms)")
        ax.set_title(f"Batch size = {bs}", pad=10)
        ax.legend(fontsize=9, loc="lower right")

    fig.suptitle("Comparação de latência",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "latency_comparison.png")
    plt.close(fig)


# ── figura 4: delta F1 por classe — heatmap ──────────────────────────────────
def _select_representative_models(
    parsed: dict[str, dict],
    models_meta: dict[str, dict],
    top_k_models: int,
) -> dict[str, dict]:
    """
    Seleciona até `top_k_models` modelos representativos por categoria.

    Estratégia:
      - Sempre inclui o melhor modelo de cada categoria (maior F1 macro).
      - Preenche as vagas restantes com os que têm maior variância de delta
        entre as classes (= mais "interessantes" para análise por classe).
      - Se top_k_models >= len(parsed), retorna todos sem filtrar.
    """
    if top_k_models <= 0 or len(parsed) <= top_k_models:
        return parsed

    # melhor de cada categoria
    by_cat: dict[str, list[str]] = {}
    for n in parsed:
        cat = classify_model(n)
        by_cat.setdefault(cat, []).append(n)

    selected: list[str] = []
    for cat, names in by_cat.items():
        best = max(names, key=lambda n: models_meta[n].get("f1_macro", 0.0))
        if best not in selected:
            selected.append(best)

    # se ainda há vagas, preenche por variância de delta
    if len(selected) < top_k_models:
        remaining = [n for n in parsed if n not in selected]

        def _variance(name: str) -> float:
            vals = list(parsed[name].values())
            return float(np.var(vals)) if vals else 0.0

        remaining.sort(key=_variance, reverse=True)
        selected += remaining[: top_k_models - len(selected)]

    return {n: parsed[n] for n in selected if n in parsed}


# ── figura 4: delta F1 por classe — gráfico de barras (piores perdas) ────────
def fig_per_class_delta(targets: dict[str, dict], baseline: dict, out: Path,
                        top_k: int = 10, **kwargs):
    """Mostra apenas as classes onde a destilação/compressão mais perde (pior delta médio)."""
    
    if "per_class_report" not in baseline or not baseline["per_class_report"]:
        print("[aviso] per_class_report ausente no baseline — fig_per_class_delta ignorada.")
        return

    base_by_class = parse_per_class_report(baseline["per_class_report"])
    
    parsed_t = {}
    for n, t in targets.items():
        rep = t.get("per_class_report", "")
        if rep:
            parsed_t[n] = parse_per_class_report(rep)

    if not parsed_t:
        print("[aviso] Nenhum per_class_report nos modelos — fig_per_class_delta ignorada.")
        return

    # Filtra apenas as classes que existem no baseline e em todos os modelos
    classes = [c for c in base_by_class if all(c in p for p in parsed_t.values())]

    # delta = aluno − baseline (negativo = pior)
    deltas = {n: {c: parsed_t[n][c]["f1"] - base_by_class[c]["f1"] for c in classes}
              for n in parsed_t}

    # ordenar por pior média entre alunos (mais negativo primeiro)
    mean_delta = {c: np.mean([deltas[n][c] for n in deltas]) for c in classes}
    classes_sorted = sorted(classes, key=lambda c: mean_delta[c])[:top_k]

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    n_models = len(targets)
    width = 0.8 / max(n_models, 1)
    y = np.arange(len(classes_sorted))[::-1]
    
    # Combina as paletas para evitar o NameError e garantir cores suficientes
    PALETTE = PALETTE_PRUNING + PALETTE_QUANT 

    for i, (name, by) in enumerate(deltas.items()):
        offset = (i - (n_models - 1) / 2) * width
        vals = [by[c] for c in classes_sorted]
        ax.barh(y + offset, vals, width, color=PALETTE[i % len(PALETTE)],
                edgecolor="black", linewidth=0.3, label=name)

    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels([c.replace("_", " ") for c in classes_sorted], fontsize=8)
    ax.set_xlabel("Δ F1 (modelo − baseline)")
    ax.set_title(f"Onde a compressão perde: top {top_k} classes com maior queda média")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5),
              frameon=False)
    
    fig.tight_layout()
    fig.savefig(out / "per_class_delta.png")
    plt.close(fig)

# ── figura 5: fronteira qualidade × custo (Pareto) ───────────────────────────
def fig_quality_vs_cost_pareto(models: dict[str, dict], baseline: dict,
                               out: Path, tolerance_pp: float = 1.0):
    """
    Gera 2 painéis (Tamanho em disco, Latência) vs Top-1 com fronteira de Pareto e
    zona de gap aceitável. Ajustado para exibir variação minúscula de acurácia.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.0), sharey=False)
    
    metrics = [
        ("file_size_mb", "Tamanho em disco (MB)", False),
        ("latency", "Latência batch=1 (ms)", False)
    ]
    
    b_top1 = baseline.get("top1", 0.0)
    
    all_ys = [t.get("top1", 0.0) for t in models.values() if "top1" in t] + [b_top1]
    
    # [NOVO] Lógica para evitar "fundo todo verde" quando o modelo for perfeito demais
    y_min_data = min(all_ys)
    tol_target = b_top1 - tolerance_pp / 100.0
    # Garante que o fundo do eixo Y desça um pouco abaixo do pior dado para vermos o limite verde,
    # mas não afunda demais se a tolerance for gigante (para não espremer os dados no topo).
    visible_tol = max(tol_target, y_min_data - 0.0015) 
    
    ylo, yhi = _tight_ylim(all_ys + [visible_tol], pad_frac=0.20)
    tol_lo = max(tol_target, ylo)
    
    pruning_i, quant_i = _build_idx(models)
    
    for ax, (metric_key, xlabel, is_log) in zip(axes, metrics):
        xs, ys, names, colors = [], [], [], []
        
        b_x = baseline.get("file_size_mb", 0) if metric_key == "file_size_mb" else baseline.get("latencies", {}).get("1", {}).get("mean_ms", None)
        
        if b_x is not None and b_x > 0:
            xs.append(b_x)
            ys.append(b_top1)
            names.append("baseline")
            colors.append("black")
            
        for name, t in models.items():
            x_val = t.get("file_size_mb", 0) if metric_key == "file_size_mb" else t.get("latencies", {}).get("1", {}).get("mean_ms", None)
            if x_val is not None and x_val > 0:
                xs.append(x_val)
                ys.append(t.get("top1", 0))
                names.append(name)
                cat = classify_model(name)
                colors.append(_color_for(name, cat, pruning_i, quant_i))
                
        if not xs:
            continue
            
        ax.axhspan(tol_lo, yhi, color="#2CA02C", alpha=0.08, 
                   label=f"gap ≤ {tolerance_pp:.1f} pp")
                   
        pts = sorted(zip(xs, ys, names), key=lambda p: (p[0], -p[1]))
        front_x, front_y = [], []
        best_y = -np.inf
        
        for x, y, n in pts:
            if y > best_y:
                front_x.append(x)
                front_y.append(y)
                best_y = y
                
        if len(front_x) > 1:
            ax.plot(front_x, front_y, color="#777777", lw=1.2, zorder=2, 
                    label="fronteira de Pareto")

        for i, (x, y, name, c) in enumerate(zip(xs, ys, names, colors)):
            if name == "baseline":
                ax.scatter(x, y, s=300, c="black", marker="*", zorder=5, label="baseline")
                ax.annotate("baseline", (x, y), xytext=(8, -8), textcoords="offset points", 
                            fontsize=9, fontweight="bold", zorder=6)
            else:
                ax.scatter(x, y, s=70, c=c, edgecolor="#444444", lw=0.8, marker="o", zorder=4, alpha=0.9)
                dy = 2 if i % 2 == 0 else -10
                ax.annotate(name, (x, y), xytext=(6, dy), textcoords="offset points", 
                            fontsize=8, zorder=6)
                
        ax.set_ylim(ylo, yhi)
        if is_log:
            ax.set_xscale("log")
            
        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel("Top-1", fontsize=11)
        
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="lower right", fontsize=9, 
                  frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

    fig.suptitle(f"Fronteira qualidade × custo (zona verde = gap ≤ {tolerance_pp:.1f} pp)", 
                 fontsize=14, y=1.05)
    fig.tight_layout()
    fig.savefig(out / "pareto_frontiers.png", bbox_inches="tight")
    plt.close(fig)
    
# ── tabela mestre ─────────────────────────────────────────────────────────────
def _per_class_stats(t: dict) -> dict[str, Any]:
    """
    Extrai do per_class_report:
      - f1_min         : pior F1 entre todas as classes
      - f1_worst_class : nome da classe com pior F1
      - f1_gap_macro   : diferença entre melhor e pior F1 de classe (spread)
    Retorna defaults vazios se o relatório estiver ausente.
    """
    report = t.get("per_class_report", "")
    parsed = parse_per_class_report(report)
    if not parsed:
        return {"f1_min": "", "f1_worst_class": "", "f1_gap_macro": ""}

    worst_cls = min(parsed, key=lambda c: parsed[c]["f1"])
    best_cls  = max(parsed, key=lambda c: parsed[c]["f1"])
    return {
        "f1_min":        round(parsed[worst_cls]["f1"], 4),
        "f1_worst_class": worst_cls,
        "f1_gap_macro":  round(parsed[best_cls]["f1"] - parsed[worst_cls]["f1"], 4),
    }


def write_summary_csv(models: dict[str, dict], baseline: dict, out: Path):
    """
    Colunas adicionadas/corrigidas em relação à versão anterior
    ────────────────────────────────────────────────────────────
    CORREÇÕES
      compression_size_x   → agora usa file_size_mb (tamanho real em disco),
                             não fp32_size_mb (que é idêntico para todos).
      compression_params_x → razão de parâmetros não-zero vs baseline;
                             substitui o antigo compression_macs_x que era
                             sempre 1.0 em pruning não estruturado.

    NOVAS COLUNAS
      latency_b1_p95_ms    → percentil 95 de latência batch=1 (cauda).
      latency_b1_std_ms    → desvio-padrão da latência batch=1 (estabilidade).
      throughput_b1        → imagens/s com batch=1.
      f1_macro_gap_pp      → gap de F1 macro vs baseline (pp), equivalente ao
                             gap_top1_pp mas para F1 (relevante em datasets
                             desbalanceados).
      f1_min               → pior F1 entre todas as classes (detecta colapso
                             em classes raras).
      f1_worst_class       → nome da classe com pior F1.
      f1_gap_macro         → spread entre melhor e pior F1 de classe (quanto
                             o modelo é homogêneo entre classes).
      param_efficiency     → top-1 / (nonzero_params / 1e6): acurácia por
                             milhão de parâmetros ativos.
      accuracy_cost_ratio  → gap_top1_pp / compression_size_x: custo de
                             acurácia por unidade de compressão; menor = melhor.
    """
    fields = [
        # ── identificação ──────────────────────────────────────────────────
        "model", "category",
        # ── parâmetros e esparsidade ───────────────────────────────────────
        "params_total", "nonzero_params", "sparsity_pct",
        # ── custo computacional ────────────────────────────────────────────
        "macs_M",
        # ── tamanho em disco ───────────────────────────────────────────────
        "fp32_MB", "fp16_MB", "int8_MB", "file_MB",
        # ── latência (batch=1) ─────────────────────────────────────────────
        "latency_b1_ms", "latency_b1_p95_ms", "latency_b1_std_ms",
        # ── throughput ─────────────────────────────────────────────────────
        "throughput_b1", "latency_b32_ms",
        # ── qualidade global ───────────────────────────────────────────────
        "top1", "f1_macro",
        # ── gaps vs baseline ───────────────────────────────────────────────
        "gap_top1_pp", "f1_macro_gap_pp",
        # ── qualidade por classe ───────────────────────────────────────────
        "f1_min", "f1_worst_class", "f1_gap_macro",
        # ── compressão ─────────────────────────────────────────────────────
        "compression_size_x", "compression_params_x", "speedup_b1",
        # ── métricas derivadas ─────────────────────────────────────────────
        "param_efficiency", "accuracy_cost_ratio",
    ]

    rows = []
    all_models = {baseline.get("arch", "baseline"): baseline, **models}
    b = baseline

    b_top1       = b.get("top1", 0.0)
    b_f1_macro   = b.get("f1_macro", 0.0)
    b_file_mb    = b.get("file_size_mb", 1.0)
    b_nz_params  = b.get("nonzero_params", b.get("params_total", 1))
    b_lat_b1     = b.get("latencies", {}).get("1", {}).get("mean_ms", 1.0)

    for name, t in all_models.items():
        cat   = classify_model(name)
        total = t.get("params_total", 0)
        nz    = t.get("nonzero_params", total)

        lat_b1_mean = t.get("latencies", {}).get("1", {}).get("mean_ms", 0.0)
        lat_b1_p95  = t.get("latencies", {}).get("1", {}).get("p95_ms", 0.0)
        lat_b1_std  = t.get("latencies", {}).get("1", {}).get("std_ms", 0.0)
        thr_b1      = t.get("latencies", {}).get("1", {}).get("throughput_imgs", 0.0)
        lat_b32_mean = t.get("latencies", {}).get("32", {}).get("mean_ms", 0.0)

        t_top1     = t.get("top1", 0.0)
        t_f1_macro = t.get("f1_macro", 0.0)
        t_file_mb  = t.get("file_size_mb", 1.0)

        gap_top1_pp    = round(100 * (b_top1 - t_top1), 3)
        f1_macro_gap   = round(100 * (b_f1_macro - t_f1_macro), 3)

        # compression_size_x: razão de tamanho real em disco
        comp_size = round(b_file_mb / t_file_mb, 2) if t_file_mb else ""

        # compression_params_x: razão de parâmetros não-zero
        #   Em pruning não estruturado os MACs não mudam, mas os parâmetros
        #   zerados são comprimidos via sparse storage → esta métrica reflete
        #   a compressão efetiva. Para quantização, nz ≈ total, então ≈ 1×.
        comp_params = round(b_nz_params / nz, 2) if nz else ""

        speedup_b1 = round(b_lat_b1 / lat_b1_mean, 2) if lat_b1_mean else ""

        # param_efficiency: top-1 por milhão de parâmetros ativos
        param_eff = round(t_top1 / (nz / 1e6), 4) if nz else ""

        # accuracy_cost_ratio: custo de acurácia por unidade de compressão
        #   gap_top1_pp / compression_size_x → quanto de top-1 (pp) se perde
        #   por cada fator de compressão obtido. Menor = mais eficiente.
        #   Para o baseline este valor é 0 por definição.
        if isinstance(comp_size, float) and comp_size > 1.0:
            acc_cost = round(gap_top1_pp / comp_size, 4)
        else:
            acc_cost = 0.0

        cls_stats = _per_class_stats(t)

        rows.append({
            "model":              name,
            "category":           cat,
            "params_total":       total,
            "nonzero_params":     nz,
            "sparsity_pct":       round(100 * (1 - nz / total) if total else 0, 2),
            "macs_M":             round(t.get("macs", 0) / 1e6, 2),
            "fp32_MB":            round(t.get("fp32_size_mb", 0), 3),
            "fp16_MB":            round(t.get("fp16_size_mb", 0), 3),
            "int8_MB":            round(t.get("int8_size_mb", 0), 3),
            "file_MB":            round(t_file_mb, 3),
            "latency_b1_ms":      round(lat_b1_mean, 3),
            "latency_b1_p95_ms":  round(lat_b1_p95, 3),
            "latency_b1_std_ms":  round(lat_b1_std, 3),
            "throughput_b1":      round(thr_b1, 1),
            "latency_b32_ms":     round(lat_b32_mean, 3),
            "top1":               round(t_top1, 4),
            "f1_macro":           round(t_f1_macro, 4),
            "gap_top1_pp":        gap_top1_pp,
            "f1_macro_gap_pp":    f1_macro_gap,
            "f1_min":             cls_stats["f1_min"],
            "f1_worst_class":     cls_stats["f1_worst_class"],
            "f1_gap_macro":       cls_stats["f1_gap_macro"],
            "compression_size_x": comp_size,
            "compression_params_x": comp_params,
            "speedup_b1":         speedup_b1,
            "param_efficiency":   param_eff,
            "accuracy_cost_ratio": acc_cost,
        })

    with open(out / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def fig_retention_and_cost(models: dict[str, dict], baseline: dict, out: Path):
    """
    Gera subplots de barras horizontais (Trellis) comparando métricas.
    Ajustado para mostrar casas decimais dinâmicas, revelando detalhes > 99.9%.
    """
    if not models:
        return

    metrics = [
        ("Top-1", "top1", "quality"),
        ("F1 macro", "f1_macro", "quality"),
        ("Tamanho", "file_size_mb", "cost"),
        ("MACs", "macs", "cost"),
        ("Latência b=1", "lat_1", "cost"),
        ("Latência b=32", "lat_32", "cost"),
    ]

    def get_val(data_dict, key):
        if key == "lat_1":
            return data_dict.get("latencies", {}).get("1", {}).get("mean_ms", None)
        if key == "lat_32":
            return data_dict.get("latencies", {}).get("32", {}).get("mean_ms", None)
        return data_dict.get(key, None)

    b_vals = {key: get_val(baseline, key) for _, key, _ in metrics}

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(3.5 * n_models, 4.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    color_qual = "#2E7D32" 
    color_cost = "#B72A3B"

    y_pos = np.arange(len(metrics))

    for ax, (name, t) in zip(axes, models.items()):
        m_vals = [get_val(t, key) for _, key, _ in metrics]

        pcts = []
        for (m_label, m_key, m_type), m_val in zip(metrics, m_vals):
            b_val = b_vals[m_key]
            if b_val and m_val is not None:
                pcts.append((m_val / b_val) * 100.0)
            else:
                pcts.append(0.0)

        colors = [color_qual if mtype == "quality" else color_cost for _, _, mtype in metrics]
        bars = ax.barh(y_pos, pcts, color=colors, edgecolor="black", linewidth=0.7, alpha=0.9)

        ax.set_title(name, fontsize=11, pad=10)
        ax.set_xlabel("% do baseline", fontsize=10)
        ax.axvline(100, color="black", linestyle="--", linewidth=0.9, alpha=0.7, zorder=0)

        max_pct = max(pcts) if pcts else 100
        x_limit = max(max_pct * 1.35, 120)
        ax.set_xlim(0, x_limit)

        # [NOVO] Aplica 3 casas decimais para métricas de qualidade e 1 para custo
        for bar, (_, _, mtype) in zip(bars, metrics):
            width = bar.get_width()
            if width > 0:
                fmt_str = f"{width:.3f}%" if mtype == "quality" else f"{width:.1f}%"
                ax.text(width + (x_limit * 0.02), bar.get_y() + bar.get_height()/2, 
                        fmt_str, 
                        va='center', ha='left', fontsize=9)

        ax.grid(axis='x', linestyle=':', alpha=0.4, zorder=0)
        ax.grid(axis='y', linestyle=':', alpha=0.4, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if ax == axes[0]:
            ax.set_yticks(y_pos)
            ax.set_yticklabels([label for label, _, _ in metrics], fontsize=11)
        else:
            ax.tick_params(axis='y', length=0)

    plt.gca().invert_yaxis()
    fig.suptitle("Retenção de qualidade e redução de custo (relativo ao baseline)", fontsize=13, y=1.05)
    plt.subplots_adjust(wspace=0.15)
    fig.savefig(out / "retention_vs_cost.png", bbox_inches="tight")
    plt.close(fig)

# ── orquestração ──────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("paths", nargs="+", type=Path,
                   help="JSONs gerados por evaluate_compressed.py")
    p.add_argument("--baseline", type=Path, default=None,
                   help="JSON de baseline separado (opcional; se omitido, "
                        "usa o campo 'baseline' dentro do primeiro JSON que o tiver)")
    p.add_argument("--out", type=Path, default=Path("output/plots_compressed"))
    p.add_argument("--tolerance-pp", type=float, default=1.0,
                   help="Gap aceitável de top-1 em pontos percentuais (zonas verdes)")
    p.add_argument("--top-k-classes", type=int, default=20,
                   help="Quantas classes mostrar no heatmap de delta F1 (padrão: 20)")
    p.add_argument("--top-k-models", type=int, default=8,
                   help="Máximo de modelos no heatmap; 0 = todos (padrão: 8)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # ── carregar baseline ──────────────────────────────────────────────────────
    baseline: dict | None = None

    if args.baseline is not None:
        raw = load_eval(args.baseline)
        baseline = raw.get("target", raw)
        if not baseline.get("arch"):
            baseline["arch"] = args.baseline.stem

    # ── carregar modelos comprimidos ───────────────────────────────────────────
    targets: dict[str, dict] = {}

    for path in args.paths:
        if args.baseline and path.resolve() == args.baseline.resolve():
            continue
        raw  = load_eval(path)
        data = raw.get("target", raw)
        if baseline is None and "baseline" in raw:
            baseline = raw["baseline"]
        name = path.stem
        if name in targets:
            name = f"{name}_{path.parent.name}"
        targets[name] = data

    if baseline is None:
        raise SystemExit(
            "Baseline não encontrado. Passe --baseline <json> ou certifique-se de que "
            "um dos JSONs contém a chave 'baseline'."
        )
    if not targets:
        raise SystemExit("Nenhum modelo comprimido encontrado nos JSONs fornecidos.")

    print(f"[info] Baseline : {baseline.get('arch', '?')}  "
          f"top-1={baseline.get('top1', 0):.4f}  "
          f"size={baseline.get('file_size_mb', 0):.2f} MB")
    for name, t in targets.items():
        cat = classify_model(name)
        sp  = sparsity_pct(t)
        print(f"[info] {name:40s}  cat={cat:12s}  top-1={t.get('top1', 0):.4f}  "
              f"sparsity={sp:.1f}%  size={t.get('file_size_mb', 0):.2f} MB")

    # fig_sparsity_vs_accuracy(targets, baseline, args.out, args.tolerance_pp)
    # fig_size_vs_accuracy    (targets, baseline, args.out, args.tolerance_pp)
    fig_latency_comparison  (targets, baseline, args.out)
    fig_per_class_delta       (targets, baseline, args.out, args.top_k_classes)
    fig_quality_vs_cost_pareto(targets, baseline, args.out, args.tolerance_pp)
    fig_retention_and_cost  (targets, baseline, args.out)
    write_summary_csv       (targets, baseline, args.out)

    print(f"\n[ok] {len(targets)} modelo(s) plotado(s) → {args.out.resolve()}")


if __name__ == "__main__":
    main()