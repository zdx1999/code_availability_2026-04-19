#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

MODEL_ORDER = ["M2", "M3", "M4"]
MODEL_LABELS = {
    "M2": "M2  Terrain + centre/activity",
    "M3": "M3  + POI volume",
    "M4": "M4  + POI richness",
}
TERM_ORDER = [
    "distance_to_centre",
    "night_time_lights",
    "population",
    "poi_log1p",
    "poi_category_richness",
]
TERM_LABELS = {
    "distance_to_centre": "Distance to centre",
    "night_time_lights": "Night-time lights",
    "population": "Population",
    "poi_log1p": "POI volume",
    "poi_category_richness": "POI richness",
}
COLORS = {
    "M2": "#4e79a7",
    "M3": "#f28e2b",
    "M4": "#59a14f",
}


def save_both(fig, out_pdf: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


def build_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    need = {"model", "term", "coef", "se"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    x = df.copy()
    x = x[x["model"].isin(MODEL_ORDER) & x["term"].isin(TERM_ORDER)].copy()
    x["coef"] = pd.to_numeric(x["coef"], errors="coerce")
    x["se"] = pd.to_numeric(x["se"], errors="coerce")
    x = x.dropna(subset=["coef", "se"])
    x["ci_low"] = x["coef"] - 1.96 * x["se"]
    x["ci_high"] = x["coef"] + 1.96 * x["se"]
    x["term_rank"] = x["term"].map({t: i for i, t in enumerate(TERM_ORDER)})
    x["model_rank"] = x["model"].map({m: i for i, m in enumerate(MODEL_ORDER)})
    return x.sort_values(["term_rank", "model_rank"]).reset_index(drop=True)


def plot_forest(ax, df: pd.DataFrame):
    offsets = {"M2": -0.18, "M3": 0.00, "M4": 0.18}
    ybase = np.array([0.0, 1.0, 2.0, 3.45, 4.45])  # extra gap before POI terms

    x_min = float(min(df["ci_low"].min(), -0.0008))
    x_max = float(max(df["ci_high"].max(), 0.0030))
    pad = 0.00018
    ax.set_xlim(x_min - pad, x_max + pad)

    # light grouping bands
    ax.axhspan(-0.45, 2.45, color="#f7f7f7", zorder=0)
    ax.axhspan(3.0, 4.9, color="#fcfcfc", zorder=0)

    for model in MODEL_ORDER:
        sub = df[df["model"] == model].copy()
        for _, row in sub.iterrows():
            y = ybase[TERM_ORDER.index(row["term"])] + offsets[model]
            ax.errorbar(
                x=row["coef"],
                y=y,
                xerr=[[row["coef"] - row["ci_low"]], [row["ci_high"] - row["coef"]]],
                fmt="o",
                color=COLORS[model],
                markerfacecolor=COLORS[model],
                markeredgecolor="white",
                markeredgewidth=0.6,
                capsize=2.5,
                markersize=6.0,
                linewidth=1.35,
                zorder=3,
            )

    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, zorder=1)
    ax.set_yticks(ybase)
    ax.set_yticklabels([TERM_LABELS[t] for t in TERM_ORDER], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Coefficient (95% CI)", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="x", alpha=0.15, linewidth=0.8)

    # subtle group labels
    ax.text(
        ax.get_xlim()[0], -0.65, "Centre / activity gradients",
        ha="left", va="bottom", fontsize=8.8, color="#666666"
    )
    ax.text(
        ax.get_xlim()[0], 2.8, "POI-based urban function gradients",
        ha="left", va="bottom", fontsize=8.8, color="#666666"
    )

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="None", color=COLORS[m], markersize=6, label=MODEL_LABELS[m])
        for m in MODEL_ORDER
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="lower right",
        fontsize=9,
        handletextpad=0.6,
        borderpad=0.0,
        labelspacing=0.7,
    )


def add_annotation_block(ax, meta: dict):
    lines = [
        "Dependent variable: NewHotspot_ge",
        "Sample: all built-up grids in extreme events",
        f"N = {meta.get('N', '864,131')}",
        f"Events = {meta.get('Events', '471')}",
        f"Cities = {meta.get('Cities', '180')}",
    ]
    ax.text(
        0.02,
        0.02,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.4,
        bbox=dict(boxstyle="square,pad=0.25", facecolor="white", edgecolor="#cccccc", linewidth=0.6, alpha=0.96),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coef-csv", required=True, help="Input CSV with columns: model,term,coef,se")
    ap.add_argument("--out", required=True, help="Output PDF path")
    ap.add_argument("--n", default="864,131")
    ap.add_argument("--events", default="471")
    ap.add_argument("--cities", default="180")
    args = ap.parse_args()

    df = pd.read_csv(args.coef_csv, low_memory=False)
    plot_df = build_plot_df(df)

    fig, ax = plt.subplots(figsize=(8.2, 4.7), dpi=300)
    plot_forest(ax, plot_df)
    ax.text(0.01, 0.98, "a", transform=ax.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")

    add_annotation_block(ax, {"N": args.n, "Events": args.events, "Cities": args.cities})

    plt.tight_layout()
    save_both(fig, Path(args.out))


if __name__ == "__main__":
    main()
