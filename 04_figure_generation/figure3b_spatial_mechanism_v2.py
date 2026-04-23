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


def save_both(fig, out_pdf: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


def load_mechanism_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        low = c.lower()
        if "median" in low and "distance" in low:
            rename_map[c] = "median_nearest_distance_km"
        if "within" in low and "3" in low:
            rename_map[c] = "share_within_3km"

    df = df.rename(columns=rename_map)

    required = {"median_nearest_distance_km", "share_within_3km"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Mechanism CSV must contain columns for event-level median nearest distance and share within 3 km. "
            f"Missing: {sorted(missing)}"
        )

    df["median_nearest_distance_km"] = pd.to_numeric(df["median_nearest_distance_km"], errors="coerce")
    df["share_within_3km"] = pd.to_numeric(df["share_within_3km"], errors="coerce")
    df = df.dropna(subset=["median_nearest_distance_km", "share_within_3km"]).copy()

    if df["share_within_3km"].max() > 1.5:
        df["share_within_3km"] = df["share_within_3km"] / 100.0

    df["share_within_3km"] = df["share_within_3km"].clip(0, 1)
    df["share_within_3km_pct"] = df["share_within_3km"] * 100.0
    return df


def add_jittered_points(ax, yvals, xpos, color, alpha=0.24, width=0.095, size=16, seed=42):
    rng = np.random.default_rng(seed)
    xs = xpos + rng.uniform(-width, width, size=len(yvals))
    ax.scatter(xs, yvals, s=size, color=color, alpha=alpha, edgecolors="none", zorder=2)


def make_single_distribution(ax, values, ylabel, fill, threshold=None, threshold_label=None,
                             median_fmt="{:.1f}", annotation=None):
    values = pd.Series(values).dropna().values

    parts = ax.violinplot([values], positions=[1], widths=0.56, showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(fill)
        body.set_edgecolor(fill)
        body.set_alpha(0.16)

    ax.boxplot(
        [values],
        positions=[1],
        widths=0.22,
        vert=True,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=fill, linewidth=1.5),
        boxprops=dict(facecolor="white", edgecolor=fill, linewidth=1.15),
        whiskerprops=dict(color=fill, linewidth=1.0),
        capprops=dict(color=fill, linewidth=1.0),
    )

    add_jittered_points(ax, values, xpos=1, color=fill)

    med = float(np.median(values))
    ax.scatter([1], [med], s=34, color=fill, edgecolors="white", linewidths=0.5, zorder=4)

    if threshold is not None:
        ax.axhline(threshold, color="#707070", linestyle="--", linewidth=1.0, zorder=1)
        if threshold_label:
            ax.text(1.33, threshold, threshold_label, ha="left", va="center", fontsize=8.2, color="#555555")

    ax.text(
        0.03, 0.93, f"Median = {median_fmt.format(med)}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=10.0, fontweight="bold"
    )

    if annotation:
        ax.text(
            0.03, 0.035, annotation,
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7.9,
            bbox=dict(boxstyle="square,pad=0.18", facecolor="white", edgecolor="#d3d3d3", linewidth=0.55, alpha=0.96),
        )

    ax.set_xticks([])
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(axis="y", alpha=0.14, linewidth=0.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mechanism-csv", required=True, help="CSV with event-level median nearest distance and share within 3 km")
    ap.add_argument("--out", required=True, help="Output PDF path")
    ap.add_argument("--sample-note", default="Event-level summaries across\nvalid hotspot-opening events")
    ap.add_argument("--reference-distance-km", type=float, default=3.0)
    ap.add_argument("--reference-share-pct", type=float, default=40.2)
    args = ap.parse_args()

    mech = load_mechanism_csv(args.mechanism_csv)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2), dpi=300, gridspec_kw={"wspace": 0.30})

    make_single_distribution(
        axes[0],
        mech["median_nearest_distance_km"],
        ylabel="Event-level median nearest distance (km)",
        fill="#4e79a7",
        threshold=args.reference_distance_km,
        threshold_label=f"{args.reference_distance_km:.0f} km",
        median_fmt="{:.1f} km",
        annotation=args.sample_note,
    )

    make_single_distribution(
        axes[1],
        mech["share_within_3km_pct"],
        ylabel="Share of new hotspots within 3 km (%)",
        fill="#f28e2b",
        threshold=args.reference_share_pct,
        threshold_label=f"{args.reference_share_pct:.1f}%",
        median_fmt="{:.1f}%",
        annotation=None,
    )

    axes[0].text(0.01, 0.98, "b", transform=axes[0].transAxes, ha="left", va="top", fontsize=12, fontweight="bold")

    axes[1].set_ylim(0, max(100, float(mech["share_within_3km_pct"].max()) * 1.08))
    axes[0].set_ylim(0, max(args.reference_distance_km * 3, float(mech["median_nearest_distance_km"].max()) * 1.08))

    plt.tight_layout()
    save_both(fig, Path(args.out))


if __name__ == "__main__":
    main()
