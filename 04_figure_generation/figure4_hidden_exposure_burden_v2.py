#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

ABS_OUTCOMES = [
    ("log1p_new_hotspot_population_absolute", "log(1 + new-hotspot\npopulation)"),
    ("log1p_new_hotspot_road_len_km_absolute", "log(1 + new-hotspot\nroad length)"),
    ("log1p_new_hotspot_poi_count_absolute", "log(1 + new-hotspot\nPOI count)"),
]

PANEL_A_KEYS = [
    ("population", "Population"),
    ("road", "Roads"),
    ("poi", "POIs"),
]


def save_both(fig, out_pdf: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


def normalize(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def resolve_share_col(df: pd.DataFrame, domain_key: str) -> str:
    cols = {normalize(c): c for c in df.columns}
    exact_candidates = [
        f"new_hotspot_{domain_key}_share_of_hotspots",
        f"new_hotspot_{domain_key}_share_of_hotspot",
        f"new_hotspot_{domain_key}_share_hotspots",
        f"new_hotspot_{domain_key}_share_hotspot",
        f"new_hotspot_{domain_key}_share_of_all_hotspots",
    ]
    if domain_key == "road":
        exact_candidates += [
            "new_hotspot_road_len_km_share_of_hotspots",
            "new_hotspot_road_len_km_share_of_hotspot",
            "new_hotspot_road_len_share_of_hotspots",
            "new_hotspot_road_share_of_hotspots",
        ]
    if domain_key == "poi":
        exact_candidates += [
            "new_hotspot_poi_count_share_of_hotspots",
            "new_hotspot_poi_share_of_hotspots",
        ]
    if domain_key == "population":
        exact_candidates += [
            "new_hotspot_population_share_of_hotspots",
            "new_hotspot_pop_share_of_hotspots",
        ]
    for cand in exact_candidates:
        if cand in cols:
            return cols[cand]

    # fallback: contains tokens
    for ncol, orig in cols.items():
        if "new_hotspot" in ncol and domain_key in ncol and "share" in ncol and "hotspot" in ncol:
            return orig
        if domain_key == "road" and "new_hotspot" in ncol and "road" in ncol and "share" in ncol and "hotspot" in ncol:
            return orig
        if domain_key == "poi" and "new_hotspot" in ncol and "poi" in ncol and "share" in ncol and "hotspot" in ncol:
            return orig
        if domain_key == "population" and "new_hotspot" in ncol and "population" in ncol and "share" in ncol and "hotspot" in ncol:
            return orig
    raise ValueError(f"Could not infer hotspot-space share column for {domain_key}.")


def binned_curve(df: pd.DataFrame, xcol: str, ycol: str, qn: int = 8) -> pd.DataFrame:
    dd = df[[xcol, ycol]].dropna().copy()
    if dd.empty:
        return pd.DataFrame(columns=["xmid", "ymed", "yq25", "yq75", "n"])
    bins = np.unique(dd[xcol].quantile(np.linspace(0, 1, qn)).values)
    if len(bins) < 4:
        bins = np.linspace(dd[xcol].min(), dd[xcol].max(), 6)
    dd["bin"] = pd.cut(dd[xcol], bins=bins, include_lowest=True, duplicates="drop")
    stat = dd.groupby("bin", observed=True).agg(
        xmid=(xcol, "median"),
        ymed=(ycol, "median"),
        yq25=(ycol, lambda s: s.quantile(0.25)),
        yq75=(ycol, lambda s: s.quantile(0.75)),
        n=(ycol, "size"),
    ).reset_index(drop=True)
    return stat[stat["n"] >= 3].copy()


def plot_hidden_share_panel(ax, event_df: pd.DataFrame):
    # infer and compute medians
    rows = []
    for key, label in PANEL_A_KEYS:
        col = resolve_share_col(event_df, key)
        vals = pd.to_numeric(event_df[col], errors="coerce").dropna()
        if vals.empty:
            share = np.nan
        else:
            if vals.max() > 1.5:
                vals = vals / 100.0
            share = float(np.nanmedian(vals.clip(0, 1)))
        rows.append((label, share))

    labels = [r[0] for r in rows]
    shares = [r[1] for r in rows]
    recurrent = [1 - s for s in shares]
    y = np.arange(len(labels))

    recurrent_color = "#c8c8c8"
    hidden_color = "#19a0c9"

    ax.barh(y, recurrent, color=recurrent_color, edgecolor="none", height=0.62)
    ax.barh(y, shares, left=recurrent, color=hidden_color, edgecolor="none", height=0.62)

    for yi, s, left in zip(y, shares, recurrent):
        ax.text(left + s / 2, yi, f"{s * 100:.1f}%", ha="center", va="center", fontsize=10, color="#1f1f1f")

    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.14, linewidth=0.8)
    ax.tick_params(axis="x", labelsize=9)

    ax.text(0.01, 0.98, "a", transform=ax.transAxes, va="top", ha="left", fontsize=12, fontweight="bold")
    ax.text(0.10, 1.03, "Recurrent hotspot share", transform=ax.transAxes, color="#7a7a7a", fontsize=8.8)
    ax.text(0.62, 1.03, "New-hotspot hidden share", transform=ax.transAxes, color=hidden_color, fontsize=8.8)

    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.8)


def plot_absolute_panel(ax, df: pd.DataFrame, ycol: str, ylabel: str, panel_letter: str, marker: str):
    stat = binned_curve(df, "peak_rain", ycol)
    line_color = "#2b5b84"
    fill_color = "#d9e4ef"
    if not stat.empty:
        ax.fill_between(stat["xmid"], stat["yq25"], stat["yq75"], color=fill_color, alpha=0.65, linewidth=0)
        ax.plot(stat["xmid"], stat["ymed"], marker=marker, color=line_color, lw=2.2, markersize=5.2)
    ax.set_xlabel("Event peak rainfall", fontsize=9.5)
    ax.set_ylabel(ylabel, fontsize=9.2, labelpad=7)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=8.5, pad=2)
    ax.grid(axis="both", alpha=0.12, linewidth=0.8)
    ax.text(0.02, 0.98, panel_letter, transform=ax.transAxes, va="top", ha="left", fontsize=12, fontweight="bold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-csv", required=True, help="event_impact_metrics_basic_v3.csv")
    ap.add_argument("--out", required=True, help="Output PDF path")
    ap.add_argument("--extreme-only", action="store_true", default=False)
    args = ap.parse_args()

    event_df = pd.read_csv(args.event_csv, low_memory=False)
    event_df.columns = [str(c).strip() for c in event_df.columns]
    if args.extreme_only and "is_extreme" in event_df.columns:
        event_df["is_extreme"] = pd.to_numeric(event_df["is_extreme"], errors="coerce").fillna(0)
        event_df = event_df[event_df["is_extreme"] == 1].copy()

    fig = plt.figure(figsize=(12.8, 8.1), dpi=300)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.03, 1.17], height_ratios=[1, 1, 1], wspace=0.31, hspace=0.34)

    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 1])

    plot_hidden_share_panel(ax_a, event_df)
    plot_absolute_panel(ax_b, event_df, ABS_OUTCOMES[0][0], ABS_OUTCOMES[0][1], "b", "o")
    plot_absolute_panel(ax_c, event_df, ABS_OUTCOMES[1][0], ABS_OUTCOMES[1][1], "c", "o")
    plot_absolute_panel(ax_d, event_df, ABS_OUTCOMES[2][0], ABS_OUTCOMES[2][1], "d", "o")

    fig.subplots_adjust(left=0.055, right=0.985, top=0.97, bottom=0.08)
    save_both(fig, Path(args.out))


if __name__ == "__main__":
    main()
