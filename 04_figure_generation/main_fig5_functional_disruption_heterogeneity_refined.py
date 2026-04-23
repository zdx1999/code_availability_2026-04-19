#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

ROAD_ORDER = ['road_l1', 'road_l2', 'road_l3']
ROAD_LABELS = {'road_l1': 'Road L1', 'road_l2': 'Road L2', 'road_l3': 'Road L3'}
POI_PRIORITY = ['commercial_life', 'residential', 'transport', 'education_culture', 'enterprise_industrial', 'medical']
POI_LABELS = {
    'commercial_life': 'Commercial/Life',
    'residential': 'Residential',
    'transport': 'Transport',
    'education_culture': 'Education/Culture',
    'enterprise_industrial': 'Enterprise/Industrial',
    'medical': 'Medical',
    'government_public': 'Government/Public',
    'finance': 'Finance',
    'other': 'Other',
}
QUARTILE_ORDER = ['Q1 small built-up', 'Q2', 'Q3', 'Q4 large built-up']
GRID = '#e8e8e8'


def save_both(fig, out_pdf: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_pdf.with_suffix('.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')


def median_iqr(values: pd.Series) -> tuple[float, float, float]:
    s = pd.to_numeric(values, errors='coerce').dropna()
    if len(s) == 0:
        return np.nan, np.nan, np.nan
    return float(s.median()), float(s.quantile(0.25)), float(s.quantile(0.75))


def road_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for road in ROAD_ORDER:
        col = f'new_hotspot_{road}_absolute'
        if col not in df.columns:
            continue
        vals = np.log1p(pd.to_numeric(df[col], errors='coerce'))
        med, q1, q3 = median_iqr(vals)
        rows.append({'label': ROAD_LABELS[road], 'median': med, 'q1': q1, 'q3': q3})
    return pd.DataFrame(rows)


def choose_poi_buckets(df: pd.DataFrame, top_n: int = 5) -> list[str]:
    chosen = [b for b in POI_PRIORITY if f'new_hotspot_poi_{b}_absolute' in df.columns]
    if len(chosen) >= top_n:
        return chosen[:top_n]
    extras = []
    for col in df.columns:
        if col.startswith('new_hotspot_poi_') and col.endswith('_absolute'):
            bucket = col[len('new_hotspot_poi_'):-len('_absolute')]
            if bucket not in chosen:
                extras.append((bucket, pd.to_numeric(df[col], errors='coerce').fillna(0).mean()))
    extras.sort(key=lambda x: x[1], reverse=True)
    return (chosen + [b for b, _ in extras])[:top_n]


def poi_summary(df: pd.DataFrame, buckets: list[str]) -> pd.DataFrame:
    rows = []
    for bucket in buckets:
        col = f'new_hotspot_poi_{bucket}_absolute'
        if col not in df.columns:
            continue
        vals = np.log1p(pd.to_numeric(df[col], errors='coerce'))
        med, q1, q3 = median_iqr(vals)
        rows.append({'label': POI_LABELS.get(bucket, bucket), 'median': med, 'q1': q1, 'q3': q3})
    return pd.DataFrame(rows).sort_values('median', ascending=True).reset_index(drop=True)


def draw_letter(ax, letter: str):
    ax.text(0.02, 0.98, letter, transform=ax.transAxes, va='top', ha='left', fontsize=12, fontweight='bold')


def draw_interval(ax, summ: pd.DataFrame, xlabel: str, letter: str, color: str):
    y = np.arange(len(summ))
    ax.hlines(y, summ['q1'], summ['q3'], color=color, linewidth=2.4, alpha=0.9)
    ax.plot(summ['median'], y, 'o', color=color, markersize=5.5)
    ax.set_yticks(y)
    ax.set_yticklabels(summ['label'])
    ax.tick_params(axis='y', labelsize=10, pad=2)
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    draw_letter(ax, letter)


def draw_quartile_boxes(ax, panel_df: pd.DataFrame, value_col: str, ylabel: str, letter: str):
    data, labels = [], []
    for q in QUARTILE_ORDER:
        vals = pd.to_numeric(panel_df.loc[panel_df['builtup_quartile'].astype(str) == q, value_col], errors='coerce').dropna()
        if len(vals):
            labels.append(q)
            data.append(vals.values)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.56, showfliers=False)
    fills = ['#d9e6f2', '#cfe8cf', '#f6d7c3', '#e6d9f2']
    for patch, c in zip(bp['boxes'], fills):
        patch.set_facecolor(c)
        patch.set_alpha(0.92)
    for median in bp['medians']:
        median.set_color('#d17c23')
        median.set_linewidth(1.2)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=14)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    draw_letter(ax, letter)


def main():
    ap = argparse.ArgumentParser(description='Refined Fig.5 without titles.')
    ap.add_argument('--event-csv', required=True)
    ap.add_argument('--panel-csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--top-poi-n', type=int, default=5)
    ap.add_argument('--extreme-only', action='store_true', default=False)
    args = ap.parse_args()

    event_df = pd.read_csv(args.event_csv, low_memory=False)
    panel_df = pd.read_csv(args.panel_csv, low_memory=False)
    if args.extreme_only and 'is_extreme' in event_df.columns:
        event_df = event_df[event_df['is_extreme'] == 1].copy()

    road = road_summary(event_df)
    poi = poi_summary(event_df, choose_poi_buckets(event_df, top_n=args.top_poi_n))

    fig = plt.figure(figsize=(13.2, 7.7), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[0.92, 1.08], width_ratios=[1.0, 1.06], hspace=0.08, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    draw_interval(ax_a, road, 'log(1 + road exposure)', 'a', '#1f4e79')
    draw_interval(ax_b, poi, 'log(1 + POI exposure)', 'b', '#16a6d1')
    draw_quartile_boxes(ax_c, panel_df, 'log1p_new_hotspot_everyday_poi_absolute', 'City-average log(1 + everyday-function POI exposure)', 'c')

    save_both(fig, Path(args.out))


if __name__ == '__main__':
    main()
