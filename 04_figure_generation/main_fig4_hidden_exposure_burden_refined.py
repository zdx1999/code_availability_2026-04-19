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

COLOR_NEW = '#16a6d1'
COLOR_RECURRENT = '#d3d3d3'
COLOR_LINE = '#1f4e79'
COLOR_BAND = '#d8e6f2'
GRID = '#e8e8e8'

STACK_CONFIG = [
    ('Population', 'new_hotspot_population_share_of_hotspots'),
    ('Roads', 'new_hotspot_road_len_km_share_of_hotspots'),
    ('POIs', 'new_hotspot_poi_count_share_of_hotspots'),
]
ABS_CONFIG = [
    ('log1p_new_hotspot_population_absolute', 'log(1 + population)', 'b'),
    ('log1p_new_hotspot_road_len_km_absolute', 'log(1 + road length)', 'c'),
    ('log1p_new_hotspot_poi_count_absolute', 'log(1 + POI count)', 'd'),
]


def save_both(fig, out_pdf: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_pdf.with_suffix('.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')


def infer_peak_col(df: pd.DataFrame) -> str:
    for c in ['peak_rain', 'peak_rain_for_screen', 'Event_Peak_Rain', 'event_peak_rain']:
        if c in df.columns:
            return c
    raise ValueError('Could not infer peak-rain column in event metrics file.')


def binned_curve(df: pd.DataFrame, xcol: str, ycol: str, qn: int = 8) -> pd.DataFrame:
    dd = df[[xcol, ycol]].dropna().copy()
    if dd.empty:
        return pd.DataFrame(columns=['xmid', 'ymed', 'yq25', 'yq75', 'n'])
    bins = np.unique(dd[xcol].quantile(np.linspace(0, 1, qn)).values)
    if len(bins) < 4:
        bins = np.linspace(dd[xcol].min(), dd[xcol].max(), min(6, max(4, dd[xcol].nunique())))
    dd['bin'] = pd.cut(dd[xcol], bins=bins, include_lowest=True, duplicates='drop')
    stat = dd.groupby('bin', observed=True).agg(
        xmid=(xcol, 'median'),
        ymed=(ycol, 'median'),
        yq25=(ycol, lambda s: s.quantile(0.25)),
        yq75=(ycol, lambda s: s.quantile(0.75)),
        n=(ycol, 'size'),
    ).reset_index(drop=True)
    return stat[stat['n'] >= 3].copy()


def draw_letter(ax, letter: str):
    ax.text(0.02, 0.98, letter, transform=ax.transAxes, va='top', ha='left', fontsize=12, fontweight='bold')


def draw_stacked_bars(ax, event_df: pd.DataFrame):
    labels = []
    hidden = []
    for label, col in STACK_CONFIG:
        if col not in event_df.columns:
            raise ValueError(f'Missing required column: {col}')
        vals = pd.to_numeric(event_df[col], errors='coerce').dropna()
        labels.append(label)
        hidden.append(float(vals.median()) if len(vals) else np.nan)

    hidden = np.array(hidden, dtype=float)
    recurrent = 1 - hidden
    y = np.arange(len(labels))

    ax.barh(y, recurrent, color=COLOR_RECURRENT, edgecolor='none', height=0.60)
    ax.barh(y, hidden, left=recurrent, color=COLOR_NEW, edgecolor='none', height=0.60)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    draw_letter(ax, 'a')

    for yi, r, h in zip(y, recurrent, hidden):
        ax.text(r + h / 2, yi, f'{h * 100:.1f}%', va='center', ha='center', fontsize=10, color='#163040')


def draw_abs_curve(ax, event_df: pd.DataFrame, peak_col: str, ycol: str, ylabel: str, letter: str):
    if ycol not in event_df.columns:
        raise ValueError(f'Missing required column: {ycol}')
    stat = binned_curve(event_df, peak_col, ycol)
    if not stat.empty:
        ax.fill_between(stat['xmid'], stat['yq25'], stat['yq75'], color=COLOR_BAND, linewidth=0)
        ax.plot(stat['xmid'], stat['ymed'], marker='o', lw=2.3, color=COLOR_LINE, markersize=5.2)
    ax.set_xlabel('Peak rainfall')
    ax.set_ylabel(ylabel)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    draw_letter(ax, letter)


def main():
    ap = argparse.ArgumentParser(description='Refined Fig.4 without titles.')
    ap.add_argument('--event-csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--extreme-only', action='store_true', default=False)
    args = ap.parse_args()

    event_df = pd.read_csv(args.event_csv, low_memory=False)
    if args.extreme_only and 'is_extreme' in event_df.columns:
        event_df = event_df[event_df['is_extreme'] == 1].copy()
    peak_col = infer_peak_col(event_df)

    fig = plt.figure(figsize=(12.4, 7.8), dpi=300, constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.08)
    ax_a = fig.add_subplot(gs[0, 0])
    gs_right = gs[0, 1].subgridspec(3, 1, hspace=0.14)
    ax_b = fig.add_subplot(gs_right[0, 0])
    ax_c = fig.add_subplot(gs_right[1, 0])
    ax_d = fig.add_subplot(gs_right[2, 0])

    draw_stacked_bars(ax_a, event_df)
    for ax, (ycol, ylabel, letter) in zip([ax_b, ax_c, ax_d], ABS_CONFIG):
        draw_abs_curve(ax, event_df, peak_col, ycol, ylabel, letter)

    save_both(fig, Path(args.out))


if __name__ == '__main__':
    main()
