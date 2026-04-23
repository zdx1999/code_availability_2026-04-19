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
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

COLOR_TOTAL = '#1f4e79'
COLOR_SHARE = '#00a6d6'
COLOR_FILL = '#9ecae1'


def save_both(fig, out_pdf: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_pdf.with_suffix('.png')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_pdf}')
    print(f'Saved: {out_png}')


def infer_peak_col(df: pd.DataFrame) -> str:
    candidates = [
        'peak_rain_for_screen', 'Event_Peak_Rain', 'peak_rain',
        'event_peak_rain', 'peak_rain_mm_h'
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError('Could not infer peak-rain column.')


def infer_group_cols(df: pd.DataFrame) -> tuple[str, str]:
    event_col = 'Event_ID' if 'Event_ID' in df.columns else 'event_id'
    if event_col not in df.columns:
        raise ValueError('Could not infer event id column.')
    city_col = 'city_clean' if 'city_clean' in df.columns else 'city'
    if city_col not in df.columns:
        raise ValueError('Could not infer city column.')
    return event_col, city_col


def build_event_table(full_df: pd.DataFrame) -> pd.DataFrame:
    peak_col = infer_peak_col(full_df)
    event_col, city_col = infer_group_cols(full_df)

    required = ['flood_count', 'hotspot_refined', 'new_hotspot_region', 'is_extreme', peak_col]
    for c in required:
        if c in full_df.columns:
            full_df[c] = pd.to_numeric(full_df[c], errors='coerce')

    event = full_df.groupby([event_col, city_col, 'is_extreme'], as_index=False).agg(
        total_event_impacts=('flood_count', 'sum'),
        hotspot_impacts=('flood_count', lambda s: s[full_df.loc[s.index, 'hotspot_refined'] == 1].sum()),
        hotspot_grids=('hotspot_refined', 'sum'),
        new_hotspot_grids=('new_hotspot_region', 'sum'),
        peak_rain=(peak_col, 'max'),
    )
    event = event[event['total_event_impacts'] > 0].copy()
    event['hotspot_burden_share'] = event['hotspot_impacts'] / event['total_event_impacts']
    event['share_newly_opened_hotspots'] = np.where(
        event['hotspot_grids'] > 0,
        event['new_hotspot_grids'] / event['hotspot_grids'],
        np.nan,
    )
    return event


def binned_curve(df: pd.DataFrame, xcol: str, ycol: str, qn: int = 7) -> pd.DataFrame:
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


def add_curve(ax, stat: pd.DataFrame, color: str, ylabel: str):
    if not stat.empty:
        ax.fill_between(stat['xmid'], stat['yq25'], stat['yq75'], color=COLOR_FILL, alpha=0.24)
        ax.plot(stat['xmid'], stat['ymed'], marker='o', lw=2.3, color=color, markersize=4.8)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis='both', labelsize=9)


def annotate_sample(ax, n: int, text: str):
    ax.text(
        0.98, 0.05, f'{text}: n={n}',
        transform=ax.transAxes,
        ha='right', va='bottom',
        fontsize=8.8, color='0.35'
    )


def main():
    ap = argparse.ArgumentParser(description='Build clean Fig.1c macro-trend panel without internal panel labels.')
    ap.add_argument('--full-csv', required=True, help='city_event_grid_full...csv')
    ap.add_argument('--out', required=True, help='Output PDF path')
    args = ap.parse_args()

    full_df = pd.read_csv(args.full_csv, low_memory=False)
    event = build_event_table(full_df)

    extreme = event[event['is_extreme'] == 1].copy()

    stat_total = binned_curve(event, 'peak_rain', 'total_event_impacts')
    stat_open = binned_curve(extreme, 'peak_rain', 'share_newly_opened_hotspots')

    fig = plt.figure(figsize=(8.4, 6.8), dpi=300)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.08, 0.92], hspace=0.22)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)

    add_curve(ax1, stat_total, COLOR_TOTAL, 'Total event flood impacts')
    add_curve(ax2, stat_open, COLOR_SHARE, 'Share of newly opened hotspots')

    ax2.set_xlabel('Event peak rainfall', fontsize=10)
    annotate_sample(ax1, len(event), 'All events')
    annotate_sample(ax2, len(extreme), 'Extreme events')

    for ax in (ax1, ax2):
        ax.grid(axis='y', alpha=0.18)
        ax.grid(axis='x', alpha=0.10)

    save_both(fig, Path(args.out))


if __name__ == '__main__':
    main()
