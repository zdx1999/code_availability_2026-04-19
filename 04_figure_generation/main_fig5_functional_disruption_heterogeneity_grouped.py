#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. 字体设定：优先使用顶级期刊偏好的 Arial 或 Helvetica
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] =['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

ROAD_ORDER =['road_l1', 'road_l2', 'road_l3']
ROAD_LABELS = {'road_l1': 'Road L1', 'road_l2': 'Road L2', 'road_l3': 'Road L3'}
ROAD_COLORS = {
    'Road L1': '#8aa8c4',   # 稍微加深了一点，确保即使数值小也能被清晰看见
    'Road L2': '#4f80a8',
    'Road L3': '#1f4e79',
}

POI_PRIORITY =['commercial_life', 'residential', 'transport', 'education_culture']
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

QUARTILE_ORDER =['Q1 small built-up', 'Q2', 'Q3', 'Q4 large built-up']
# 箱线图X轴多行展示映射字典 (保持水平排版)
QUARTILE_DISPLAY_LABELS = {
    'Q1 small built-up': 'Q1\n(Smallest)',
    'Q2': 'Q2',
    'Q3': 'Q3',
    'Q4 large built-up': 'Q4\n(Largest)'
}

GRID = '#e8e8e8'
PANEL_C_FILLS =['#d9e6f2', '#cfe8cf', '#f6d7c3', '#e6d9f2']


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
    rows =[]
    for road in ROAD_ORDER:
        col = f'new_hotspot_{road}_absolute'
        if col not in df.columns:
            continue
        vals = np.log1p(pd.to_numeric(df[col], errors='coerce'))
        med, q1, q3 = median_iqr(vals)
        rows.append({'label': ROAD_LABELS[road], 'median': med, 'q1': q1, 'q3': q3})
    out = pd.DataFrame(rows)
    out['order'] = out['label'].map({'Road L1': 0, 'Road L2': 1, 'Road L3': 2})
    return out.sort_values('order').drop(columns='order').reset_index(drop=True)


def choose_poi_buckets(df: pd.DataFrame, top_n: int = 4) -> list[str]:
    chosen =[b for b in POI_PRIORITY if f'new_hotspot_poi_{b}_absolute' in df.columns]
    if len(chosen) >= top_n:
        return chosen[:top_n]
    extras =[]
    for col in df.columns:
        if col.startswith('new_hotspot_poi_') and col.endswith('_absolute'):
            bucket = col[len('new_hotspot_poi_'):-len('_absolute')]
            if bucket not in chosen:
                extras.append((bucket, pd.to_numeric(df[col], errors='coerce').fillna(0).mean()))
    extras.sort(key=lambda x: x[1], reverse=True)
    return (chosen +[b for b, _ in extras])[:top_n]


def poi_summary(df: pd.DataFrame, buckets: list[str]) -> pd.DataFrame:
    rows =[]
    for bucket in buckets:
        col = f'new_hotspot_poi_{bucket}_absolute'
        if col not in df.columns:
            continue
        vals = np.log1p(pd.to_numeric(df[col], errors='coerce'))
        med, q1, q3 = median_iqr(vals)
        rows.append({'label': POI_LABELS.get(bucket, bucket), 'median': med, 'q1': q1, 'q3': q3})
    return pd.DataFrame(rows).sort_values('median', ascending=True).reset_index(drop=True)


def draw_letter(ax, letter: str, x_offset: float = -0.1, y_offset: float = 1.05):
    # transform=ax.transAxes 下，x<0 表示向左移出坐标轴，y>1 表示向上移出坐标轴
    # clip_on=False 保证文字移出绘图区后仍然可见
    ax.text(x_offset, y_offset, letter, transform=ax.transAxes, va='bottom', ha='left',
            fontsize=16, fontweight='bold', clip_on=False)


def draw_interval(ax, summ: pd.DataFrame, xlabel: str, color, lighten_labels: set[str] | None = None,
                  invert_y: bool = True):
    y = np.arange(len(summ))
    lighten_labels = lighten_labels or set()

    if isinstance(color, dict):
        colors = [color.get(lbl, '#1f4e79') for lbl in summ['label']]
    else:
        colors = [color] * len(summ)

    for yi, (_, row), c in zip(y, summ.iterrows(), colors):
        is_light = row['label'] in lighten_labels
        # 3. 极大地加粗了线宽与点的大小，增加图面张力
        lw = 3.0 if is_light else 4.0
        alpha = 0.90 if is_light else 1.0
        msize = 8.0 if is_light else 10.0
        
        ax.hlines(yi, row['q1'], row['q3'], color=c, linewidth=lw, alpha=alpha)
        ax.plot(row['median'], yi, 'o', color=c, markersize=msize, alpha=alpha)

    ax.set_yticks(y)
    ax.set_yticklabels(summ['label'], fontsize=11)
    if invert_y:
        ax.invert_yaxis()
    ax.tick_params(axis='y', labelsize=11, pad=4)
    ax.set_xlabel(xlabel, fontsize=11, labelpad=6)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def draw_quartile_boxes(ax, panel_df: pd.DataFrame, value_col: str, ylabel: str):
    data, labels = [],[]
    for q in QUARTILE_ORDER:
        vals = pd.to_numeric(
            panel_df.loc[panel_df['builtup_quartile'].astype(str) == q, value_col],
            errors='coerce'
        ).dropna()
        if len(vals):
            # 4. 使用转换后的多行水平标签
            labels.append(QUARTILE_DISPLAY_LABELS.get(q, q))
            data.append(vals.values)

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.56, showfliers=False)
    for patch, c in zip(bp['boxes'], PANEL_C_FILLS):
        patch.set_facecolor(c)
        patch.set_alpha(0.92)
    for median in bp['medians']:
        median.set_color('#d17c23')
        median.set_linewidth(1.5)

    ax.set_ylabel(ylabel, fontsize=11, labelpad=8)
    # 5. X轴标签改为完全水平对齐
    ax.tick_params(axis='x', rotation=0, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser(
        description='Figure 5 regrouped layout: upper panel combines Roads and POIs, lower panel shows city heterogeneity.'
    )
    ap.add_argument('--event-csv', required=True)
    ap.add_argument('--panel-csv', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--top-poi-n', type=int, default=4)
    ap.add_argument('--extreme-only', action='store_true', default=False)
    args = ap.parse_args()

    event_df = pd.read_csv(args.event_csv, low_memory=False)
    panel_df = pd.read_csv(args.panel_csv, low_memory=False)

    if args.extreme_only and 'is_extreme' in event_df.columns:
        event_df['is_extreme'] = pd.to_numeric(event_df['is_extreme'], errors='coerce').fillna(0)
        event_df = event_df[event_df['is_extreme'] == 1].copy()

    road = road_summary(event_df)
    poi = poi_summary(event_df, choose_poi_buckets(event_df, top_n=args.top_poi_n))

    fig = plt.figure(figsize=(13.0, 7.6), dpi=300, constrained_layout=False)
    
    # 6. 核心排版调整：大幅压扁上排（1.0），拉长下排（2.2），并增加间隔防止标签打架
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.2], hspace=0.25)

    # Upper combined information block (panel a): Roads + POIs
    top = outer[0].subgridspec(1, 2, width_ratios=[0.74, 1.26], wspace=0.22)
    ax_a1 = fig.add_subplot(top[0, 0])
    ax_a2 = fig.add_subplot(top[0, 1])

    draw_interval(
        ax_a1,
        road,
        'log(1 + new-hotspot road exposure)',
        ROAD_COLORS,
        lighten_labels={'Road L1'},
        invert_y=True,
    )
    draw_interval(
        ax_a2,
        poi,
        'log(1 + new-hotspot POI exposure)',
        '#16a6d1',
        invert_y=True,
    )
    draw_letter(ax_a1, 'a', x_offset=-0.28, y_offset=1.02)

    # Lower city heterogeneity block (panel b)
    ax_b = fig.add_subplot(outer[1, 0])
    draw_quartile_boxes(
        ax_b,
        panel_df,
        'log1p_new_hotspot_everyday_poi_absolute',
        'City-average log(1 + everyday-function POI exposure)',
    )
    draw_letter(ax_b, 'b', x_offset=-0.06, y_offset=1.02)

    # 预留左侧及底部一定边距，防止被无情截断
    plt.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.08)
    save_both(fig, Path(args.out))


if __name__ == '__main__':
    main()