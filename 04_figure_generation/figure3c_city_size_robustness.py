#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

DEFAULT_GRID_BASE = Path(
    r"outputs_spatial_burden\refined_outputs\grid_universe\city_grid_base_centroids_gaia_with_gee_terrain_repaired_253cities.csv"
)
DEFAULT_FULL = Path(
    r"outputs_spatial_burden\refined_outputs\grid_universe\poi2018_outputs\city_event_grid_full_gaia_v4_poi_repaired_253cities.csv"
)
DEFAULT_SIGN_CSV = Path(
    r"code414\outputs_reframed_heterogeneity\heterogeneity_quartile_stability_table.csv"
)
QUARTILE_ORDER = ["Q1 smallest", "Q2", "Q3", "Q4 largest"]
QUARTILE_SHORT = ["Q1", "Q2", "Q3", "Q4"]
QUARTILE_COLORS = ["#c9d4e3", "#cfe2cc", "#edd1bc", "#d8cfe8"]
VAR_ORDER = ["distance_to_centre", "night_time_lights", "population", "poi_activity"]
VAR_LABELS = {
    "distance_to_centre": "Distance",
    "night_time_lights": "NTL",
    "population": "Population",
    "poi_activity": "POI",
}
SYMBOL_MAP = {
    "-": ("−", "#4e79a7"),
    "+": ("+", "#59a14f"),
    "+/-": ("±", "#9c755f"),
    "0": ("0", "#999999"),
    "weak": ("±", "#9c755f"),
}


def save_both(fig, out_pdf: Path):
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


def resolve_input_path(user_arg: str) -> Path:
    p = Path(user_arg)
    if p.exists():
        return p.resolve()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    candidates = []
    if not p.is_absolute():
        candidates.extend([
            Path.cwd() / p,
            script_dir / p,
            repo_root / p,
        ])

    seen = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand.resolve()

    raise FileNotFoundError(f"Could not find input file: {user_arg}")


def normalize_city(x):
    if pd.isna(x):
        return ""
    s = str(x).strip().replace(" ", "")
    for suf in ["特别行政区", "自治州", "地区", "市辖区", "盟", "市", "县"]:
        if s.endswith(suf) and len(s) > len(suf):
            s = s[:-len(suf)]
    return s


def build_city_quartile_share_panel(grid_base_csv: str, full_csv: str) -> pd.DataFrame:
    grid_base_csv = resolve_input_path(grid_base_csv)
    full_csv = resolve_input_path(full_csv)

    base = pd.read_csv(grid_base_csv, usecols=["city_clean", "grid_id", "cell_size_m"], low_memory=False)
    base["city_clean"] = base["city_clean"].map(normalize_city)
    base["cell_size_m"] = pd.to_numeric(base["cell_size_m"], errors="coerce").fillna(1000)
    builtup = (
        base.groupby("city_clean", as_index=False)
        .agg(n_builtup_grids=("grid_id", "nunique"), cell_size_m=("cell_size_m", "median"))
    )
    builtup["builtup_area"] = builtup["n_builtup_grids"] * (builtup["cell_size_m"] ** 2) / 1_000_000.0
    rank = builtup["builtup_area"].rank(method="average", pct=True)
    builtup["builtup_quartile"] = pd.cut(
        rank,
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=QUARTILE_ORDER,
        include_lowest=True,
    ).astype(str)
    builtup["quartile_label"] = builtup["builtup_quartile"]

    full = pd.read_csv(full_csv, usecols=["city_clean", "Event_ID", "is_extreme", "new_hotspot_region", "grid_id"], low_memory=False)
    full["city_clean"] = full["city_clean"].map(normalize_city)
    full["is_extreme"] = pd.to_numeric(full["is_extreme"], errors="coerce").fillna(0)
    full["new_hotspot_region"] = pd.to_numeric(full["new_hotspot_region"], errors="coerce").fillna(0)
    full = full[full["is_extreme"] == 1].copy()

    event = (
        full.groupby(["city_clean", "Event_ID"], as_index=False)
        .agg(
            event_new_hotspot_share=("new_hotspot_region", "mean"),
            n_event_grids=("grid_id", "nunique"),
        )
    )
    city_panel = (
        event.groupby("city_clean", as_index=False)
        .agg(
            city_avg_new_hotspot_share=("event_new_hotspot_share", "mean"),
            n_extreme_events=("Event_ID", "nunique"),
        )
    )
    city_panel = city_panel.merge(
        builtup[["city_clean", "builtup_quartile", "builtup_area", "quartile_label"]],
        on="city_clean",
        how="inner",
    )
    city_panel["city_std"] = city_panel["city_clean"]
    city_panel["share_pct"] = city_panel["city_avg_new_hotspot_share"].clip(0, 1) * 100.0
    city_panel = city_panel[city_panel["n_extreme_events"] > 0].copy()
    return city_panel.sort_values(["builtup_quartile", "city_clean"]).reset_index(drop=True)


def load_share_csv(path: str) -> pd.DataFrame:
    path = resolve_input_path(path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for c in df.columns:
        low = c.lower()
        if "quart" in low:
            rename_map[c] = "builtup_quartile"
        if ("share" in low and "hotspot" in low) or ("new_hotspot_share" in low):
            rename_map[c] = "city_avg_new_hotspot_share"

    df = df.rename(columns=rename_map)
    need = {"builtup_quartile", "city_avg_new_hotspot_share"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Share CSV missing columns: {sorted(missing)}")

    df["builtup_quartile"] = df["builtup_quartile"].astype(str).str.strip()
    df["city_avg_new_hotspot_share"] = pd.to_numeric(df["city_avg_new_hotspot_share"], errors="coerce")
    df = df.dropna(subset=["city_avg_new_hotspot_share"]).copy()

    if df["city_avg_new_hotspot_share"].max() > 1.5:
        df["city_avg_new_hotspot_share"] = df["city_avg_new_hotspot_share"] / 100.0

    df["share_pct"] = df["city_avg_new_hotspot_share"].clip(0, 1) * 100.0
    return df


def simplify_sign(v: str) -> str:
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return "0"
    if s.startswith("-"):
        return "-"
    if s.startswith("+"):
        return "+"
    if s in {"0", "zero"}:
        return "0"
    return "weak"


def load_sign_csv(path: str) -> pd.DataFrame:
    path = resolve_input_path(path)
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for c in df.columns:
        low = c.lower()
        if "quart" in low:
            rename[c] = "builtup_quartile"
        if "distance" in low or "distcenter" in low:
            rename[c] = "distance_to_centre"
        if "night" in low or "ntl" in low:
            rename[c] = "night_time_lights"
        if "population" in low or low == "pop":
            rename[c] = "population"
        if "poi" in low:
            rename[c] = "poi_activity"
    df = df.rename(columns=rename)

    need = {"builtup_quartile"} | set(VAR_ORDER)
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Sign CSV missing columns: {sorted(missing)}")

    df["builtup_quartile"] = df["builtup_quartile"].astype(str).str.strip()

    if df["builtup_quartile"].duplicated().any():
        rows = []
        for q, sub in df.groupby("builtup_quartile", dropna=False):
            out = {"builtup_quartile": q}
            for var in VAR_ORDER:
                signs = [simplify_sign(v) for v in sub[var].tolist() if str(v).strip() != ""]
                uniq = sorted(set(signs))
                if not uniq:
                    out[var] = "0"
                elif len(uniq) == 1:
                    out[var] = uniq[0]
                else:
                    out[var] = "weak"
            rows.append(out)
        df = pd.DataFrame(rows)

    return df


def make_box(ax, share_df: pd.DataFrame):
    positions = np.arange(1, len(QUARTILE_ORDER) + 1)
    arrays = []
    for q in QUARTILE_ORDER:
        vals = share_df.loc[share_df["builtup_quartile"] == q, "share_pct"].dropna().values
        arrays.append(vals)

    bp = ax.boxplot(
        arrays,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#cc7a00", linewidth=1.4),
        boxprops=dict(edgecolor="#444444", linewidth=0.9),
        whiskerprops=dict(color="#444444", linewidth=0.85),
        capprops=dict(color="#444444", linewidth=0.85),
    )
    for patch, c in zip(bp["boxes"], QUARTILE_COLORS):
        patch.set_facecolor(c)
        patch.set_alpha(0.88)

    rng = np.random.default_rng(42)
    for x, q in zip(positions, QUARTILE_ORDER):
        vals = share_df.loc[share_df["builtup_quartile"] == q, "share_pct"].dropna().values
        if len(vals) == 0:
            continue
        xs = x + rng.uniform(-0.10, 0.10, len(vals))
        ax.scatter(xs, vals, s=12, color="#666666", alpha=0.18, edgecolors="none", zorder=2)

    ax.set_xticks(positions)
    ax.set_xticklabels(QUARTILE_ORDER, rotation=15)
    ax.set_ylabel("City-average new-hotspot rate among built-up grids (%)", fontsize=10)
    ax.tick_params(axis="both", labelsize=9)
    ax.grid(axis="y", alpha=0.16, linewidth=0.8)

    # downward trend cue
    medians = [np.nanmedian(v) if len(v) > 0 else np.nan for v in arrays]
    x_use = [p for p, m in zip(positions, medians) if np.isfinite(m)]
    y_use = [m for m in medians if np.isfinite(m)]
    if len(x_use) >= 2:
        ax.plot(x_use, y_use, color="#777777", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)

    ax.text(
        0.03, 0.95, "Higher city-average opening\nin smaller built-up cities",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.7, color="#555555"
    )


def make_sign_matrix(ax, sign_df: pd.DataFrame):
    ax.set_xlim(0, len(VAR_ORDER))
    ax.set_ylim(0, len(QUARTILE_ORDER))
    ax.invert_yaxis()

    for i, q in enumerate(QUARTILE_ORDER):
        row = sign_df[sign_df["builtup_quartile"] == q]
        for j, var in enumerate(VAR_ORDER):
            ax.add_patch(
                Rectangle((j, i), 1, 1, facecolor="#fafafa", edgecolor="#dddddd", linewidth=0.8)
            )
            symbol_raw = ""
            if not row.empty:
                symbol_raw = str(row.iloc[0][var]).strip()
            symbol, color = SYMBOL_MAP.get(symbol_raw, (symbol_raw, "#777777"))
            ax.text(j + 0.5, i + 0.52, symbol, ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    ax.set_xticks(np.arange(len(VAR_ORDER)) + 0.5)
    ax.set_xticklabels([VAR_LABELS[v] for v in VAR_ORDER], fontsize=9)
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_yticks(np.arange(len(QUARTILE_ORDER)) + 0.5)
    ax.set_yticklabels(QUARTILE_SHORT, fontsize=9)
    ax.tick_params(axis="y", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0.00, -0.38,
        "Directional stability across\nquartile-specific reduced-set models",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.7, color="#555555"
    )
    ax.text(
        0.00, -0.60,
        "Reduced-set quartile checks omit\nGAIA development year",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.0, color="#777777"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--share-csv", default="", help="Optional prebuilt share CSV; if omitted, build the 180-city panel from grid base + full table")
    ap.add_argument("--grid-base-csv", default=str(DEFAULT_GRID_BASE), help="Repaired 253-city built-up grid base")
    ap.add_argument("--full-csv", default=str(DEFAULT_FULL), help="Full repaired event-grid table")
    ap.add_argument("--sign-csv", default=str(DEFAULT_SIGN_CSV), help="Quartile-wise sign summary or stability table")
    ap.add_argument("--out-share-csv", default="", help="Optional path to save the constructed 180-city quartile share panel")
    ap.add_argument("--out", required=True, help="Output PDF path")
    args = ap.parse_args()

    if str(args.share_csv).strip():
        share_df = load_share_csv(args.share_csv)
    else:
        share_df = build_city_quartile_share_panel(args.grid_base_csv, args.full_csv)
        if str(args.out_share_csv).strip():
            out_share = Path(args.out_share_csv)
            out_share.parent.mkdir(parents=True, exist_ok=True)
            share_df.to_csv(out_share, index=False, encoding="utf-8-sig")
            print(f"Saved: {out_share}")
    sign_df = load_sign_csv(args.sign_csv)

    fig = plt.figure(figsize=(8.9, 4.6), dpi=300)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1.05], wspace=0.22)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    make_box(ax1, share_df)
    make_sign_matrix(ax2, sign_df)

    ax1.text(0.01, 0.98, "c", transform=ax1.transAxes, ha="left", va="top", fontsize=12, fontweight="bold")

    plt.tight_layout()
    save_both(fig, Path(args.out))


if __name__ == "__main__":
    main()
