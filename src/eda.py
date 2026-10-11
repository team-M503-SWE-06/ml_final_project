"""
eda.py — exploratory figures for Problem 1(b).

Saves PNGs to report/figures/ and prints a short summary. Uses only the
cleaned frame plus the split indices (for the tier class balance), so it
never touches anything that the models fit on.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

FIG_DIR = "report/figures"

# Palette (validated categorical slots 1-2, single-hue blue ramp for magnitude)
BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK_2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.labelsize": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "font.size": 9, "figure.dpi": 150,
})


def _save(fig, name: str, out_dir: str) -> str:
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _thousands(x, _pos=None) -> str:
    return f"{x / 1e3:,.0f}k" if x < 1e6 else f"{x / 1e6:,.1f}M"


def price_distribution(df, out_dir):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4))
    a.hist(df["price"], bins=80, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    a.set(title="Price (AZN) — heavy right tail", xlabel="price (AZN)", ylabel="listings")
    a.xaxis.set_major_formatter(_thousands)
    b.hist(np.log(df["price"]), bins=80, color=BLUE, edgecolor=SURFACE, linewidth=0.4)
    b.set(title="log(price) — close to symmetric", xlabel="log(price)", ylabel="listings")
    return _save(fig, "eda_price_distribution.png", out_dir)


def area_vs_price(df, out_dir, seed):
    rng = np.random.default_rng(seed)
    sub = df.iloc[rng.permutation(len(df))[:15_000]]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.scatter(sub["area"], sub["price"], s=4, alpha=0.25, color=BLUE, linewidths=0)
    ax.set(xscale="log", yscale="log", title="Area vs price (log–log, 15k sample)",
           xlabel="area (m²)", ylabel="price (AZN)")
    ax.yaxis.set_major_formatter(_thousands)
    return _save(fig, "eda_area_vs_price.png", out_dir)


def price_by_rooms(df, out_dir):
    d = df.dropna(subset=["rooms"])
    rooms = [r for r in sorted(d["rooms"].unique()) if (d["rooms"] == r).sum() >= 50]
    data = [d.loc[d["rooms"] == r, "price"] for r in rooms]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.boxplot(data, tick_labels=[str(int(r)) for r in rooms], showfliers=False,
               patch_artist=True, widths=0.6,
               boxprops=dict(facecolor="#b7d3f6", edgecolor=BLUE),
               medianprops=dict(color=INK, linewidth=1.5),
               whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    ax.set(yscale="log", title="Price by number of rooms (rooms with ≥ 50 listings)",
           xlabel="rooms", ylabel="price (AZN, log scale)")
    ax.yaxis.set_major_formatter(_thousands)
    return _save(fig, "eda_price_by_rooms.png", out_dir)


def location_map(df, out_dir):
    d = df.dropna(subset=["lat", "lng"])
    # zoom to the Baku/Absheron core where 99 % of listings are
    lat0, lat1 = d["lat"].quantile([0.005, 0.995])
    lng0, lng1 = d["lng"].quantile([0.005, 0.995])
    d = d[d["lat"].between(lat0, lat1) & d["lng"].between(lng0, lng1)]
    d = d.iloc[np.argsort(d["price"].to_numpy())]   # expensive points drawn on top
    fig, ax = plt.subplots(figsize=(7, 4.8))
    sc = ax.scatter(d["lng"], d["lat"], c=np.log(d["price"]), cmap=BLUE_RAMP, s=3,
                    alpha=0.7, linewidths=0)
    cb = fig.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label("log(price)", color=INK_2)
    cb.outline.set_visible(False)
    ax.set(title="Location effect: log(price) by coordinates (Baku / Absheron)",
           xlabel="longitude", ylabel="latitude", aspect="equal")
    return _save(fig, "eda_location_map.png", out_dir)


def price_by_category(df, out_dir):
    order = df.groupby("category")["price"].median().sort_values().index
    data = [df.loc[df["category"] == c, "price"] for c in order]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.boxplot(data, tick_labels=list(order), showfliers=False, patch_artist=True, widths=0.6,
               boxprops=dict(facecolor="#b7d3f6", edgecolor=BLUE),
               medianprops=dict(color=INK, linewidth=1.5),
               whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
    ax.set(yscale="log", title="Price by property category", ylabel="price (AZN, log scale)")
    ax.yaxis.set_major_formatter(_thousands)
    return _save(fig, "eda_price_by_category.png", out_dir)


def tier_balance(tiers: dict, threshold: float, out_dir):
    """tiers: {split_name: 0/1 array}. Stacked share of standard vs premium per split."""
    names = list(tiers)
    prem = np.array([tiers[n].mean() for n in names])
    std = 1 - prem
    fig, ax = plt.subplots(figsize=(6, 2.6))
    y = np.arange(len(names))
    ax.barh(y, std, color=BLUE, height=0.55, label="standard (0)", edgecolor=SURFACE, linewidth=2)
    ax.barh(y, prem, left=std, color=ORANGE, height=0.55, label="premium (1)",
            edgecolor=SURFACE, linewidth=2)
    for i, n in enumerate(names):
        ax.text(std[i] / 2, i, f"{std[i]:.1%}", va="center", ha="center", color="white", fontsize=8)
        ax.text(std[i] + prem[i] / 2, i, f"{prem[i]:.1%}", va="center", ha="center",
                color="white", fontsize=8)
    ax.set_yticks(y, [f"{n} (n={len(tiers[n]):,})" for n in names])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.grid(False)
    ax.set(title=f"Price-tier class balance (threshold = train median {threshold:,.0f} AZN)",
           xlabel="share of listings")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, frameon=False)
    return _save(fig, "eda_tier_balance.png", out_dir)


def summary(df) -> str:
    q = df["price"].quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    lines = [f"rows: {len(df):,}",
             "price quantiles (AZN): " + ", ".join(f"p{int(k * 100)}={v:,.0f}" for k, v in q.items()),
             "missing: " + ", ".join(f"{c}={df[c].isna().mean():.1%}"
                                     for c in df.columns if df[c].isna().any()),
             "category counts: " + ", ".join(f"{k}={v:,}" for k, v in df["category"].value_counts().items()),
             f"corr(log area, log price) = "
             f"{np.corrcoef(np.log(df['area']), np.log(df['price']))[0, 1]:.3f}"]
    return "\n".join(lines)


def run_eda(df, tiers: dict, threshold: float, out_dir: str = FIG_DIR, seed: int = 42):
    """Make every EDA figure; return the list of saved paths."""
    os.makedirs(out_dir, exist_ok=True)
    return [
        price_distribution(df, out_dir),
        area_vs_price(df, out_dir, seed),
        price_by_rooms(df, out_dir),
        price_by_category(df, out_dir),
        location_map(df, out_dir),
        tier_balance(tiers, threshold, out_dir),
    ]
