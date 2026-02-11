#!/usr/bin/env python
"""
Generate analysis charts from Aedes albopictus processed data and summary stats.

Reads:
  - ../output_stats/albopictus_summary.json  (pipeline run statistics)
  - ../output_data/albopictus.csv.zip        (final processed dataset)

Saves plots to:
  - ../output_stats/plots/
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent
STATS_DIR = (SCRIPT_DIR / "../output_stats").resolve()
DATA_DIR = (SCRIPT_DIR / "../output_data").resolve()
PLOTS_DIR = STATS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = STATS_DIR / "albopictus_summary.json"
CSV_PATH = DATA_DIR / "albopictus.csv.zip"

# Also load raw occurrence for the zero/non-zero species breakdown
RAW_DATA_DIR = (SCRIPT_DIR / "../input_data/dwca-aimsurv-v2.3").resolve()

# ── style defaults ───────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="notebook", palette="deep", font_scale=1.0)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.bbox": "tight",
})
PALETTE = sns.color_palette("deep", 10)


def load_inputs():
    """Return (summary_dict, dataframe)."""
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(f"Summary JSON not found: {SUMMARY_PATH}")
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Processed CSV not found: {CSV_PATH}")

    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summary = json.load(f)

    df = pd.read_csv(CSV_PATH, low_memory=False,
                     parse_dates=["start_date", "end_date"])
    for col in ("start_date", "end_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return summary, df


# ── individual plot functions ────────────────────────────────────────────────

def plot_top_species(summary):
    """Stacked horizontal bar chart: zero vs non-zero counts for top 15 species."""
    species_totals = summary.get("raw", {}).get("top_scientific_names", {})
    if not species_totals:
        logger.warning("No top_scientific_names in summary – skipping.")
        return

    # Try to load raw occurrence to compute zero/non-zero per species
    raw_occ_path = RAW_DATA_DIR / "occurrence.txt"
    if raw_occ_path.exists():
        logger.info("Loading raw occurrence for zero/non-zero species breakdown …")
        raw = pd.read_csv(raw_occ_path, delimiter="\t", low_memory=False,
                          usecols=["scientificName", "individualCount"])
        top_names = list(species_totals.keys())
        raw = raw[raw["scientificName"].isin(top_names)].copy()
        raw["is_zero"] = raw["individualCount"] == 0
        agg = (
            raw.groupby("scientificName")["is_zero"]
            .value_counts()
            .unstack(fill_value=0)
            .reindex(top_names)
            .fillna(0)
        )
        # Ensure columns exist
        zeros = agg.get(True, pd.Series(0, index=agg.index)).values.astype(int)
        nonzeros = agg.get(False, pd.Series(0, index=agg.index)).values.astype(int)
        names = list(agg.index)
    else:
        logger.warning("Raw occurrence file not found – falling back to totals only.")
        names = list(species_totals.keys())
        nonzeros = np.array(list(species_totals.values()))
        zeros = np.zeros_like(nonzeros)

    short = [n.split("(")[0].strip() if "(" in n else n for n in names]
    y_pos = np.arange(len(short))

    fig, ax = plt.subplots(figsize=(11, 7))
    bars_nz = ax.barh(y_pos, nonzeros, color=PALETTE[0], edgecolor="white", label="Non-zero counts")
    bars_z = ax.barh(y_pos, zeros, left=nonzeros, color=PALETTE[3], edgecolor="white", label="Zero counts")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(short, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Number of occurrence records")
    ax.set_title("Top 15 Species — Zero vs Non-Zero Counts", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)

    # Annotate totals
    for i, (nz, z) in enumerate(zip(nonzeros, zeros)):
        total = nz + z
        ax.text(total + max(nonzeros + zeros) * 0.008, i, f"{total:,}", va="center", fontsize=7)

    sns.despine(left=True)
    fig.savefig(PLOTS_DIR / "01_top_species.png")
    plt.close(fig)
    logger.info("Saved 01_top_species.png")


def plot_pipeline_funnel(summary):
    """Funnel / waterfall chart showing record counts at each pipeline stage."""
    stages, labels = [], []

    raw_total = summary.get("raw", {}).get("occurrence_records_total")
    if raw_total:
        stages.append(raw_total); labels.append("Raw\noccurrences")
    post_concat = summary.get("extraction", {}).get("post_concat_total_records")
    if post_concat:
        stages.append(post_concat); labels.append("After extraction\n(albopictus + zeros)")
    after_coords = summary.get("coordinates", {}).get("records_after")
    if after_coords:
        stages.append(after_coords); labels.append("After coord\ncleaning")
    after_life = summary.get("final_filtering", {}).get("records_after_life_stage_filter")
    if after_life:
        stages.append(after_life); labels.append("After life-stage\nfilter")
    final = summary.get("final_filtering", {}).get("final_records")
    if final:
        stages.append(final); labels.append("Final\ndataset")

    if len(stages) < 2:
        logger.warning("Not enough funnel data – skipping.")
        return

    colors = sns.color_palette("Blues_d", len(stages))[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(stages)), stages, color=colors, edgecolor="white", width=0.6)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Number of records")
    ax.set_title("Pipeline Funnel – Record Counts per Stage", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, v in zip(bars, stages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    sns.despine()
    fig.savefig(PLOTS_DIR / "02_pipeline_funnel.png")
    plt.close(fig)
    logger.info("Saved 02_pipeline_funnel.png")


def plot_life_stage_comparison(summary):
    """Grouped bar chart: life-stage counts before vs after filtering."""
    before = summary.get("final_filtering", {}).get("life_stage_counts_before", {})
    after = summary.get("final_filtering", {}).get("life_stage_counts_after", {})
    if not before:
        logger.warning("No life_stage_counts data – skipping.")
        return

    all_stages = sorted(set(list(before.keys()) + list(after.keys())), key=str)
    rows = []
    for s in all_stages:
        rows.append({"Life Stage": str(s), "Phase": "Before filtering", "Records": before.get(s, 0)})
        rows.append({"Life Stage": str(s), "Phase": "After filtering", "Records": after.get(s, 0)})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(data=plot_df, x="Life Stage", y="Records", hue="Phase",
                palette=[PALETTE[1], PALETTE[2]], edgecolor="white", ax=ax)
    ax.set_title("Life-Stage Distribution Before vs After Filtering", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.xticks(rotation=30, ha="right", fontsize=9)
    sns.despine()
    fig.savefig(PLOTS_DIR / "03_life_stage_comparison.png")
    plt.close(fig)
    logger.info("Saved 03_life_stage_comparison.png")


def plot_sampling_effort_validation(summary):
    """Donut chart of sampling effort validation results."""
    sev = summary.get("sampling_effort_validation", {})
    kept = sev.get("kept_true", 0)
    discarded = sev.get("kept_false", 0)
    nan_val = sev.get("kept_nan", 0)

    if kept + discarded + nan_val == 0:
        logger.warning("No sampling effort validation data – skipping.")
        return

    labels, sizes, colors = [], [], []
    if kept:
        labels.append(f"Kept ({kept:,})"); sizes.append(kept); colors.append(PALETTE[2])
    if discarded:
        labels.append(f"Discarded ({discarded:,})"); sizes.append(discarded); colors.append(PALETTE[3])
    if nan_val:
        labels.append(f"NaN ({nan_val:,})"); sizes.append(nan_val); colors.append(PALETTE[7])

    fig, ax = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.78,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_fontweight("bold")
    ax.set_title("Sampling Effort Validation", fontsize=13, fontweight="bold")
    fig.savefig(PLOTS_DIR / "04_sampling_effort_validation.png")
    plt.close(fig)
    logger.info("Saved 04_sampling_effort_validation.png")


def plot_weekly_rate_distribution(df):
    """Histogram of weekly occurrence rates (log-scaled y-axis)."""
    if "weeklyRate" not in df.columns:
        logger.warning("weeklyRate column missing – skipping.")
        return

    rates = df["weeklyRate"].dropna()
    rates = rates[rates.between(0, rates.quantile(0.99))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(rates, bins=80, color=PALETTE[0], edgecolor="white", log=True)
    ax.set_xlabel("Weekly Rate (individuals / week)")
    ax.set_ylabel("Frequency (log scale)")
    ax.set_title("Distribution of Weekly Occurrence Rates", fontsize=13, fontweight="bold")
    sns.despine()
    fig.savefig(PLOTS_DIR / "05_weekly_rate_distribution.png")
    plt.close(fig)
    logger.info("Saved 05_weekly_rate_distribution.png")


def plot_individual_count_distribution(df):
    """Donut + histogram of individualCount."""
    if "individualCount" not in df.columns:
        logger.warning("individualCount column missing – skipping.")
        return

    counts = df["individualCount"].dropna()
    zero_n = int((counts == 0).sum())
    pos_n = int((counts > 0).sum())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: donut
    wedges, texts, autotexts = axes[0].pie(
        [zero_n, pos_n],
        labels=[f"Zero ({zero_n:,})", f"Positive ({pos_n:,})"],
        colors=[PALETTE[3], PALETTE[2]], autopct="%1.1f%%", startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2),
        pctdistance=0.78,
    )
    for t in autotexts:
        t.set_fontsize(10); t.set_fontweight("bold")
    axes[0].set_title("Zero vs Positive Counts", fontsize=11, fontweight="bold")

    # Right: histogram of positive counts (capped at 99th percentile)
    pos = counts[counts > 0]
    cap = pos.quantile(0.99)
    axes[1].hist(pos[pos <= cap], bins=60, color=PALETTE[4], edgecolor="white", log=True)
    axes[1].set_xlabel("Individual Count")
    axes[1].set_ylabel("Frequency (log)")
    axes[1].set_title("Positive Count Distribution (≤ 99th pctl)", fontsize=11, fontweight="bold")

    fig.suptitle("Individual Count Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.savefig(PLOTS_DIR / "06_individual_count_distribution.png")
    plt.close(fig)
    logger.info("Saved 06_individual_count_distribution.png")


def plot_records_per_trap(df):
    """Histogram of records per trap."""
    if "id_trap" not in df.columns:
        logger.warning("id_trap column missing – skipping.")
        return

    per_trap = df.groupby("id_trap").size().rename("records")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(per_trap, bins=min(80, per_trap.nunique()), color=PALETTE[5], edgecolor="white", log=True)
    ax.axvline(per_trap.median(), color=PALETTE[3], ls="--", lw=2,
               label=f"Median = {per_trap.median():.0f}")
    ax.set_xlabel("Records per Trap")
    ax.set_ylabel("Number of Traps (log)")
    ax.set_title(f"Records per Trap (n_traps={per_trap.nunique():,})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    sns.despine()
    fig.savefig(PLOTS_DIR / "07_records_per_trap.png")
    plt.close(fig)
    logger.info("Saved 07_records_per_trap.png")


def plot_monthly_time_series(df):
    """Monthly time series of observation counts, split by life stage."""
    if "end_date" not in df.columns or "lifeStage" not in df.columns:
        logger.warning("Required columns missing – skipping time series.")
        return

    df = df.dropna(subset=["end_date"]).copy()
    df["year_month"] = df["end_date"].dt.to_period("M")

    monthly = df.groupby(["year_month", "lifeStage"]).size().unstack(fill_value=0)
    monthly.index = monthly.index.to_timestamp()

    pal = sns.color_palette("Set2", monthly.shape[1])

    fig, ax = plt.subplots(figsize=(15, 5))
    monthly.plot.bar(stacked=True, ax=ax, width=0.9, edgecolor="none", color=pal)
    tick_labels = [d.strftime("%Y-%m") if i % 3 == 0 else "" for i, d in enumerate(monthly.index)]
    ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Number of records")
    ax.set_title("Monthly Observation Counts by Life Stage", fontsize=13, fontweight="bold")
    ax.legend(title="Life Stage", fontsize=9)
    sns.despine()
    fig.savefig(PLOTS_DIR / "08_monthly_time_series.png")
    plt.close(fig)
    logger.info("Saved 08_monthly_time_series.png")


def plot_geographic_scatter(df):
    """Lat/lon scatter coloured by life stage with an OpenStreetMap basemap."""
    if "decimalLatitude" not in df.columns or "decimalLongitude" not in df.columns:
        logger.warning("Coordinate columns missing – skipping map.")
        return

    # Try to use contextily for a basemap
    try:
        import contextily as ctx
        has_ctx = True
    except ImportError:
        has_ctx = False
        logger.warning("contextily not installed – plotting without basemap. "
                       "Install with:  pip install contextily")

    # Try to use geopandas for proper CRS handling
    try:
        import geopandas as gpd
        has_gpd = True
    except ImportError:
        has_gpd = False

    stages = df["lifeStage"].unique() if "lifeStage" in df.columns else [None]
    pal = sns.color_palette("bright", len(stages))

    if has_ctx and has_gpd:
        # Build GeoDataFrame in EPSG:4326, then reproject to Web Mercator for contextily
        gdf = gpd.GeoDataFrame(
            df, geometry=gpd.points_from_xy(df["decimalLongitude"], df["decimalLatitude"]),
            crs="EPSG:4326",
        ).to_crs(epsg=3857)

        fig, ax = plt.subplots(figsize=(12, 10))
        for i, stage in enumerate(stages):
            sub = gdf[gdf["lifeStage"] == stage] if stage else gdf
            sub.plot(ax=ax, markersize=4, alpha=0.4, color=pal[i], label=stage)

        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto")
        except Exception as e:
            logger.warning(f"Could not add basemap tile: {e}")

        ax.set_axis_off()
        ax.set_title("Geographic Distribution of Observations", fontsize=14, fontweight="bold")
        ax.legend(markerscale=5, fontsize=9, loc="lower left", frameon=True,
                  facecolor="white", edgecolor="grey")
    elif has_ctx:
        # contextily without geopandas – manual Web Mercator conversion
        import pyproj
        transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

        fig, ax = plt.subplots(figsize=(12, 10))
        for i, stage in enumerate(stages):
            sub = df[df["lifeStage"] == stage] if stage else df
            x, y = transformer.transform(sub["decimalLongitude"].values, sub["decimalLatitude"].values)
            ax.scatter(x, y, s=4, alpha=0.4, color=pal[i], label=stage)

        try:
            ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom="auto")
        except Exception as e:
            logger.warning(f"Could not add basemap tile: {e}")

        ax.set_axis_off()
        ax.set_title("Geographic Distribution of Observations", fontsize=14, fontweight="bold")
        ax.legend(markerscale=5, fontsize=9, loc="lower left", frameon=True,
                  facecolor="white", edgecolor="grey")
    else:
        # Fallback: plain scatter with seaborn styling
        fig, ax = plt.subplots(figsize=(12, 10))
        for i, stage in enumerate(stages):
            sub = df[df["lifeStage"] == stage] if stage else df
            ax.scatter(sub["decimalLongitude"], sub["decimalLatitude"],
                       s=4, alpha=0.4, color=pal[i], label=stage)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Geographic Distribution of Observations", fontsize=14, fontweight="bold")
        ax.legend(markerscale=5, fontsize=9)
        sns.despine()

    fig.savefig(PLOTS_DIR / "09_geographic_scatter.png")
    plt.close(fig)
    logger.info("Saved 09_geographic_scatter.png")


def plot_time_diff_distribution(df):
    """Stacked bar chart of sampling time-diff values, split by life stage and zero vs non-zero."""
    if "time_diff" not in df.columns:
        logger.warning("time_diff column missing – skipping.")
        return

    cols = ["time_diff", "individualCount", "lifeStage"]
    td = df[cols].dropna().copy()
    td["is_zero"] = td["individualCount"] == 0

    # Keep only top 30 time_diff values by total frequency
    top_td = td["time_diff"].value_counts().sort_index().head(30).index
    td = td[td["time_diff"].isin(top_td)]

    life_stages = sorted(td["lifeStage"].unique())
    n_stages = len(life_stages)
    stage_colors_nz = sns.color_palette("deep", n_stages)
    stage_colors_z = sns.color_palette("pastel", n_stages)

    # Aggregate: for each (time_diff, lifeStage) get zero and non-zero counts
    agg = (
        td.groupby(["time_diff", "lifeStage", "is_zero"])
        .size()
        .unstack(fill_value=0)
    )
    # Ensure both columns exist
    for c in (False, True):
        if c not in agg.columns:
            agg[c] = 0

    td_values = sorted(td["time_diff"].unique())
    x = np.arange(len(td_values))
    td_to_idx = {v: i for i, v in enumerate(td_values)}

    fig, ax = plt.subplots(figsize=(14, 6))

    # Stack: for each time_diff, stack life stages; within each life stage show non-zero then zero
    bottom = np.zeros(len(td_values))
    legend_handles = []

    for s_i, stage in enumerate(life_stages):
        nz_vals = np.zeros(len(td_values))
        z_vals = np.zeros(len(td_values))
        for td_val in td_values:
            idx = td_to_idx[td_val]
            if (td_val, stage) in agg.index:
                row = agg.loc[(td_val, stage)]
                nz_vals[idx] = row.get(False, 0)
                z_vals[idx] = row.get(True, 0)

        b1 = ax.bar(x, nz_vals, bottom=bottom, color=stage_colors_nz[s_i],
                     edgecolor="white", linewidth=0.5)
        bottom += nz_vals
        b2 = ax.bar(x, z_vals, bottom=bottom, color=stage_colors_z[s_i],
                     edgecolor="white", linewidth=0.5, hatch="//")
        bottom += z_vals

        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=stage_colors_nz[s_i],
                                            edgecolor="white", label=f"{stage} (non-zero)"))
        legend_handles.append(plt.Rectangle((0, 0), 1, 1, fc=stage_colors_z[s_i],
                                            edgecolor="white", hatch="//", label=f"{stage} (zero)"))

    # Annotate totals
    for i, total in enumerate(bottom):
        if total > 0:
            ax.text(i, total, f"{int(total):,}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in td_values], rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Time Difference (days)")
    ax.set_ylabel("Frequency")
    ax.set_title("Sampling Time Differences by Life Stage & Zero/Non-Zero (top 30)",
                 fontsize=13, fontweight="bold")
    ax.legend(handles=legend_handles, fontsize=8, ncol=n_stages, loc="upper right")
    sns.despine()
    fig.savefig(PLOTS_DIR / "10_time_diff_distribution.png")
    plt.close(fig)
    logger.info("Saved 10_time_diff_distribution.png")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("Loading data …")
    summary, df = load_inputs()
    logger.info(f"Loaded summary with keys: {list(summary.keys())}")
    logger.info(f"Loaded CSV with {len(df):,} rows, {len(df.columns)} columns")

    plot_top_species(summary)
    plot_pipeline_funnel(summary)
    plot_life_stage_comparison(summary)
    plot_sampling_effort_validation(summary)
    plot_weekly_rate_distribution(df)
    plot_individual_count_distribution(df)
    plot_records_per_trap(df)
    plot_monthly_time_series(df)
    plot_geographic_scatter(df)
    plot_time_diff_distribution(df)

    logger.info(f"All plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
