
"""Exploratory Data Analysis: SF Food Facility Inspections."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_PATH = Path("data/cleaned-data/cleaned_data_eda.csv")
OUTPUT_DIR = Path("figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent colors
TEAL = "#168B83"
CORAL = "#D75A49"
NAVY = "#182B40"
GRAY = "#68798A"
BG = "#FAFBFD"

sns.set_theme(style="white", font="DejaVu Sans")

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": NAVY,
    "axes.labelcolor": NAVY,
    "xtick.color": GRAY,
    "ytick.color": NAVY,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
})


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def add_title(fig, number, heading, subtitle):
    """Add a consistent numbered title and subtitle."""
    fig.text(
        0.08, 0.96,
        f"{number}  |  {heading}",
        fontsize=17,
        fontweight="bold",
        va="top",
    )

    fig.text(
        0.08, 0.89,
        subtitle,
        fontsize=10,
        color=GRAY,
        va="top",
    )


def save_chart(fig, filename):
    """Save and close a figure."""
    path = OUTPUT_DIR / filename

    fig.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
        pad_inches=0.25,
    )

    plt.close(fig)
    print(f"Saved: {path}")


def calculate_rates(df, column):
    """Calculate inspection counts and unfavorable rates."""
    data = (
        df.groupby(column)["target"]
        .agg(["size", "mean"])
        .reset_index()
    )

    data.columns = [column, "count", "rate"]
    data["rate"] *= 100

    return data


# ---------------------------------------------------------
# 01. Overall Inspection Outcomes — Pie Chart
# ---------------------------------------------------------

def overall_chart(df):
    """Show the overall proportion of inspection outcomes."""
    total = len(df)
    unfavorable = int(df["target"].sum())
    passed = total - unfavorable

    fig, ax = plt.subplots(figsize=(9, 5.8))

    fig.subplots_adjust(
        left=0.09,
        right=0.94,
        top=0.79,
        bottom=0.13,
    )

    add_title(
        fig,
        "01",
        "The Big Picture",
        "Most rated inspections pass; unfavorable = "
        "Conditional Pass or Closure",
    )

    ax.pie(
        [passed, unfavorable],
        colors=[TEAL, CORAL],
        startangle=90,
        counterclock=False,
        wedgeprops={
            "linewidth": 3,
            "edgecolor": BG,
        },
        radius=0.97,
    )

    # Percentages and inspection counts
    ax.text(
        1.35, 0.35,
        f"{passed / total:.1%}",
        fontsize=26,
        fontweight="bold",
        color=TEAL,
    )

    ax.text(
        1.35, 0.14,
        f"PASS  ·  {passed:,} inspections",
        fontsize=10,
        color=GRAY,
    )

    ax.text(
        1.35, -0.27,
        f"{unfavorable / total:.1%}",
        fontsize=26,
        fontweight="bold",
        color=CORAL,
    )

    ax.text(
        1.35, -0.48,
        f"UNFAVORABLE  ·  {unfavorable:,} inspections",
        fontsize=10,
        color=GRAY,
    )

    ax.set_xlim(-1.2, 3.6)
    ax.axis("off")

    fig.text(
        0.09, 0.07,
        f"Based on {total:,} inspections with a recorded rating",
        color=GRAY,
        fontsize=9,
    )

    save_chart(fig, "eda_01_overall.png")


# ---------------------------------------------------------
# 02. Facility Type — Lollipop Chart
# ---------------------------------------------------------

def facility_chart(df):
    """Compare unfavorable rates across facility types."""
    data = calculate_rates(df, "permit_type")

    data = (
        data[data["count"] >= 50]
        .sort_values("rate")
    )

    baseline = df["target"].mean() * 100

    fig, ax = plt.subplots(figsize=(11, 7))

    fig.subplots_adjust(
        left=0.30,
        right=0.86,
        top=0.78,
        bottom=0.15,
    )

    add_title(
        fig,
        "02",
        "Which Facility Categories Stand Out?",
        "Unfavorable inspection rates by facility type"
        "  ·  Minimum 50 inspections",
    )

    y = np.arange(len(data))

    # Horizontal lines
    ax.hlines(
        y,
        0,
        data["rate"],
        color="#B7C9D6",
        linewidth=3,
    )

    # Dots
    colors = [
        CORAL if rate > baseline else TEAL
        for rate in data["rate"]
    ]

    ax.scatter(
        data["rate"],
        y,
        color=colors,
        s=110,
        zorder=3,
        edgecolor="white",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(data["permit_type"])

    # Labels
    for i, row in enumerate(data.itertuples()):
        ax.text(
            row.rate + 0.15,
            i,
            f"{row.rate:.1f}%  ·  n={row.count:,}",
            va="center",
            fontsize=9,
        )

    ax.axvline(
        baseline,
        color=NAVY,
        linestyle="--",
        linewidth=1.2,
        label=f"Overall: {baseline:.1f}%",
    )

    ax.set_xlim(0, data["rate"].max() + 3.5)
    ax.set_xlabel("Unfavorable Inspection Rate (%)")

    ax.legend(frameon=False, loc="lower right")

    ax.grid(axis="x", color="#DFE5EB")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    save_chart(fig, "eda_02_facility_type.png")


# ---------------------------------------------------------
# 03. Neighborhoods — Diverging Bar Chart
# ---------------------------------------------------------

def neighborhood_chart(df):
    """Compare neighborhood rates with the citywide average."""
    data = calculate_rates(df, "analysis_neighborhood")

    data = data[
        (data["count"] >= 150)
        & (data["analysis_neighborhood"] != "Unknown")
    ].copy()

    baseline = df["target"].mean() * 100

    data["difference"] = data["rate"] - baseline
    data = data.sort_values("difference")

    fig, ax = plt.subplots(figsize=(11, 10))

    fig.subplots_adjust(
        left=0.30,
        right=0.90,
        top=0.81,
        bottom=0.12,
    )

    add_title(
        fig,
        "03",
        "Which Neighborhoods Deviate From the City Average?",
        f"Percentage-point difference from the "
        f"{baseline:.1f}% citywide unfavorable rate"
        "  ·  Minimum 150 inspections",
    )

    colors = [
        CORAL if value >= 0 else TEAL
        for value in data["difference"]
    ]

    bars = ax.barh(
        data["analysis_neighborhood"],
        data["difference"],
        color=colors,
        height=0.65,
    )

    ax.axvline(
        0,
        color=NAVY,
        linewidth=1.5,
    )

    # Direct labels
    for bar, row in zip(bars, data.itertuples()):
        difference = row.difference

        ax.annotate(
            f"{difference:+.1f} pp",
            (
                difference,
                bar.get_y() + bar.get_height() / 2,
            ),
            xytext=(6 if difference >= 0 else -6, 0),
            textcoords="offset points",
            ha="left" if difference >= 0 else "right",
            va="center",
            fontsize=8,
        )

    limit = max(abs(data["difference"])) + 2.5
    ax.set_xlim(-limit, limit)

    ax.set_xlabel(
        "Difference From Citywide Rate (Percentage Points)"
    )

    ax.grid(axis="x", color="#DFE5EB")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    save_chart(fig, "eda_03_neighborhoods.png")


# ---------------------------------------------------------
# 04. Neighborhood × Facility Type — Heatmap
# ---------------------------------------------------------

def heatmap_chart(df):
    """Compare neighborhood and facility-type combinations."""

    top_neighborhoods = (
        df.loc[
            df["analysis_neighborhood"] != "Unknown",
            "analysis_neighborhood",
        ]
        .value_counts()
        .head(12)
        .index
    )

    top_facilities = (
        df.loc[
            df["permit_type"] != "Other",
            "permit_type",
        ]
        .value_counts()
        .head(5)
        .index
    )

    subset = df[
        df["analysis_neighborhood"].isin(top_neighborhoods)
        & df["permit_type"].isin(top_facilities)
    ]

    grouped = subset.groupby(
        ["analysis_neighborhood", "permit_type"]
    )["target"]

    counts = (
        grouped.size()
        .unstack()
        .reindex(
            index=top_neighborhoods,
            columns=top_facilities,
        )
    )

    rates = (
        (grouped.mean() * 100)
        .unstack()
        .reindex(
            index=top_neighborhoods,
            columns=top_facilities,
        )
    )

    # Hide groups with fewer than 25 inspections
    rates = rates.where(counts >= 25)

    fig, ax = plt.subplots(figsize=(12, 8.5))

    fig.subplots_adjust(
        left=0.25,
        right=0.93,
        top=0.79,
        bottom=0.24,
    )

    add_title(
        fig,
        "04",
        "Where Facility Type Meets Geography",
        "Unfavorable rates across neighborhoods and facility types"
        "  ·  Minimum 25 inspections per cell",
    )

    sns.heatmap(
        rates,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        linewidths=1,
        linecolor=BG,
        cbar_kws={
            "label": "Unfavorable Rate (%)",
        },
        ax=ax,
    )

    ax.set_xlabel("Facility Category", labelpad=12)
    ax.set_ylabel("")

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=25,
        ha="right",
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
    )

    fig.text(
        0.25, 0.07,
        "Blank cells contain fewer than 25 inspections.",
        fontsize=9,
        color=GRAY,
    )

    save_chart(fig, "eda_04_heatmap.png")


# ---------------------------------------------------------
# 05. Inspection Frequency — Vertical Bar Chart
# ---------------------------------------------------------

def frequency_chart(df):
    """Compare outcomes by inspection frequency category."""
    data = calculate_rates(df, "inspection_frequency_type")

    data = data[
        (data["count"] >= 30)
        & (
            data["inspection_frequency_type"].astype(str)
            != "Unknown"
        )
    ].sort_values("rate")

    baseline = df["target"].mean() * 100

    fig, ax = plt.subplots(figsize=(9, 6))

    fig.subplots_adjust(
        left=0.13,
        right=0.94,
        top=0.78,
        bottom=0.18,
    )

    add_title(
        fig,
        "05",
        "Does Inspection Frequency Track Outcomes?",
        "Unfavorable inspection rates by recorded frequency category",
    )

    colors = [
        CORAL if rate > baseline else TEAL
        for rate in data["rate"]
    ]

    bars = ax.bar(
        data["inspection_frequency_type"].astype(str),
        data["rate"],
        color=colors,
        width=0.45,
    )

    ax.axhline(
        baseline,
        color=NAVY,
        linestyle="--",
        linewidth=1.2,
        label=f"Overall: {baseline:.1f}%",
    )

    for bar, row in zip(bars, data.itertuples()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{row.rate:.1f}%\nn={row.count:,}",
            ha="center",
            fontsize=10,
        )

    ax.set_ylim(
        0,
        max(data["rate"].max(), baseline) * 1.5,
    )

    ax.set_xlabel("Inspection Frequency Category")
    ax.set_ylabel("Unfavorable Inspection Rate (%)")

    ax.legend(frameon=False)

    ax.grid(axis="y", color="#DFE5EB")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    save_chart(fig, "eda_05_frequency.png")


# ---------------------------------------------------------
# 06. Inspection Type — 100% Stacked Bar Chart
# ---------------------------------------------------------

def inspection_type_chart(df):
    """Compare pass and unfavorable shares by inspection type."""
    data = calculate_rates(df, "inspection_type")

    data = (
        data[data["count"] >= 50]
        .sort_values("rate")
    )

    fig, ax = plt.subplots(figsize=(11, 7))

    fig.subplots_adjust(
        left=0.26,
        right=0.86,
        top=0.75,
        bottom=0.14,
    )

    add_title(
        fig,
        "06",
        "How Does the Outcome Mix Vary by Inspection Type?",
        "Share of Pass vs. Conditional Pass / Closure"
        "  ·  Minimum 50 inspections",
    )

    y = np.arange(len(data))

    ax.barh(
        y,
        100 - data["rate"],
        color=TEAL,
        label="Pass",
        height=0.65,
    )

    ax.barh(
        y,
        data["rate"],
        left=100 - data["rate"],
        color=CORAL,
        label="Unfavorable",
        height=0.65,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(data["inspection_type"])

    for i, row in enumerate(data.itertuples()):
        ax.text(
            101,
            i,
            f"{row.rate:.1f}%  (n={row.count:,})",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel("Percentage of Rated Inspections")

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
    )

    ax.grid(axis="x", color="#DFE5EB")
    ax.set_axisbelow(True)
    ax.tick_params(length=0)

    save_chart(fig, "eda_06_inspection_types.png")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    """Load cleaned data and generate all EDA charts."""
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "target",
        "permit_type",
        "analysis_neighborhood",
        "inspection_frequency_type",
        "inspection_type",
    ]

    missing = set(required_columns) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["target"]).copy()
    df["target"] = df["target"].astype(int)

    if not df["target"].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1.")

    for column in required_columns[1:]:
        df[column] = (
            df[column]
            .fillna("Unknown")
            .astype(str)
        )

    print(f"Loaded {len(df):,} inspections.")
    print(
        f"Overall unfavorable rate: "
        f"{df['target'].mean():.1%}"
    )

    overall_chart(df)
    facility_chart(df)
    neighborhood_chart(df)
    heatmap_chart(df)
    frequency_chart(df)
    inspection_type_chart(df)

    print("\nAll six visualizations generated successfully.")

if __name__ == "__main__":
    main()