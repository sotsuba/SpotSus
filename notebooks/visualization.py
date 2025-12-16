"""
Visualization utilities for fraud detection EDA.

This module provides plotting functions for temporal analysis of fraud patterns.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns
from dataclasses import dataclass

# =============================================================================
# Helper Functions
# =============================================================================


def _get_unique_categories(df: pl.DataFrame, category_col: str) -> list:
    """Get sorted unique non-null categories from a column."""
    return (
        df.filter(pl.col(category_col).is_not_null())[category_col]
        .unique()
        .sort()
        .to_list()
    )


def _set_tick_visibility(ax, tick_interval: int) -> None:
    """Set x-axis tick visibility based on interval."""
    for i, label in enumerate(ax.get_xticklabels()):
        label.set_visible(i % tick_interval == 0)


def _create_day_labels(
    ax,
    time_min: int,
    time_max: int,
    tick_interval: int,
    rotation: int = 45,
    fontsize: int = 8,
) -> None:
    """Set x-axis ticks with day index labels (D0, D15, D30, ...)."""
    tick_positions = list(range(int(time_min), int(time_max) + 1, tick_interval))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(
        [f"{t}" for t in tick_positions],
        rotation=rotation,
        ha="right",
        fontsize=fontsize,
    )


def _compute_temporal_stats(
    df: pl.DataFrame,
    time_col: str,
    target: str = "isFraud",
    include_amount: bool = False,
) -> pl.DataFrame:
    """
    Compute aggregated temporal statistics.

    Args:
        df: Input DataFrame
        time_col: Time column to group by
        target: Target column name
        include_amount: Whether to include average transaction amount

    Returns:
        Aggregated DataFrame with columns: time_col, total_transactions,
        fraud_count, fraud_rate, and optionally avg_transaction_amount
    """
    aggs = [
        pl.count().alias("total_transactions"),
        pl.col(target).sum().alias("fraud_count"),
        pl.col(target).mean().alias("fraud_rate"),
    ]

    if include_amount and "TransactionAmt" in df.columns:
        aggs.append(pl.col("TransactionAmt").mean().alias("avg_transaction_amount"))

    return df.group_by(time_col).agg(*aggs).sort(time_col)


# =============================================================================
# FFT Analysis
# =============================================================================


def analyze_fft(
    df: pl.DataFrame,
    target_col: str = "isFraud",
    time_col: str = "hour_step",
    max_period: int = 200,
    figsize: tuple = (12, 6),
    return_fig: bool = False,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Visualize FFT power spectrum to detect cyclical patterns in fraud data.

    Uses Fast Fourier Transform to identify periodic patterns (daily, weekly)
    in fraudulent transactions.

    Args:
        df: Polars DataFrame with time and target columns
        target_col: Target column name (default: 'isFraud')
        time_col: Time column for grouping (default: 'hour_step')
        max_period: Maximum period (hours) to display on x-axis
        figsize: Figure size tuple
    """
    # Convert to pandas for groupby operation
    pdf = df.to_pandas()

    # Count fraud cases per time unit
    signal = pdf[pdf[target_col] == 1].groupby(time_col)["TransactionID"].count()

    # Fill missing time steps with 0 (FFT requires constant time steps)
    full_idx = np.arange(signal.index.min(), signal.index.max() + 1)
    signal = signal.reindex(full_idx, fill_value=0)

    # Compute FFT power spectrum
    power = np.abs(np.fft.fft(signal.values)) ** 2
    freqs = np.fft.fftfreq(len(signal), d=1)
    periods = 1 / freqs

    # Visualize
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(periods, power, color="black", linewidth=0.8)
    ax.set_title(f"FFT Power Spectrum of {target_col}", fontsize=14)
    ax.set_xlabel("Cycle Period (Hours)")
    ax.set_ylabel("Power (Strength of Cycle)")
    ax.set_xlim(0, max_period)

    # Add markers for expected cycles
    ax.axvline(24, color="r", linestyle="--", label="24h (Daily)", alpha=0.7)
    ax.axvline(24 * 7, color="g", linestyle="--", label="168h (Weekly)", alpha=0.7)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()

    if return_fig:
        return fig, ax
    plt.show()


# =============================================================================
# Main Plotting Functions
# =============================================================================


def plot_temporal_stats(
    stats: pl.DataFrame,
    x: str,
    tick_interval: int,
    label: str,
    y_lim: float,
    plot_value: bool = False,
    figsize: tuple = (20, 16),
    return_fig: bool = False,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot total transactions, fraud rate, and fraud count over time.

    Creates a 2-row figure:
    - Row 1: Transaction volume (bars) with fraud rate overlay (line)
    - Row 2: Fraud count (bars)

    Args:
        stats: DataFrame with columns: x, total_transactions, fraud_count, fraud_rate
        x: Column name for x-axis (time column)
        tick_interval: Interval for x-axis tick visibility
        label: Label for time unit (e.g., "Day", "Month")
        y_lim: Upper limit for fraud rate y-axis
        plot_value: Whether to display bar values as labels
        figsize: Figure size tuple
    """
    _, axs = plt.subplots(2, 1, figsize=figsize)

    # Row 1: Transaction volume + Fraud rate
    sns.barplot(
        data=stats, x=x, y="total_transactions", ax=axs[0], color="skyblue", alpha=0.5
    )
    axs[0].set_title(f"Total Transactions per {label}", fontsize=16)
    axs[0].set_xlabel(f"{label}s since Reference Date")
    axs[0].set_ylabel("Transaction Volume")
    axs[0].tick_params(axis="y", labelcolor="steelblue")

    # Fraud rate on secondary axis
    ax_rate = axs[0].twinx()
    sns.lineplot(data=stats, x=x, y="fraud_rate", ax=ax_rate, color="red", marker="o")
    ax_rate.set_ylabel("Fraud Rate", color="red")
    ax_rate.tick_params(axis="y", labelcolor="red")
    ax_rate.set_ylim(0, y_lim)

    _set_tick_visibility(axs[0], tick_interval)

    # Row 2: Fraud count
    axs[1].set_title(f"Fraud Count per {label}", fontsize=16)
    sns.barplot(data=stats, x=x, y="fraud_count", ax=axs[1], color="gold")
    axs[1].set_xlabel(f"{label}s since Reference Date")
    axs[1].set_ylabel("Fraud Count")
    axs[1].tick_params(axis="y", labelcolor="darkorange")

    _set_tick_visibility(axs[1], tick_interval)

    # Add value labels if requested
    if plot_value:
        for ax in axs:
            if ax.containers:
                ax.bar_label(
                    ax.containers[0],
                    fmt="%d",
                    padding=3,
                    color="black",
                    fontsize=12,
                )

    if return_fig:
        return plt.gcf(), axs
    plt.tight_layout()
    plt.show()


def plot_temporal_by_category(
    df: pl.DataFrame,
    time_col: str,
    category_col: str,
    target: str = "isFraud",
    tick_interval: int = 10,
    figsize: tuple = (20, 12),
    log_scale: bool = False,
    show_volume: bool = True,
    return_fig: bool = False,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Plot fraud rate over time for each category value in separate subplots.

    Each subplot shows:
    - Primary y-axis: Fraud rate (line)
    - Secondary y-axis: Transaction volume (bars, if show_volume=True)
    - Reference line: Global fraud rate

    Args:
        df: Polars DataFrame
        time_col: Time column name (e.g., 'day_step', 'month_step')
        category_col: Categorical column to split by (e.g., 'card4')
        target: Target column name
        tick_interval: Interval for x-axis tick visibility
        figsize: Figure size
        log_scale: Use log scale for volume y-axis
        show_volume: Show transaction volume as bar chart
    """
    categories = _get_unique_categories(df, category_col)
    n_categories = len(categories)

    # Create subplot grid (no shared y-axis for different scales)
    n_cols = 2
    n_rows = (n_categories + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    global_fraud_rate = df[target].mean()
    time_min, time_max = df[time_col].min(), df[time_col].max()
    colors = sns.color_palette("husl", n_categories)

    for idx, cat in enumerate(categories):
        ax = axes[idx]
        cat_df = df.filter(pl.col(category_col) == cat)
        stats = _compute_temporal_stats(cat_df, time_col, target).to_pandas()

        # Volume bars (secondary axis)
        if show_volume:
            ax2 = ax.twinx()
            ax2.bar(
                stats[time_col],
                stats["total_transactions"],
                color=colors[idx],
                alpha=0.2,
                label="Volume",
            )
            ax2.set_ylabel("Transaction Volume", color="gray", fontsize=9)
            ax2.tick_params(axis="y", labelcolor="gray")
            if log_scale:
                ax2.set_yscale("log")

        # Fraud rate line (primary axis)
        ax.plot(
            stats[time_col],
            stats["fraud_rate"],
            color=colors[idx],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{cat}",
        )

        # Global reference line
        ax.axhline(
            y=global_fraud_rate,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Global: {global_fraud_rate:.3f}",
        )

        # Formatting
        ax.set_title(f"{category_col} = {cat} (n={cat_df.height:,})", fontsize=12)
        ax.set_xlabel(f"{time_col} (Day Index)")
        ax.set_ylabel("Fraud Rate")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(time_min, time_max)

        _create_day_labels(ax, time_min, time_max, tick_interval)

    # Hide unused subplots
    for idx in range(n_categories, len(axes)):
        axes[idx].set_visible(False)

    if return_fig:
        return plt.gcf(), axes
    plt.suptitle(f"Fraud Rate over {time_col} by {category_col}", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_temporal_overlay(
    df: pl.DataFrame,
    time_col: str,
    category_col: str,
    target: str = "isFraud",
    tick_interval: int = 10,
    figsize: tuple = (16, 8),
    log_scale: bool = False,
    return_fig: bool = False,
) -> Optional[Tuple[plt.Figure, plt.Axes]]:
    """
    Overlay all categories on a single plot for comparison.

    Args:
        df: Polars DataFrame
        time_col: Time column name
        category_col: Categorical column to split by
        target: Target column name
        tick_interval: Interval for x-axis tick visibility
        figsize: Figure size
        log_scale: Use log scale for y-axis
    """
    categories = _get_unique_categories(df, category_col)

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("husl", len(categories))

    global_fraud_rate = df[target].mean()
    time_min, time_max = df[time_col].min(), df[time_col].max()

    for idx, cat in enumerate(categories):
        cat_df = df.filter(pl.col(category_col) == cat)
        stats = (
            cat_df.group_by(time_col)
            .agg(pl.col(target).mean().alias("fraud_rate"))
            .sort(time_col)
            .to_pandas()
        )

        ax.plot(
            stats[time_col],
            stats["fraud_rate"],
            color=colors[idx],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=f"{cat} (n={cat_df.height:,})",
        )

    # Global reference line
    ax.axhline(
        y=global_fraud_rate,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Global: {global_fraud_rate:.3f}",
    )

    if log_scale:
        ax.set_yscale("log")

    ax.set_title(f"Fraud Rate over {time_col} by {category_col}", fontsize=14)
    ax.set_xlabel(f"{time_col} (Day Index)")
    ax.set_ylabel("Fraud Rate" + (" (log)" if log_scale else ""))
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    _create_day_labels(ax, time_min, time_max, tick_interval)

    if return_fig:
        return plt.gcf(), ax
    plt.tight_layout()
    plt.show()


def plot_heatmap_info(
    df: pl.DataFrame, on: str, index: str, values: str, return_fig: bool = False
):
    """Plot mean, count and sum pivot heatmaps for `values` grouped by `index` and `on`.

    Args:
        df: Polars DataFrame
        on: Column name to pivot into columns
        index: Column name to use as rows
        values: Column to aggregate (e.g., target)
    """
    common_args = {"on": on, "index": index, "values": values}
    heatmap_data_mean = (
        df.pivot(**common_args, aggregate_function="mean").to_pandas().set_index(index)
    )
    heatmap_data_count = (
        df.pivot(**common_args, aggregate_function="count").to_pandas().set_index(index)
    )
    heatmap_data_sum = (
        df.pivot(**common_args, aggregate_function="sum").to_pandas().set_index(index)
    )

    _, axs = plt.subplots(1, 3, figsize=(20, 7))
    common_plot_args = {"annot": True, "cmap": "coolwarm", "linewidths": 0.5}

    sns.heatmap(heatmap_data_mean, ax=axs[0], fmt=".2f", **common_plot_args)
    axs[0].set_title(f"Mean {values}: {index} vs {on}")

    sns.heatmap(heatmap_data_count, ax=axs[1], fmt=".0f", **common_plot_args)
    axs[1].set_title(f"Total Transaction Count: {index} vs {on}")

    sns.heatmap(heatmap_data_sum, ax=axs[2], fmt=".0f", **common_plot_args)
    axs[2].set_title(f"Sum {values}: {index} vs {on}")

    if return_fig:
        return fig, axs
    plt.tight_layout()
    plt.show()


@dataclass
class PlotConfig:
    """Holder for common plotting options.

    Using a dataclass makes function signatures cleaner and easier to pass around
    in testable ways.
    """

    tick_interval: int = 10
    figsize: tuple[int, int] = (20, 12)
    log_scale: bool = False
    show_volume: bool = True


__all__ = [
    "PlotConfig",
    "analyze_fft",
    "plot_temporal_stats",
    "plot_temporal_by_category",
    "plot_temporal_overlay",
    "plot_heatmap_info",
]
