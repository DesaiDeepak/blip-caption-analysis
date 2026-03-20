"""
Visualize ablation experiment results.

Reads ablation_results.csv (produced by run_ablation.py) and generates
a set of publication-ready plots saved to  ablation_plots/ .

Usage:
    python visualize_ablation.py
    python visualize_ablation.py --csv ablation_results.csv --outdir ablation_plots
"""

import os
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---- Constants -------------------------------------------------------------

METRICS = ["bleu", "meteor", "rouge_l", "cider"]
METRIC_LABELS = {
    "bleu": "BLEU",
    "meteor": "METEOR",
    "rouge_l": "ROUGE-L",
    "cider": "CIDEr",
    "loss": "Loss",
}
COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
})


# ---- Plotting helpers ------------------------------------------------------

def plot_metric_vs_lr(df: pd.DataFrame, outdir: str):
    """Bar chart: each metric grouped by learning rate, one bar per epoch count."""
    epoch_vals = sorted(df["epochs"].unique())
    lr_vals = sorted(df["lr"].unique())

    # Average across data_fraction for a cleaner picture
    grouped = df.groupby(["lr", "epochs"])[METRICS].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        x = np.arange(len(lr_vals))
        width = 0.8 / len(epoch_vals)

        for i, ep in enumerate(epoch_vals):
            subset = grouped[grouped["epochs"] == ep]
            vals = [subset[subset["lr"] == lr][metric].values[0]
                    if lr in subset["lr"].values else 0
                    for lr in lr_vals]
            ax.bar(x + i * width, vals, width, label=f"{ep} epochs",
                   color=COLORS[i % len(COLORS)], edgecolor="white")

        ax.set_xticks(x + width * (len(epoch_vals) - 1) / 2)
        ax.set_xticklabels([f"{lr:.0e}" for lr in lr_vals])
        ax.set_xlabel("Learning Rate")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} vs Learning Rate")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Metrics vs Learning Rate (averaged over data fractions)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, "metrics_vs_lr.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_metric_vs_epochs(df: pd.DataFrame, outdir: str):
    """Line plot: metric on y-axis, epochs on x-axis, one line per LR."""
    grouped = df.groupby(["lr", "epochs"])[METRICS].mean().reset_index()
    lr_vals = sorted(df["lr"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        for i, lr in enumerate(lr_vals):
            sub = grouped[grouped["lr"] == lr].sort_values("epochs")
            ax.plot(sub["epochs"], sub[metric], marker="o",
                    label=f"lr={lr:.0e}", color=COLORS[i % len(COLORS)],
                    linewidth=2, markersize=6)

        ax.set_xlabel("Epochs")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} vs Epochs")
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle("Metrics vs Epochs (averaged over data fractions)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, "metrics_vs_epochs.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_metric_vs_data_fraction(df: pd.DataFrame, outdir: str):
    """Line plot: metric vs data fraction, one line per LR."""
    grouped = df.groupby(["lr", "data_fraction"])[METRICS].mean().reset_index()
    lr_vals = sorted(df["lr"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        for i, lr in enumerate(lr_vals):
            sub = grouped[grouped["lr"] == lr].sort_values("data_fraction")
            ax.plot(sub["data_fraction"] * 100, sub[metric], marker="s",
                    label=f"lr={lr:.0e}", color=COLORS[i % len(COLORS)],
                    linewidth=2, markersize=6)

        ax.set_xlabel("Data Fraction (%)")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} vs Data Fraction")
        ax.legend()
        ax.grid(alpha=0.3)

    fig.suptitle("Metrics vs Data Fraction (averaged over epoch counts)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(outdir, "metrics_vs_data_fraction.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_loss_curve(df: pd.DataFrame, outdir: str):
    """Line plot: loss vs epochs, one line per LR."""
    grouped = df.groupby(["lr", "epochs"])["loss"].mean().reset_index()
    lr_vals = sorted(df["lr"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, lr in enumerate(lr_vals):
        sub = grouped[grouped["lr"] == lr].sort_values("epochs")
        ax.plot(sub["epochs"], sub["loss"], marker="o",
                label=f"lr={lr:.0e}", color=COLORS[i % len(COLORS)],
                linewidth=2, markersize=7)

    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Evaluation Loss vs Epochs", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    path = os.path.join(outdir, "loss_vs_epochs.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_heatmap(df: pd.DataFrame, outdir: str):
    """Heatmap: BLEU score for each (LR, epochs) combination at 100% data."""
    full_data = df[df["data_fraction"] == df["data_fraction"].max()]
    if full_data.empty:
        full_data = df  # fallback: use all data

    pivot = full_data.pivot_table(
        values="bleu", index="lr", columns="epochs", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{lr:.0e}" for lr in pivot.index])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Learning Rate")
    ax.set_title("BLEU Score Heatmap (LR × Epochs)", fontsize=14,
                 fontweight="bold")

    # Annotate cells with values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = "white" if val > pivot.values.mean() else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="BLEU")
    fig.tight_layout()
    path = os.path.join(outdir, "bleu_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")


def plot_radar(df: pd.DataFrame, outdir: str):
    """Radar (spider) chart comparing the best config per LR across metrics."""
    # Pick the best row per LR (highest BLEU)
    best_per_lr = df.loc[df.groupby("lr")["bleu"].idxmax()]

    categories = [METRIC_LABELS[m] for m in METRICS]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for i, (_, row) in enumerate(best_per_lr.iterrows()):
        values = [row[m] for m in METRICS]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2,
                label=f"lr={row['lr']:.0e} (ep={int(row['epochs'])})",
                color=COLORS[i % len(COLORS)])
        ax.fill(angles, values, alpha=0.1, color=COLORS[i % len(COLORS)])

    ax.set_thetagrids(np.degrees(angles[:-1]), categories)
    ax.set_title("Best Config per LR — Metric Radar", fontsize=14,
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    path = os.path.join(outdir, "metric_radar.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")


def print_summary_table(df: pd.DataFrame):
    """Print a readable summary to the console."""
    print("\n" + "=" * 70)
    print("ABLATION RESULTS SUMMARY")
    print("=" * 70)

    # Best overall by each metric
    for m in METRICS:
        best = df.loc[df[m].idxmax()]
        print(f"\n  Best {METRIC_LABELS[m]:8s}: {best[m]:.4f}  "
              f"(lr={best['lr']:.0e}, epochs={int(best['epochs'])}, "
              f"data={best['data_fraction']*100:.0f}%)")

    # Lowest loss
    best_loss = df.loc[df["loss"].idxmin()]
    print(f"\n  Lowest Loss : {best_loss['loss']:.4f}  "
          f"(lr={best_loss['lr']:.0e}, epochs={int(best_loss['epochs'])}, "
          f"data={best_loss['data_fraction']*100:.0f}%)")

    print("\n" + "=" * 70)


# ---- Main ------------------------------------------------------------------

def main(csv_path: str, outdir: str):
    if not os.path.exists(csv_path):
        print(f"Error: '{csv_path}' not found. Run run_ablation.py first.")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} experiment rows from {csv_path}")

    os.makedirs(outdir, exist_ok=True)

    print("\nGenerating plots …")
    plot_metric_vs_lr(df, outdir)
    plot_metric_vs_epochs(df, outdir)
    plot_metric_vs_data_fraction(df, outdir)
    plot_loss_curve(df, outdir)
    plot_heatmap(df, outdir)
    plot_radar(df, outdir)

    print_summary_table(df)
    print(f"\nAll plots saved to  {outdir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize ablation experiment results")
    parser.add_argument("--csv", default="ablation_results.csv",
                        help="Path to the ablation CSV file")
    parser.add_argument("--outdir", default="ablation_plots",
                        help="Directory to save the generated plots")
    args = parser.parse_args()
    main(args.csv, args.outdir)
