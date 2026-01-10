#!/usr/bin/env python3
"""
スケーリング則の可視化

run_scaling_experiment.py の結果を読み込み、log-log プロットを生成する。

Usage:
    python plot_scaling.py scaling_results.json
    python plot_scaling.py scaling_results.json --output scaling_plot.png
"""

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np


def plot_scaling_law(results_path: str, output_path: str = None, show: bool = True):
    """スケーリング則をプロット"""
    # Load results
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data["results"]

    # Extract data
    names = [r["name"] for r in results]
    n_params = np.array([r["n_params"] for r in results])
    val_loss = np.array([r["val_loss"] for r in results])

    # Fit power law: L = a * N^(-b)
    # log(L) = log(a) - b * log(N)
    log_params = np.log10(n_params)
    log_loss = np.log10(val_loss)

    # Linear regression in log-log space
    coeffs = np.polyfit(log_params, log_loss, 1)
    slope, intercept = coeffs
    fitted_log_loss = np.polyval(coeffs, log_params)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Log-log plot
    ax1 = axes[0]
    ax1.scatter(n_params, val_loss, s=100, c="blue", zorder=5)
    for i, name in enumerate(names):
        ax1.annotate(
            name,
            (n_params[i], val_loss[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
        )

    # Fitted line
    x_fit = np.logspace(np.log10(n_params.min() * 0.8), np.log10(n_params.max() * 1.2), 100)
    y_fit = 10 ** (slope * np.log10(x_fit) + intercept)
    ax1.plot(x_fit, y_fit, "r--", alpha=0.7, label=f"L = {10**intercept:.2f} × N^({slope:.3f})")

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Parameters (N)", fontsize=12)
    ax1.set_ylabel("Validation Loss (L)", fontsize=12)
    ax1.set_title("Scaling Law: Loss vs Parameters", fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Residuals (to check fit quality)
    ax2 = axes[1]
    residuals = log_loss - fitted_log_loss
    ax2.bar(names, residuals, color="steelblue", alpha=0.7)
    ax2.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Model Size", fontsize=12)
    ax2.set_ylabel("Residual (log scale)", fontsize=12)
    ax2.set_title("Fit Residuals", fontsize=14)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add experiment info
    info_text = (
        f"Data: {data['data_size']:,} chars\n"
        f"Steps: {data['steps']}\n"
        f"Batch: {data['batch_size']}\n"
        f"Slope: {slope:.3f}"
    )
    fig.text(0.02, 0.02, info_text, fontsize=9, family="monospace", va="bottom")

    plt.tight_layout()

    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")

    if show:
        plt.show()

    # Print summary
    print("\n" + "=" * 50)
    print("Scaling Law Analysis")
    print("=" * 50)
    print(f"Fitted equation: L = {10**intercept:.4f} × N^({slope:.4f})")
    print(f"Power law exponent: {-slope:.4f}")
    print(f"\nTheoretical reference (Chinchilla): ~0.076")
    print(f"Your measured exponent: {-slope:.4f}")
    print("\nNote: Exponent depends on data size and training tokens.")
    print("=" * 50)

    return {"slope": slope, "intercept": intercept, "r_squared": 1 - np.var(residuals) / np.var(log_loss)}


def main():
    parser = argparse.ArgumentParser(description="Plot Scaling Law Results")
    parser.add_argument("results", type=str, help="Path to scaling_results.json")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output image path")
    parser.add_argument("--no-show", action="store_true", help="Don't display plot")

    args = parser.parse_args()

    plot_scaling_law(args.results, args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
