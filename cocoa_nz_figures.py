"""Plot the two compiled n(z) reconstructions and their bin-probability changes.

One two-panel figure uses saved reader samples and integrated bin areas.
No new spline fit or cosmology calculation is performed here.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_nz_reconstruction(report_file, arrays_file, output_directory):
    """Save one two-panel figure as nz_reconstruction.png.

    Arguments:
      report_file: JSON measurement record from the project's comparison.
      arrays_file: NPZ probability audit, including dense compiled-reader
        samples around the illustrated interior histogram transition.
      output_directory: folder receiving the PNG.

    The left panel overlays both curves on the same input histogram. The
    right panel compares probability redistribution for every source/lens
    distribution. Total-area errors are tabulated in the project README.
    """
    report = json.loads(Path(report_file).read_text())
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    originals = []
    akimas = {}
    for row in report["probability_audit"]["rows"]:
        identity = (row["sample"], row["tomographic_bin"])
        if row["arm"] == "original":
            originals.append(row)
        else:
            akimas[identity] = row

    # Illustrate the distribution with the largest original redistribution;
    # within it, show the largest interior density jump, not an outer cutoff.
    selected = max(originals, key=lambda row: row["bin_redistribution_pct"])
    sample = selected["sample"]
    tomo = selected["tomographic_bin"] - 1
    colours = ("#bd5124", "#168568")
    names = ("Original midpoint cubic", "Fixed 3× histogram + Akima")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 4.8))
    with np.load(file=arrays_file) as data:
        edges = data[sample + "_original_edges"]
        probabilities = data[sample + "_original_target"][tomo]
        heights = probabilities / np.diff(edges)
        centres = .5 * (edges[:-1] + edges[1:])
        cell = int(np.argmax(np.abs(np.diff(heights)))) + 1
        left = max(0, cell - 2)
        right = min(probabilities.size, cell + 3)
        ax = axes[0]
        ax.stairs(values=heights, edges=edges, fill=True, color="0.6", alpha=.25,
                  label="Normalized input histogram")
        ax.stairs(values=heights, edges=edges, color="0.45", linewidth=.8)
        zoom_values = [heights[left:right]]
        for column, arm in enumerate(("original", "akima")):
            prefix = sample + "_" + arm
            z = data[prefix + "_zoom_z"]
            density = data[prefix + "_zoom_density"]
            ax.plot(z, density, color=colours[column], linewidth=2, label=names[column])
            zoom_values.append(density)
        ax.scatter(centres[left:right], heights[left:right], s=32,
                   facecolor="none", edgecolor="black", zorder=4)
        values = np.concatenate(zoom_values)
        lower = np.min(values)
        upper = np.max(values)
        margin = .08 * max(upper - lower, .01 * np.max(heights))
        ax.set_ylim(lower - margin, upper + margin)
        ax.set_xlim(edges[left], edges[right])
        ax.set_xlabel("Redshift z")
        ax.set_ylabel("Density n(z)")
        ax.set_title(f"Same histogram · {sample} bin {tomo + 1}")
        ax.legend(fontsize=9)
        ax.grid(alpha=.15)

    labels = []
    original_values = []
    akima_values = []
    for row in originals:
        identity = (row["sample"], row["tomographic_bin"])
        labels.append(row["sample"][0].upper() + str(row["tomographic_bin"]))
        original_values.append(row["bin_redistribution_pct"])
        akima_values.append(akimas[identity]["bin_redistribution_pct"])
    x = np.arange(len(labels))
    axes[1].bar(x - .19, original_values, width=.38, color=colours[0], label=names[0])
    axes[1].bar(x + .19, akima_values, width=.38, color=colours[1], label=names[1])
    axes[1].set_title("Probability redistributed among original cells")
    axes[1].set_ylabel("% of total probability")
    axes[1].set_xticks(x, labels, rotation=60, fontsize=8)
    axes[1].set_xlabel("Tomographic distribution: S = source, L = lens")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=.15)
    fig.suptitle(report["project"] + " · two interpretations of n(z)", fontsize=15)
    fig.text(.5, .025, "Compiled readers; zero shift, unit stretch. Hollow circles: original centres.\n"
             "Redistribution = 50 Σ|q/Σq − p|; p: input masses, q: reconstructed cell areas.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .09, 1, .93))
    fig.savefig(output / "nz_reconstruction.png", dpi=180)
    plt.close(fig)
