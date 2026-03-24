#!/usr/bin/env python3
"""
Bimanual Coordination Phase Analysis
=====================================

Discrete relative phase extraction and Shannon entropy computation
from raw pendulum position time-series data.

Methodology follows Sections 2.5-2.6 of:

    Park, C. (2026). Thermodynamic Entropy Management in Human Motor Control
    Across Circadian and Thermal Challenges. Journal of The Royal Society
    Interface. DOI: 10.1098/rsif.2025.1023

Input:  cwa17w1.txt — sample raw pendulum position data from DC potentiometers
Output: Discrete relative phase, circular statistics, Shannon entropy

Note: Statistical analyses reported in the manuscript were computed in
SPSS Version 27. This script demonstrates the data processing pipeline
for phase extraction and entropy quantification.

Repository: https://github.com/pcw8531/thermodynamic-motor-control
Archived:   https://doi.org/10.5281/zenodo.19201270
Licence:    MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.signal import argrelmax
from scipy.stats import circmean, circstd

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 10,
    "font.family": "Arial",
    "axes.linewidth": 0.8,
})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filename):
    """Load raw pendulum position data and separate left/right hands.

    Parameters
    ----------
    filename : str
        Path to tab-separated data file. Columns: hand (1=left, 2=right),
        x, y, z positions, and 3 additional channels. Sampling rate: 100 Hz.

    Returns
    -------
    left : ndarray of shape (n_samples, 3)
    right : ndarray of shape (n_samples, 3)
    """
    data = pd.read_csv(
        filename, sep=r"\s+", engine="python",
        names=["hand", "x", "y", "z", "ch4", "ch5", "ch6"],
        comment="e",
    )
    left = data.loc[data["hand"] == 1, ["x", "y", "z"]].values.astype(float)
    right = data.loc[data["hand"] == 2, ["x", "y", "z"]].values.astype(float)
    return left, right


# ---------------------------------------------------------------------------
# Dimensionality reduction
# ---------------------------------------------------------------------------

def collapse(data):
    """Reduce 3D pendulum positions to 1D via PCA (first component)."""
    return np.squeeze(PCA(n_components=1).fit_transform(data))


# ---------------------------------------------------------------------------
# Peak detection
# ---------------------------------------------------------------------------

def find_peaks(vector, cutoff=0):
    """Find local maxima above a given amplitude threshold.

    Parameters
    ----------
    vector : ndarray
        One-dimensional oscillation signal.
    cutoff : float
        Minimum peak amplitude.

    Returns
    -------
    peak_idxs : ndarray
        Sample indices of detected peaks.
    """
    peak_idxs = argrelmax(vector)[0]
    return peak_idxs[vector[peak_idxs] > cutoff]


# ---------------------------------------------------------------------------
# Discrete relative phase (Equation 23 in manuscript)
# ---------------------------------------------------------------------------

def wrap(vector):
    """Wrap phase values to [-π, π]."""
    return ((vector + np.pi) % (2 * np.pi)) - np.pi


def discrete_relative_phase(left, right):
    """Compute discrete relative phase between left and right pendulums.

    Implements Equation 23:
        φ_i = 2π (t_maxL_i - t_maxR_i) / (t_maxL_{i+1} - t_maxL_i)

    Parameters
    ----------
    left : ndarray of shape (n_samples, 3)
    right : ndarray of shape (n_samples, 3)

    Returns
    -------
    rel_phase : ndarray
        Discrete relative phase values in radians, wrapped to [-π, π].
    """
    left_peaks = find_peaks(collapse(left))
    right_peaks = find_peaks(collapse(right))

    # Ensure both vectors are the same length
    min_length = min(left_peaks.shape[0], right_peaks.shape[0])
    left_peaks = left_peaks[:min_length]
    right_peaks = right_peaks[:min_length]

    return wrap(
        2 * np.pi * (left_peaks - right_peaks)[:-1] / np.diff(left_peaks)
    )


# ---------------------------------------------------------------------------
# Shannon entropy (Equation 24 in manuscript)
# ---------------------------------------------------------------------------

def entropy(data, **kwargs):
    """Compute Shannon entropy of a distribution.

    H(φ) = -Σ p_i log(p_i)

    Parameters
    ----------
    data : ndarray
        Phase values.
    **kwargs
        Additional arguments passed to np.histogram (e.g., bins=20).

    Returns
    -------
    H : float
        Shannon entropy value.
    """
    heights, _ = np.histogram(data, density=True, **kwargs)
    heights = heights[np.nonzero(heights)]
    return -(heights * np.log(heights)).sum(axis=0)


# ===========================================================================
# Main analysis
# ===========================================================================

if __name__ == "__main__":
    print("Bimanual Coordination Phase Analysis")
    print("Park (2026) — J. R. Soc. Interface")
    print("DOI: 10.1098/rsif.2025.1023\n")
    print("=" * 50)

    # --- Load data ---
    filename = "cwa17w1.txt"
    print(f"Loading: {filename}")
    left, right = load_data(filename)
    print(f"  Left hand samples:  {left.shape[0]}")
    print(f"  Right hand samples: {right.shape[0]}")
    print(f"  Duration: {left.shape[0] / 100:.1f} s at 100 Hz")

    # --- Reduce to 1D ---
    left_1d = collapse(left)
    right_1d = collapse(right)
    t = np.arange(left.shape[0])

    # --- Plot raw 3D data ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.suptitle("Raw Pendulum Oscillations (3D)", fontweight="bold")
    labels = ["x", "y", "z"]
    for dim in range(3):
        axes[dim].plot(t, left[:, dim], label="Left")
        axes[dim].plot(t, right[:, dim], label="Right")
        axes[dim].set_ylabel(f"{labels[dim]} position")
        axes[dim].legend(loc="upper right", fontsize=8)
        axes[dim].grid(True, alpha=0.3)
    axes[2].set_xlabel("Sample")
    plt.tight_layout()
    plt.savefig("figure_raw_3d_oscillations.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --- Plot 1D collapsed data with peaks ---
    left_peaks = find_peaks(left_1d)
    right_peaks = find_peaks(right_1d)

    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    fig.suptitle("PCA-Reduced Oscillations with Detected Peaks", fontweight="bold")
    axes[0].plot(t, left_1d, color="#2E86AB")
    axes[0].plot(left_peaks, left_1d[left_peaks], ".", color="#FF6B6B", markersize=6)
    axes[0].set_ylabel("Left pendulum")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t, right_1d, color="#2E86AB")
    axes[1].plot(right_peaks, right_1d[right_peaks], ".", color="#FF6B6B", markersize=6)
    axes[1].set_ylabel("Right pendulum")
    axes[1].set_xlabel("Sample")
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure_pca_with_peaks.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --- Compute discrete relative phase ---
    rel_phase = discrete_relative_phase(left, right)
    print(f"\n  Oscillation cycles detected: {len(rel_phase)}")

    # --- Plot relative phase time series ---
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.rad2deg(rel_phase), ".", color="#2E86AB", markersize=5)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Oscillation Cycle")
    ax.set_ylabel("Relative Phase (°)")
    ax.set_title("Discrete Relative Phase", fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure_relative_phase.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --- Circular statistics ---
    mean_deg = np.rad2deg(circmean(rel_phase))
    sd_deg = np.rad2deg(circstd(rel_phase))
    print(f"\n  Circular mean: {mean_deg:.2f}°")
    print(f"  Circular SD:   {sd_deg:.2f}°")

    # --- Phase distribution histogram ---
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.rad2deg(rel_phase), bins=20, color="#2E86AB", edgecolor="white",
            alpha=0.8)
    ax.axvline(x=0, color="#FF6B6B", linestyle="--", linewidth=2, label="Target (0°)")
    ax.set_xlabel("Relative Phase (°)")
    ax.set_ylabel("Frequency")
    ax.set_title("Phase Distribution", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("figure_phase_distribution.png", dpi=300, bbox_inches="tight")
    plt.show()

    # --- Shannon entropy ---
    H = entropy(rel_phase, bins=20)
    print(f"\n  Shannon entropy H(φ) = {H:.3f}  (20 bins)")

    print("\n" + "=" * 50)
    print("Phase analysis complete.")
    print("=" * 50)
