#!/usr/bin/env python3
"""
Discrete Relative Phase Extraction and Entropy Quantification
=============================================================

This script demonstrates the data processing pipeline for computing
discrete relative phase and Shannon entropy from raw bimanual
coordination time-series data.

Methodology follows Section 2.5 and 2.6 of:

    Park, C. (2026). Thermodynamic Entropy Management in Human Motor Control
    Across Circadian and Thermal Challenges. Journal of The Royal Society
    Interface. DOI: 10.1098/rsif.2025.1023

Pipeline:
    1. Load raw pendulum position data (100 Hz, 3D)
    2. Reduce to 1D via Principal Component Analysis
    3. Detect oscillation peaks (local maxima)
    4. Compute discrete relative phase (Equation 23)
    5. Quantify Shannon entropy of phase distribution (Equation 24)

Input:  cwa17w1.txt  — sample raw pendulum position time-series
Output: Printed phase statistics and entropy value

Repository: https://github.com/pcw8531/thermodynamic-motor-control
Archived:   https://doi.org/10.5281/zenodo.19201270
Licence:    MIT
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.signal import argrelmax
from scipy.stats import circmean, circstd


# ===========================================================================
# SECTION 1: DATA LOADING
# ===========================================================================

def load_pendulum_data(filename):
    """Load raw pendulum position data from DC potentiometers.

    The data file contains columns:
        hand (1=left, 2=right), x, y, z positions, and 3 additional channels.

    Sampling rate: 100 Hz.

    Parameters
    ----------
    filename : str
        Path to the tab-separated data file.

    Returns
    -------
    left : ndarray of shape (n_samples, 3)
        Left pendulum 3D positions.
    right : ndarray of shape (n_samples, 3)
        Right pendulum 3D positions.
    """
    data = pd.read_csv(
        filename,
        sep=r"\s+",
        names=["hand", "x", "y", "z", "ch4", "ch5", "ch6"],
        engine="python",
        comment="e",  # skip 'end' line if present
    )

    # Separate left (hand=1) and right (hand=2) pendulum data
    left = data.loc[data["hand"] == 1, ["x", "y", "z"]].values.astype(float)
    right = data.loc[data["hand"] == 2, ["x", "y", "z"]].values.astype(float)

    return left, right


# ===========================================================================
# SECTION 2: DIMENSIONALITY REDUCTION
# ===========================================================================

def reduce_to_1d(positions):
    """Reduce 3D pendulum positions to 1D using PCA.

    The first principal component captures the primary axis of oscillation,
    which dominates variance in sagittal-plane constrained movements.

    Parameters
    ----------
    positions : ndarray of shape (n_samples, 3)
        3D positional data from DC potentiometers.

    Returns
    -------
    signal_1d : ndarray of shape (n_samples,)
        One-dimensional projection along the primary oscillation axis.
    explained_variance : float
        Proportion of variance explained by the first component.
    """
    pca = PCA(n_components=1)
    signal_1d = pca.fit_transform(positions).squeeze()
    explained_variance = pca.explained_variance_ratio_[0]

    return signal_1d, explained_variance


# ===========================================================================
# SECTION 3: PEAK DETECTION
# ===========================================================================

def detect_peaks(signal, amplitude_threshold=0.0):
    """Identify oscillation peaks (local maxima) in the 1D signal.

    Parameters
    ----------
    signal : ndarray of shape (n_samples,)
        One-dimensional oscillation signal.
    amplitude_threshold : float
        Minimum amplitude for a peak to be considered valid.
        Helps reject noise-induced false peaks.

    Returns
    -------
    peak_indices : ndarray
        Sample indices of detected peaks.
    """
    peak_indices = argrelmax(signal)[0]

    # Apply amplitude threshold
    if amplitude_threshold > 0:
        peak_indices = peak_indices[signal[peak_indices] > amplitude_threshold]

    return peak_indices


# ===========================================================================
# SECTION 4: DISCRETE RELATIVE PHASE (Equation 23)
# ===========================================================================

def compute_discrete_relative_phase(left_3d, right_3d, amplitude_threshold=1.75):
    """Compute discrete relative phase between left and right pendulums.

    Implements Equation 23 from the manuscript:

        φᵢ = 2π (t_maxL_i - t_maxR_i) / (t_maxL_{i+1} - t_maxL_i)

    where t_maxL_i is the time of the i-th maximum extension of the left
    pendulum. Phase is wrapped to [-π, π].

    Parameters
    ----------
    left_3d : ndarray of shape (n_samples, 3)
        Left pendulum 3D positions.
    right_3d : ndarray of shape (n_samples, 3)
        Right pendulum 3D positions.
    amplitude_threshold : float
        Minimum peak amplitude for valid oscillation detection.

    Returns
    -------
    relative_phase : ndarray
        Discrete relative phase values in radians, wrapped to [-π, π].
    left_peaks : ndarray
        Detected peak indices for the left pendulum.
    right_peaks : ndarray
        Detected peak indices for the right pendulum.
    """
    # Step 1: Reduce to 1D
    left_1d, var_left = reduce_to_1d(left_3d)
    right_1d, var_right = reduce_to_1d(right_3d)

    # Step 2: Detect peaks
    left_peaks = detect_peaks(left_1d, amplitude_threshold)
    right_peaks = detect_peaks(right_1d, amplitude_threshold)

    # Step 3: Ensure equal length
    n = min(len(left_peaks), len(right_peaks))
    left_peaks = left_peaks[:n]
    right_peaks = right_peaks[:n]

    # Step 4: Compute discrete relative phase (Equation 23)
    numerator = left_peaks[:-1] - right_peaks[:-1]
    denominator = np.diff(left_peaks)

    # Avoid division by zero
    valid = denominator > 0
    relative_phase = np.full(len(numerator), np.nan)
    relative_phase[valid] = 2 * np.pi * numerator[valid] / denominator[valid]

    # Remove NaN values
    relative_phase = relative_phase[~np.isnan(relative_phase)]

    # Wrap to [-π, π]
    relative_phase = (relative_phase + np.pi) % (2 * np.pi) - np.pi

    return relative_phase, left_peaks, right_peaks


# ===========================================================================
# SECTION 5: SHANNON ENTROPY (Equation 24)
# ===========================================================================

def compute_shannon_entropy(phase_data, n_bins=20):
    """Compute Shannon information entropy of phase distribution.

    Implements Equation 24 from the manuscript:

        H(φ) = -Σ pᵢ log₂ pᵢ

    where pᵢ is the probability of observing phase states within
    discretised bins of width π/20 rad (9°).

    Parameters
    ----------
    phase_data : ndarray
        Relative phase values in radians.
    n_bins : int
        Number of histogram bins spanning [-π, π].
        Default: 20 (bin width = π/20 rad = 9°).

    Returns
    -------
    entropy : float
        Shannon entropy in bits.
    probabilities : ndarray
        Probability distribution across bins.
    bin_edges : ndarray
        Bin edge positions.
    """
    counts, bin_edges = np.histogram(phase_data, bins=n_bins,
                                     range=(-np.pi, np.pi))

    # Convert to probability distribution
    probabilities = counts / counts.sum()

    # Compute entropy (excluding zero-probability bins)
    nonzero = probabilities > 0
    entropy = -np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero]))

    return entropy, probabilities, bin_edges


# ===========================================================================
# SECTION 6: COORDINATION STABILITY METRICS (Supplement S1.1)
# ===========================================================================

def compute_coordination_metrics(relative_phase, target_phase=0.0):
    """Compute standard coordination stability measures.

    Parameters
    ----------
    relative_phase : ndarray
        Discrete relative phase values in radians.
    target_phase : float
        Intended phase relationship (0.0 for in-phase coordination).

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - mean_phase: circular mean of relative phase (degrees)
        - sd_phase: circular standard deviation (degrees)
        - fixed_point_shift: deviation from target (degrees)
        - n_cycles: number of oscillation cycles analysed
    """
    mean_rad = circmean(relative_phase, low=-np.pi, high=np.pi)
    sd_rad = circstd(relative_phase, low=-np.pi, high=np.pi)

    return {
        "mean_phase_deg": np.rad2deg(mean_rad),
        "sd_phase_deg": np.rad2deg(sd_rad),
        "fixed_point_shift_deg": np.rad2deg(mean_rad - target_phase),
        "n_cycles": len(relative_phase),
    }


# ===========================================================================
# SECTION 7: MAIN EXECUTION
# ===========================================================================

def analyse_trial(filename, amplitude_threshold=1.75):
    """Run complete analysis pipeline on a single trial.

    Parameters
    ----------
    filename : str
        Path to raw data file.
    amplitude_threshold : float
        Peak detection threshold.

    Returns
    -------
    results : dict
        Complete analysis results.
    """
    print(f"Loading data from: {filename}")
    print("-" * 60)

    # Load raw data
    left, right = load_pendulum_data(filename)
    print(f"  Samples loaded: {left.shape[0]} per hand")
    print(f"  Duration: {left.shape[0] / 100:.1f} s at 100 Hz")

    # Dimensionality reduction
    left_1d, var_l = reduce_to_1d(left)
    right_1d, var_r = reduce_to_1d(right)
    print(f"  PCA variance explained: left={var_l:.3f}, right={var_r:.3f}")

    # Phase extraction
    rel_phase, l_peaks, r_peaks = compute_discrete_relative_phase(
        left, right, amplitude_threshold)
    print(f"  Peaks detected: left={len(l_peaks)}, right={len(r_peaks)}")
    print(f"  Phase values extracted: {len(rel_phase)}")

    # Coordination metrics
    metrics = compute_coordination_metrics(rel_phase)
    print(f"\n  Coordination Stability:")
    print(f"    Mean relative phase:  {metrics['mean_phase_deg']:>8.2f}°")
    print(f"    Phase variability:    {metrics['sd_phase_deg']:>8.2f}°")
    print(f"    Fixed-point shift:    {metrics['fixed_point_shift_deg']:>8.2f}°")
    print(f"    Oscillation cycles:   {metrics['n_cycles']:>8d}")

    # Shannon entropy
    entropy, probs, edges = compute_shannon_entropy(rel_phase, n_bins=20)
    print(f"\n  Shannon Entropy:")
    print(f"    H(φ) = {entropy:.3f} bits  (20 bins, width = π/20 rad)")

    # Entropy sensitivity to bin count
    print(f"\n  Bin Size Sensitivity:")
    for n_bins in [10, 15, 20, 25, 30]:
        h, _, _ = compute_shannon_entropy(rel_phase, n_bins=n_bins)
        print(f"    {n_bins:>3} bins: H(φ) = {h:.3f}")

    # Phase distribution summary
    print(f"\n  Phase Distribution:")
    print(f"    Range: [{np.rad2deg(rel_phase.min()):.1f}°, "
          f"{np.rad2deg(rel_phase.max()):.1f}°]")
    in_phase_pct = np.mean(np.abs(np.rad2deg(rel_phase)) < 30) * 100
    print(f"    Within ±30° of target: {in_phase_pct:.1f}%")

    return {
        "relative_phase": rel_phase,
        "entropy": entropy,
        "metrics": metrics,
        "probabilities": probs,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("Discrete Relative Phase and Entropy Analysis")
    print("Park (2026) — J. R. Soc. Interface")
    print("DOI: 10.1098/rsif.2025.1023")
    print("=" * 60)
    print()

    try:
        results = analyse_trial("cwa17w1.txt")
        print("\n" + "=" * 60)
        print("Analysis complete.")
        print("=" * 60)
    except FileNotFoundError:
        print("\nError: cwa17w1.txt not found in the current directory.")
        print("Please run this script from the repository root directory:")
        print("  cd thermodynamic-motor-control")
        print("  python phase_analysis.py")
