#!/usr/bin/env python3
"""
Entropy Analysis for Bimanual Coordination Data
================================================

Data visualisation and descriptive statistics for:

    Park, C. (2026). Thermodynamic Entropy Management in Human Motor Control
    Across Circadian and Thermal Challenges. Journal of The Royal Society
    Interface. DOI: 10.1098/rsif.2025.1023

This script loads the experimental data from Tables S1-S3 of the
Supplementary Information, computes descriptive statistics, and generates
individual participant visualisations for all three experiments.

Note: Inferential statistics (ANOVA, interaction tests) were computed in
SPSS Version 27 as reported in the manuscript. This script provides data
presentation, descriptive summaries, and visual inspection of individual
participant patterns.

Repository: https://github.com/pcw8531/thermodynamic-motor-control
Archived:   https://doi.org/10.5281/zenodo.19201270
Licence:    MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import ttest_rel
import warnings

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 10,
    "font.family": "Arial",
    "axes.linewidth": 0.8,
})

COLORS = {
    "entropy": "#2E86AB",
    "temperature": "#FF6B6B",
    "normal": "#2E86AB",
    "heat": "#FF6B6B",
    "cold": "#45B7D1",
}

TIME_POINTS = ["5:00", "12:00", "17:00", "0:00"]
TIME_NUMERIC = [5, 12, 17, 24]


# ---------------------------------------------------------------------------
# Experiment I: Normal circadian entropy data (Table S1a)
# ---------------------------------------------------------------------------

EXP1_ENTROPY = {
    "P1": {"5:00": 3.929, "12:00": 2.882, "17:00": 4.023, "0:00": 5.819},
    "P2": {"5:00": 5.830, "12:00": 5.351, "17:00": 4.041, "0:00": 5.377},
    "P3": {"5:00": 4.194, "12:00": 3.820, "17:00": 3.691, "0:00": 4.728},
    "P4": {"5:00": 5.880, "12:00": 5.888, "17:00": 4.520, "0:00": 3.898},
    "P5": {"5:00": 5.779, "12:00": 5.254, "17:00": 4.431, "0:00": 4.313},
    "P6": {"5:00": 4.912, "12:00": 5.853, "17:00": 5.771, "0:00": 5.798},
    "P7": {"5:00": 5.757, "12:00": 5.625, "17:00": 4.059, "0:00": 5.829},
    "P8": {"5:00": 5.688, "12:00": 5.405, "17:00": 5.815, "0:00": 5.839},
}

# Core body temperatures — group means (Table S1a)
TEMPERATURES = {
    "5:00": 36.607,
    "12:00": 36.834,
    "17:00": 37.023,
    "0:00": 36.681,
}


# ---------------------------------------------------------------------------
# Experiment II: Heat perturbation — normalised Z-scores (Table S2)
# ---------------------------------------------------------------------------

EXP2_HEAT = {
    "P1": {"N_5:00": -0.667, "N_17:00": -0.590, "Ab_5:00":  0.885, "Ab_17:00": -0.994},
    "P2": {"N_5:00":  0.887, "N_17:00": -0.576, "Ab_5:00":  0.903, "Ab_17:00": -0.906},
    "P3": {"N_5:00": -0.451, "N_17:00": -0.862, "Ab_5:00": -0.012, "Ab_17:00": -0.595},
    "P4": {"N_5:00":  0.928, "N_17:00": -0.184, "Ab_5:00":  0.874, "Ab_17:00": -2.334},
    "P5": {"N_5:00":  0.845, "N_17:00": -0.257, "Ab_5:00": -0.780, "Ab_17:00": -0.515},
    "P6": {"N_5:00":  0.137, "N_17:00":  0.839, "Ab_5:00":  0.915, "Ab_17:00": -1.089},
    "P7": {"N_5:00":  0.828, "N_17:00": -0.561, "Ab_5:00":  0.892, "Ab_17:00": -0.202},
    "P8": {"N_5:00":  0.771, "N_17:00":  0.875, "Ab_5:00":  0.834, "Ab_17:00":  0.162},
}


# ---------------------------------------------------------------------------
# Experiment III: Cold perturbation — normalised Z-scores (Table S3)
# ---------------------------------------------------------------------------

EXP3_COLD = {
    "P1": {"N_5:00":  0.387, "N_17:00": -1.906, "Ab_5:00": -0.317, "Ab_17:00": -1.920},
    "P2": {"N_5:00":  0.735, "N_17:00":  0.508, "Ab_5:00": -0.003, "Ab_17:00": -1.853},
    "P3": {"N_5:00":  0.523, "N_17:00": -0.418, "Ab_5:00":  1.050, "Ab_17:00": -2.004},
    "P4": {"N_5:00":  0.979, "N_17:00":  0.865, "Ab_5:00":  0.929, "Ab_17:00":  0.315},
    "P5": {"N_5:00":  0.314, "N_17:00": -0.004, "Ab_5:00":  0.729, "Ab_17:00": -0.537},
    "P6": {"N_5:00":  0.259, "N_17:00":  0.442, "Ab_5:00":  1.057, "Ab_17:00": -0.027},
    "P7": {"N_5:00": -0.535, "N_17:00": -1.505, "Ab_5:00":  0.950, "Ab_17:00": -0.973},
    "P8": {"N_5:00":  0.574, "N_17:00":  0.639, "Ab_5:00":  0.468, "Ab_17:00":  0.278},
}


# ===========================================================================
# Experiment I — Circadian Entropy Analysis
# ===========================================================================

def analyse_experiment_1():
    """Print descriptive statistics and plot individual circadian patterns."""

    df = pd.DataFrame(EXP1_ENTROPY).T.reset_index()
    df.columns = ["Participant", "5:00", "12:00", "17:00", "0:00"]

    print("=" * 60)
    print("Experiment I — Entropy Values H(φ)  [Table S1a]")
    print("=" * 60)
    print(df.to_string(index=False))

    print("\nCore Body Temperatures (°C):")
    for time, temp in TEMPERATURES.items():
        print(f"  {time}: {temp:.3f} °C")

    # --- Individual participant statistics ---
    print("\nIndividual Participant Statistics:")
    print("-" * 60)
    for idx, row in df.iterrows():
        values = [row["5:00"], row["12:00"], row["17:00"], row["0:00"]]
        peak_time = TIME_POINTS[np.argmax(values)]
        min_time = TIME_POINTS[np.argmin(values)]
        print(
            f"  {row['Participant']}: "
            f"Mean = {np.mean(values):.3f} ± {np.std(values):.3f},  "
            f"Peak at {peak_time} ({np.max(values):.3f}),  "
            f"Min at {min_time} ({np.min(values):.3f})"
        )

    # --- Figure: Individual entropy patterns across circadian cycle ---
    fig, axes = plt.subplots(2, 4, figsize=(14, 6))

    for idx, row in df.iterrows():
        ax = axes[idx // 4, idx % 4]
        entropy_values = [row["5:00"], row["12:00"], row["17:00"], row["0:00"]]
        temp_values = [TEMPERATURES[t] for t in TIME_POINTS]

        # Entropy
        ax.plot(
            TIME_NUMERIC, entropy_values, "o-",
            color=COLORS["entropy"], linewidth=2, markersize=8,
            markerfacecolor="white", markeredgewidth=2,
            markeredgecolor=COLORS["entropy"],
        )

        # Temperature (secondary axis)
        ax2 = ax.twinx()
        ax2.plot(
            TIME_NUMERIC, temp_values, "--",
            color=COLORS["temperature"], alpha=0.5, linewidth=1.5,
        )

        ax.set_title(f"Participant {idx + 1}", fontweight="bold")
        ax.set_xlabel("Circadian Time")
        ax.set_ylabel("Entropy H(φ)", color=COLORS["entropy"])
        ax2.set_ylabel("Temperature (°C)", color=COLORS["temperature"])
        ax.set_xticks(TIME_NUMERIC)
        ax.set_xticklabels(["5:00", "12:00", "17:00", "00:00"])
        ax.grid(True, alpha=0.3)
        ax.set_ylim([2, 6.5])
        ax2.set_ylim([36.5, 37.1])
        ax.tick_params(axis="y", labelcolor=COLORS["entropy"])
        ax2.tick_params(axis="y", labelcolor=COLORS["temperature"])

    plt.tight_layout()
    plt.savefig("figure_exp1_individual_patterns.png", dpi=300, bbox_inches="tight")
    plt.show()


# ===========================================================================
# Experiments II & III — Thermal Perturbation Analysis
# ===========================================================================

def analyse_perturbation_experiments():
    """Print descriptive statistics and plot individual perturbation responses."""

    df_heat = pd.DataFrame(EXP2_HEAT).T.reset_index()
    df_heat.columns = ["Participant", "N_5:00", "N_17:00", "Ab_5:00", "Ab_17:00"]

    df_cold = pd.DataFrame(EXP3_COLD).T.reset_index()
    df_cold.columns = ["Participant", "N_5:00", "N_17:00", "Ab_5:00", "Ab_17:00"]

    print("\n" + "=" * 60)
    print("Experiment II — Heat Perturbation (Z-scores)  [Table S2]")
    print("=" * 60)
    print(df_heat.to_string(index=False))

    print("\n" + "=" * 60)
    print("Experiment III — Cold Perturbation (Z-scores)  [Table S3]")
    print("=" * 60)
    print(df_cold.to_string(index=False))

    # --- Figure: Individual responses to thermal perturbations ---
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.suptitle(
        "Individual Entropy Responses to Thermal Perturbations",
        fontsize=14, fontweight="bold",
    )

    x = [5, 17]

    # Top 2 rows: Heat perturbation
    for idx in range(8):
        ax = axes[idx // 4, idx % 4]
        participant = f"P{idx + 1}"

        normal = [df_heat.iloc[idx]["N_5:00"], df_heat.iloc[idx]["N_17:00"]]
        perturbed = [df_heat.iloc[idx]["Ab_5:00"], df_heat.iloc[idx]["Ab_17:00"]]

        ax.plot(
            x, normal, "o-", color=COLORS["normal"], linewidth=2,
            markersize=8, label="Normal", markerfacecolor="white",
            markeredgewidth=2,
        )
        ax.plot(
            x, perturbed, "s-", color=COLORS["heat"], linewidth=2,
            markersize=8, label="Heat", markerfacecolor="white",
            markeredgewidth=2,
        )

        interaction = (perturbed[0] - normal[0]) - (perturbed[1] - normal[1])

        ax.set_title(f"{participant}: Heat\n\u0394 = {interaction:.2f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Circadian Time")
        ax.set_ylabel("Normalised Entropy (Z)")
        ax.set_xticks([5, 17])
        ax.set_xticklabels(["5:00", "17:00"])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim([-3, 2])

    # Bottom 2 rows: Cold perturbation
    for idx in range(8):
        ax = axes[(idx // 4) + 2, idx % 4]
        participant = f"P{idx + 1}"

        normal = [df_cold.iloc[idx]["N_5:00"], df_cold.iloc[idx]["N_17:00"]]
        perturbed = [df_cold.iloc[idx]["Ab_5:00"], df_cold.iloc[idx]["Ab_17:00"]]

        ax.plot(
            x, normal, "o-", color=COLORS["normal"], linewidth=2,
            markersize=8, label="Normal", markerfacecolor="white",
            markeredgewidth=2,
        )
        ax.plot(
            x, perturbed, "^-", color=COLORS["cold"], linewidth=2,
            markersize=8, label="Cold", markerfacecolor="white",
            markeredgewidth=2,
        )

        interaction = (perturbed[0] - normal[0]) - (perturbed[1] - normal[1])

        ax.set_title(f"{participant}: Cold\n\u0394 = {interaction:.2f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Circadian Time")
        ax.set_ylabel("Normalised Entropy (Z)")
        ax.set_xticks([5, 17])
        ax.set_xticklabels(["5:00", "17:00"])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize=8)
        ax.set_ylim([-3, 2])

    plt.tight_layout()
    plt.savefig("figure_exp23_perturbation_responses.png", dpi=300, bbox_inches="tight")
    plt.show()


# ===========================================================================
# Main
# ===========================================================================

if __name__ == "__main__":
    print("Thermodynamic Entropy Management in Human Motor Control")
    print("Park (2026) — J. R. Soc. Interface")
    print("DOI: 10.1098/rsif.2025.1023\n")

    analyse_experiment_1()
    analyse_perturbation_experiments()

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("Note: Inferential statistics (ANOVA, interaction tests) were")
    print("computed in SPSS Version 27. See manuscript Tables 3.1-3.2")
    print("and Supplement S6 for full statistical results.")
    print("=" * 60)
