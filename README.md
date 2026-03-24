# Thermodynamic Entropy Management in Human Motor Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19201270.svg)](https://doi.org/10.5281/zenodo.19201270)

## Overview

This repository contains data and analysis code for the study:

**Park, C. (2026). Thermodynamic Entropy Management in Human Motor Control Across Circadian and Thermal Challenges. *Journal of The Royal Society Interface*. DOI: [10.1098/rsif.2025.1023](https://doi.org/10.1098/rsif.2025.1023)**

The study provides preliminary evidence that human motor control operates according to thermodynamic principles, managing entropy distribution to maintain functional stability—analogous to how refrigerators achieve local cooling through global heating.

## Repository Structure

```
/thermodynamic-motor-control/
├── README.md                        # This file
├── LICENSE                          # MIT license
├── Entropy Analysis.ipynb           # Complete analysis with embedded data
├── cwa17w1.ipynb                    # Bimanual coordination phase calculations
├── cwa17w1.txt                      # Sample raw pendulum position data
├── experimental_data_complete.csv   # Complete experimental dataset (Tables S1-S3)
├── requirements.txt                 # Python package dependencies
└── .gitignore                       # Standard Python gitignore
```

## Data Files

### Primary Analysis Notebook
**Entropy Analysis.ipynb** contains the complete analysis pipeline with embedded data:
- Shannon entropy calculations: H(φ) = -Σ pᵢ log₂ pᵢ
- Individual participant visualisations across circadian cycles
- Thermal perturbation response patterns
- Statistical comparisons and interaction effects

### Experimental Data
**experimental_data_complete.csv** provides the complete dataset from all three experiments (576 total observations from 16 participants):

| Column | Description |
|--------|-------------|
| `Experiment` | Experiment identifier (Exp1_Circadian, Exp2_Heat, Exp3_Cold) |
| `Participant` | Participant ID (P1-P8) |
| `Group` | Experimental group (1 = circadian only, 2 = thermal perturbation) |
| `Circadian_Time` | Time of measurement (05:00, 12:00, 17:00, 00:00) |
| `Condition` | Temperature condition (Normal, Heat, Cold) |
| `Entropy_H_phi` | Shannon entropy H(φ) or normalised Z-score |
| `Core_Temp_Celsius` | Core body temperature (°C) - Exp1 only |
| `Note` | Data source reference (Table S1, S2, or S3) |

**Data Summary:**
- **Experiment 1 (Circadian):** 8 participants × 4 time points × 6 trials = 192 observations with entropy and temperature
- **Experiment 2 (Heat):** 8 participants × 2 times × 2 conditions × 6 trials = 192 observations (Z-scores)
- **Experiment 3 (Cold):** 8 participants × 2 times × 2 conditions × 6 trials = 192 observations (Z-scores)

### Raw Movement Data
**cwa17w1.txt** contains sample raw pendulum position time-series:
- Sampling rate: 100 Hz
- Variables: Left/right pendulum angular positions (radians)
- Example trial demonstrating in-phase bimanual coordination at 1.21s period

## Key Findings

| Measure | Value |
|---------|-------|
| Circadian temperature range | 36.575°C (05:00) to 37.023°C (17:00) |
| Peak entropy (05:00) | H(φ) = 5.246 |
| Minimum entropy (17:00) | H(φ) = 4.544 |
| Circadian entropy modulation | 15.4% increase at temperature minimum |
| Heat perturbation amplification | 139% |
| Cold perturbation amplification | 151% (19.8% stronger than heat) |
| Temperature-entropy correlation | r = -0.678, p < 0.001 |
| Achieved statistical power | 0.82-0.89 across experiments |

## Reproducibility

To reproduce all analyses and figures:

```bash
# Clone the repository
git clone https://github.com/pcw8531/thermodynamic-motor-control.git
cd thermodynamic-motor-control

# Install dependencies
pip install -r requirements.txt

# Open Jupyter Notebook
jupyter notebook "Entropy Analysis.ipynb"
```

### Requirements
- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- SciPy
- Jupyter Notebook

## Experimental Protocol

### Participants
- N = 16 healthy adults (two groups of 8)
- Group 1: Circadian baseline (Experiment 1)
- Group 2: Thermal perturbation (Experiments 2 & 3)
- Total observations: 576 (16 participants × 6 trials × 6 conditions)

### Task
Bimanual pendulum coordination maintaining in-phase (φ = 0°) pattern at 1.21s period

### Conditions
- **Normal:** Standard laboratory conditions
- **Heat Perturbation:** 30-min heat vest application (42-45°C surface)
- **Cold Perturbation:** 30-min ice vest application (5-10°C surface)

### Measurements
- Core body temperature: HuBDIC HFS-100 infrared thermometer
- Movement: DC potentiometers (0.25° resolution) at 100 Hz
- Entropy: Shannon entropy of phase distribution (π/20 radian bins)

## Citation

If you use this data or code, please cite:

```bibtex
@article{park2026thermodynamic,
  title={Thermodynamic Entropy Management in Human Motor Control Across Circadian and Thermal Challenges},
  author={Park, Chulwook},
  journal={Journal of The Royal Society Interface},
  year={2026},
  doi={10.1098/rsif.2025.1023}
}
```

**Archived data and code:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19201270.svg)](https://doi.org/10.5281/zenodo.19201270)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Chulwook Park, Ph.D.**  
BK21 Four, Institute of Sport Science  
Seoul National University  
Email: pcw8531@snu.ac.kr

## Acknowledgments

This work was supported by the Basic Science Research Program through the National Research Foundation of Korea (NRF), funded by the Ministry of Education (Grant No. 2020R1I1A1A01056967). Additional support was provided by the Seoul National University BK21 Four Program.
