# Thermodynamic Entropy Management in Human Motor Control

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19201270.svg)](https://doi.org/10.5281/zenodo.19201270)

## Overview

Data and analysis code for:

**Park, C. (2026). Thermodynamic Entropy Management in Human Motor Control Across Circadian and Thermal Challenges. *Journal of The Royal Society Interface*. DOI: [10.1098/rsif.2025.1023](https://doi.org/10.1098/rsif.2025.1023)**

## Files

| File | Description |
|------|-------------|
| `Analysis.py` | Data presentation and individual participant visualisations (Experiments I–III) |
| `phase_analysis.py` | Discrete relative phase extraction and Shannon entropy computation from raw data |
| `experimental_data_complete.csv` | Complete experimental dataset from Tables S1–S3 (576 total observations from 16 participants) |
| `cwa17w1.txt` | Sample raw pendulum position time-series (100 Hz) |
| `requirements.txt` | Python dependencies |

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
| Achieved statistical power | 0.82–0.89 across experiments |

## Usage

```bash
git clone https://github.com/pcw8531/thermodynamic-motor-control.git
cd thermodynamic-motor-control
pip install -r requirements.txt
python Analysis.py
python phase_analysis.py
```

**Note:** Inferential statistics (ANOVA, interaction tests, power analysis) were computed in SPSS Version 27 as reported in the manuscript (Tables 3.1–3.2, Supplement S6).

## Citation

```bibtex
@article{park2026thermodynamic,
  title={Thermodynamic Entropy Management in Human Motor Control Across Circadian and Thermal Challenges},
  author={Park, Chulwook},
  journal={Journal of The Royal Society Interface},
  year={2026},
  doi={10.1098/rsif.2025.1023}
}
```

## Contact

**Chulwook Park, Ph.D.** — BK21 Four, Associate Professor, Department of Physical Education, Seoul National University — pcw8531@snu.ac.kr

## Acknowledgments

Supported by the National Research Foundation of Korea (NRF), Ministry of Education (Grant No. 2020R1I1A1A01056967) and Seoul National University BK21 Four Program.
