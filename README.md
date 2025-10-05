# Thermodynamic Entropy Management in Human Motor Control

This repository contains data and analysis code for the manuscript:
"Thermodynamic Entropy Management in Human Motor Control Across Circadian and Thermal Challenges"

## Citation
Park, C. (2025). Thermodynamic Entropy Management in Human Motor Control. 
Journal of The Royal Society Interface. [DOI pending]

## Data Description
- `experiment1_circadian.csv`: Entropy values for 8 participants across 4 circadian time points
- `experiment2_heat.csv`: Normalized entropy under heat perturbation
- `experiment3_cold.csv`: Normalized entropy under cold perturbation

## Analysis Files
### Main Analysis
- `Entropy Analysis.ipynb`: Complete analysis pipeline reproducing all manuscript figures and statistics

### Supplementary Analysis
- `cwa17w1.ipynb`: Prototype implementation of bimanual coordination phase calculations demonstrating the underlying computational methods used in data processing
- `cwa17w1.txt`: Raw time-series data sample from bimanual pendulum coordination trials showing position data from DC potentiometers

## Code Requirements
Python 3.8+ with packages listed in requirements.txt

## Usage
1. To reproduce manuscript results: Open and run `Entropy Analysis.ipynb`
2. To examine phase calculation methods: See `cwa17w1.ipynb` for detailed computational procedures
All data is embedded within the notebooks for convenience.

## Repository Contents
- Main entropy analysis with embedded experimental data
- Sample bimanual coordination calculation methodology
- Raw data example from pendulum coordination trials
- Required package specifications

## License
MIT License - See LICENSE file for details
