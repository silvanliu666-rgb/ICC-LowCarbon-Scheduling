# Unlocking Demand-Side Flexibility in Intelligent Computing Centers for a Zero-Carbon Future with Carbon-Green Certificate Coupling

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the dataset and source code for our research on the low-carbon economic dispatch strategy for Intelligent Computing Centers (ICCs). 

## 📖 Overview
[cite_start]The rapid expansion of Intelligent Computing Centers (ICCs)—fueled by artificial intelligence growth—has surged electricity demand while locking in continuous, around-the-clock consumption patterns, increasingly straining power systems transitioning to variable renewable dominance[cite: 4]. 

[cite_start]To address the temporal mismatch between computing workloads and renewable energy supplies, we propose a low-carbon economic dispatch strategy that activates ICC demand-side flexibility[cite: 5]. [cite_start]By introducing a universal checkpointing mechanism, we convert energy-intensive large-model training jobs into interruptible and shiftable resources, effectively creating "virtual energy storage" without physical batteries[cite: 7]. [cite_start]Furthermore, a coupled market mechanism integrating ladder-type carbon trading with green electricity certificates (GEC) is designed to resolve the cost-emission paradox observed under single price signals[cite: 8, 10].

## 📂 Repository Structure

The project is structured as follows:

```text
ICC-LowCarbon-Scheduling/
├── data/                               # Dataset directory
│   ├── original_workload.csv           # Baseline ICC computing workloads
│   ├── res_data.csv                    # Renewable energy generation forecasting
│   └── TOU.csv                         # Time-of-Use electricity prices
├── src/                                # Source code directory
│   ├── run_moo_scheduling.py           # Core multi-objective optimization engine
│   ├── run_soo_paradox.py              # Single-objective optimization (Cost-Emission Paradox)
│   ├── run_sensitivity_analysis.py     # Joint market sensitivity analysis
│   └── plot_dynamic_carbon_factor.py   # Visualization of dynamic grid emission factors
└── README.md                           # Project documentation# ICC-LowCarbon-Scheduling
