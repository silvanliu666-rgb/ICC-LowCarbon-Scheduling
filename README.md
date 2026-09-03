# ICC-LowCarbon-Scheduling

This repository contains the original datasets and source code for the manuscript: **"Turning interruptible AI training into virtual energy storage for renewable integration in computing centers"**[cite: 1]. The paper has been submitted to *Renewable and Sustainable Energy Reviews*[cite: 1].

This project proposes a low-carbon dispatch framework that models interruptible Artificial Intelligence (AI) training workloads as virtual energy storage[cite: 1]. By dynamically reshaping compute-intensive workloads to align with local renewable energy generation (Wind/PV) and dynamic carbon emission factors, the framework significantly reduces renewable curtailment, operational costs, and physical carbon emissions[cite: 1].

## Repository Structure

The repository is highly structured to directly correspond to the sections and figures in the manuscript. Each sub-directory contains its own required dataset (`.csv` or `.xlsx`) and a dedicated Python execution script for seamless reproducibility.

### 1. Base (Sections 4.1 - 4.3)
Contains the fundamental experimental setup, load reshaping simulations, and the characterization of the Virtual Energy Storage (VES) effect[cite: 1]. 
* **Script:** `run_moo_scheduling.py`
* **Outputs:** `Fig2_Baseline_Workload.png` (Fig. 2), `Fig3_Optimized_Workload.png` (Fig. 3), `Fig4_VES_Effect.png` (Fig. 4), `FigA1_RES_Profile.png`, `FigA2_Price_Curve.png`, and detailed 15-min baseline numerical data (`Baseline_Schedule_96Steps.csv`).

### 2. Method (Section 3.5.3)
Contains the improved genetic algorithm (GA) implementation and statistical validation scripts[cite: 1].
* **Script:** `Method.py`
* **Output:** `FigA4_Algorithm_Statistics.png` (Fig. A4) demonstrating algorithmic robustness over 10 independent runs.

### 3. Benchmark Evaluation (Section 4.4)
Contains comprehensive comparative benchmarks against isolated baselines and physical storage[cite: 1].
* **Workload Flexibility:** Runs ablation studies on computing flexibility parameters. Outputs `Fig5_Ablation_LMS_only.png` (Fig. 5) and `Fig6_Ablation_LMI_only.png` (Fig. 6)[cite: 1].
* **Market Signals:** Decoupling analysis for price, carbon, and certificate signals. Outputs `Fig7_Objective_Comparison.png` (Fig. 7)[cite: 1].
* **Physical Battery Storage:** Compares software-defined virtual storage against a 10 MWh physical battery benchmark. Outputs comparative `.csv` logs and distribution profiles[cite: 1].

### 4. Multi-Scenario Validation (Section 4.5)
Contains extensive stress tests validating the framework's robustness across diverse environmental and structural constraints[cite: 1].
* **AI Lifecycle Phases:** Evaluates structural shifts from pre-training to inference. Outputs `Fig10_Lifecycle_Optimization_Grid.png` (Fig. 10)[cite: 1].
* **Diverse Renewable Generation (RES-domi & RES-season):** Evaluates extreme RES patterns (`RES-domi.py`) and seasonal variations (`RES-season.py`). Outputs `Fig9_Extreme_Generation_Patterns.png` (Fig. 9) and `Fig8_Seasonal_Generation_Patterns.png` (Fig. 8)[cite: 1].
* **Grid Carbon Emission Factors:** Evaluates the system against dynamic and future duck-curve macro-grid emissions (`E_GRID_DYNAMIC.py`). Outputs `Fig11_CEF_Scenarios.png` (Fig. 11)[cite: 1].

## Data Availability
The workload baseline is reconstructed using empirical data from the Alibaba PAI (Platform for Artificial Intelligence) GPU cluster trace, scaled to represent a core computing cluster consuming ~40,000 kWh daily[cite: 1]. Raw data files (`Base_workload.csv`, `res_data.csv`, `TOU.csv`) are consistently provided within their respective execution directories.

## Requirements and Execution
To run the scripts, ensure you have Python 3.8+ installed along with the following primary packages:
```bash
pip install pandas numpy matplotlib openpyxl
