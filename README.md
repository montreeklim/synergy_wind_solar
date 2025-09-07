# Synergy between wind and solar in energy generation with chance-constrained models

This project provides a framework for quantifying the synergistic effects of combining wind and solar power generation. The goal is to analyze how a mixed portfolio of these intermittent renewable sources can lead to a more stable and reliable aggregate power output. 

We provide code to calculate a **synergy ratio** based on time-series generation data.

---

## Formulation for Output Variability

Let $P_{wind}(t)$ and $P_{solar}(t)$ be the normalized power output for wind and solar at time $t$, respectively. We seek a weighting factor $\alpha \in [0, 1]$ that defines the mix of generation capacity. The combined power output is:

$$P_{total}(t, \alpha) = \alpha \cdot P_{wind}(t) + (1-\alpha) \cdot P_{solar}(t)$$

The objective is to find the optimal mix $\alpha^*$ that minimizes the variance (or standard deviation) of the total power output over a given period $T$:

$$\alpha^* = \arg\min_{0 \leq \alpha \leq 1} \quad \text{Var}(P_{total}(t, \alpha))$$

A lower variance in the combined output signifies greater synergy between the two sources.

---

Repository Content
The repository contains the following content:

code/synergy_analyzer.py: A Python module containing the core functions for the analysis.

The function run_complementation_analysis takes two pandas Series (wind and solar time-series data) as input. It returns a dictionary containing key metrics: the Pearson correlation, the optimal mixing parameter $\\alpha^\*$, the percentage reduction in variance compared to individual sources, and the computation time.

code/analysis_notebook.ipynb: A Jupyter Notebook that provides a step-by-step walkthrough of the analysis. It includes data loading, cleaning, executing the analysis, and visualizing the results. This is the main file for reproducing the project's findings.

data/: This directory should contain the input CSV files. Each file is expected to have at least two columns: a timestamp column and a power_kw column representing the power output at that time.

results/: This directory will store any output files, such as plots of the power generation curves or CSV files with summary statistics.

---

Requirements to Run Code
The code relies on standard Python data science libraries. An optimization solver like Gurobi or SciPy's optimizer may be required for more complex formulations.

Python 3.8+

pandas

NumPy

Matplotlib

scikit-learn
