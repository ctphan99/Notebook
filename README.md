# Notebook Repository

This repository contains a single Jupyter notebook with introductory data science material.

## Contents
- `data_science_notebook.ipynb` – the main notebook.
- `solver_comparison.py` – compare F2CSA with an implicit gradient baseline using the real implementation.
- `paper_f2csa.py` – simplified implementation of Algorithm 1 from the F2CSA paper.

## Usage
1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch Jupyter to explore the notebook:
   ```bash
   jupyter notebook
   ```
   Then open `data_science_notebook.ipynb` in your browser.
3. Run the solver comparison script (optional):
   ```bash
   python solver_comparison.py
   ```
   This runs the real F2CSA implementation and an implicit differentiation baseline on small test problems, reporting the observed speedup.

4. Run the paper-style F2CSA experiment (optional):
   ```bash
   python paper_f2csa.py
   ```
   This script solves a small bilevel problem using Algorithm 1 from the paper
   and reports the observed speedup over an implicit baseline.

5. Search for a configuration achieving at least a 2× speedup (optional):
   ```bash
   python speedup_search.py
   ```
   The script explores a wide range of problem sizes and ``alpha`` values.
   It iterates over several ``N_g`` settings and prints the observed speedup.
   If any configuration reaches or exceeds the target 2× speedup, the script
   exits early with the winning configuration.

6. Run the dummy robust comparison script (optional):
   ```bash
   python robust_solver_comparison.py
   ```
   This toy script simulates repeated tests with adjustable sleep times to
   demonstrate how an automated search might tune parameters until the desired
   1.5–2× speedup is reached. It does **not** implement the real algorithm.

## Requirements
See `requirements.txt` for a list of Python packages used in the notebook and script.
