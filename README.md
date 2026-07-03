# AutoDockAnalyzer

Interactive PyMOL 3.X plugin for the analysis of AutoDock4 and AutoDock Vina docking results.

## 🧪 Description

AutoDockAnalyzer is a Python plugin for PyMOL designed to analyze docking results from AutoDock (.dlg) and AutoDock Vina (.pdbqt). It provides AutoDock-like clustering, thermodynamic data, and visualization tools comparable to AutoDockTools, with extended interactive capabilities.

## ✨ Features

- AutoDock-like clustering based on RMSD
- Interactive table (click to visualize clusters)
- Extraction of:
  - Best pose
  - Medoid (representative structure)
- Thermodynamic calculations:
  - Binding free energy (ΔG)
  - Dissociation constant (Kd)
  - pKd (−log10 Kd)
- Cluster population visualization (bar plot)
- Table sorting (like Excel)
- Highlight selected cluster interactively
- AutoDock Vina poses visualization

## 📊 Output

The plugin provides:

| Cluster | Size | Energy | Kd | pKd | Mean |
|--------|------|--------|----|-----|------|

