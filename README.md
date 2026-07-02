# AutoDockAnalyzer

Interactive PyMOL 3.X plugin for the analysis of AutoDock4 and AutoDock Vina docking results.

## 🧪 Description

AutoDockAnalyzer is a Python plugin for PyMOL designed to analyze docking results from AutoDock (.dlg files) and AutoDock Vina (.pdbqt). It provides clustering, thermodynamic interpretation, and visualization tools comparable to AutoDockTools, with extended interactive capabilities.

## ✨ Features

- AutoDock-like clustering based on RMSD and energy
- Interactive table (click to visualize clusters)
- Automatic cluster coloring (consistent with plots)
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

## 🎨 Visualization

- Each cluster has a consistent color in:
  - PyMOL structures
  - Bar plot
- Selected cluster is highlighted automatically
