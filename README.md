# AutoDockAnalyzer (ADA)

Interactive PyMOL 3.X plugin for the analysis of AutoDock4 and AutoDock Vina docking results.

## 🧪 Description

AutoDockAnalyzer is a light and portable plugin for PyMOL designed to analyze docking results from AutoDock (.dlg) and AutoDock Vina (.pdbqt). It provides AutoDock-like clustering, thermodynamic data, and visualization tools comparable to AutoDockTools, with extended interactive capabilities.

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

| Cluster | Size | Energy | Kd | pKd | Mean | SD |
|--------|------|--------|----|-----|------|-----|

## Installing the plugin
🖥️ Download the repo and extract the files. Then, in PyMOL click on Plugin -> Plugin Manager -> Install New Plugin -> Install from local file (Choose File ...) and select the init.py file in the plugin folder. You can access the plugin from the Plugin tab in PyMOL.

## 📄 How to cite
Garcia-Marin, J., AutoDockAnalyzer, 2026, https://github.com/reibag/AutoDockAnalyzer/

