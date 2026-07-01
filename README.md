# AutoDockAnalyzer

Interactive PyMOL plugin for the analysis of AutoDock docking results.

## 🧪 Description

AutoDockAnalyzer2 is a Python plugin for PyMOL designed to analyze docking results from AutoDock (.dlg files). It provides clustering, thermodynamic interpretation, and visualization tools comparable to AutoDockTools, with extended interactive capabilities.

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

## 📊 Output

The plugin provides:

| Cluster | Size | Energy | Kd | pKd | Mean |
|--------|------|--------|----|-----|------|

## 🎨 Visualization

- Each cluster has a consistent color in:
  - PyMOL structures
  - Bar plot
- Selected cluster is highlighted automatically

## 🚀 Installation

1. Copy the file:
