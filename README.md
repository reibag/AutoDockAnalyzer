# AutoDockAnalyzer (ADA)

[![Version](https://img.shields.io/badge/version-1.6-blue.svg)](https://github.com/reibag/AutoDockAnalyzer)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23374149.svg)](https://doi.org/10.5281/zenodo.23374149)

**AutoDockAnalyzer (ADA) v1.6** is an open-source, lightweight, single-file Python 3 plugin natively integrated into PyMOL (supporting both PyMOL 2.x and 3.x) for post-docking conformational analysis, clustering, and empirical score-derived affinity evaluation of AutoDock 4 (`.dlg`) and AutoDock Vina (`.pdbqt`) simulation outputs.

---

## 🧪 Description

Modern structure-based virtual screening workflows generate hundreds or thousands of poses that must be organized into conformational families and quantitatively evaluated. While legacy tools like AutoDockTools (ADT) rely on deprecated Python 2 environments, and contemporary molecular graphics suites like UCSF ChimeraX lack native parsing for raw text-based `.dlg` files, ADA bridges this operational gap.

ADA parses raw docking outputs directly inside PyMOL without requiring external dependencies or command-line scripting. It features **dual clustering modalities** to reconcile native AutoDock 4 symmetry-aware clustering with interactive AutoDockTools-style Cartesian RMSD filtering, non-blocking visualization modules, automated polar contact mapping, and binding-site residue inspection.

---

## ✨ Key Features

- **Dual Clustering Support for AutoDock 4 (`.dlg`):**
  - **AutoDock-type (`Read .dlg Table`, Default):** Directly parses the precomputed `RMSD TABLE` (`RANKING` records) and `CLUSTERING HISTOGRAM` from `.dlg` logs upon loading, faithfully preserving AutoDock 4's native type-matched, symmetry-aware clustering (accounting for equivalent chemical groups such as carboxylates, ammonium hydrogens, and symmetrical alkyl chains).
  - **AutoDockTools-type (`Index-Matched RMSD`):** Allows interactive greedy re-clustering on the fly at any custom RMSD cutoff (e.g., `2.0 Å`, `1.5 Å`) using strict 1-to-1 atom-index Cartesian RMSD (`rmsnosym`), reproducing the interactive clustering behavior of legacy AutoDockTools (ADT).
- **Conformational Representatives:**
  - **Best Pose:** Instantly extracts the lowest-energy conformer for each cluster.
  - **Medoid Extraction (Centroid):** Identifies the true geometric medoid minimizing average pairwise RMSD across all cluster members.
- **Interactive Visual Navigation:** Selecting any cluster or pose in the sortable data table automatically renders and centers that conformational family in the 3D PyMOL viewport.
- **Hydrogen Bonding & Binding-Site Pocket Profiling:**
  - **Find H-Bonds:** Dynamic polar contact detection between receptor and ligand using PyMOL's native `cmd.distance(..., mode=2)` geometry (default distance cutoff $\le 3.5\text{ \AA}$ combined with donor/acceptor angle constraints).
  - **Show Pocket Residues (3 Å):** One-click selection and `sticks` rendering of all complete receptor residues located within a customizable distance cutoff (default `3.0 Å`) of the ligand (`byres ((receptor) within cutoff of (ligand))`) to inspect side-chain orientations and non-covalent interactions.
- **Nominal Binding Affinity Estimations (apparent Kd / pKd at 298.15 K):** Computes empirical score-derived affinity estimations ($K_d\text{ (app.)}$ and $\text{p}K_d\text{ (app.)}$ at 298.15 K) to facilitate relative ranking across conformational clusters and binding modes.
- **AutoDock Vina Compatibility:** Parses multi-model `.pdbqt` files from AutoDock Vina, ranking poses, extracting RMSD l.b./u.b. metrics, and linking table entries directly to 3D states.
- **Dual-Panel Publication Plotting:** Generates high-resolution dual-panel charts with Matplotlib (Panel 1: Cluster population histogram; Panel 2: Intracluster mean binding score with $\pm\text{SD}$ error bars) using the non-interactive `Agg` backend for cross-platform stability.
- **CSV Data Export:** Exports structured docking score and clustering metrics into standardized `.csv` files (UTF-8 encoded) that explicitly include the methodological scoring disclaimer in their file header for downstream reporting.

---

## ⚠️ Empirical Score-Derived Affinity Estimations

> **Note on Binding Affinities & Scoring:**
> Calculated binding energies, scores, and derived apparent dissociation constants ($K_d\text{ (app.)}$ / $\text{p}K_d\text{ (app.)}$ at 298.15 K) are nominal empirical estimations provided by the respective AutoDock 4 and AutoDock Vina scoring functions. They must not be interpreted as experimental thermodynamic free energies ($\Delta G$) or rigorous biophysical equilibrium constants ($K_d / K_i$).

Accordingly, calculated nominal binding affinity estimations ($K_d\text{ (app.)}$ and $\text{p}K_d\text{ (app.)}$ at $298.15\text{ K}$) must be interpreted strictly as relative ranking metrics under standard ideal-gas assumptions ($298.15\text{ K}$, $\Delta G = -RT \ln K_d$) to compare docking poses relative to one another, rather than calibrated experimental constants. Clear caveat notices are embedded directly in the GUI and in the header of all exported CSV reports:

```text
# AutoDockAnalyzer (ADA) - Results Export
# NOTE: Docking scores and apparent Kd/pKd values (298.15 K) are nominal empirical estimations, not experimental thermodynamic free energies (Delta G) or equilibrium constants (Kd/Ki).
# ------------------------------------------------------------------------------------------------------------------------
```

### 📊 Summary Table Columns

| Column | Description |
| :--- | :--- |
| **Cluster** | Cluster ranking index. |
| **Size** | Total number of conformations belonging to the cluster. |
| **Best Energy** | Lowest empirical binding score in the cluster ($\text{kcal/mol}$). |
| **Kd (app.) [M]** | Nominal apparent dissociation constant estimated from the empirical docking score at $298.15\text{ K}$ ($K_d\text{ (app.)} = e^{\text{Score} / RT}$). |
| **pKd (app.)** | Negative base-10 logarithm of the apparent dissociation constant ($-\log_{10} K_d\text{ (app.)}$). |
| **Mean Energy** | Intracluster mean empirical binding score ($\text{kcal/mol}$). |
| **Energy SD** | Standard deviation of intracluster binding scores, reflecting energetic homogeneity. |

---

## 🛠️ Installation

**AutoDockAnalyzer** is designed as a **standalone single-file plugin** (`AutoDockAnalyzer.py`) to maximize portability.

1. **Download the repository:**
   - Download the latest release or click **`Code` → `Download ZIP`** on GitHub and **extract (unzip) the folder** on your computer (or clone via `git clone https://github.com/reibag/AutoDockAnalyzer.git`).
2. **Install in PyMOL:**
   - Open PyMOL (version 2.x or 3.x).
   - Navigate to **`Plugin` → `Plugin Manager`** in the top menu bar.
   - Switch to the **`Install New Plugin`** tab.
   - Under *Install from local file*, click **`Choose file...`**.
   - Select the extracted **`AutoDockAnalyzer.py`** file directly and confirm the installation directory.
3. **Launch the plugin:**
   - Once installed, open the plugin anytime from **`Plugin` → `AutoDockAnalyzer`** or by typing `AutoDockAnalyzer` in the PyMOL command line.

---

## 📄 How to Cite

García Marín, J. (2026). *AutoDockAnalyzer (ADA): An Intuitive PyMOL Plugin for Streamlined Post-Docking Analysis of AutoDock and AutoDock Vina* (Version 1.6). Zenodo. https://doi.org/10.5281/zenodo.23374149
