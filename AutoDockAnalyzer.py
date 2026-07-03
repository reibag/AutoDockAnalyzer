#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# PLUGIN: AutoDockAnalyzer (ADA)
# VERSION: 1.2
# DATE: 2026-07-03
# AUTHOR: Javier García Marín (assisted with Copilot 365)
# AFFILIATION: University of Alcalá (Department of Organic and Inorganic Chemistry)
# LICENSE: GNU GENERAL PUBLIC LICENSE (GPLv3)
# ==============================================================================
"""
DESCRIPTION:
    Interactive PyMOL 3.X plugin partitioned into multi-tabs for processing both
    AutoDock Classic (.dlg) and AutoDock Vina multi-state (.pdbqt) outputs.

FEATURES:
    - Tab 1 (AutoDock4): Geometric greedy clustering, medoid extraction, thermodynamic 
      calculations (Kd, pKd), population charting with energy SD, and standalone CSV reports.
    - Tab 2 (AutoDock Vina): Parsing of Vina multi-model states, affinity mapping tables, 
      thermodynamic profiling, and native Vina CSV reporting.
    - Tab 3 (About): Institutional credits and non-commercial licensing information.
VERSION HISTORY:
    v1.2 (2026-07-03):
        - Corrected critical bugs 
    v1.1 (2026-07-02):
        - Added intracluster Energy SD 
        - Fixed a critical UI freezing issue and CPU overhead caused by blocking 'plt.show()' calls in Matplotlib.
        - Resolved plugin startup crashes by implementing lazy loading for 'matplotlib'.
    v1.0 (2026-06-29):
        - Initial implementation of VINA poses loading and basic PyMOL integration.
"""

import numpy as np
import os
import csv
from pymol import cmd
from pymol.Qt import QtWidgets, QtCore

# ==============================================================================
# DEPENDENCY CHECK & FALLBACK HANDLING
# ==============================================================================
# Safely verify if matplotlib is available without crashing the plugin at startup.
# This ensures core docking parsing and CSV exports remain fully functional.
try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

version = 1.2

# Personal Logo Data (encoded in Base64)
PERSONAL_LOGO_BASE64 = b"""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
"""
class NumericTableWidgetItem(QtWidgets.QTableWidgetItem):
    """Custom table widget item that allows proper numerical sorting."""
    def __init__(self, value):
        super().__init__()
        if value is None or value == "" or value == "Inf" or value == "-Inf":
            self.setData(QtCore.Qt.DisplayRole, str(value))
        else:
            try:
                if isinstance(value, int):
                    self.setData(QtCore.Qt.DisplayRole, value)
                else:
                    self.setData(QtCore.Qt.DisplayRole, float(value))
            except ValueError:
                self.setData(QtCore.Qt.DisplayRole, value)


class ADA:
    """Core logic controller for processing AutoDock4 results."""
    def __init__(self):
        self.poses = []
        self.energies = []
        self.clusters = []
        self.current_cutoff = 2.0

    def apply_visual_style(self, obj):
        """Optimizes representation styling and applies a strict state limit frame."""
        cmd.hide("everything", obj)
        cmd.show("sticks", obj)
        cmd.hide("spheres", obj)
        cmd.hide("nonbonded", obj)
        cmd.set("sphere_scale", 0.0, obj)
        
        # PERFORMANCE FIX: Prevent viewport rendering loops from drawing all states at once
        # during mouse manipulation. This forces PyMOL to only compute active frames.
        cmd.set("all_states", 0, obj)

    def load_dlg(self, filename):
        if not os.path.exists(filename):
            print("[ADA3] Error: File not found")
            return False

        self.poses = []
        self.energies = []
        reading = False
        pose = []
        energy = None

        with open(filename, "r") as f:
            for line in f:
                if "Estimated Free Energy of Binding" in line:
                    try:
                        energy = float(line.split("=")[1].split()[0])
                    except Exception:
                        energy = None

                if line.startswith("DOCKED: MODEL"):
                    pose = []
                    reading = True
                elif reading and line.startswith("DOCKED: ATOM"):
                    pose.append(line.replace("DOCKED: ", ""))
                elif reading and line.startswith("DOCKED: ENDMDL"):
                    reading = False
                    self.poses.append(pose)
                    self.energies.append(energy)

        print(f"[ADA 3] Loaded {len(self.poses)} AutoDock poses successfully.")
        return True

    def load_into_pymol(self):
        if not self.poses: return
        obj = "lig_all"
        cmd.delete(obj)
        
        # Performance optimization to block canvas refreshes during massive state initialization
        cmd.set("suspend_updates", 1)
        for i, pose in enumerate(self.poses):
            cmd.read_pdbstr("".join(pose), obj, state=i+1)
        self.apply_visual_style(obj)
        cmd.set("suspend_updates", 0)

    def coords(self, pose):
        coordinates = []
        for line in pose:
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                coordinates.append([x, y, z])
            except ValueError:
                parts = line.split()
                if len(parts) >= 8:
                    try:
                        coordinates.append([float(parts[6]), float(parts[7]), float(parts[8])])
                    except ValueError:
                        continue
        return np.array(coordinates)

    def rmsd(self, a, b):
        if len(a) != len(b) or len(a) == 0: return 999.0
        return np.sqrt(((a - b) ** 2).sum() / len(a))

    def cluster(self, cutoff=2.0):
        if not self.poses: return
        self.current_cutoff = cutoff
        order = sorted(range(len(self.poses)), key=lambda i: self.energies[i] if self.energies[i] is not None else 999)
        assigned = [False] * len(self.poses)
        clusters = []

        for i in order:
            if assigned[i]: continue
            seed = self.coords(self.poses[i])
            cl = [i]
            assigned[i] = True
            for j in order:
                if assigned[j]: continue
                if self.rmsd(seed, self.coords(self.poses[j])) <= cutoff:
                    cl.append(j)
                    assigned[j] = True
            clusters.append(cl)

        self.clusters = clusters
        self.create_clusters()

    def table(self):
        R, T = 0.001987, 298.15
        data = []
        for i, c in enumerate(self.clusters):
            energies = [self.energies[j] for j in c if self.energies[j] is not None]
            best = min(energies) if energies else None
            mean = sum(energies) / len(energies) if energies else None
            std_dev = np.std(energies) if energies else None
            if best is not None:
                try:
                    # OVERFLOW PROTECTION: Max binding constraint threshold avoids exponential crashes
                    safe_energy = max(best, -100.0)
                    kd = np.exp(safe_energy / (R * T))
                    pkd = -np.log10(kd) if kd > 0 else float('inf')
                except (OverflowError, RuntimeWarning):
                    kd, pkd = float('inf'), float('-inf')
            else:
                kd, pkd = None, None
            data.append([int(i+1), int(len(c)), best, kd, pkd, mean, std_dev])
        data.sort(key=lambda x: x[2] if x[2] is not None else 999)
        return data

    def create_clusters(self):
        cmd.set("suspend_updates", 1)  # Disable screen rendering updates
        cutoff_str = str(self.current_cutoff).replace('.', '_')
        for i, c in enumerate(self.clusters):
            name = f"cluster_{i+1}_rc{cutoff_str}"
            cmd.delete(name)
            for state, idx in enumerate(c):
                cmd.read_pdbstr("".join(self.poses[idx]), name, state=state+1)
            self.apply_visual_style(name)
        cmd.set("suspend_updates", 0)  # Re-enable and refresh everything at once

    def best(self):
        for i, c in enumerate(self.clusters):
            idx = min(c, key=lambda x: self.energies[x] if self.energies[x] is not None else 999)
            name = f"best_{i+1}"
            cmd.delete(name)
            cmd.read_pdbstr("".join(self.poses[idx]), name)
            self.apply_visual_style(name)

    def medoid(self):
        for i, c in enumerate(self.clusters):
            best_idx, best_score = None, float("inf")
            for idx_i in c:
                coords_i = self.coords(self.poses[idx_i])
                total = sum(self.rmsd(coords_i, self.coords(self.poses[idx_j])) for idx_j in c)
                avg = total / len(c)
                if avg < best_score:
                    best_score, best_idx = avg, idx_i
            if best_idx is not None:
                name = f"medoid_{i+1}"
                cmd.delete(name)
                cmd.read_pdbstr("".join(self.poses[best_idx]), name)
                self.apply_visual_style(name)

    def plot(self):
        """
        Generates a population bar chart using the energy standard deviation 
        directly as error bars (yerr) over each cluster bar.
        
        This method uses a non-blocking asynchronous pipeline to prevent 
        PyMOL Qt thread freezing and includes lazy loading for matplotlib.
        """
        table = self.table()
        if not table: 
            return
        
        if not HAS_MATPLOTLIB:
            print("[ADA] Error: Matplotlib is not installed. Plotting functionality is disabled.")
            return

        # DEFERRED IMPORTS: Loaded locally to prevent startup dependency failures
        import matplotlib
        matplotlib.use('Agg', force=True) 
        import matplotlib.pyplot as plt
        
        cluster_ids = [row[0] for row in table]
        populations = [row[1] for row in table]
        energy_sds = [row[6] if (row[6] is not None) else 0.0 for row in table]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        color_bars = '#3498db'    
        color_errors = '#e74c3c'  
        
        # Plot the primary population bars mapping the energy SD array directly to 'yerr'
        ax.bar(cluster_ids, populations, 
               color=color_bars, 
               alpha=0.7, 
               edgecolor='#2980b9',
               yerr=energy_sds, 
               error_kw={
                   'ecolor': color_errors, 
                   'elinewidth': 2, 
                   'capsize': 6, 
                   'capthick': 2
               },
               label='Population')
        
        ax.set_xlabel("Cluster ID", fontweight='bold', labelpad=10)
        ax.set_ylabel("Cluster Population (# poses)", fontweight='bold', labelpad=10)
        ax.set_xticks(cluster_ids)  
        ax.set_title("Cluster Population with Energy SD as Error Bars", fontsize=12, fontweight='bold', pad=15)
        
        fig.tight_layout()
        output_path = os.path.join(os.path.expanduser("~"), "ADA3_cluster_plot.png")
        fig.savefig(output_path, dpi=150)
        plt.close(fig)  
        
        import platform
        import subprocess
        if platform.system() == "Windows":
            os.startfile(output_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.Popen(["open", output_path])
        else:  # Linux / Unix desktop environments
            subprocess.Popen(["xdg-open", output_path])

    def export_csv(self, filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Cluster", "Size", "Best Energy", "Kd (M)", "pKd", "Mean Energy", "Energy SD"])
            for row in self.table():
                kd_out = f"{row[3]:.2e}" if isinstance(row[3], float) else ""
                pkd_out = f"{row[4]:.2f}" if isinstance(row[4], float) else ""
                sd_out = f"{row[6]:.2f}" if isinstance(row[6], float) else ""
                writer.writerow([row[0], row[1], row[2], kd_out, pkd_out, row[5], sd_out])


class VinaAnalyzer:
    """Module dedicated to parsing and exporting AutoDock Vina data files."""
    def __init__(self):
        self.models = []
        self.affinities = []

    def parse_vina_file(self, filename):
        if not os.path.exists(filename): return False
        self.models = []
        self.affinities = []
        current_model = []
        with open(filename, "r") as f:
            for line in f:
                if "REMARK VINA RESULT" in line:
                    try:
                        self.affinities.append(float(line.split()[3]))
                    except (IndexError, ValueError):
                        self.affinities.append(0.0)
                if line.startswith("MODEL"):
                    current_model = []
                elif line.startswith("ATOM") or line.startswith("HETATM"):
                    current_model.append(line)
                elif line.startswith("ENDMDL"):
                    self.models.append(current_model)
        return True

    def load_into_pymol(self):
        cmd.delete("vina_all")
        cmd.set("suspend_updates", 1)
        for i, model in enumerate(self.models):
            name = f"vina_mode_{i+1}"
            cmd.delete(name)
            cmd.read_pdbstr("".join(model), name)
            cmd.hide("everything", name)
            cmd.show("sticks", name)
            cmd.set("all_states", 0, name)  # Viewport protection frame rule
        cmd.set("suspend_updates", 0)

    def export_vina_csv(self, filename):
        R, T = 0.001987, 298.15 #Constants to calculate KD
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Mode", "Affinity (kcal/mol)", "Kd (M)", "pKd"])
            for i, delta_g in enumerate(self.affinities):
                try:
                    safe_energy = max(delta_g, -100.0)
                    kd = np.exp(safe_energy / (R * T))
                    pkd = -np.log10(kd) if kd > 0 else float('inf')
                except (OverflowError, RuntimeWarning):
                    kd, pkd = float('inf'), float('-inf')
                writer.writerow([i + 1, delta_g, f"{kd:.2e}", f"{pkd:.2f}"])


ada = ADA()
vina = VinaAnalyzer()


# ================= GUI INTERFACE =================

class GUI(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoDockAnalyzer v1.2")
        self.resize(700, 580)
        
        main_layout = QtWidgets.QVBoxLayout()
        self.tabs = QtWidgets.QTabWidget()

        self.init_autodock_tab()
        self.init_vina_tab()
        self.init_acknowledgements_tab()

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def init_autodock_tab(self):
        """Tab 1: AutoDock4 Layout."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        self.file = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse AutoDock Output (.dlg)")
        cutoff_label = QtWidgets.QLabel("RMSD cutoff (Å)")
        self.cutoff = QtWidgets.QLineEdit("2.0")
        load_btn = QtWidgets.QPushButton("Load Poses")
        cluster_btn = QtWidgets.QPushButton("Run Clustering")
        best_btn = QtWidgets.QPushButton("Extract Best Poses")
        medoid_btn = QtWidgets.QPushButton("Extract Centroids (Medoids)")
        plot_btn = QtWidgets.QPushButton("Plot Population")
        export_btn = QtWidgets.QPushButton("Export CSV Report")
        self.table_widget = QtWidgets.QTableWidget()

        # Connections
        browse.clicked.connect(self.browse_classic)
        load_btn.clicked.connect(self.load_classic)
        cluster_btn.clicked.connect(self.cluster_classic)
        best_btn.clicked.connect(self.run_best)
        medoid_btn.clicked.connect(self.run_medoid)
        plot_btn.clicked.connect(self.run_plot)
        export_btn.clicked.connect(self.save_csv_classic)
        self.table_widget.cellClicked.connect(self.handle_classic_click)

        # REORDERED COMPONENT LAYOUT
        layout.addWidget(self.file)
        layout.addWidget(browse)
        layout.addWidget(load_btn)          
        layout.addWidget(cutoff_label)      
        layout.addWidget(self.cutoff)        
        layout.addWidget(cluster_btn)       
        layout.addWidget(best_btn)
        layout.addWidget(medoid_btn)
        
        # DYNAMIC UI HANDLING: Disable plotting element if dependencies are not met
        if not HAS_MATPLOTLIB:
            plot_btn.setEnabled(False)
            plot_btn.setToolTip("Plotting requires the 'matplotlib' library to be installed in Python.")
            
        layout.addWidget(plot_btn)
        layout.addWidget(export_btn)
        layout.addWidget(self.table_widget)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "AutoDock")

    def init_vina_tab(self):
        """Tab 2: AutoDock Vina Layout."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        self.vina_file_line = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Browse Vina Output (.pdbqt / .pdb)")
        load_btn = QtWidgets.QPushButton("Load Poses")
        vina_export_btn = QtWidgets.QPushButton("Export Vina CSV Report")
        self.vina_table = QtWidgets.QTableWidget()

        browse_btn.clicked.connect(self.browse_vina)
        load_btn.clicked.connect(self.load_vina_workflow)
        vina_export_btn.clicked.connect(self.save_csv_vina)
        self.vina_table.cellClicked.connect(self.handle_vina_click)

        layout.addWidget(self.vina_file_line)
        layout.addWidget(browse_btn)
        layout.addWidget(load_btn)
        layout.addWidget(vina_export_btn)
        layout.addWidget(self.vina_table)

        tab.setLayout(layout)
        self.tabs.addTab(tab, "AutoDock Vina")

    def init_acknowledgements_tab(self):
        """Tab 3: About, Affiliation & License."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        title = QtWidgets.QLabel(f"<h1 align='center'>AutoDockAnalyzer {version}</h1>")
        title.setAlignment(QtCore.Qt.AlignCenter)
        
        info = QtWidgets.QLabel(
            "<p align='center' style='font-size:11pt; line-height:140%;'>"
            "<b>Developer:</b> Javier García Marín<br>"
            "<i>Department of Organic and Inorganic Chemistry</i><br>"
            "<b>University of Alcalá</b><br>"
            "<a href='mailto:javier.garciamarin@uah.es'>javier.garciamarin@uah.es</a><br><br>"
            "<font color='#880088'><b>License: GPLv3</b></font><br>"
            #"Academic & non-commercial use only."
            "</p>"
        )
        info.setAlignment(QtCore.Qt.AlignCenter)
        info.setOpenExternalLinks(True)
        
        logo_label = QtWidgets.QLabel()
        logo_label.setText(
            f"<p align='center'><img src='data:image/png;base64,{PERSONAL_LOGO_BASE64.decode('utf-8')}' "
            "width='100' height='100'></p>"
        )
        logo_label.setAlignment(QtCore.Qt.AlignCenter)
        
        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(info)
        layout.addWidget(logo_label)
        layout.addStretch()

        tab.setLayout(layout)
        self.tabs.addTab(tab, "About")

    # ---------- CLASSIC SLOTS ----------
    def browse_classic(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select AutoDock File", "", "DLG files (*.dlg *.DLG)")
        if f: self.file.setText(f)

    def load_classic(self):
        if self.file.text() and ada.load_dlg(self.file.text()):
            ada.load_into_pymol()

    def cluster_classic(self):
        if not ada.poses: return
        self.table_widget.setSortingEnabled(False)
        ada.cluster(float(self.cutoff.text()))
        table_data = ada.table()

        self.table_widget.setRowCount(len(table_data))
        self.table_widget.setColumnCount(7) 
        self.table_widget.setHorizontalHeaderLabels(["Cluster", "Size", "Best Energy", "Kd (M)", "pKd", "Mean Energy", "Energy SD"])

        for i, row in enumerate(table_data):
            self.table_widget.setItem(i, 0, NumericTableWidgetItem(row[0]))
            self.table_widget.setItem(i, 1, NumericTableWidgetItem(row[1]))
            self.table_widget.setItem(i, 2, NumericTableWidgetItem(row[2]))
            if isinstance(row[3], float) and row[3] != float('inf'):
                item = NumericTableWidgetItem(row[3]); item.setText(f"{row[3]:.2e}")
                self.table_widget.setItem(i, 3, item)
            else:
                self.table_widget.setItem(i, 3, NumericTableWidgetItem(""))
            if isinstance(row[4], float) and row[4] != float('-inf'):
                item = NumericTableWidgetItem(row[4]); item.setText(f"{row[4]:.2f}")
                self.table_widget.setItem(i, 4, item)
            else:
                self.table_widget.setItem(i, 4, NumericTableWidgetItem(""))
            self.table_widget.setItem(i, 5, NumericTableWidgetItem(row[5]))
            
            if row[6] is not None:
                sd_item = NumericTableWidgetItem(row[6]); sd_item.setText(f"{row[6]:.2f}")
                self.table_widget.setItem(i, 6, sd_item)
            else:
                self.table_widget.setItem(i, 6, NumericTableWidgetItem(""))

        self.table_widget.resizeColumnsToContents()
        self.table_widget.setSortingEnabled(True)

    def handle_classic_click(self, row, column):
        if not ada.clusters: return
        cluster_id = self.table_widget.item(row, 0).text()
        cutoff_str = str(ada.current_cutoff).replace('.', '_')
        cmd.hide("everything", "cluster_*")
        cmd.show("sticks", f"cluster_{cluster_id}_rc{cutoff_str}")

    def run_best(self): 
        if ada.clusters: ada.best()
    def run_medoid(self): 
        if ada.clusters: ada.medoid()
    def run_plot(self): 
        if ada.clusters: ada.plot()
    def save_csv_classic(self):
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Classic Report", "", "*.csv")
        if f: ada.export_csv(f)

    # ---------- VINA SLOTS ----------
    def browse_vina(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Vina File", "", "Vina Outputs (*.pdbqt *.pdb)")
        if f: self.vina_file_line.setText(f)

    def load_vina_workflow(self):
        if not self.vina_file_line.text(): return
        if not vina.parse_vina_file(self.vina_file_line.text()): return
        vina.load_into_pymol()

        self.vina_table.setSortingEnabled(False)
        self.vina_table.clear()
        self.vina_table.setRowCount(len(vina.models))
        self.vina_table.setColumnCount(4)
        self.vina_table.setHorizontalHeaderLabels(["Mode", "Affinity (kcal/mol)", "Kd (M)", "pKd"])

        R, T = 0.001987, 298.15
        for i, delta_g in enumerate(vina.affinities):
            try:
                safe_energy = max(delta_g, -100.0)
                kd = np.exp(safe_energy / (R * T))
                pkd = -np.log10(kd) if kd > 0 else float('inf')
            except (OverflowError, RuntimeWarning):
                kd, pkd = float('inf'), float('-inf')

            self.vina_table.setItem(i, 0, NumericTableWidgetItem(i + 1))
            self.vina_table.setItem(i, 1, NumericTableWidgetItem(delta_g))
            kd_item = NumericTableWidgetItem(kd); kd_item.setText(f"{kd:.2e}")
            pkd_item = NumericTableWidgetItem(pkd); pkd_item.setText(f"{pkd:.2f}")
            self.vina_table.setItem(i, 2, kd_item)
            self.vina_table.setItem(i, 3, pkd_item)

        self.vina_table.resizeColumnsToContents()
        self.vina_table.setSortingEnabled(True)

    def handle_vina_click(self, row, column):
        if not vina.models: return
        mode_id = self.vina_table.item(row, 0).text()
        cmd.hide("everything", "vina_mode_*")
        cmd.show("sticks", f"vina_mode_{mode_id}")

    def save_csv_vina(self):
        if not vina.models: return
        f, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Vina Report", "", "*.csv")
        if f: vina.export_vina_csv(f)


# ================= PLUGIN REGISTRATION =================

dialog = None

def run_plugin():
    global dialog
    dialog = GUI()
    dialog.show()

def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt("ADA", run_plugin)

cmd.extend("AutoDockAnalyzer", run_plugin)
print(f"✅ AutoDockAnalyzer v{version} LOADED")
