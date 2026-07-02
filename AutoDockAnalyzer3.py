#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# PLUGIN: AutoDockAnalyzer 3 (ADA)
# VERSION: 3.1
# DATE: 2026-07-02
# AUTHOR: Javier García Marín (assisted by M365 Copilot & Gemini Flash)
# AFFILIATION: University of Alcalá (Department of Organic & Inorganic Chemistry)
# LICENSE: MIT License
# ==============================================================================
"""
DESCRIPTION:
    Interactive PyMOL 3.X plugin partitioned into multi-tabs for processing both
    AutoDock Classic (.dlg) and AutoDock Vina multi-state outputs.

FEATURES:
    - Tab 1 (AutoDock Classic): Greedy clustering, medoid generation, statistical calculations (Kd, pKd),
      population plotting, and standalone CSV reporting.
    - Tab 2 (AutoDock Vina): Direct parsing of Vina multi-model states, affinity
      mapping tables, thermodynamic profiling, and standalone Vina CSV reporting.
    - Tab 3 (Acknowledgements): Institutional credits.
"""

import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from pymol import cmd
from pymol.Qt import QtWidgets, QtCore


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
    """Core logic controller for processing AutoDock Classic results."""
    def __init__(self):
        self.poses = []
        self.energies = []
        self.clusters = []
        self.current_cutoff = 2.0

    def apply_visual_style(self, obj):
        cmd.hide("everything", obj)
        cmd.show("sticks", obj)
        cmd.hide("spheres", obj)
        cmd.hide("nonbonded", obj)
        cmd.set("sphere_scale", 0.0, obj)

    def load_dlg(self, filename):
        if not os.path.exists(filename):
            print("[ADA] Error: File not found")
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

        print(f"[ADA] Loaded {len(self.poses)} Classic poses successfully.")
        return True

    def load_into_pymol(self):
        if not self.poses: return
        obj = "lig_all"
        cmd.delete(obj)
        for i, pose in enumerate(self.poses):
            cmd.read_pdbstr("".join(pose), obj, state=i+1)
        self.apply_visual_style(obj)

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
            if best is not None:
                try:
                    kd = np.exp(best / (R * T))
                    pkd = -np.log10(kd)
                except OverflowError:
                    kd, pkd = float('inf'), float('-inf')
            else:
                kd, pkd = None, None
            data.append([int(i+1), int(len(c)), best, kd, pkd, mean])
        data.sort(key=lambda x: x[2] if x[2] is not None else 999)
        return data

    def create_clusters(self):
        cutoff_str = str(self.current_cutoff).replace('.', '_')
        for i, c in enumerate(self.clusters):
            name = f"cluster_{i+1}_rc{cutoff_str}"
            cmd.delete(name)
            for state, idx in enumerate(c):
                cmd.read_pdbstr("".join(self.poses[idx]), name, state=state+1)
            self.apply_visual_style(name)

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
        table = self.table()
        if not table: return
        plt.figure()
        plt.bar([row[0] for row in table], [row[1] for row in table])
        plt.xlabel("Cluster ID")
        plt.ylabel("Population (#poses)")
        plt.title("Cluster Population")
        plt.show()

    def export_csv(self, filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Cluster", "Size", "Best Energy", "Kd (M)", "pKd", "Mean Energy"])
            for row in self.table():
                kd_out = f"{row[3]:.2e}" if isinstance(row[3], float) else ""
                pkd_out = f"{row[4]:.2f}" if isinstance(row[4], float) else ""
                writer.writerow([row[0], row[1], row[2], kd_out, pkd_out, row[5]])


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
                    except:
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
        for i, model in enumerate(self.models):
            name = f"vina_mode_{i+1}"
            cmd.delete(name)
            cmd.read_pdbstr("".join(model), name)
            cmd.hide("everything", name)
            cmd.show("sticks", name)

    def export_vina_csv(self, filename):
        R, T = 0.001987, 298.15
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Mode", "Affinity (kcal/mol)", "Kd (M)", "pKd"])
            for i, delta_g in enumerate(self.affinities):
                try:
                    kd = np.exp(delta_g / (R * T))
                    pkd = -np.log10(kd)
                except OverflowError:
                    kd, pkd = float('inf'), float('-inf')
                writer.writerow([i + 1, delta_g, f"{kd:.2e}", f"{pkd:.2f}"])


ada = ADA()
vina = VinaAnalyzer()


# ================= GUI =================

class GUI(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoDockAnalyzer v3.0")
        self.resize(700, 580)
        
        main_layout = QtWidgets.QVBoxLayout()
        self.tabs = QtWidgets.QTabWidget()

        # Build tabs
        self.init_autodock_tab()
        self.init_vina_tab()
        self.init_acknowledgements_tab()

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

    def init_autodock_tab(self):
        """Tab 1: AutoDock Classic Layout."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        self.file = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse .DLG")
        cutoff_label = QtWidgets.QLabel("RMSD cutoff (Å)")
        self.cutoff = QtWidgets.QLineEdit("2.0")
        load_btn = QtWidgets.QPushButton("Load DLG")
        cluster_btn = QtWidgets.QPushButton("Run Clustering")
        best_btn = QtWidgets.QPushButton("Extract Best Poses")
        medoid_btn = QtWidgets.QPushButton("Extract Centroids (Medoids)")
        plot_btn = QtWidgets.QPushButton("Plot Population")
        
        # FIXED: Re-added Classic CSV Export button
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

        layout.addWidget(self.file)
        layout.addWidget(browse)
        layout.addWidget(cutoff_label)
        layout.addWidget(self.cutoff)
        layout.addWidget(load_btn)
        layout.addWidget(cluster_btn)
        layout.addWidget(best_btn)
        layout.addWidget(medoid_btn)
        layout.addWidget(plot_btn)
        layout.addWidget(export_btn)
        layout.addWidget(self.table_widget)
        
        tab.setLayout(layout)
        self.tabs.addTab(tab, "AutoDock4 (.DLG)")

    def init_vina_tab(self):
        """Tab 2: AutoDock Vina Layout."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        self.vina_file_line = QtWidgets.QLineEdit()
        browse_btn = QtWidgets.QPushButton("Browse Vina Output (.pdbqt / .pdb)")
        load_btn = QtWidgets.QPushButton("Load Poses & Compute Thermodynamics")
        
        # FIXED: Added Vina CSV Export button
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
        """Tab 3: Acknowledgements & Academic Affiliation Info."""
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout()

        # Enclosing values inside labels with rich text parsing formatting
        title = QtWidgets.QLabel("<h1 align='center'>AutoDockAnalyzer 3</h1>")
        title.setAlignment(QtCore.Qt.AlignCenter)
        
        info = QtWidgets.QLabel(
            "<p align='center' style='font-size:12pt; line-height:150%;'>"
            "<b>Developer:</b> Javier García Marín<br>"
            "<i>Organic and Inorganic Chemistry Department</i><br>"
            "<b>University of Alcalá</b><br>"
            "<a href='mailto:javier.garciamarin@uah.es'>javier.garciamarin@uah.es</a>"
            "</p>"
        )
        info.setAlignment(QtCore.Qt.AlignCenter)
        # Asegúrate de habilitar la apertura de enlaces externos si es necesario
        info.setOpenExternalLinks(True)
        #
        layout.addStretch()
        layout.addWidget(title)
        layout.addWidget(info)
        layout.addStretch()

        tab.setLayout(layout)
        self.tabs.addTab(tab, "About")

    # ---------- CLASSIC SLOTS ----------
    def browse_classic(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select DLG", "", "DLG files (*.dlg *.DLG)")
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
        self.table_widget.setColumnCount(6)
        self.table_widget.setHorizontalHeaderLabels(["Cluster", "Size", "Best Energy", "Kd (M)", "pKd", "Mean Energy"])

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
                kd = np.exp(delta_g / (R * T))
                pkd = -np.log10(kd)
            except OverflowError:
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
    addmenuitemqt("AutoDockAnalyzer 3", run_plugin)

cmd.extend("AutoDockAnalyzer 3", run_plugin)