#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
# PLUGIN: AutoDockAnalyzer (ADA)
# VERSION: 2.6
# DATE: 2026-07-01
# AUTHOR: Javier García Marín (assisted by M365 Copilot & Gemini Flash)
# AFFILIATION: University of Alcalá (Department of Organic & Inorganic Chemistry)
# LICENSE: MIT License
# ==============================================================================
"""
DESCRIPTION:
    Interactive PyMOL 3.X plugin for the comprehensive analysis of AutoDock 
    docking results (DLG files).

FEATURES:
    - AutoDock-like clustering algorithm with automatic PyMOL object generation.
    - Dynamic object naming incorporating the user-defined RMSD cutoff.
    - Interactive table with single-click cluster isolation in PyMOL viewer.
    - Sortable data columns and thermodynamic analysis (Kd calculations).
    - Cluster population bar plotting with matplotlib.

VERSION HISTORY:
    v2.6 (2026-07-01):
        - Added: Native numeric sorting for table columns (Size, Energy, Kd, Mean).
        - Fixed: Prevented string-sorting issues (e.g., '10' coming before '2').
    v2.5 (2026-07-01):
        - Removed: 'Create Cluster States' button; creation is now fully automated.
        - Changed: Cluster object names now dynamically include the RMSD cutoff value.
        - Added: Single-click isolation on table rows to toggle visibility in PyMOL.
    v2.5 (2026-06-28):
        - Removed: 'Create Cluster States' button; creation is now fully automated.
        - Changed: Cluster object names now dynamically include the RMSD cutoff value.
        - Added: Single-click isolation on table rows to toggle visibility in PyMOL.
    v2.2 (2026-06-15): 
        - Fixed: Robust coordinate parsing using split() fallback.
        - Added: Crash-safe checks in GUI (validation for file loading).
    v2.1 (2026-04-01):
        - Initial clustering implementation and basic PyMOL integration.
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

    def __init__(self):
        self.poses = []
        self.energies = []
        self.clusters = []
        self.current_cutoff = 2.0

    # ---------- VISUAL ----------
    def apply_visual_style(self, obj):
        cmd.hide("everything", obj)
        cmd.show("sticks", obj)
        cmd.hide("spheres", obj)
        cmd.hide("nonbonded", obj)
        cmd.set("sphere_scale", 0.0, obj)

    # ---------- LOAD ----------
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

        print(f"[ADA] Loaded {len(self.poses)} poses successfully.")
        return True

    def load_into_pymol(self):
        if not self.poses:
            return
        obj = "lig_all"
        cmd.delete(obj)

        for i, pose in enumerate(self.poses):
            cmd.read_pdbstr("".join(pose), obj, state=i+1)

        self.apply_visual_style(obj)

    # ---------- GEOMETRY ----------
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
        if len(a) != len(b) or len(a) == 0:
            return 999.0
        return np.sqrt(((a - b) ** 2).sum() / len(a))

    # ---------- CLUSTER ----------
    def cluster(self, cutoff=2.0):
        if not self.poses:
            print("[ADA] No poses loaded to cluster.")
            return

        self.current_cutoff = cutoff
        order = sorted(
            range(len(self.poses)),
            key=lambda i: self.energies[i] if self.energies[i] is not None else 999
        )

        assigned = [False] * len(self.poses)
        clusters = []

        for i in order:
            if assigned[i]:
                continue

            seed = self.coords(self.poses[i])
            cl = [i]
            assigned[i] = True

            for j in order:
                if assigned[j]:
                    continue

                if self.rmsd(seed, self.coords(self.poses[j])) <= cutoff:
                    cl.append(j)
                    assigned[j] = True

            clusters.append(cl)

        self.clusters = clusters
        print("[ADA] Clusters generated:", len(clusters))
        self.create_clusters()

    # ---------- TABLE ----------
    def table(self):
        R = 0.001987
        T = 298.15
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
                    kd = float('inf')
                    pkd = float('-inf')
            else:
                kd = None
                pkd = None

            # Return raw values (or defaults) for strict row mapping
            data.append([int(i+1), int(len(c)), best, kd, pkd, mean])

        data.sort(key=lambda x: x[2] if x[2] is not None else 999)
        return data

    # ---------- OBJECTS ----------
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
            best_idx = None
            best_score = float("inf")

            for idx_i in c:
                coords_i = self.coords(self.poses[idx_i])
                total = 0
                for idx_j in c:
                    total += self.rmsd(coords_i, self.coords(self.poses[idx_j]))
                avg = total / len(c)

                if avg < best_score:
                    best_score = avg
                    best_idx = idx_i

            if best_idx is not None:
                name = f"medoid_{i+1}"
                cmd.delete(name)
                cmd.read_pdbstr("".join(self.poses[best_idx]), name)
                self.apply_visual_style(name)

    # ---------- PLOT ----------
    def plot(self):
        table = self.table()
        if not table:
            return
        clusters = [row[0] for row in table]
        sizes = [row[1] for row in table]

        plt.figure()
        plt.bar(clusters, sizes)
        plt.xlabel("Cluster ID")
        plt.ylabel("Population (#poses)")
        plt.title("Cluster Population")
        plt.savefig("cluster_population.png", dpi=300)
        plt.show()

    # ---------- EXPORT CSV ----------
    def export_csv(self, filename):
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Cluster", "Size", "Best Energy", "Kd (M)", "pKd", "Mean Energy"])
            for row in self.table():
                kd_out = f"{row[3]:.2e}" if isinstance(row[3], float) else ""
                pkd_out = f"{row[4]:.2f}" if isinstance(row[4], float) else ""
                energy_out = row[2] if row[2] is not None else ""
                mean_out = row[5] if row[5] is not None else ""
                writer.writerow([row[0], row[1], energy_out, kd_out, pkd_out, mean_out])
        print("[ADA] CSV exported:", filename)


ada = ADA()


# ================= GUI =================

class GUI(QtWidgets.QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoDockAnalyzer v2.6")
        
        # Enforce a slightly wider default size so all 6 columns fit comfortably
        self.resize(650, 500)
        
        layout = QtWidgets.QVBoxLayout()

        self.file = QtWidgets.QLineEdit()
        browse = QtWidgets.QPushButton("Browse")

        cutoff_label = QtWidgets.QLabel("RMSD cutoff (Å)")
        self.cutoff = QtWidgets.QLineEdit("2.0")

        load_btn = QtWidgets.QPushButton("Load DLG")
        cluster_btn = QtWidgets.QPushButton("Run Clustering")

        best_btn = QtWidgets.QPushButton("Extract Best Poses")
        medoid_btn = QtWidgets.QPushButton("Extract Centroids (Medoids)")
        plot_btn = QtWidgets.QPushButton("Plot Population")
        export_btn = QtWidgets.QPushButton("Export CSV Report")

        self.table_widget = QtWidgets.QTableWidget()

        # Connections
        browse.clicked.connect(self.browse)
        load_btn.clicked.connect(self.load)
        cluster_btn.clicked.connect(self.cluster)
        best_btn.clicked.connect(self.run_best)
        medoid_btn.clicked.connect(self.run_medoid)
        plot_btn.clicked.connect(self.run_plot)
        export_btn.clicked.connect(self.save_csv)
        
        self.table_widget.cellClicked.connect(self.handle_table_click)

        # Layout Setup
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

        self.setLayout(layout)

    def browse(self):
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select DLG", "", "DLG files (*.dlg *.DLG);;All files (*)"
        )
        if file_name:
            self.file.setText(file_name)

    def load(self):
        if self.file.text():
            if ada.load_dlg(self.file.text()):
                ada.load_into_pymol()
        else:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please select a DLG file first.")

    def cluster(self):
        if not ada.poses:
            QtWidgets.QMessageBox.warning(self, "Warning", "No poses loaded. Load a DLG file first.")
            return

        try:
            cutoff_val = float(self.cutoff.text())
        except ValueError:
            QtWidgets.QMessageBox.critical(self, "Error", "Invalid RMSD cutoff value.")
            return

        # Explicitly disable sorting while populating to avoid indexing mismatches
        self.table_widget.setSortingEnabled(False)
        self.table_widget.clear()

        ada.cluster(cutoff_val)
        table_data = ada.table()

        self.table_widget.setRowCount(len(table_data))
        self.table_widget.setColumnCount(6)
        self.table_widget.setHorizontalHeaderLabels(
            ["Cluster", "Size", "Best Energy", "Kd (M)", "pKd", "Mean Energy"]
        )

        for i, row in enumerate(table_data):
            # 0. Cluster ID
            self.table_widget.setItem(i, 0, NumericTableWidgetItem(row[0]))
            # 1. Size
            self.table_widget.setItem(i, 1, NumericTableWidgetItem(row[1]))
            # 2. Best Energy
            self.table_widget.setItem(i, 2, NumericTableWidgetItem(row[2]))
            
            # 3. Kd (M) string representation + numeric backing
            if isinstance(row[3], float) and row[3] != float('inf'):
                kd_item = NumericTableWidgetItem(row[3])
                kd_item.setText(f"{row[3]:.2e}")
                self.table_widget.setItem(i, 3, kd_item)
            else:
                self.table_widget.setItem(i, 3, NumericTableWidgetItem(str(row[3]) if row[3] is not None else ""))
                
            # 4. pKd string representation + numeric backing
            if isinstance(row[4], float) and row[4] != float('-inf'):
                pkd_item = NumericTableWidgetItem(row[4])
                pkd_item.setText(f"{row[4]:.2f}")
                self.table_widget.setItem(i, 4, pkd_item)
            else:
                self.table_widget.setItem(i, 4, NumericTableWidgetItem(str(row[4]) if row[4] is not None else ""))
                
            # 5. Mean Energy
            self.table_widget.setItem(i, 5, NumericTableWidgetItem(row[5]))

        # Dynamic layout adjustments to guarantee column deployment
        self.table_widget.resizeColumnsToContents()
        self.table_widget.setSortingEnabled(True)

    def handle_table_click(self, row, column):
        if not ada.clusters:
            return
        try:
            cluster_id = self.table_widget.item(row, 0).text()
            cutoff_str = str(ada.current_cutoff).replace('.', '_')
            target_object = f"cluster_{cluster_id}_rc{cutoff_str}"
            
            cmd.hide("everything", "cluster_*")
            cmd.show("sticks", target_object)
            print(f"[ADA] Isolated PyMOL workspace view for object: {target_object}")
        except Exception as e:
            print(f"[ADA] Interaction mapping exception: {str(e)}")

    def _check_clusters(self):
        if not ada.clusters:
            QtWidgets.QMessageBox.warning(self, "Warning", "Please run clustering first.")
            return False
        return True

    def run_best(self):
        if self._check_clusters(): ada.best()

    def run_medoid(self):
        if self._check_clusters(): ada.medoid()

    def run_plot(self):
        if self._check_clusters(): ada.plot()

    def save_csv(self):
        if not self._check_clusters():
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save CSV", "", "*.csv"
        )
        if filename:
            ada.export_csv(filename)


# ================= PLUGIN REGISTRATION =================

dialog = None

def run_plugin():
    global dialog
    dialog = GUI()
    dialog.show()

def __init_plugin__(app=None):
    from pymol.plugins import addmenuitemqt
    addmenuitemqt("AutoDockAnalyzer", run_plugin)

cmd.extend("AutoDockAnalyzer", run_plugin)
print("✅ AutoDockAnalyzer v2.6 LOADED (Strict layout constraints fixed)")
