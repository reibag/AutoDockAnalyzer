#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoDockAnalyzer (ADA) - Runtime Scaling Benchmark Tool
Measures execution time across increasing pose counts (Clustering + Medoid Extraction).
"""

import time
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


class HeadlessADA:
    def __init__(self):
        self.poses = []
        self.energies = []
        self.clusters = []
        self.current_cutoff = 2.0

    def load_dlg(self, filename):
        if not os.path.exists(filename):
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
        return True

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

    def cluster(self, cutoff=2.0):
        if not self.poses: return
        self.current_cutoff = cutoff
        order = sorted(range(len(self.poses)), key=lambda i: self.energies[i] if self.energies[i] is not None else 999)
        assigned = [False] * len(self.poses)
        clusters = []

        # Pre-compute coordinates to optimize vector operations
        cached_coords = [self.coords(p) for p in self.poses]

        for i in order:
            if assigned[i]: continue
            seed = cached_coords[i]
            cl = [i]
            assigned[i] = True
            for j in order:
                if assigned[j]: continue
                if self.rmsd(seed, cached_coords[j]) <= cutoff:
                    cl.append(j)
                    assigned[j] = True
            clusters.append(cl)

        self.clusters = clusters

    def calculate_medoids(self):
        if not self.clusters: return
        cached_coords = [self.coords(p) for p in self.poses]
        medoid_indices = []

        for c in self.clusters:
            best_idx, best_score = None, float("inf")
            for idx_i in c:
                coords_i = cached_coords[idx_i]
                total = sum(self.rmsd(coords_i, cached_coords[idx_j]) for idx_j in c)
                avg = total / len(c)
                if avg < best_score:
                    best_score, best_idx = avg, idx_i
            medoid_indices.append(best_idx)
        return medoid_indices


# ==============================================================================
# BENCHMARK SUITE
# ==============================================================================

def quadratic_model(n, a, b, c):
    """Polynomial model O(N^2) for curve fitting"""
    return a * (n ** 2) + b * n + c

def run_benchmark(dlg_path, output_png="ADA_runtime_scaling.png", output_txt="ADA_benchmark_report.txt", repeats=5, cutoff=2.0):
    ada = HeadlessADA()
    if not ada.load_dlg(dlg_path):
        raise FileNotFoundError(f"Error: Unable to locate or read the DLG file at: '{dlg_path}'")

    total_available = len(ada.poses)
    if total_available == 0:
        raise ValueError(f"The input file '{dlg_path}' does not contain valid docking conformations (DOCKED: MODEL).")

    print(f"[BENCHMARK] Input file: {os.path.abspath(dlg_path)}")
    print(f"[BENCHMARK] Total conformations loaded: {total_available}")

    # Automatically generate step intervals based on the number of poses available
    step_size = max(10, total_available // 7)
    pose_steps = list(range(step_size, total_available, step_size))
    if not pose_steps or pose_steps[-1] != total_available:
        pose_steps.append(total_available)

    mean_times_cluster = []
    sd_times_cluster = []
    mean_times_total = []
    sd_times_total = []

    header = f"{'Poses':>8} | {'Clustering (s)':>18} | {'Total (+Medoids) (s)':>22}"
    separator = "-" * 54

    print("\nStarting execution timing benchmarks:")
    print(header)
    print(separator)

    # Pre-allocate lines for the text report
    report_lines = [
        "=" * 70,
        "AutoDockAnalyzer (ADA) - Computational Runtime Scaling Benchmark",
        "=" * 70,
        f"Input file: {os.path.abspath(dlg_path)}",
        f"Total available poses: {total_available}",
        f"RMSD cutoff: {cutoff} Å",
        f"Repeats per pose count: {repeats}",
        f"Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "-" * 70,
        f"{'Poses':>8} | {'Clustering Mean (s)':>20} | {'Clustering SD (s)':>18} | {'Total Mean (s)':>16} | {'Total SD (s)':>14}",
        "-" * 70
    ]

    for n in pose_steps:
        t_cluster_runs = []
        t_total_runs = []

        for _ in range(repeats):
            bench_ada = HeadlessADA()
            bench_ada.poses = ada.poses[:n]
            bench_ada.energies = ada.energies[:n]

            # 1. Benchmark Greedy Clustering
            t0 = time.perf_counter()
            bench_ada.cluster(cutoff=cutoff)
            t_cluster = time.perf_counter() - t0

            # 2. Benchmark Medoid Extraction
            t0_med = time.perf_counter()
            bench_ada.calculate_medoids()
            t_medoid = time.perf_counter() - t0_med

            t_cluster_runs.append(t_cluster)
            t_total_runs.append(t_cluster + t_medoid)

        mean_c = np.mean(t_cluster_runs)
        std_c = np.std(t_cluster_runs)
        mean_tot = np.mean(t_total_runs)
        std_tot = np.std(t_total_runs)

        mean_times_cluster.append(mean_c)
        sd_times_cluster.append(std_c)
        mean_times_total.append(mean_tot)
        sd_times_total.append(std_tot)

        console_row = f"{n:8d} | {mean_c:10.4f} ± {std_c:5.4f} | {mean_tot:14.4f} ± {std_tot:5.4f}"
        print(console_row)

        report_lines.append(f"{n:8d} | {mean_c:20.5f} | {std_c:18.5f} | {mean_tot:16.5f} | {std_tot:14.5f}")

    # ==============================================================================
    # GENERATE PLOT
    # ==============================================================================
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)

    fitted_equation_str = "Quadratic fit unavailable"
    if len(pose_steps) >= 3:
        x_smooth = np.linspace(min(pose_steps), max(pose_steps), 200)
        try:
            popt, _ = curve_fit(quadratic_model, pose_steps, mean_times_total)
            ax.plot(x_smooth, quadratic_model(x_smooth, *popt), '--', color='#7f8c8d', 
                    label=r'Quadratic Fit ($\mathcal{O}(N^2)$)', linewidth=1.5, zorder=2)
            fitted_equation_str = f"Time(N) = {popt[0]:.6e}*N^2 + {popt[1]:.6e}*N + {popt[2]:.6e}"
        except Exception:
            pass

    ax.errorbar(pose_steps, mean_times_total, yerr=sd_times_total, fmt='-o', color='#2980b9',
                ecolor='#34495e', elinewidth=1.2, capsize=4, markersize=6, 
                label='Total (Clustering + Medoids)', linewidth=1.8, zorder=3)

    ax.errorbar(pose_steps, mean_times_cluster, yerr=sd_times_cluster, fmt='--s', color='#27ae60',
                ecolor='#2c3e50', elinewidth=1.0, capsize=3, markersize=5,
                label='Greedy Clustering only', linewidth=1.4, zorder=3)

    ax.set_title("AutoDockAnalyzer (ADA) Runtime Scaling", fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Poses ($N$)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Execution Time (seconds)", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(frameon=True, fontsize=10, loc='upper left')

    fig.tight_layout()
    fig.savefig(output_png, dpi=300)
    plt.close(fig)

    # Save summary report to text file
    report_lines.extend([
        "-" * 70,
        f"Quadratic Curve Fit: {fitted_equation_str}",
        "=" * 70
    ])
    with open(output_txt, "w") as f_out:
        f_out.write("\n".join(report_lines) + "\n")

    print(f"\n[OK] Plot successfully generated: {os.path.abspath(output_png)}")
    print(f"[OK] Summary report saved: {os.path.abspath(output_txt)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AutoDockAnalyzer (ADA) - Computational Runtime Scaling Benchmark CLI"
    )
    parser.add_argument(
        "--dlg", "-d",
        required=True,
        type=str,
        help="Path to the input AutoDock .dlg file."
    )
    parser.add_argument(
        "--output-fig", "-of",
        type=str,
        default="ADA_runtime_scaling.png",
        help="Output image filename/path for the scaling plot (default: ADA_runtime_scaling.png)."
    )
    parser.add_argument(
        "--output-txt", "-ot",
        type=str,
        default="ADA_benchmark_report.txt",
        help="Output text report filename/path (default: ADA_benchmark_report.txt)."
    )
    parser.add_argument(
        "--repeats", "-r",
        type=int,
        default=5,
        help="Number of iterations per pose interval to calculate mean and standard deviation."
    )
    parser.add_argument(
        "--cutoff", "-c",
        type=float,
        default=2.0,
        help="Cartesian RMSD clustering cutoff threshold in Angstroms."
    )

    args = parser.parse_args()

    run_benchmark(
        dlg_path=args.dlg,
        output_png=args.output_fig,
        output_txt=args.output_txt,
        repeats=args.repeats,
        cutoff=args.cutoff
    )