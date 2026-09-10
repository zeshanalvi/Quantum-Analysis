#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# Quantum Annealing (Ideal/Noiseless) via Mean-Field Spin-Vector Dynamics
# ----------------------------------------------------------------------------------------------------------------

# Notes: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Code focuses on Performance, Scalability, Reliability, Carbon footprints
# Output folders are:
    #   Performance = runtime
    #   Scalability = growth of runtime as N increases (digits/bits).
    #   Reliability = success rate + dispersion across repetitions.
    #   Carbon footprints = derived *only* from PERFORMANCE totals
    #   (sum of successful wall-times), device power 65 Watts fixed for comparasion, PUE 1.2 fixed, 
    #   country carbon intensity (kg CO2/kWh). I did not use scalability or reliability results for CO2.

#      Carbon Data File: 
#      Uses ONLY Performance runtimes (not Scalability/Reliability), 
#      Placed on my Desktop Excel of country intensities "Filtered CO2 intensity 236 Countries.xlsx"
#      (CSV or XLSX; year column optional. I have selected --year-select latest.)
# ----------------------------------------------------------------------------------------------------------------
"""
Quantum Annealing (Ideal/Noiseless) via Mean-Field Spin-Vector Dynamics
Hamiltonian path (linear):
  H(s) = -(1-s) * Γ * sum_i σ_i^x  - s * sum_(i,j) J_ij σ_i^z σ_j^z  - s * sum_i h_i σ_i^z
Dynamics (noiseless, mean-field):
  d m_i / dt = 2 * m_i × B_i(s),   with B_i(s) = ( (1-s)Γ, 0, s(∑_j J_ij m_{j,z} + h_i) )

Round sign(m_{i,z}) at the end to obtain a final bitstring and measure optimality gap.

Default CLI mirrors my classical annealing script 1.1.
"""


# ------------********------------ Imports
#These imports collect standard utilities for filesystem access, timing, CLI parsing, warnings, 
#    and memory profiling, plus core math/numerics via numpy.
#They set up data handling with pandas and headless plotting by 
#  forcing Matplotlib’s "Agg" backend for non-interactive environments.
#ProcessPoolExecutor/as_completed are included to parallelize experiments across processes 
#  and gather results as workers finish.

import os, sys, time, argparse, warnings, pathlib, tracemalloc, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------********------------
# ------------------------- Instance generation (planted optimum) 
# ------------********------------

#def make_planted_ising constructs a planted Ising problem on an Erdős–Rényi ER(N,p) graph with a hidden optimal spin configuration s★.
#It assigns couplings J_ij = s★_i s★_j on sampled edges and aligns local fields h_i with s★ so that s★ is energetically favored.
#The function returns the graph, parameters, planted spins, and the analytic optimum energy E★ = −(#edges) − field_strength·N.

def make_planted_ising(N: int, p: float, field_strength: float, seed: int):
    """
    ER(N,p) graph; planted s★ ∈ {±1}; set J_ij = s★_i s★_j on edges; h_i = field_strength * s★_i.
    Then E(s★) = -(#edges) - field_strength * N  (unique optimum if field_strength>0).
    """
    rng = np.random.default_rng(seed)
    s_star = rng.choice([-1, 1], size=N).astype(np.int8)
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if rng.random() < p:
                edges.append((i, j))
    m = len(edges)
    J = {(i, j): int(s_star[i]) * int(s_star[j]) for (i, j) in edges}
    h = field_strength * s_star.astype(np.float64)
    E_star = -m - field_strength * N
    return edges, J, h, s_star, E_star

#def energy_of evaluates the Ising Hamiltonian by summing all pairwise interaction terms over edges and subtracting the linear field contribution.
#It computes −∑(i,j) J_ij s_i s_j − h·s for a given spin configuration and returns the total energy as a float.

def energy_of(spins: np.ndarray, edges, J, h) -> float:
    e = 0.0
    for (i, j) in edges:
        e -= J[(i, j)] * spins[i] * spins[j]
    e -= np.dot(h, spins)
    return float(e)

#def build_neighbors creates an adjacency list for the Ising graph, 
#   mapping each spin index to its neighboring spins and corresponding couplings J_ij.
#It symmetrically inserts each edge so local energy changes can be 
#   computed efficiently in O(degree) time per spin update.

def build_neighbors(N, edges, J):
    nbrs_by_i = {i: [] for i in range(N)}
    for (i, j) in edges:
        Jij = J[(i, j)]
        nbrs_by_i[i].append((j, Jij))
        nbrs_by_i[j].append((i, Jij))
    return nbrs_by_i

# ------------********------------
# ------------------------- Mean-field QA dynamics
# ------------********------------


#def mean_field_rhs computes the mean-field equations of motion dm/dt for all spins under a transverse-field Ising schedule at interpolation parameter s.
#It builds the effective field B_i = ( (1−s)Γ, 0, s(∑_j J_ij m_j^z + h_i) ) from neighbor magnetizations and local fields.
#The time derivatives are obtained via the classical spin precession rule dm_i = 2·(m_i × B_i), returned as an (N,3) array.

def mean_field_rhs(m: np.ndarray, s: float, Gamma: float, nbrs_by_i, h: np.ndarray) -> np.ndarray:
    """
     RHS dm/dt for all spins at schedule point s.
    m shape: (N,3) with rows [mx, my, mz] per spin; |m_i|≈1 ideally.
    """
    N = m.shape[0]
    Bx = (1.0 - s) * Gamma
    z_field = np.zeros(N, dtype=np.float64)
    # z_field[i] = sum_j J_ij * m[j,2]
    for i in range(N):
        acc = 0.0
        for j, Jij in nbrs_by_i[i]:
            acc += Jij * m[j, 2]
        z_field[i] = s * (acc + h[i])

    dm = np.zeros_like(m)
    # dm_i = 2 * m_i × B_i ;  B_i = (Bx, 0, z_field[i])
    # Cross product:
    # mx,my,mz crossed with (Bx,0,Bz)
    # = ( my*Bz - mz*0, mz*Bx - mx*Bz, mx*0 - my*Bx )
    dm[:, 0] = 2.0 * (m[:, 1] * z_field)                   # dmx
    dm[:, 1] = 2.0 * (m[:, 2] * Bx - m[:, 0] * z_field)    # dmy
    dm[:, 2] = -2.0 * (m[:, 1] * Bx)                       # dmz
    return dm

def renorm_rows(m: np.ndarray):
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    m[:] = m / norms

def evolve_mean_field_QA(N: int, steps: int, Gamma: float, nbrs_by_i, h: np.ndarray, seed: int):
    """
    Heun (RK2) integrator over s ∈ [0,1]; t discretized into `steps`.
    Init state = +x for all spins with tiny seed-dependent perturbation to break ties.
    Returns final magnetizations m (N,3).
    """
    rng = np.random.default_rng(seed)
    m = np.zeros((N, 3), dtype=np.float64)
    m[:, 0] = 1.0  # +x ground state at s=0
    # tiny coherent perturbations (noiseless but breaks symmetry deterministically per seed)
    eps = 1e-3
    m[:, 1] += eps * rng.standard_normal(N)
    m[:, 2] += eps * rng.standard_normal(N)
    renorm_rows(m)

    dt = 1.0 / max(1, steps)
    s = 0.0
    for k in range(steps):
        s_k = s
        s_k1 = min(1.0, s + dt)
        k1 = mean_field_rhs(m, s_k,  Gamma, nbrs_by_i, h)
        m_pred = m + dt * k1
        renorm_rows(m_pred)
        k2 = mean_field_rhs(m_pred, s_k1, Gamma, nbrs_by_i, h)
        m += 0.5 * dt * (k1 + k2)
        renorm_rows(m)
        s = s_k1
    return m

# ------------********------------
# ------------------------- One benchmark run
# ------------********------------

#def run_single executes one mean-field quantum annealing trial on a planted Ising instance 
#        and evaluates the resulting solution quality.
#It evolves spin magnetizations under a transverse-field schedule, 
#        then performs readout by taking the sign of the final z-magnetization.
#The function profiles runtime and memory, 
#        computes the normalized energy gap to the planted optimum, 
#        flags success by gap_tol, 
#        and returns all metrics in a result dictionary.

def run_single(N: int, steps: int, p: float, field_strength: float,
               Gamma: float, gap_tol: float, seed: int):
    edges, J, h, s_star, E_star = make_planted_ising(N, p, field_strength, seed)
    nbrs_by_i = build_neighbors(N, edges, J)

    tracemalloc.start()
    t0 = time.perf_counter()

    m = evolve_mean_field_QA(N, steps, Gamma, nbrs_by_i, h, seed+1)
    # Readout: round by sign of m_z
    spins = np.sign(m[:, 2]).astype(np.int8)
    spins[spins == 0] = 1  # tie -> +1
    best_E = energy_of(spins, edges, J, h)

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gap = (best_E - E_star) / (abs(E_star) + 1e-12)
    success = bool(gap <= gap_tol)

    return {
        "n": N,
        "edges": len(edges),
        "p": p,
        "steps": steps,
        "Gamma": Gamma,
        "field_strength": field_strength,
        "seed": seed,
        "best_energy": float(best_E),
        "opt_energy": float(E_star),
        "gap": float(gap),
        "success": success,
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024**2)),
    }

# ------------********------------
# ------------------------- Carbon footprint (Performance only)
# ------------********------------

def resolve_excel_path(excel_arg: str) -> str:
    p = pathlib.Path(excel_arg)
    if p.is_absolute() and p.exists(): return str(p)
    desktop = pathlib.Path.home() / "Desktop" / excel_arg
    return str(desktop if desktop.exists() else p)

def load_carbon_excel(path: str, year_select: str = "latest") -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}
    country = next((v for k,v in cols.items() if "country" in k or "nation" in k or k=="location"), None)
    if country is None: raise ValueError("No 'Country' column found.")
    year = next((v for k,v in cols.items() if "year" in k or "date" in k), None)
    intensity = next((v for k,v in cols.items()
                      if "intensity" in k or ("co2" in k and ("kwh" in k or "/kwh" in k)) or "kgco2" in k or "gco2" in k), None)
    if intensity is None:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if year in numeric: numeric.remove(year)
        if not numeric: raise ValueError("No numeric intensity column detected.")
        intensity = numeric[0]
    keep = [country] + ([year] if year else []) + [intensity]
    df = df[keep].copy()
    df.columns = ["Country","Year","Intensity"] if len(keep)==3 else ["Country","Intensity"]
    if "Year" in df.columns and year_select.lower()=="latest":
        df = df.sort_values(["Country","Year"]).groupby("Country", as_index=False).tail(1)
    med = float(df["Intensity"].dropna().median())
    if med > 50: df["Intensity"] = df["Intensity"] / 1000.0  # g/kWh -> kg/kWh
    return df.dropna(subset=["Country","Intensity"]).reset_index(drop=True)

def compute_carbon(perf_df: pd.DataFrame, intensity_df: pd.DataFrame,
                   power_watts: float, pue: float):
    total_runtime_s = float(perf_df["runtime_s"].sum())
    kWh_total = power_watts * pue * total_runtime_s / 3_600_000.0
    df = intensity_df.copy()
    df["kWh"] = kWh_total
    df["kgCO2e"] = df["Intensity"] * df["kWh"]
    summary = pd.DataFrame({
        "total_runtime_s": [total_runtime_s],
        "power_watts": [power_watts],
        "PUE": [pue],
        "kWh_total": [kWh_total],
        "median_intensity_kg_per_kWh": [float(df["Intensity"].median())],
        "mean_kgCO2e_across_countries": [float(df["kgCO2e"].mean())],
    })
    return df.sort_values("kgCO2e", ascending=False), summary

# ------------********------------
# ------------------------- Plotting
# ------------********------------

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    try:
        plt.boxplot(data, tick_labels=labels)
    except TypeError:
        plt.boxplot(data, labels=labels)

# ------------********------------
def plot_performance(df, outdir):
    g = df.groupby("n")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["runtime_s"], marker="o")
    plt.title("Performance: Runtime vs N"); plt.xlabel("N (spins)"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("n")["gap"].mean().reset_index()
    plt.figure(); plt.plot(g2["n"], g2["gap"], marker="s")
    plt.title("Performance: Mean optimality gap vs N"); plt.xlabel("N (spins)"); plt.ylabel("Mean gap")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_gap_vs_N.png"), **plt_kwargs); plt.close()

# ------------********------------
def plot_scalability(df, outdir):
    g = df.groupby("n")["runtime_s"].mean().reset_index()
    plt.figure(); plt.loglog(g["n"], g["runtime_s"], marker="o")
    plt.title("Scalability: Runtime (log–log)"); plt.xlabel("N (spins)"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, which="both", alpha=0.3); plt.savefig(os.path.join(outdir, "scal_loglog_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("n")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g2["n"], g2["peak_mem_mb"], marker="^")
    plt.title("Scalability: Peak memory vs N"); plt.xlabel("N (spins)"); plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_N.png"), **plt_kwargs); plt.close()

# ------------********------------
def plot_reliability(df, outdir):
    g = df.groupby("n")["success"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["success"], marker="o"); plt.ylim(0,1.05)
    plt.title("Reliability: Success rate vs N (gap≤tol)"); plt.xlabel("N"); plt.ylabel("Success rate")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_N.png"), **plt_kwargs); plt.close()

    data = [df[df["n"]==k]["gap"].values for k in sorted(df["n"].unique())]
    plt.figure(); _boxplot_with_labels(data, labels=sorted(df["n"].unique()))
    plt.title("Reliability: Optimality gap distribution"); plt.xlabel("N"); plt.ylabel("Gap")
    plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, "rel_gap_boxplot.png"), **plt_kwargs); plt.close()

# ------------********------------
def plot_carbon(df, outdir):
    top = df.nlargest(15, "kgCO2e")
    plt.figure(figsize=(8,5)); plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
    plt.title("Carbon: Top 15 countries (kgCO2e)"); plt.xlabel("kg CO2e")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "carbon_top15.png"), **plt_kwargs); plt.close()

    plt.figure(); plt.hist(df["kgCO2e"], bins=30)
    plt.title("Carbon: Emission distribution"); plt.xlabel("kg CO2e"); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "carbon_distribution.png"), **plt_kwargs); plt.close()

# ------------********------------
# ------------------------- Main 
# ------------********------------

def main():
    parser = argparse.ArgumentParser(description="Quantum Annealing (Ideal/Noiseless) Benchmarks via Mean-Field Dynamics")
    # Keep CLI aligned with classical SA
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 192], help="Problem sizes N (spins).")
    parser.add_argument("--steps", type=int, default=1000, help="Anneal integration steps (Heun RK2).")
    parser.add_argument("--trials", type=int, default=10, help="Trials per N (different seeds).")
    parser.add_argument("--p", type=float, default=0.2, help="ER graph edge probability.")
    parser.add_argument("--field-strength", type=float, default=0.1, help="h_i = field_strength * s★_i.")
    parser.add_argument("--Gamma", type=float, default=1.0, help="Initial transverse field Γ.")
    parser.add_argument("--gap-tol", type=float, default=0.01, help="Success if (E- E*)/|E*| ≤ gap_tol.")
    # carbon & infra
    parser.add_argument("--excel", type=str, default="Filtered CO2 intensity 236 Countries.xlsx")
    parser.add_argument("--device-power-watts", type=float, default=65.0)
    parser.add_argument("--pue", type=float, default=1.2)
    parser.add_argument("--year-select", type=str, default="latest")
    parser.add_argument("--outdir", type=str, default="carbon_by_country")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    args = parser.parse_args()

    # Output folders
    perf_dir = os.path.join(os.getcwd(), "Performance")
    scal_dir = os.path.join(os.getcwd(), "Scalability")
    rel_dir  = os.path.join(os.getcwd(), "Reliability")
    carb_root = os.path.join(os.getcwd(), "Carbon footprints")
    carb_dir  = os.path.join(carb_root, args.outdir)
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]:
        os.makedirs(d, exist_ok=True)

    print("[run] Starting ideal quantum annealing benchmarks...")
    jobs = [(N, args.steps, args.p, args.field_strength, args.Gamma, args.gap_tol, 1000 + 17*N + i)
            for N in args.sizes for i in range(args.trials)]
    rows = []

    def consume(res):
        rows.append(res)
        print(f"  - N={res['n']} seed={res['seed']} gap={res['gap']:.3e} "
              f"runtime={res['runtime_s']:.4f}s success={int(res['success'])}")

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_single, *job) for job in jobs]
            for f in as_completed(futs):
                consume(f.result())
    else:
        for job in jobs:
            consume(run_single(*job))

    df = pd.DataFrame(rows)

# ------------********------------
    # ----- Performance -----
#This block saves detailed and aggregated performance results to an Excel workbook while suppressing non-critical warnings from the writer.
#It groups runs by problem size n to compute mean runtime, memory, gap, and success rate, writing each table to a separate sheet.
#Finally, it generates performance plots from the raw results and stores them in the performance output directory.

    perf_xlsx = os.path.join(perf_dir, "performance_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw_runs")
            agg = df.groupby("n").agg(
                mean_runtime_s=("runtime_s","mean"),
                mean_peak_mem_mb=("peak_mem_mb","mean"),
                mean_gap=("gap","mean"),
                success_rate=("success","mean"),
            ).reset_index()
            agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(df, perf_dir)

# ------------********------------
    # ----- Scalability -----
#This block writes scalability measurements to an Excel file, storing per-run runtime and memory
#    in a raw sheet and size-averaged metrics in an aggregated sheet.
#It suppresses writer warnings for cleaner logs and uses pandas groupby to compute mean scaling behavior by problem size n.
#After saving the tables, it generates and saves scalability plots in the designated output directory.
    scal_xlsx = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_xlsx, engine="openpyxl") as w:
            df[["n","runtime_s","peak_mem_mb"]].to_excel(w, index=False, sheet_name="raw")
            df.groupby("n").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(df, scal_dir)

# ------------********------------
    # ----- Reliability -----
#This block records reliability results to an Excel workbook, saving per-run gaps 
#   and success flags alongside aggregated statistics by problem size.
#It computes success rates and gap mean/std via groupby aggregation while suppressing non-essential writer warnings.
#Finally, it produces reliability plots from the run-level data and 
#  stores them in the reliability output directory.

    rel_xlsx = os.path.join(rel_dir, "reliability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(rel_xlsx, engine="openpyxl") as w:
            df[["n","seed","gap","success"]].to_excel(w, index=False, sheet_name="runs")
            rel_agg = df.groupby("n").agg(
                success_rate=("success","mean"),
                mean_gap=("gap","mean"),
                std_gap=("gap","std"),
            ).reset_index()
            rel_agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_reliability(df, rel_dir)

# ------------********------------
    # ----- Carbon (PERFORMANCE runtime only) -----
#This block resolves and loads a carbon-intensity Excel dataset, 
#   then computes energy use and CO₂ emissions from the experiment runtimes.
#It applies the specified device power and PUE to estimate total kWh and per-country kgCO2e, 
#   producing both detailed and summary tables.
#All carbon results, along with the original intensity input, are written to a multi-sheet Excel workbook with warnings suppressed.

    excel_path = resolve_excel_path(args.excel)
    try:
        intensity_df = load_carbon_excel(excel_path, year_select=args.year_select)
        carbon_df, summary_df = compute_carbon(df, intensity_df,
                                               power_watts=args.device_power_watts,
                                               pue=args.pue)
        carb_xlsx = os.path.join(carb_dir, "carbon_results.xlsx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pd.ExcelWriter(carb_xlsx, engine="openpyxl") as w:
                carbon_df.to_excel(w, index=False, sheet_name="per_country")
                summary_df.to_excel(w, index=False, sheet_name="summary")
                intensity_df.to_excel(w, index=False, sheet_name="intensity_input")

# ------------********------------quick plots
    
        top = carbon_df.nlargest(15, "kgCO2e")
        plt.figure(figsize=(8,5)); plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
        plt.title("Carbon: Top 15 countries (kgCO2e)"); plt.xlabel("kg CO2e")
        plt.tight_layout(); plt.savefig(os.path.join(carb_dir, "carbon_top15.png"), **plt_kwargs); plt.close()

        plt.figure(); plt.hist(carbon_df["kgCO2e"], bins=30)
        plt.title("Carbon: Emission distribution"); plt.xlabel("kg CO2e"); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(carb_dir, "carbon_distribution.png"), **plt_kwargs); plt.close()
        print(f"[carbon] Wrote Excel and plots to: {carb_dir}")
    except Exception as e:
        print(f"[carbon] ERROR reading Excel '{excel_path}': {e}")
        print("[carbon] Skipping carbon benchmark. Re-run once the file is available.")

    print("\n=== Summary ===")
    print("Performance Excel:", perf_xlsx)
    print("Scalability Excel:", scal_xlsx)
    print("Reliability Excel:", rel_xlsx)
    print("Carbon Excel:", os.path.join(carb_dir, "carbon_results.xlsx"))
    print("Folders created:\n -", perf_dir, "\n -", scal_dir, "\n -", rel_dir, "\n -", carb_dir)

if __name__ == "__main__":
    main()
