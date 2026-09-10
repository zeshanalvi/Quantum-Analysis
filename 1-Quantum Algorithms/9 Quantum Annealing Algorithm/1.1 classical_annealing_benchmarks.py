#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------------------------------------------
# Empirical Evaluation of: Classical Simulated Annealing (Ising/QUBO)
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
Problem: Ising on an ER random graph with a *planted* ground state s★ ∈ {±1}^N
Energy:  E(s) = - sum_{(i,j)∈E} J_ij s_i s_j - sum_i h_i s_i
Instance construction (planted optimum):
  J_ij = s★_i s★_j  for edges (i,j),    h_i = field_strength * s★_i
Then E(s★) = -(#edges) - field_strength * N  (unique optimum if field_strength>0)
"""


# ------------********------------ Imports
import os, sys, time, argparse, warnings, pathlib, tracemalloc, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------********------------
# ------------------------- Instance generation
# ------------********------------

#def make_planted_ising builds a planted Ising instance 
#      by sampling an Erdős–Rényi graph ER(N,p) and drawing a hidden spin assignment s★.
#It sets couplings on sampled edges as J_ij = s★_i s★_j (±1) 
#      and aligns local fields as h_i = field_strength * s★_i, making s★ energetically favored.
#The function returns the edge list, coupling dict, field vector, planted spins, 
#      and the analytic energy of the planted state E★ = −m − field_strength*N.

def make_planted_ising(N: int, p: float, field_strength: float, seed: int):
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

    E_star = -m - field_strength * N  # energy at s★
    return edges, J, h, s_star, E_star

# ------------********------------
# ------------------------- Energy & SA kernel
# ------------********------------

#def energy_of computes the Ising Hamiltonian energy by summing
#     pairwise edge interactions (−J_ij s_i s_j) and the linear field term (−h·s).
def energy_of(spins: np.ndarray, edges, J, h) -> float:
    e = 0.0
    for (i, j) in edges:
        e -= J[(i, j)] * spins[i] * spins[j]
    e -= np.dot(h, spins)
    return float(e)

#def delta_energy_if_flip returns the local energy change from flipping spin i using its neighbor contributions 
#   plus the on-site field, enabling fast Metropolis-style updates.
def delta_energy_if_flip(i: int, spins: np.ndarray, nbrs_by_i, J, h) -> float:
    s_i = spins[i]
    field_term = h[i]
    neigh_sum = 0.0
    for j, Jij in nbrs_by_i[i]:
        neigh_sum += Jij * spins[j]
    return 2.0 * s_i * (neigh_sum + field_term)

#def build_neighbors converts the edge/coupling representation into an 
#   adjacency list with J values for O(degree) local energy calculations per spin.
def build_neighbors(N, edges, J):
    nbrs_by_i = {i: [] for i in range(N)}
    for (i, j) in edges:
        Jij = J[(i, j)]
        nbrs_by_i[i].append((j, Jij))
        nbrs_by_i[j].append((i, Jij))
    return nbrs_by_i


#def simulated_annealing performs Metropolis simulated annealing on an Ising model 
#   using a linear temperature schedule from T0 to Tf over a given number of sweeps.
#Each sweep visits all spins in a random order, proposing flips with acceptance based 
#    on the local ΔE (always accept downhill moves, uphill with exp(−ΔE/T)).
#It tracks and returns the best (lowest) energy encountered along with the corresponding spin configuration.

def simulated_annealing(N: int, edges, J, h, steps: int, T0: float, Tf: float, seed: int):

    rng = np.random.default_rng(seed)
    spins = rng.choice([-1, 1], size=N).astype(np.int8)
    nbrs_by_i = build_neighbors(N, edges, J)

    E = energy_of(spins, edges, J, h)
    best_E, best_spins = E, spins.copy()

    if steps <= 1:
        temps = [Tf]
    else:
        temps = [T0 + (Tf - T0) * k / (steps - 1) for k in range(steps)]

    for T in temps:
        # one sweep
        for i in rng.permutation(N):
            dE = delta_energy_if_flip(i, spins, nbrs_by_i, J, h)
            if dE <= 0.0 or (T > 0.0 and rng.random() < math.exp(-dE / T)):
                spins[i] = -spins[i]
                E += dE
                if E < best_E:
                    best_E = E
                    best_spins = spins.copy()
    return best_E, best_spins

# ------------********------------------------
# ------------------------- One benchmark run
# ------------********------------------------

#def run_single generates a planted Ising instance, runs simulated annealing once, and records performance and solution quality metrics.
#It profiles runtime and peak memory with tracemalloc, 
#     then computes a normalized optimality gap relative to the planted optimum energy E★.
#The function flags success if the gap is within gap_tol and returns a results 
#     dictionary including graph size, energies, gap, timing, and memory.

def run_single(N: int, steps: int, p: float, field_strength: float, T0: float, Tf: float, gap_tol: float, seed: int):
    edges, J, h, s_star, E_star = make_planted_ising(N, p, field_strength, seed)

    tracemalloc.start()
    t0 = time.perf_counter()
    best_E, best_spins = simulated_annealing(N, edges, J, h, steps, T0, Tf, seed+1)
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
        "T0": T0,
        "Tf": Tf,
        "seed": seed,
        "best_energy": float(best_E),
        "opt_energy": float(E_star),
        "gap": float(gap),
        "success": success,
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024**2)),
    }

# ------------********------------
# ------------------------- Carbon footprint
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

def plot_performance(df, outdir):
    g = df.groupby("n")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["runtime_s"], marker="o")
    plt.title("Performance: Runtime vs N"); plt.xlabel("N (variables)"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("n")["gap"].mean().reset_index()
    plt.figure(); plt.plot(g2["n"], g2["gap"], marker="s")
    plt.title("Performance: Mean optimality gap vs N"); plt.xlabel("N (variables)"); plt.ylabel("Mean gap")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_gap_vs_N.png"), **plt_kwargs); plt.close()

def plot_scalability(df, outdir):
    g = df.groupby("n")["runtime_s"].mean().reset_index()
    plt.figure(); plt.loglog(g["n"], g["runtime_s"], marker="o")
    plt.title("Scalability: Runtime (log–log)"); plt.xlabel("N (variables)"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, which="both", alpha=0.3); plt.savefig(os.path.join(outdir, "scal_loglog_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("n")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g2["n"], g2["peak_mem_mb"], marker="^")
    plt.title("Scalability: Peak memory vs N"); plt.xlabel("N (variables)"); plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_N.png"), **plt_kwargs); plt.close()

def plot_reliability(df, outdir):
    g = df.groupby("n")["success"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["success"], marker="o"); plt.ylim(0,1.05)
    plt.title("Reliability: Success rate vs N (gap≤tol)"); plt.xlabel("N"); plt.ylabel("Success rate")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_N.png"), **plt_kwargs); plt.close()

    data = [df[df["n"]==k]["gap"].values for k in sorted(df["n"].unique())]
    plt.figure(); _boxplot_with_labels(data, labels=sorted(df["n"].unique()))
    plt.title("Reliability: Optimality gap distribution"); plt.xlabel("N"); plt.ylabel("Gap")
    plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, "rel_gap_boxplot.png"), **plt_kwargs); plt.close()

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
    parser = argparse.ArgumentParser(description="Classical Simulated Annealing Benchmarks (Ising/QUBO, planted optimum)")
   
    parser.add_argument("--sizes", nargs="+", type=int, default=[64, 128, 192], help="Problem sizes N.")
    parser.add_argument("--steps", type=int, default=1000, help="Annealing sweeps (each sweep visits N spins).")
    parser.add_argument("--trials", type=int, default=10, help="Trials per N (different seeds).")
    parser.add_argument("--p", type=float, default=0.2, help="ER graph edge probability.")
    parser.add_argument("--field-strength", type=float, default=0.1, help="Breaks ±s* degeneracy; ensures unique optimum.")
    parser.add_argument("--T0", type=float, default=2.0, help="Initial temperature.")
    parser.add_argument("--Tf", type=float, default=0.01, help="Final temperature.")
    parser.add_argument("--gap-tol", type=float, default=0.01, help="Success if (best_E - E*)/|E*| ≤ gap_tol.")
    # carbon/infra — unchanged
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
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]: os.makedirs(d, exist_ok=True)

    print("[run] Starting classical simulated annealing benchmarks...")
    jobs = [(N, args.steps, args.p, args.field_strength, args.T0, args.Tf, args.gap_tol, 1000 + 17*N + i)
            for N in args.sizes for i in range(args.trials)]
    rows = []

    def consume(res):
        rows.append(res)
        print(f"  - N={res['n']} seed={res['seed']} gap={res['gap']:.3e} "
              f"runtime={res['runtime_s']:.4f}s success={int(res['success'])}")

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_single, *job) for job in jobs]
            for f in as_completed(futs): consume(f.result())
    else:
        for job in jobs: consume(run_single(*job))

    df = pd.DataFrame(rows)

# ------------********------------
    # ----- Performance -----

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

    scal_xlsx = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_xlsx, engine="openpyxl") as w:
            df[["n","runtime_s","peak_mem_mb"]].to_excel(w, index=False, sheet_name="raw")
            df.groupby("n").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(df, scal_dir)

# ------------********------------
    # ----- Reliability -----

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
    # ----- Carbon (from PERFORMANCE runtime only) -----

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
        plot_carbon(carbon_df, carb_dir)
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
