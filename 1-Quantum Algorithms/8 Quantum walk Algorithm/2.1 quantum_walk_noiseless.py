#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# Ideal (Noiseless) Discrete-Time Quantum Walk on a 1-D Lattice (Reflecting Boundaries)
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
Ideal (Noiseless) Discrete-Time Quantum Walk on a 1-D Lattice (Reflecting Boundaries)
Default problem:
  - N (num_positions) = 21 (odd, start at center)
  - T (num_steps)     = 1000
  - Coin: Hadamard; shift: reflecting hard walls (unitary permutation)
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
# ========================= Quantum walk core
# ------------********------------

#def hadamard_coin() function constructs and returns the 2×2 Hadamard coin matrix.
#It normalizes the matrix entries to preserve unitary evolution.
#It is used as a coin operator in quantum or coined random-walk simulations.

def hadamard_coin():
    inv = 1.0 / math.sqrt(2.0)
    return np.array([[inv, inv],
                     [inv, -inv]], dtype=np.complex128)


#def initial_coin_vec_from_seed () function selects a deterministic, 
#   normalized coin state based on a given seed.
#It constructs several canonical coin states using the Hadamard operator.
#It ensures reproducible yet varied initial conditions for multiple trials.

def initial_coin_vec_from_seed(seed: int) -> np.ndarray:
    """
    Deterministic set of normalized coin states; seed picks one.
    Keeps 'trials' meaningful while staying noiseless.
    """
    H = hadamard_coin()
    L = np.array([1.0, 0.0], dtype=np.complex128)
    R = np.array([0.0, 1.0], dtype=np.complex128)
    plus  = H @ R           # (|L>+|R>)/sqrt2
    minus = H @ L           # (|L>-|R>)/sqrt2
    ip    = (L + 1j*R) / math.sqrt(2.0)
    im    = (L - 1j*R) / math.sqrt(2.0)
    bank = [L, R, plus, minus, ip, im]
    return bank[seed % len(bank)]

def apply_coin_all_sites(psi: np.ndarray, H: np.ndarray):
    """
    psi shape: (N, 2) complex; apply 2x2 coin to coin subspace at each site.
    Row-wise multiply by H^T (since rows are site states, columns coin components).
    """
    # psi[:, :] = psi @ H.T  (explicit to avoid extra allocs)
    left  = psi[:, 0].copy()
    right = psi[:, 1].copy()
    psi[:, 0] = H[0,0]*left + H[1,0]*right
    psi[:, 1] = H[0,1]*left + H[1,1]*right

#def shift_reflecting () function implements a reflecting boundary shift operator for a coined quantum walk.
#It moves left and right coin components inward while reflecting amplitudes at the boundaries.
#It returns the updated state after applying the unitary permutation.

def shift_reflecting(psi: np.ndarray):
    """
    Reflecting hard-wall shift (unitary permutation):
      interior: |L,i> -> |L,i-1>, |R,i> -> |R,i+1>
      left wall:  |L,0>   -> |R,0>
      right wall: |R,N-1> -> |L,N-1>
    """
    N = psi.shape[0]
    new = np.zeros_like(psi)
    # interior left
    if N > 1:
        new[0:N-1, 0] += psi[1:N, 0]      # from i=1..N-1 (L) -> i-1
        new[1:N,   1] += psi[0:N-1, 1]    # from i=0..N-2 (R) -> i+1
    # reflections
    new[0, 1]     += psi[0, 0]            # L at 0 -> R at 0
    new[N-1, 0]   += psi[N-1, 1]          # R at N-1 -> L at N-1
    return new

#def evolve_iterative () evolves a coined quantum walk state 
#   for a fixed number of time steps.
#It alternates between applying the Hadamard coin at all sites and a reflecting shift.
#It returns the final position–coin state after T iterations.

def evolve_iterative(N: int, T: int, seed: int):
    """
    Efficient iterative evolution: (H⊗I) then reflecting shift, T times.
    Returns final psi (N,2).
    """
    H = hadamard_coin()
    psi = np.zeros((N, 2), dtype=np.complex128)
    pos0 = N // 2
    psi[pos0, :] = initial_coin_vec_from_seed(seed)

    for _ in range(T):
        apply_coin_all_sites(psi, H)
        psi = shift_reflecting(psi)
    return psi

#def build_dense_U () function constructs the dense global step operator for a coined quantum walk.
#It builds a block-diagonal coin operator 
#    using Hadamard gates at each position.
#It prepares the structure for a reflecting shift operator 
#    in the combined position–coin basis.

def build_dense_U(N: int) -> np.ndarray:
    """
    Build global step operator U = S @ (C), where
      C = ⊕_{i=0}^{N-1} H (block-diagonal Hadamard),
      S = reflecting shift permutation in the {|L,i>, |R,i>} basis.
    Basis ordering: [(L,0),(R,0),(L,1),(R,1),..., (L,N-1),(R,N-1)]
    """
    H = hadamard_coin()
    d = 2 * N
    C = np.zeros((d, d), dtype=np.complex128)
    for i in range(N):
        C[2*i:2*i+2, 2*i:2*i+2] = H

    S = np.zeros((d, d), dtype=np.complex128)

    def idx(i, coin):  # coin 0=L, 1=R
        return 2*i + coin

    # interior shifts
    for i in range(1, N):      # L,i -> L,i-1
        S[idx(i-1, 0), idx(i, 0)] = 1.0
    for i in range(0, N-1):    # R,i -> R,i+1
        S[idx(i+1, 1), idx(i, 1)] = 1.0

    # reflections
    S[idx(0,   1), idx(0,   0)] = 1.0      # L,0 -> R,0
    S[idx(N-1, 0), idx(N-1, 1)] = 1.0      # R,N-1 -> L,N-1

    U = S @ C
    return U

#def evolve_dense_reference () function performs a reference evolution of a
#       coined quantum walk using a dense unitary matrix.
#It initializes the state at the center position with a deterministic coin state.
#It repeatedly applies the global unitary operator 
#       and returns the final state reshaped by position and coin.

def evolve_dense_reference(N: int, T: int, seed: int, U: np.ndarray):
    """
    Reference evolution using dense unitary U: psi <- U^T psi.
    """
    d = 2 * N
    psi = np.zeros(d, dtype=np.complex128)
    pos0 = N // 2
    coin = initial_coin_vec_from_seed(seed)   # [L,R]
    psi[2*pos0:2*pos0+2] = coin
    for _ in range(T):
        psi = U @ psi
    # reshape to (N,2)
    return psi.reshape(N, 2)

def position_marginal(psi: np.ndarray) -> np.ndarray:
    """Return p[i] = |psi_L(i)|^2 + |psi_R(i)|^2."""
    return (np.abs(psi[:, 0])**2 + np.abs(psi[:, 1])**2).astype(np.float64)

# ------------********------------
# ========================= Metrics
# ------------********------------

def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    return 0.5 * float(np.abs(p - q).sum())

def run_single(N: int, T: int, tv_tol: float, seed: int, U_cache: dict) -> dict:
    """
    One experiment: evolve iteratively and by dense-unitary reference; compare TV(p_iter, p_ref).
    """
    if N not in U_cache:
        U_cache[N] = build_dense_U(N)

    tracemalloc.start()
    t0 = time.perf_counter()

    psi_iter = evolve_iterative(N, T, seed)
    psi_ref  = evolve_dense_reference(N, T, seed, U_cache[N])

    p_iter = position_marginal(psi_iter)
    p_ref  = position_marginal(psi_ref)

    tv = tv_distance(p_iter, p_ref)

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "n": N,
        "steps": T,
        "seed": seed,
        "tv_distance": float(tv),
        "success": bool(tv <= tv_tol),
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024**2)),
    }

# ------------********------------
# ========================= Carbon (Performance only)
# ------------********------------

def resolve_excel_path(excel_arg: str) -> str:
    p = pathlib.Path(excel_arg)
    if p.is_absolute() and p.exists():
        return str(p)
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
    if med > 50:  # likely g/kWh
        df["Intensity"] = df["Intensity"] / 1000.0
    return df.dropna(subset=["Country","Intensity"]).reset_index(drop=True)

def compute_carbon(perf_df: pd.DataFrame, intensity_df: pd.DataFrame,
                   power_watts: float, pue: float) -> tuple[pd.DataFrame, pd.DataFrame]:
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
# ========================= Plotting
# ------------********------------

#This following block defines common plotting keyword arguments for consistent figure output.
#It provides a helper function to create boxplots with labels, 
#    handling version differences in matplotlib.
#It ensures compatibility across environments 
#    by falling back to alternative parameter names.

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    try:
        plt.boxplot(data, tick_labels=labels)
    except TypeError:
        plt.boxplot(data, labels=labels)

def plot_performance(df: pd.DataFrame, outdir: str):
    g = df.groupby("n")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["runtime_s"], marker="o")
    plt.title("Performance: Runtime vs N (positions)"); plt.xlabel("N"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("n")["tv_distance"].mean().reset_index()
    plt.figure(); plt.plot(g2["n"], g2["tv_distance"], marker="s")
    plt.title("Performance: Mean TV distance vs N"); plt.xlabel("N"); plt.ylabel("TV distance")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_tv_vs_N.png"), **plt_kwargs); plt.close()

def plot_scalability(df: pd.DataFrame, outdir: str):
    g = df.groupby("n")["runtime_s"].mean().reset_index()
    plt.figure(); plt.loglog(g["n"], g["runtime_s"], marker="o")
    plt.title("Scalability: Runtime (log–log)"); plt.xlabel("N (positions)"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, which="both", alpha=0.3); plt.savefig(os.path.join(outdir, "scal_loglog_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("n")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g2["n"], g2["peak_mem_mb"], marker="^")
    plt.title("Scalability: Peak memory vs N"); plt.xlabel("N (positions)"); plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_N.png"), **plt_kwargs); plt.close()

def plot_reliability(df: pd.DataFrame, outdir: str):
    g = df.groupby("n")["success"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["success"], marker="o"); plt.ylim(0,1.05)
    plt.title("Reliability: Success rate vs N (TV<=tol)"); plt.xlabel("N"); plt.ylabel("Success rate")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_N.png"), **plt_kwargs); plt.close()

    data = [df[df["n"]==k]["tv_distance"].values for k in sorted(df["n"].unique())]
    plt.figure(); _boxplot_with_labels(data, labels=sorted(df["n"].unique()))
    plt.title("Reliability: TV distance distribution"); plt.xlabel("N"); plt.ylabel("TV distance")
    plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, "rel_tv_boxplot.png"), **plt_kwargs); plt.close()

def plot_carbon(df: pd.DataFrame, outdir: str):
    top = df.nlargest(15, "kgCO2e")
    plt.figure(figsize=(8,5)); plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
    plt.title("Carbon: Top 15 countries (kgCO2e)"); plt.xlabel("kg CO2e")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "carbon_top15.png"), **plt_kwargs); plt.close()

    plt.figure(); plt.hist(df["kgCO2e"], bins=30)
    plt.title("Carbon: Emission distribution"); plt.xlabel("kg CO2e"); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "carbon_distribution.png"), **plt_kwargs); plt.close()

# ------------********------------
# ========================= Main Function
# ------------********------------

#This main function configures and runs ideal 1D quantum walk benchmarks 
#   using command-line arguments.
#It sets up output directories, prepares experiment jobs, 
#   and manages parallel or sequential execution.
#It collects results for performance and correctness analysis into a pandas DataFrame.

def main():
    parser = argparse.ArgumentParser(description="Ideal 1D Quantum Walk (Reflecting) Benchmarks")
    parser.add_argument("--sizes", nargs="+", type=int, default=[21], help="Positions N (odd recommended).")
    parser.add_argument("--steps", type=int, default=1000, help="Time steps T.")
    parser.add_argument("--trials", type=int, default=10, help="Trials per N (different seeds => coin state).")
    parser.add_argument("--tv-tol", type=float, default=1e-12, help="TV threshold for success.")

    # carbon/infra (same as other scripts)
    parser.add_argument("--excel", type=str, default="Filtered CO2 intensity 236 Countries.xlsx")
    parser.add_argument("--device-power-watts", type=float, default=65.0)
    parser.add_argument("--pue", type=float, default=1.2)
    parser.add_argument("--year-select", type=str, default="latest")
    parser.add_argument("--outdir", type=str, default="carbon_by_country")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    args = parser.parse_args()

    # Output dirs
    perf_dir = os.path.join(os.getcwd(), "Performance")
    scal_dir = os.path.join(os.getcwd(), "Scalability")
    rel_dir  = os.path.join(os.getcwd(), "Reliability")
    carb_root = os.path.join(os.getcwd(), "Carbon footprints")
    carb_dir  = os.path.join(carb_root, args.outdir)
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]:
        os.makedirs(d, exist_ok=True)

    print("[run] Starting ideal quantum-walk experiments...")
    jobs = [(N, 1000 + 17*N + i) for N in args.sizes for i in range(args.trials)]
    rows = []
    U_cache = {}

    def consume(res: dict):
        rows.append(res)
        print(f"  - N={res['n']} seed={res['seed']} TV={res['tv_distance']:.2e} "
              f"runtime={res['runtime_s']:.4f}s success={int(res['success'])}")

    if args.workers > 1 and len(jobs) > 1:
        
        def task(N, seed):  # small wrapper that builds its own U
            return run_single(N, args.steps, args.tv_tol, seed, {})
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(task, N, seed) for (N, seed) in jobs]
            for f in as_completed(futs):
                consume(f.result())
    else:
        for (N, seed) in jobs:
            consume(run_single(N, args.steps, args.tv_tol, seed, U_cache))

    df = pd.DataFrame(rows)

# ------------********------------
    # ------------------ Performance ------------------
    perf_xlsx = os.path.join(perf_dir, "performance_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw_runs")
            agg = df.groupby("n").agg(
                mean_runtime_s=("runtime_s","mean"),
                mean_peak_mem_mb=("peak_mem_mb","mean"),
                mean_tv_distance=("tv_distance","mean"),
                success_rate=("success","mean"),
            ).reset_index()
            agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(df, perf_dir)

# ------------********------------
    # ------------------ Scalability ------------------
    scal_xlsx = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_xlsx, engine="openpyxl") as w:
            df[["n","runtime_s","peak_mem_mb"]].to_excel(w, index=False, sheet_name="raw")
            df.groupby("n").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(df, scal_dir)

# ------------********------------
    # ------------------ Reliability ------------------
    rel_xlsx = os.path.join(rel_dir, "reliability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(rel_xlsx, engine="openpyxl") as w:
            df[["n","seed","tv_distance","success"]].to_excel(w, index=False, sheet_name="runs")
            rel_agg = df.groupby("n").agg(
                success_rate=("success","mean"),
                mean_tv=("tv_distance","mean"),
                std_tv=("tv_distance","std"),
            ).reset_index()
            rel_agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_reliability(df, rel_dir)

# ------------********------------
    # ------------------ Carbon (PERFORMANCE runtime only) ------------------
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
