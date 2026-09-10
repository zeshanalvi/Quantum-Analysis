#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# Simon's Algorithm — Ideal (Noiseless) Quantum Baseline
# ----------------------------------------------------------------------------------------------------------------

#     We simulate the measurement outcomes of Simon's algorithm exactly:
#     Each oracle call yields a random y in {0,1}^n such that y·s = 0 (mod 2).
#     We collect independent equations until rank = n-1, then solve for s via GF(2) elimination.

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
Simon's Algorithm — Ideal (Noiseless) Quantum Baseline

Benchmarks produced:
  1) Performance     – runtime, queries_used, peak memory
  2) Scalability     – runtime vs queries (log-log), peak memory vs n
  3) Reliability     – success rate, queries distribution
  4) Carbon Footprint– CO₂ from PERFORMANCE runtime only

Folders created:
  Performance/
  Scalability/
  Reliability/
  Carbon footprints/<outdir>/
"""

import os, sys, time, argparse, warnings, pathlib, tracemalloc, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ========================= Utilities: GF(2) linear algebra =========================

def gf2_rank(A: np.ndarray) -> int:
    """Return rank of binary matrix A over GF(2)."""
    A = (A.astype(np.uint8) & 1).copy()
    m, n = A.shape
    row = 0
    for col in range(n):
        # find pivot
        pivot = None
        for r in range(row, m):
            if A[r, col]:
                pivot = r; break
        if pivot is None:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        # eliminate in all other rows
        for r in range(m):
            if r != row and A[r, col]:
                A[r, :] ^= A[row, :]
        row += 1
        if row == m:
            break
    return row

def gf2_solve_homogeneous_one(A: np.ndarray) -> np.ndarray:
    """
    Solve A s = 0 over GF(2) and return one non-zero solution (if exists).
    Assumes nullity >= 1; for Simon we expect rank = n-1 so nullity = 1.
    """
    A = (A.astype(np.uint8) & 1).copy()
    m, n = A.shape
    row = 0
    pivots = []
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if A[r, col]:
                pivot = r; break
        if pivot is None:
            continue
        if pivot != row:
            A[[row, pivot]] = A[[pivot, row]]
        for r in range(m):
            if r != row and A[r, col]:
                A[r, :] ^= A[row, :]
        pivots.append(col)
        row += 1
        if row == m:
            break

    free = [c for c in range(n) if c not in pivots]
    s = np.zeros(n, dtype=np.uint8)
    if not free:
        return s  # only zero solution
    fc = free[0]          # only one free var when rank = n-1
    s[fc] = 1
    # back-substitute pivot variables from RREF
    for r in range(len(pivots)-1, -1, -1):
        pc = pivots[r]
        val = 0
        for j in range(pc+1, n):
            if A[r, j] and s[j]:
                val ^= 1
        s[pc] = val
    return s

# ========================= Ideal Simon measurement sampler =========================

def sample_y_orthogonal_to_s(n: int, s: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample y uniformly from { y ∈ {0,1}^n : y·s = 0 mod 2 }."""
    while True:
        y = rng.integers(0, 2, size=n, dtype=np.uint8)
        if int(np.bitwise_and(y, s).sum() & 1) == 0:
            return y

def simon_quantum_noiseless(n: int, s: np.ndarray, seed: int,
                            max_rounds: int | None = None):
    """
    Ideal Simon algorithm: collect random y with y·s=0 until rank(Y)=n-1.
    One oracle query per round.
    """
    if max_rounds is None:
        max_rounds = 4 * n  # generous cap; expected O(n) rounds

    rng = np.random.default_rng(seed)
    Ys = []
    rank = 0
    queries = 0
    for _ in range(max_rounds):
        y = sample_y_orthogonal_to_s(n, s, rng)
        Ys.append(y)
        queries += 1
        rank = gf2_rank(np.array(Ys, dtype=np.uint8))
        if rank >= n - 1:
            break

    if rank < n - 1:
        return np.zeros(n, dtype=np.uint8), queries, False

    s_guess = gf2_solve_homogeneous_one(np.array(Ys, dtype=np.uint8))
    success = bool(np.array_equal(s_guess, s))
    return s_guess, queries, success

# ========================= One experiment =========================

def run_single(n: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    # Generate non-zero hidden string s
    s = np.zeros(n, dtype=np.uint8)
    while not s.any():
        s = rng.integers(0, 2, size=n, dtype=np.uint8)

    tracemalloc.start()
    t0 = time.perf_counter()
    s_guess, queries, success = simon_quantum_noiseless(n, s, seed=seed)
    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "n": n,
        "seed": seed,
        "queries_used": int(queries),
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024**2)),
        "success": bool(success)
    }

# ========================= Carbon I/O =========================

def resolve_excel_path(excel_arg: str) -> str:
    p = pathlib.Path(excel_arg)
    if p.is_absolute() and p.exists():
        return str(p)
    desktop = pathlib.Path.home() / "Desktop" / excel_arg
    return str(desktop if desktop.exists() else p)

def load_carbon_excel(path: str, year_select: str = "latest") -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}
    cand_country = next((v for k,v in cols.items() if "country" in k or "nation" in k or k=="location"), None)
    if cand_country is None: raise ValueError("No 'Country' column found.")
    cand_year = next((v for k,v in cols.items() if "year" in k or "date" in k), None)
    cand_intensity = next((v for k,v in cols.items()
                           if "intensity" in k or ("co2" in k and ("kwh" in k or "/kwh" in k)) or "kgco2" in k or "gco2" in k), None)
    if cand_intensity is None:
        numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if cand_year in numeric: numeric.remove(cand_year)
        if not numeric: raise ValueError("No numeric intensity column detected.")
        cand_intensity = numeric[0]
    keep = [cand_country] + ([cand_year] if cand_year else []) + [cand_intensity]
    df = df[keep].copy()
    df.columns = ["Country","Year","Intensity"] if len(keep)==3 else ["Country","Intensity"]
    if "Year" in df.columns and year_select.lower()=="latest":
        df = df.sort_values(["Country","Year"]).groupby("Country", as_index=False).tail(1)

    # If it looks like g/kWh, convert to kg/kWh
    med = float(df["Intensity"].dropna().median())
    if med > 50:
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

# ========================= Plotting helpers =========================

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def plot_performance(df: pd.DataFrame, outdir: str):
    g = df.groupby("n")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["runtime_s"], marker="o")
    plt.title("Performance: Runtime vs n"); plt.xlabel("n (bits)"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "perf_runtime_vs_n.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("n")["queries_used"].mean().reset_index()
    plt.figure(); plt.plot(g2["n"], g2["queries_used"], marker="s")
    plt.title("Performance: Mean queries vs n"); plt.xlabel("n (bits)"); plt.ylabel("Mean queries used")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "perf_queries_vs_n.png"), **plt_kwargs); plt.close()

def plot_scalability(df: pd.DataFrame, outdir: str):
    plt.figure(); plt.loglog(df["queries_used"], df["runtime_s"], "o", alpha=0.6)
    plt.title("Scalability: Runtime vs Queries (log-log)")
    plt.xlabel("Queries used"); plt.ylabel("Runtime (s)")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(outdir, "scal_runtime_vs_queries.png"), **plt_kwargs); plt.close()

    g = df.groupby("n")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["peak_mem_mb"], marker="^")
    plt.title("Scalability: Peak memory vs n"); plt.xlabel("n (bits)"); plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "scal_peakmem_vs_n.png"), **plt_kwargs); plt.close()

def plot_reliability(df: pd.DataFrame, outdir: str):
    g = df.groupby("n")["success"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["success"], marker="o"); plt.ylim(0,1.05)
    plt.title("Reliability: Success rate vs n"); plt.xlabel("n (bits)"); plt.ylabel("Success rate")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "rel_success_vs_n.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("n")["queries_used"].agg(["mean","std"]).reset_index()
    plt.figure(); plt.errorbar(g2["n"], g2["mean"], yerr=g2["std"], fmt="-s")
    plt.title("Reliability: Mean±Std of queries vs n"); plt.xlabel("n (bits)"); plt.ylabel("Queries used")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "rel_queries_mean_std.png"), **plt_kwargs); plt.close()

def plot_carbon(df: pd.DataFrame, outdir: str):
    top = df.nlargest(15, "kgCO2e")
    plt.figure(figsize=(8,5)); plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
    plt.title("Carbon: Top 15 countries (kgCO2e)"); plt.xlabel("kg CO2e")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "carbon_top15.png"), **plt_kwargs); plt.close()

    plt.figure(); plt.hist(df["kgCO2e"], bins=30)
    plt.title("Carbon: Emission distribution"); plt.xlabel("kg CO2e"); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "carbon_distribution.png"), **plt_kwargs); plt.close()

# ========================= Main =========================================

def main():
    parser = argparse.ArgumentParser(description="Simon's Algorithm (Ideal, Noiseless) — Quantum Baseline")
    parser.add_argument("--sizes", nargs="+", type=int, default=[6, 8, 10],
                        help="Input sizes n (bits)")
    parser.add_argument("--trials", type=int, default=10,
                        help="Trials per size")
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

    print("[run] Starting ideal Simon (noiseless) experiments...")
    jobs = [(n, 1000 + 17*n + i) for n in args.sizes for i in range(args.trials)]
    rows = []

    def consume(res: dict):
        rows.append(res)
        print(f"  - n={res['n']} seed={res['seed']} queries={res['queries_used']} "
              f"runtime={res['runtime_s']:.6f}s success={int(res['success'])}")

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_single, n, seed) for (n, seed) in jobs]
            for f in as_completed(futs):
                consume(f.result())
    else:
        for (n, seed) in jobs:
            consume(run_single(n, seed))

    df = pd.DataFrame(rows)

    # ------------------ Performance ------------------
    perf_xlsx = os.path.join(perf_dir, "performance_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw")
            df.groupby("n").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(df, perf_dir)

    # ------------------ Scalability ------------------
    scal_xlsx = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw")
            df.groupby("n").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(df, scal_dir)

    # ------------------ Reliability ------------------
    rel_xlsx = os.path.join(rel_dir, "reliability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(rel_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw")
            df.groupby("n").agg(success_rate=("success","mean"),
                                mean_queries=("queries_used","mean"),
                                std_queries=("queries_used","std")).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_reliability(df, rel_dir)

    # ------------------ Carbon (from PERFORMANCE runtimes only) ------------------
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
        # quick plots
        top = carbon_df.nlargest(15, "kgCO2e")
        plt.figure(figsize=(8,5)); plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
        plt.title("Carbon: Top 15 countries (kgCO2e)"); plt.xlabel("kg CO2e")
        plt.tight_layout(); plt.savefig(os.path.join(carb_dir, "carbon_top15.png"), dpi=140, bbox_inches="tight"); plt.close()

        plt.figure(); plt.hist(carbon_df["kgCO2e"], bins=30)
        plt.title("Carbon: Emission distribution"); plt.xlabel("kg CO2e"); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(carb_dir, "carbon_distribution.png"), dpi=140, bbox_inches="tight"); plt.close()
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
