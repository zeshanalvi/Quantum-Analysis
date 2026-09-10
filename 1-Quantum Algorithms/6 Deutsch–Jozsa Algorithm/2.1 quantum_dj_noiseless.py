#!/usr/bin/env python3
# -*- coding: utf-8 -*-



# ----------------------------------------------------------------------------------------------------------------
# Deutsch–Jozsa (Noiseless Quantum) : DEmpirical Evaluation
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
This code script evaluates the Deutsch–Jozsa quantum algorithm on a statevector (ideal) simulator under the DJ promise:
  - CONSTANT oracles: f(x) ≡ c ∈ {0,1}
  - BALANCED oracles: parity family f_s(x) = (s · x) mod 2 with random non-zero s

Quantum circuit (noiseless, analytic):
  |0^n> --H^n-- phase_oracle[(-1)^{f(x)}] --H^n--> measure first n
  Prediction: 'CONSTANT' iff all measured bits are 0^n; else 'BALANCED'.
  (For parity-balanced, the final state is |s> deterministically.)

  python quantum_dj_noiseless.py ^
    --sizes 6 8 10 12 ^
    --trials 10 ^
    --excel "Filtered CO2 intensity 236 Countries.xlsx" ^
    --device-power-watts 65 --pue 1.2 ^
    --year-select latest ^
    --outdir carbon_by_country ^
    --combine
"""

 # ------------********------------ Imports

import os, sys, time, argparse, tracemalloc, pathlib, warnings, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

 # ------------********------------ 
# ========================= Minimal statevector simulator =========================

#This function applies a Hadamard gate to a specified qubit using little-endian bit ordering.
#It iterates over the state vector in blocks to combine amplitude pairs affected by the Hadamard transform.
#It updates the quantum state in place using normalized sum and difference operations.

def hadamard_on_qubit(state: np.ndarray, q: int, n: int):
    """Apply H on qubit q (little-endian bit ordering)."""
    dim = state.shape[0]
    m = 1 << q          # block size
    stride = m << 1
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    for base in range(0, dim, stride):
        i0 = base
        i1 = base + m
        for off in range(m):
            a = state[i0 + off]
            b = state[i1 + off]
            state[i0 + off] = (a + b) * inv_sqrt2
            state[i1 + off] = (a - b) * inv_sqrt2

#This function applies a Z gate to a specified qubit by flipping 
# the sign of amplitudes where that qubit is set.
#It computes a bit mask to identify state indices affected by the Z operation.
#It updates the quantum state vector in place using a vectorized sign inversion.

def z_on_qubit(state: np.ndarray, q: int):
    """Apply Z on qubit q: multiply amplitudes with bit(q)==1 by -1."""
    dim = state.shape[0]
    bit = 1 << q
    # Vectorized: build mask of indices with this bit set, flip sign
    idx = np.arange(dim, dtype=np.uint64)
    mask = (idx & bit) != 0
    state[mask] = -state[mask]

#This function initializes an n-qubit quantum state vector in the |0...0⟩ basis state.
#It allocates a complex-valued array of size 2^n to represent the full state space.
#It sets the first amplitude to one and returns the prepared state.

def prepare_zero_state(n: int) -> np.ndarray:
    st = np.zeros(1 << n, dtype=np.complex128)
    st[0] = 1.0
    return st

def apply_h_layer(state: np.ndarray, n: int):
    for q in range(n):
        hadamard_on_qubit(state, q, n)

# ------------********------------ 
# ------------********------------  DJ phase-oracle

#This function constructs a phase oracle for a constant Boolean function 
#  using a global phase factor.
#It returns an inner function that conditionally multiplies the entire quantum state by −1.
#It models constant oracles where the global phase does not affect measurement outcomes.

def oracle_constant_phase(c: int):
    """
    Phase oracle for constant f(x)=c:
      c=0 -> identity; c=1 -> global phase (-1).
     Multiplying whole state by (-1)^c.
    """
    def apply(state: np.ndarray, n: int):
        if c & 1:
            state *= -1.0
    return apply

# def oracle_parity_phase function builds a balanced parity phase oracle based on a given bit string s.
#It applies a Z gate to each qubit whose corresponding bit in s is set to one.
#It returns an oracle function that modifies the quantum state via phase kickback.

def oracle_parity_phase(s_bits: np.ndarray):
    """
    Balanced parity oracle: multiply amplitude of |x> by (-1)^{s·x}.
    Equivalently, apply Z on each qubit i where s_i=1, after H-layer (kickback style).
    """
    s = s_bits.astype(np.uint8)
    def apply(state: np.ndarray, n: int):
        for i in range(n):
            if s[i]:
                z_on_qubit(state, i)
    return apply

# ------------********------------  Deutsch–Jozsa run (one instance) =========================

#This function runs a single Deutsch–Jozsa experiment instance using a seeded random generator for reproducibility.
def run_single(n: int, seed: int, oracle_type: str) -> dict:
    """
    Build a promised DJ oracle instance and run *one* ideal DJ circuit.
    Measure wall-clock runtime, peak memory; queries_used=1; correctness vs ground truth.
    """
    rng = np.random.default_rng(seed)

    if oracle_type == "CONSTANT":
        c = int(rng.integers(0, 2))
        oracle = oracle_constant_phase(c)
        true_label = "CONSTANT"
    elif oracle_type == "BALANCED":
        s = np.zeros(n, dtype=np.uint8)
        while not s.any():
            s = rng.integers(0, 2, size=n, dtype=np.uint8)
        oracle = oracle_parity_phase(s)
        true_label = "BALANCED"
    else:
        raise ValueError("oracle_type must be CONSTANT or BALANCED")

    # ------------********------------  Ideal DJ circuit, statevector ---
    tracemalloc.start()
    t0 = time.perf_counter()

    st = prepare_zero_state(n)         # |0^n>
    apply_h_layer(st, n)               # H^n
    oracle(st, n)                      # phase oracle (-1)^{f(x)}
    apply_h_layer(st, n)               # H^n

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Deterministic readout (noiseless): measure first n.
    # We don't sample; check amplitude at |0^n|.
    p_zero = float((st[0].conjugate() * st[0]).real)
    pred = "CONSTANT" if p_zero > 1.0 - 1e-12 else "BALANCED"
    correct = (pred == true_label)

    return {
        "n": n,
        "seed": seed,
        "oracle": true_label,
        "prediction": pred,
        "correct": bool(correct),
        "queries_used": 1,                       # DJ uses exactly one oracle call
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024 ** 2))
    }

# ------------********------------  Carbon accounting (Performance only)

#Resolving excel location path
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

    
    med = float(df["Intensity"].dropna().median())
    if med > 50:
        df["Intensity"] = df["Intensity"]/1000.0

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

# ------------********------------  Plot helpers

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    try:
        plt.boxplot(data, tick_labels=labels)
    except TypeError:
        plt.boxplot(data, labels=labels)


# ------------********------------ PERF
def plot_performance(df: pd.DataFrame, outdir: str):
    g = df.groupby("n")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["runtime_s"], marker="o")
    plt.title("Performance: Runtime vs n"); plt.xlabel("n (input bits)"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_n.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("n")["queries_used"].mean().reset_index()
    plt.figure(); plt.plot(g2["n"], g2["queries_used"], marker="s")
    plt.title("Performance: Mean queries vs n (DJ = 1)"); plt.xlabel("n (input bits)"); plt.ylabel("Mean queries used")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_queries_vs_n.png"), **plt_kwargs); plt.close()

# ------------********------------ SCAL
def plot_scalability(df: pd.DataFrame, outdir: str):
    plt.figure()
    plt.loglog(df["queries_used"], df["runtime_s"], "o", alpha=0.5)
    plt.title("Scalability: Runtime vs Queries (log-log)"); plt.xlabel("Queries used"); plt.ylabel("Runtime (s)")
    plt.grid(True, which="both", alpha=0.3); plt.savefig(os.path.join(outdir, "scal_loglog_runtime_vs_queries.png"), **plt_kwargs); plt.close()

    g = df.groupby("n")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["peak_mem_mb"], marker="^")
    plt.title("Scalability: Peak memory vs n"); plt.xlabel("n (input bits)"); plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_n.png"), **plt_kwargs); plt.close()

# ------------********------------ RELI
def plot_reliability(df: pd.DataFrame, outdir: str):
    g = df.groupby("n")["correct"].mean().reset_index()
    plt.figure(); plt.plot(g["n"], g["correct"], marker="o"); plt.ylim(0,1.05)
    plt.title("Reliability: Success rate vs n"); plt.xlabel("n (input bits)"); plt.ylabel("Success rate")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_n.png"), **plt_kwargs); plt.close()

    data = [df[df["n"]==k]["queries_used"].values for k in sorted(df["n"].unique())]
    plt.figure(); _boxplot_with_labels(data, labels=sorted(df["n"].unique()))
    plt.title("Reliability: Queries distribution by n (DJ = 1)"); plt.xlabel("n (input bits)"); plt.ylabel("Queries used")
    plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, "rel_queries_boxplot.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("n")["queries_used"].agg(["mean","std"]).reset_index()
    plt.figure(); plt.errorbar(g2["n"], g2["mean"], yerr=g2["std"], fmt="-s")
    plt.title("Reliability: Mean±Std of queries vs n (DJ = 1)"); plt.xlabel("n (input bits)"); plt.ylabel("Queries used")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_queries_mean_std.png"), **plt_kwargs); plt.close()

# ------------********------------ CARB
def plot_carbon(df: pd.DataFrame, outdir: str):
    top = df.nlargest(15, "kgCO2e")
    plt.figure(figsize=(8,5)); plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
    plt.title("Carbon: Top 15 countries (kgCO2e)"); plt.xlabel("kg CO2e")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "carbon_top15.png"), **plt_kwargs); plt.close()

    bot = df.nsmallest(15, "kgCO2e")
    plt.figure(figsize=(8,5)); plt.barh(bot["Country"][::-1], bot["kgCO2e"][::-1])
    plt.title("Carbon: Bottom 15 countries (kgCO2e)"); plt.xlabel("kg CO2e")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "carbon_bottom15.png"), **plt_kwargs); plt.close()

    plt.figure(); plt.hist(df["kgCO2e"], bins=30)
    plt.title("Carbon: Emission distribution"); plt.xlabel("kg CO2e"); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "carbon_distribution.png"), **plt_kwargs); plt.close()

    xs = np.sort(df["kgCO2e"].values); ys = np.arange(1, len(xs)+1)/len(xs)
    plt.figure(); plt.plot(xs, ys)
    plt.title("Carbon: CDF of emissions"); plt.xlabel("kg CO2e"); plt.ylabel("CDF")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "carbon_cdf.png"), **plt_kwargs); plt.close()

# ------------********------------  ------------********------------ 
# ========================= Main ===================================
# ------------********------------  ------------********------------ 

def main():
    parser = argparse.ArgumentParser(description="Deutsch–Jozsa (Noiseless Quantum) Benchmarks")
    parser.add_argument("--sizes", nargs="+", type=int, default=[6, 8, 10, 12],
                        help="Input sizes n (bits).")
    parser.add_argument("--trials", type=int, default=10,
                        help="Trials per (n, oracle_type). Half constant, half balanced.")
    parser.add_argument("--excel", type=str, default="Filtered CO2 intensity 236 Countries.xlsx",
                        help="Excel with country carbon intensities on Desktop or absolute path.")
    parser.add_argument("--device-power-watts", type=float, default=65.0)
    parser.add_argument("--pue", type=float, default=1.2)
    parser.add_argument("--year-select", type=str, default="latest")
    parser.add_argument("--outdir", type=str, default="carbon_by_country")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2),
                        help="Parallel processes for different seeds.")
    args = parser.parse_args()

    # Output folders (same structure as classical)
    perf_dir = os.path.join(os.getcwd(), "Performance")
    scal_dir = os.path.join(os.getcwd(), "Scalability")
    rel_dir  = os.path.join(os.getcwd(), "Reliability")
    carb_root = os.path.join(os.getcwd(), "Carbon footprints")
    carb_dir  = os.path.join(carb_root, args.outdir)
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]:
        os.makedirs(d, exist_ok=True)

    print("[run] Starting noiseless Deutsch–Jozsa experiments...")
    rows = []
    jobs = []
    # Equal number of CONSTANT and BALANCED per n
    for n in args.sizes:
        for i in range(args.trials):
            jobs.append((n, 1000 + 17*n + i, "CONSTANT"))
            jobs.append((n, 2000 + 17*n + i, "BALANCED"))
        
#This block defines a result-handling helper that records each experiment outcome and prints a concise status line.
#It executes experiment jobs in parallel when multiple workers are available, 
# otherwise IT will fall back to sequential execution.
#It collects all results into a pandas DataFrame for downstream analysis and reporting.

    def _consume(res: dict):
        rows.append(res)
        print(f"  - n={res['n']}, oracle={res['oracle']}, seed={res['seed']}, "
              f"queries={res['queries_used']}, runtime={res['runtime_s']:.6f}s, "
              f"correct={int(res['correct'])}")

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_single, n, seed, typ) for (n, seed, typ) in jobs]
            for f in as_completed(futs):
                _consume(f.result())
    else:
        for (n, seed, typ) in jobs:
            _consume(run_single(n, seed, typ))

    df = pd.DataFrame(rows)

    # ------------********------------  Performance ------------------
    perf_path = os.path.join(perf_dir, "performance_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_path, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw_runs")
            agg = df.groupby("n").agg(
                mean_runtime_s=("runtime_s","mean"),
                mean_queries=("queries_used","mean"),
                mean_peak_mem_mb=("peak_mem_mb","mean")
            ).reset_index()
            agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(df, perf_dir)

    # ------------********------------  Scalability ------------------
    scal_path = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_path, engine="openpyxl") as w:
            df[["n","oracle","queries_used","runtime_s","peak_mem_mb"]].to_excel(w, index=False, sheet_name="raw")
            df.groupby(["n","oracle"]).mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(df, scal_dir)

    # ------------********------------  Reliability ------------------
    rel_path = os.path.join(rel_dir, "reliability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(rel_path, engine="openpyxl") as w:
            df[["n","seed","oracle","prediction","correct","queries_used"]].to_excel(w, index=False, sheet_name="runs")
            rel_agg = df.groupby("n").agg(
                success_rate=("correct","mean"),
                mean_queries=("queries_used","mean"),
                std_queries=("queries_used","std")
            ).reset_index()
            rel_agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_reliability(df, rel_dir)

    # ------------------ Carbon (Performance only) ------------------
    excel_path = resolve_excel_path(args.excel)
    try:
        intensity_df = load_carbon_excel(excel_path, year_select=args.year_select)
        carbon_df, summary_df = compute_carbon(df, intensity_df,
                                               power_watts=args.device_power_watts,
                                               pue=args.pue)
        carb_path = os.path.join(carb_dir, "carbon_results.xlsx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pd.ExcelWriter(carb_path, engine="openpyxl") as w:
                carbon_df.to_excel(w, index=False, sheet_name="per_country")
                summary_df.to_excel(w, index=False, sheet_name="summary")
                intensity_df.to_excel(w, index=False, sheet_name="intensity_input")
        plot_carbon(carbon_df, carb_dir)
        print(f"[carbon] Wrote Excel and plots to: {carb_dir}")
    except Exception as e:
        print(f"[carbon] ERROR reading Excel '{excel_path}': {e}")
        print("[carbon] Skipping carbon benchmark. Re-run once the file is available.")

    # ------------********------------  Summary ------------------
    print("\n=== Summary (console) ===")
    print("Performance Excel:", perf_path)
    print("Scalability Excel:", scal_path)
    print("Reliability Excel:", rel_path)
    print("Carbon Excel:", os.path.join(carb_dir, "carbon_results.xlsx"))
    print("Folders created:\n -", perf_dir, "\n -", scal_dir, "\n -", rel_dir, "\n -", carb_dir)

if __name__ == "__main__":
    main()
