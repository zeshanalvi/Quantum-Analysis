#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
"""
Quantum Hamiltonian Simulation (Ideal/Noiseless) — Benchmarks

Hamiltonian: 1D Transverse-Field Ising
  H = -J * sum Z_i Z_{i+1}  -  h * sum X_i
  (open chain by default; --periodic 1 to enable ZZ between last and first)

Simulation: product-formula (Trotter) approximations
  - order=1 : Lie–Trotter   (ZZ layer then X layer each step)
  - order=2 : Strang/second (X half, ZZ full, X half per step)

Accuracy metric:
  For each random initial state |ψ>, evolve with Trotter steps (coarse)
  and with (ref_mult × steps) (reference), at same total time T.
  Compute fidelity F = |<ψ_ref | ψ_coarse>|^2 and RMSE between statevectors.
  Batch success = (mean_fidelity >= --acc-tol).
"""
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


# ------------********------------ Imports

import os, time, math, argparse, warnings, pathlib, tracemalloc
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ----------------------------- State & layers ----------------------------------

def _idx_arrays(n):
    dim = 1 << n
    idx = np.arange(dim, dtype=np.int64)
    return dim, idx

def _apply_rx_qubit(state: np.ndarray, n: int, q: int, theta: float):
    """
    Apply Rx(theta) = exp(-i theta X / 2) on qubit q of an n-qubit statevector.
    Vectorized via bit-masks.
    """
    mask = 1 << q
    dim = state.size
    idx = np.arange(dim, dtype=np.int64)
    idx0 = idx[(idx & mask) == 0]
    idx1 = idx0 | mask
    c = math.cos(theta/2.0)
    s = math.sin(theta/2.0)
    a = state[idx0].copy()
    b = state[idx1].copy()
    # Rx: [c  -i s; -i s  c]
    state[idx0] = c*a - 1j*s*b
    state[idx1] = c*b - 1j*s*a

def _precompute_zz_phase(n: int, pair, theta: float, idx: np.ndarray):
    """
    For Z_i Z_j rotation with angle theta': exp(-i theta' Z_i Z_j).
    For computational basis |z>, phase = exp(-i theta' * z_i z_j), with z∈{+1,-1}.
    Returns a vector 'phase' of shape (2^n,) to multiply elementwise.
    """
    i, j = pair
    zi = 1 - 2*((idx >> i) & 1)  # +1 if bit=0, -1 if bit=1
    zj = 1 - 2*((idx >> j) & 1)
    sgn = zi * zj
    return np.exp(-1j * theta * sgn)

def _apply_phase_layer(state: np.ndarray, phase: np.ndarray):
    state *= phase

def simulate_trotter(n: int, T: float, r: int, order: int, J: float, h: float,
                     periodic: bool, init_state: np.ndarray,
                     cache=None):
    """
    Simulate e^{-i H T} |psi> by a product formula with r steps.
    order: 1 (Lie-Trotter) or 2 (Strang)
    cache: dict to reuse precomputed phase vectors for ZZ layer.
    """
    state = init_state.copy()
    dim, idx = _idx_arrays(n)

    # neighbor pairs for ZZ
    pairs = [(i, i+1) for i in range(n-1)]
    if periodic and n >= 2:
        pairs.append((n-1, 0))

    dt = T / max(1, r)
    # angles
    theta_x = 2.0 * h * dt       # because e^{-i h dt X} = Rx(2 h dt)
    theta_zz = J * dt            # e^{-i J dt Z Z}

    # Precompute ZZ phase vectors once
    if cache is None:
        cache = {}
    key = (n, periodic, J, dt)
    if key not in cache:
        phase_list = [ _precompute_zz_phase(n, pr, theta_zz, idx) for pr in pairs ]
        cache[key] = phase_list
    phase_list = cache[key]

    # One step kernels
    def layer_X_half():
        th = theta_x/2.0
        for q in range(n):
            _apply_rx_qubit(state, n, q, th)
    def layer_X_full():
        for q in range(n):
            _apply_rx_qubit(state, n, q, theta_x)
    def layer_ZZ_full():
        for ph in phase_list:
            _apply_phase_layer(state, ph)

    # Evolve
    if order == 1:
        for _ in range(r):
            layer_ZZ_full()
            layer_X_full()
    else:  # order==2
        for _ in range(r):
            layer_X_half()
            layer_ZZ_full()
            layer_X_half()

    # Normalize (protect from tiny numerical drift)
    norm = np.linalg.norm(state)
    if norm == 0:
        return state
    return state / norm

def random_state(n: int, rng: np.random.Generator) -> np.ndarray:
    dim = 1 << n
    a = rng.normal(size=dim) + 1j*rng.normal(size=dim)
    a /= np.linalg.norm(a)
    return a.astype(np.complex128)

# ----------------------------- Batch runner ------------------------------------

def run_single(N: int, steps: int, seed: int,
               T: float, r_steps: int, order: int,
               J: float, h: float, periodic: int,
               acc_tol: float, ref_mult: int) -> dict:
    """
    One batch for N qubits & seed.
    - Accuracy computed vs reference simulation with r_ref = ref_mult * r_steps.
    - Runtime/memory measured on the COARSE simulation only.
    """
    rng = np.random.default_rng(seed)
    periodic = bool(periodic)
    order = int(order)
    r_steps = int(r_steps)
    r_ref = max(1, int(ref_mult) * max(1, r_steps))

    tracemalloc.start()
    t0 = time.perf_counter()

    # Coarse simulation runtime only
    cache_coarse = {}
    cache_ref = {}  # for ref phase precomputation
    fid_sum = 0.0
    rmse_sum = 0.0

    # Measure coarse-only wall time
    for _ in range(steps):
        psi0 = random_state(N, rng)
        _ = simulate_trotter(N, T, r_steps, order, J, h, periodic, psi0, cache=cache_coarse)
    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Now compute accuracy (not counted in runtime on purpose)
    for _ in range(steps):
        psi0 = random_state(N, rng)
        psi_coarse = simulate_trotter(N, T, r_steps, order, J, h, periodic, psi0, cache=cache_coarse)
        psi_ref    = simulate_trotter(N, T, r_ref,   2,     J, h, periodic, psi0, cache=cache_ref)  # high-accuracy Strang
        inner = np.vdot(psi_ref, psi_coarse)
        fid = float(np.abs(inner)**2)
        err = float(np.sqrt(np.mean(np.abs(psi_ref - psi_coarse)**2)))
        fid_sum += fid
        rmse_sum += err

    accuracy_mean = fid_sum / max(1, steps)
    rmse_mean = rmse_sum / max(1, steps)

    # Work units (coarse only): number of exponentials per Trotter step
    zz_ops = N if periodic and N>=2 else max(0, N-1)
    x_ops = N
    if order == 1:
        exp_per_step = zz_ops + x_ops
    else:
        exp_per_step = zz_ops + 2*x_ops
    ops_total = exp_per_step * steps * r_steps

    success = bool(accuracy_mean >= acc_tol)

    return {
        "N": int(N),
        "steps": int(steps),
        "seed": int(seed),
        "T": float(T),
        "r_steps": int(r_steps),
        "order": int(order),
        "J": float(J),
        "h": float(h),
        "periodic": int(periodic),
        "ref_mult": int(ref_mult),
        "accuracy_mean": float(accuracy_mean),
        "rmse_mean": float(rmse_mean),
        "success": success,
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024**2)),
        "exp_ops_total": int(ops_total),
    }

# ----------------------------- Carbon accounting -------------------------------

def resolve_excel_path(excel_arg: str) -> str:
    p = pathlib.Path(excel_arg)
    if p.is_absolute() and p.exists(): return str(p)
    desktop = pathlib.Path.home() / "Desktop" / excel_arg
    return str(desktop if desktop.exists() else p)

def load_carbon_excel(path: str, year_select: str = "latest") -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}
    country = next((v for k,v in cols.items() if "country" in k or "nation" in k or k=="location"), None)
    if country is None: raise ValueError("No 'Country' column found in Excel.")
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

# ----------------------------- Plotting helpers --------------------------------

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    try:
        plt.boxplot(data, tick_labels=labels)  # Matplotlib ≥3.9
    except TypeError:
        plt.boxplot(data, labels=labels)

def plot_performance(df: pd.DataFrame, outdir: str):
    g = df.groupby("N")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["N"], g["runtime_s"], marker="o")
    plt.title("Quantum H-Sim Performance: Runtime vs N"); plt.xlabel("qubits N"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("N")["exp_ops_total"].mean().reset_index()
    plt.figure(); plt.plot(g2["N"], g2["exp_ops_total"], marker="s")
    plt.title("Quantum H-Sim Performance: Exponential ops (coarse) vs N"); plt.xlabel("qubits N"); plt.ylabel("Mean #exp ops")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_ops_vs_N.png"), **plt_kwargs); plt.close()

def plot_scalability(df: pd.DataFrame, outdir: str):
    g = df.groupby("N")["runtime_s"].mean().reset_index()
    plt.figure(); plt.loglog(g["N"], g["runtime_s"], "o")
    plt.title("Quantum H-Sim Scalability: Runtime (log–log) vs N"); plt.xlabel("qubits N"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, which="both", alpha=0.3); plt.savefig(os.path.join(outdir, "scal_loglog_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("N")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g2["N"], g2["peak_mem_mb"], marker="^")
    plt.title("Quantum H-Sim Scalability: Peak memory vs N"); plt.xlabel("qubits N"); plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_N.png"), **plt_kwargs); plt.close()

def plot_reliability(df: pd.DataFrame, outdir: str):
    g = df.groupby("N")["accuracy_mean"].mean().reset_index()
    plt.figure(); plt.plot(g["N"], g["accuracy_mean"], marker="o"); plt.ylim(0,1.05)
    plt.title("Quantum H-Sim Reliability: Mean fidelity vs N"); plt.xlabel("qubits N"); plt.ylabel("Mean fidelity")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_fidelity_vs_N.png"), **plt_kwargs); plt.close()

    data = [df[df["N"]==k]["accuracy_mean"].values for k in sorted(df["N"].unique())]
    plt.figure(); _boxplot_with_labels(data, labels=sorted(df["N"].unique()))
    plt.title("Quantum H-Sim Reliability: Fidelity distribution by N"); plt.xlabel("qubits N"); plt.ylabel("Fidelity")
    plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, "rel_fidelity_boxplot.png"), **plt_kwargs); plt.close()

# ----------------------------- Main --------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quantum Hamiltonian Simulation (Ideal/Noiseless) — Benchmarks")
    parser.add_argument("--sizes", nargs="+", type=int, default=[6, 8, 10],
                        help="Number of qubits N.")
    parser.add_argument("--steps", type=int, default=50, help="Random initial states per (N, seed).")
    parser.add_argument("--trials", type=int, default=10, help="Seeds per N.")
    parser.add_argument("--time", type=float, default=1.0, help="Total evolution time T.")
    parser.add_argument("--trotter-steps", type=int, default=100, dest="r",
                        help="Number of Trotter steps for the COARSE simulation.")
    parser.add_argument("--order", type=int, choices=[1,2], default=2, help="Product formula order: 1=Lie, 2=Strang.")
    parser.add_argument("--J", type=float, default=1.0, help="Ising ZZ coupling.")
    parser.add_argument("--h", type=float, default=1.0, help="Transverse X field.")
    parser.add_argument("--periodic", type=int, default=0, help="1 for periodic boundary conditions.")
    parser.add_argument("--acc-tol", type=float, default=0.90, help="Require mean fidelity ≥ this for success.")
    parser.add_argument("--ref-mult", type=int, default=20, help="Reference has ref_mult × trotter-steps.")
    # Carbon/infra (unchanged)
    parser.add_argument("--excel", type=str, default="Filtered CO2 intensity 236 Countries.xlsx")
    parser.add_argument("--device-power-watts", type=float, default=65.0)
    parser.add_argument("--pue", type=float, default=1.2)
    parser.add_argument("--year-select", type=str, default="latest")
    parser.add_argument("--outdir", type=str, default="carbon_by_country")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    # fast toggle (optional)
    parser.add_argument("--fast", action="store_true", help="Quick mode: smaller sizes and fewer steps.")
    args = parser.parse_args()

    if args.fast:
        args.sizes = [6, 8]
        args.steps = 20
        args.r = max(20, args.r//5)

    # Output folders
    cwd = os.getcwd()
    perf_dir = os.path.join(cwd, "Performance")
    scal_dir = os.path.join(cwd, "Scalability")
    rel_dir  = os.path.join(cwd, "Reliability")
    carb_root = os.path.join(cwd, "Carbon footprints")
    carb_dir  = os.path.join(carb_root, args.outdir)
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]:
        os.makedirs(d, exist_ok=True)

    print("[run] Starting Quantum Hamiltonian Simulation (ideal, noiseless) benchmarks...")
    rows, jobs = [], []
    for N in args.sizes:
        for i in range(args.trials):
            seed = 1000 + 53*int(N) + i
            jobs.append((N, args.steps, seed, args.time, args.r, args.order,
                         args.J, args.h, args.periodic, args.acc_tol, args.ref_mult))

    def consume(res: dict):
        rows.append(res)
        print(f"  - N={res['N']} seed={res['seed']} "
              f"fid={res['accuracy_mean']:.3f} rmse={res['rmse_mean']:.4f} "
              f"r={res['r_steps']} ord={res['order']} runtime={res['runtime_s']:.3f}s "
              f"success={int(res['success'])}")

    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_single, *job) for job in jobs]
            for f in as_completed(futs):
                consume(f.result())
    else:
        for job in jobs:
            consume(run_single(*job))

    df = pd.DataFrame(rows)

    # ------------------- Performance (Excel + plots) -------------------
    perf_xlsx = os.path.join(perf_dir, "performance_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw_runs")
            agg = df.groupby("N").agg(
                mean_runtime_s=("runtime_s","mean"),
                mean_peak_mem_mb=("peak_mem_mb","mean"),
                mean_fidelity=("accuracy_mean","mean"),
                mean_ops=("exp_ops_total","mean"),
            ).reset_index()
            agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(df, perf_dir)

    # ------------------- Scalability (Excel + plots) -------------------
    scal_xlsx = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_xlsx, engine="openpyxl") as w:
            df[["N","runtime_s","peak_mem_mb","exp_ops_total"]].to_excel(w, index=False, sheet_name="raw")
            df.groupby("N").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(df, scal_dir)

    # ------------------- Reliability (Excel + plots) -------------------
    rel_xlsx = os.path.join(rel_dir, "reliability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(rel_xlsx, engine="openpyxl") as w:
            df[["N","seed","accuracy_mean","rmse_mean","success"]].to_excel(w, index=False, sheet_name="runs")
            rel_agg = df.groupby("N").agg(
                mean_fidelity=("accuracy_mean","mean"),
                std_fidelity=("accuracy_mean","std"),
                success_rate=("success","mean"),
            ).reset_index()
            rel_agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_reliability(df, rel_dir)

    # ------------------- Carbon (Performance runtime only) --------------
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
        # Plots
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
