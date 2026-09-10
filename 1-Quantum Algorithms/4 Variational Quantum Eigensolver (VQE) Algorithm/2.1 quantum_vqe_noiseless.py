# Noiseless (ideal) QUANTUM baseline: 
# VQE on 1D TFIM using a statevector simulator

# ----------------------------------------------------------------------------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
# How to run from Windows CMD (example):
# > python filename (Dear Researcher! You can change as per your requirements/logic).py ^


"""I ran it with the following baseline:
  python quantum_vqe_noiseless.py ^
    --fast --workers 4 ^
    --sizes 6 8 10 ^
    --J 1.0 --h 1.05 --periodic 0 ^
    --excel "Filtered CO2 intensity 236 Countries.xlsx" ^
    --device-power-watts 65 --pue 1.2 --year-select latest ^
    --outdir carbon_by_country --combine
"""

# Key features are:
# Ansatz: L-layer hardware-efficient circuit: [RY(θ_{l,i}) for all qubits] + CZ entanglers on a line (periodic optional).
# Simulator: exact statevector (NumPy), noiseless; energy computed analytically (no shots).
# Optimizer: parameter-shift gradients + Adam (stable and fast for small N).
# Reliability: success if |E_final - E_exact| ≤ max(abs_tol, rel_tol*|E_exact|)  (default rel_tol=5%).

# ------------********------------   Imports

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless savefig
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------********------------   Imports
from __future__ import annotations
import os, sys, time, math, argparse, tracemalloc, pathlib, warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ------------********------------  Dependency bootstrap 

def _ensure(pkgs: List[str]):
    import importlib
    missing = []
    for p in pkgs:
        try:
            importlib.import_module(p)     # Check if package is already installed
        except Exception:
            missing.append(p)              # Record missing packages
    if missing:
        print(f"[setup] Installing missing packages: {missing}")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])  # Auto-install missing packages
        except Exception:
            print("[setup] Auto-install failed. Please install manually:", missing)
            raise
_ensure(["numpy", "pandas", "matplotlib", "openpyxl"])   # Ensure required dependencies exist

# ------------********------------  TFIM model

def neighbors_open(i: int, N: int) -> Tuple[int|None, int|None]:
    left = i - 1 if i - 1 >= 0 else None          # Left neighbor exists only if within bounds
    right = i + 1 if i + 1 < N else None          # Right neighbor exists only if within bounds
    return left, right

def neighbors_periodic(i: int, N: int) -> Tuple[int, int]:
    return (i - 1) % N, (i + 1) % N               # Wrap-around neighbors for periodic boundary conditions

@dataclass
class TFIM:
    N: int
    J: float
    h: float
    periodic: bool = False                        # Whether to use periodic or open boundaries


# ------------********------------  Statevector simulator utils ----------------------------
#This function applies a single-qubit RY(θ) rotation directly to a full statevector in little-endian ordering.
#It iterates over amplitude blocks that correspond to the target qubit’s 0/1 subspaces 
# and updates them using the standard RY rotation matrix.
#By modifying the statevector in place, it avoids circuit simulation overhead and enables fast low-level quantum-state evolution.

def apply_ry(state: np.ndarray, q: int, theta: float):
    """Apply RY(theta) on qubit q (little-endian: bit 0 = qubit 0)."""
    dim = state.shape[0]
    m = 1 << q
    stride = m << 1
    c = math.cos(theta/2.0)
    s = math.sin(theta/2.0)
    for base in range(0, dim, stride):
        i0 = base
        i1 = base + m
        for off in range(m):
            a = state[i0 + off]
            b = state[i1 + off]
            state[i0 + off] = c*a - s*b
            state[i1 + off] = s*a + c*b

#This function applies a controlled-Z gate directly to a statevector by f
# lipping the sign of amplitudes where both target qubits are in the |1⟩ state.
#It uses bitmask checks to identify indices satisfying the CZ control condition.
#The operation is performed in place for efficiency during low-level simulation.
def apply_cz(state: np.ndarray, q: int, r: int):
    """Apply CZ(q, r). Multiply amplitudes by -1 where both bits are 1."""
    dim = state.shape[0]
    mask = (1 << q) | (1 << r)
    for idx in range(dim):
        if (idx & mask) == mask:
            state[idx] = -state[idx]


#This following function prepares a variational quantum state
# by sequentially applying RY rotations and CZ entanglers across multiple layers.
#Each layer applies specified RY angles to every qubit, followed by CZ gates between nearest neighbors 
# (and optionally between the endpoints for periodic lattices).
#The statevector is built explicitly, starting from |0…0⟩ and 
# evolving in place for efficient simulation.

def prepare_state(N: int, thetas: np.ndarray, layers: int, periodic: bool):
    """
    thetas shape: (layers, N), RY angles.
    Entanglers: CZ between (i, i+1) for i=0..N-2; if periodic, also (N-1,0).
    """
    state = np.zeros(1<<N, dtype=np.complex128)
    state[0] = 1.0  # |0...0>
    for l in range(layers):

        # single-qubit RY
        for q in range(N):
            apply_ry(state, q, float(thetas[l, q]))

        # CZ line
        for i in range(N-1):
            apply_cz(state, i, i+1)
        if periodic and N>2:
            apply_cz(state, N-1, 0)
    return state

# ------------********------------  Energy (noiseless) --------------------------------

#This function precomputes Z-basis eigenvalue signs for every qubit and every computational basis state.
#It produces an (N, 2^N) array where each entry is +1 or −1 depending on whether the corresponding bit is 0 or 1.
#These precomputed signs accelerate expectation-value calculations by avoiding repeated bitwise operations.
def precompute_z_signs(N: int) -> np.ndarray:
    """
    Returns an (N, 2^N) array with z_signs[q, s] = +1 if bit(q) of s is 0 else -1.
    """
    dim = 1<<N
    idx = np.arange(dim, dtype=np.uint64)
    signs = np.empty((N, dim), dtype=np.int8)
    for q in range(N):
        bit = (idx >> q) & 1
        signs[q] = 1 - 2*bit  # 0 -> +1, 1 -> -1
    return signs

#This function computes exact expectation values of Σ ZᵢZᵢ₊₁ and Σ Xᵢ from a full statevector.
#Z–Z correlations are obtained using precomputed sign tables weighted by probabilities; X expectations are computed 
#via amplitude pairings corresponding to bit flips.
#It supports both open and periodic boundary conditions 
#and returns the two expectation values as floats.
def expect_zz_and_x(state: np.ndarray, N: int, periodic: bool, z_signs: np.ndarray) -> Tuple[float, float]:
    """
    Compute <Σ Z_i Z_{i+1}>, <Σ X_i> exactly from statevector.
    """
    probs = np.real(state.conj() * state)  # length 2^N

    # <ZiZj>
    zz_sum = 0.0
    for i in range(N-1):
        zz_sum += float(np.sum(probs * (z_signs[i] * z_signs[i+1])))
    if periodic and N>1:
        zz_sum += float(np.sum(probs * (z_signs[N-1] * z_signs[0])))

    # <Xi>  via pair dot products
    x_sum = 0.0
    dim = state.shape[0]
    for i in range(N):
        m = 1 << i

        # indices with bit i == 0
        base = np.arange(0, dim, 2*m)
        offs = np.arange(m)
        i0 = (base[:, None] + offs[None, :]).ravel()
        i1 = i0 + m
        x_sum += 2.0 * float(np.real(np.vdot(state[i0], state[i1])))
    return zz_sum, x_sum

def vqe_energy(N: int, J: float, h: float, periodic: bool, thetas: np.ndarray, z_signs: np.ndarray) -> float:
    st = prepare_state(N, thetas, thetas.shape[0], periodic)
    zz, xs = expect_zz_and_x(st, N, periodic, z_signs)
    return -J * zz - h * xs

# ------------********------------  Exact reference -----------------------------------

#This function builds the full TFIM Hamiltonian matrix in the computational basis 
# for a given system size, couplings, and boundary type.
#If the Hilbert space exceeds a safety threshold, it skips computation; 
# otherwise it diagonalizes the Hamiltonian.
#It returns the exact ground-state energy as a float, enabling accuracy checks for variational methods.

def exact_tfim_energy(N: int, J: float, h: float, periodic: bool=False, max_dim: int=8192) -> float | None:
    dim = 1 << N
    if dim > max_dim:
        return None
    H = np.zeros((dim, dim), dtype=float)
    for state in range(dim):
        z = 1 - 2*np.array([(state >> i) & 1 for i in range(N)], dtype=int)
        zz = int(np.sum(z[:-1]*z[1:]))
        if periodic and N>1:
            zz += int(z[-1]*z[0])
        H[state, state] += -J * zz
        for i in range(N):
            H[state, state ^ (1<<i)] += -h
    evals = np.linalg.eigvalsh(H)
    return float(evals[0])

# ------------********------------  Optimizer (Adam) --------------------------------

@dataclass
class OptimConfig:
    steps: int = 150
    layers: int = 3
    lr: float = 0.05
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    early_stop: bool = False
    patience: int = 10
    min_delta: float = 1e-4
    seed: int = 123


#This function computes the parameter-shift gradient for a VQE ansatz by evaluating energies at θ±π/2 for each parameter.
#It restores parameters after each perturbation and returns both the gradient array and the unshifted energy.
#Although not optimized for reuse of intermediate states, it provides a clear, correct implementation of the parameter-shift rule.

def parameter_shift_grad(N: int, J: float, h: float, periodic: bool, thetas: np.ndarray, z_signs: np.ndarray) -> Tuple[np.ndarray, float]:
    """Compute grad dE/dtheta via 0.5*(E(theta+π/2) - E(theta-π/2)) for each parameter."""
    L, Q = thetas.shape
    grad = np.zeros_like(thetas)
    base_E = vqe_energy(N, J, h, periodic, thetas, z_signs)
    shift = math.pi/2.0

    # For efficiency we can reuse state if we implemented light updates; for clarity we re-evaluate.

    for l in range(L):
        for q in range(Q):
            thetas[l, q] += shift
            Ep = vqe_energy(N, J, h, periodic, thetas, z_signs)
            thetas[l, q] -= 2*shift
            Em = vqe_energy(N, J, h, periodic, thetas, z_signs)
            thetas[l, q] += shift  # restore
            grad[l, q] = 0.5 * (Ep - Em)
    return grad, base_E

def optimize_vqe(N: int, J: float, h: float, periodic: bool, layers: int, steps: int, seed: int,
                 lr: float, beta1: float, beta2: float, eps: float,
                 early_stop: bool, patience: int, min_delta: float) -> Dict[str, object]:
    rng = np.random.default_rng(seed)

    # Initialize RY angles small random
    thetas = 0.1 * rng.standard_normal((layers, N))
    z_signs = precompute_z_signs(N)

    m = np.zeros_like(thetas)
    v = np.zeros_like(thetas)
    hist = []
    best_E = float("inf")
    best_thetas = thetas.copy()
    no_improve = 0

    t0 = time.perf_counter()
    for k in range(1, steps+1):
        g, E = parameter_shift_grad(N, J, h, periodic, thetas, z_signs)

        # Adam update
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g * g)
        m_hat = m / (1 - beta1**k)
        v_hat = v / (1 - beta2**k)
        thetas = thetas - lr * m_hat / (np.sqrt(v_hat) + eps)

        hist.append(float(E))
        if E < best_E - min_delta:
            best_E = float(E)
            best_thetas = thetas.copy()
            no_improve = 0
        else:
            no_improve += 1
            if early_stop and no_improve >= patience:
                break

    runtime = time.perf_counter() - t0

    # Evaluates final (best) energy

    final_E = vqe_energy(N, J, h, periodic, best_thetas, z_signs)
    grad_norm = float(np.linalg.norm(g))
    return {
        "final_energy": float(final_E),
        "history": [float(x) for x in hist],
        "runtime_s": float(runtime),
        "avg_grad_norm": grad_norm,
        "layers": layers,
    }

# ------------********------------  Orchestration (one run) ------------------------------
#This function runs a full VQE optimization, tracking runtime, memory usage, convergence, and accuracy 
#  relative to the exact TFIM solution.
#It computes the baseline energy at zero parameters, evaluates exact energies when feasible,
#    and aggregates all results into a structured dictionary.
#Returned metrics include final energy, error, gradient norms, memory footprint, and the full optimization history.

def run_single(N: int, seed: int, steps: int, layers: int, J: float, h: float, periodic: bool,
               lr: float, early_stop: bool, patience: int, min_delta: float) -> Dict[str, object]:
    tracemalloc.start()
    t0 = time.perf_counter()
    res = optimize_vqe(
        N=N, J=J, h=h, periodic=periodic, layers=layers, steps=steps, seed=seed,
        lr=lr, beta1=0.9, beta2=0.999, eps=1e-8,
        early_stop=early_stop, patience=patience, min_delta=min_delta
    )
    runtime = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # starting energy at thetas=0 (all RY=0 -> |0...0>)

    z_signs = precompute_z_signs(N)
    init_thetas = np.zeros((layers, N))
    init_energy = vqe_energy(N, J, h, periodic, init_thetas, z_signs)

    e_exact = exact_tfim_energy(N, J, h, periodic=periodic, max_dim=8192)
    error = None if e_exact is None else float(res["final_energy"] - e_exact)

    return {
        "N": N,
        "seed": seed,
        "steps": steps,
        "layers": layers,
        "runtime_s": float(runtime),
        "final_energy": float(res["final_energy"]),
        "init_energy": float(init_energy),
        "exact_energy": None if e_exact is None else float(e_exact),
        "error": error,
        "avg_grad_norm": float(res["avg_grad_norm"]),
        "peak_mem_mb": float(peak/(1024*1024)),
        "history": res["history"],
    }

# ------------********------------  Carbon intensity I/O --------------------------------

def resolve_excel_path(excel_arg: str) -> str:
    p = pathlib.Path(excel_arg)
    if p.is_absolute() and p.exists(): return str(p)
    desktop = pathlib.Path.home() / "Desktop" / excel_arg
    return str(desktop if desktop.exists() else p)


#This function loads a carbon-intensity spreadsheet and auto-detects the Country, Year, and Intensity columns using flexible name matching.
#It normalizes the table, optionally selects the latest year per country, and 
#   heuristically converts g/kWh values to kg/kWh when magnitudes indicate unit mismatch.
#The cleaned DataFrame—standardized and filtered for valid rows—is returned for downstream carbon-emission calculations.

def load_carbon_excel(path: str, year_select: str="latest") -> pd.DataFrame:
    df = pd.read_excel(path)
    cols = {c.lower(): c for c in df.columns}
    cand_country = next((v for k,v in cols.items() if "country" in k or "nation" in k or k=="location"), None)
    if cand_country is None: raise ValueError("No 'Country' column found.")
    cand_year = next((v for k,v in cols.items() if "year" in k or "date" in k), None)
    cand_intensity = next((v for k,v in cols.items() if "intensity" in k or ("co2" in k and ("kwh" in k or "/kwh" in k)) or "kgco2" in k or "gco2" in k), None)
    if cand_intensity is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if cand_year in numeric_cols: numeric_cols.remove(cand_year)
        if not numeric_cols: raise ValueError("No numeric intensity column detected.")
        cand_intensity = numeric_cols[0]
    keep = [cand_country] + ([cand_year] if cand_year else []) + [cand_intensity]
    df = df[keep].copy()
    df.columns = ["Country","Year","Intensity"] if len(keep)==3 else ["Country","Intensity"]
    if "Year" in df.columns and year_select.lower()=="latest":
        df = df.sort_values(["Country","Year"]).groupby("Country", as_index=False).tail(1)
    med = float(df["Intensity"].dropna().median())
    if med > 50: df["Intensity"] = df["Intensity"]/1000.0  # g/kWh -> kg/kWh
    return df.dropna(subset=["Country","Intensity"]).reset_index(drop=True)


#This function converts total benchmark runtime into energy consumption 
#   and multiplies it by country-specific carbon intensities to estimate CO₂ emissions.
#It augments the intensity table with kWh and kg-CO₂e columns, 
#    and builds a summary containing aggregate energy and emission statistics.
#The detailed per-country emissions (sorted highest first) 
#    and the summary table are both returned.

def compute_carbon(perf_df: pd.DataFrame, intensity_df: pd.DataFrame,
                   power_watts: float, pue: float, combine: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_runtime_s = float(perf_df["runtime_s"].sum())
    kWh_total = power_watts * pue * total_runtime_s / 3_600_000.0
    df = intensity_df.copy()
    df["kWh"] = kWh_total
    df["kgCO2e"] = df["Intensity"] * df["kWh"]
    summary = pd.DataFrame({
        "total_runtime_s":[total_runtime_s],
        "power_watts":[power_watts], "PUE":[pue], "kWh_total":[kWh_total],
        "median_intensity_kg_per_kWh":[float(df["Intensity"].median())],
        "mean_kgCO2e_across_countries":[float(df["kgCO2e"].mean())],
    })
    return df.sort_values("kgCO2e", ascending=False), summary

# ------------********------------  plotting helpers --------------------------------
#This plotting function generates performance visualizations, including runtime scaling, final-energy statistics, 
# and energy errors when exact results are available.
#It also plots a detailed convergence curve for the largest system size using stored optimization histories.
#All figures are saved to the specified directory using consistent formatting options.

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def plot_performance(perf_df: pd.DataFrame, histories: Dict[Tuple[int,int], List[float]], outdir: str):
    g = perf_df.groupby("N")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["N"], g["runtime_s"], marker="o")
    plt.xlabel("System size N"); plt.ylabel("Mean runtime (s)"); plt.title("Performance: Runtime vs N")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = perf_df.groupby("N")["final_energy"].agg(["mean","std"]).reset_index()
    plt.figure(); plt.errorbar(g2["N"], g2["mean"], yerr=g2["std"], fmt="-o")
    plt.xlabel("System size N"); plt.ylabel("Final energy (mean±std)"); plt.title("Performance: Final energy vs N")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_energy_vs_N.png"), **plt_kwargs); plt.close()

    sub = perf_df.dropna(subset=["error"])
    if not sub.empty:
        g3 = sub.groupby("N")["error"].agg(["mean","std"]).reset_index()
        plt.figure(); plt.errorbar(g3["N"], g3["mean"], yerr=g3["std"], fmt="-o")
        plt.xlabel("System size N (exact available)"); plt.ylabel("Energy error (mean±std)")
        plt.title("Performance: Energy error vs N"); plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(outdir, "perf_error_vs_N.png"), **plt_kwargs); plt.close()

    maxN = int(perf_df["N"].max()); seeds = sorted(perf_df[perf_df["N"]==maxN]["seed"].unique())
    if seeds:
        s = seeds[0]; key = (maxN, s)
        if key in histories:
            plt.figure(); plt.plot(range(1, len(histories[key])+1), histories[key])
            plt.xlabel("Iteration"); plt.ylabel("Energy"); plt.title(f"Performance: Convergence N={maxN}, seed={s}")
            plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, f"perf_convergence_N{maxN}_seed{s}.png"), **plt_kwargs); plt.close()


#This function creates scalability plots showing how runtime, memory usage, per-iteration cost, 
#   and gradient norms grow with system size.
#It aggregates metrics over seeds, then produces four separate line plots with consistent styling.
#Each plot is saved to the provided output directory for reporting or comparison.

def plot_scalability(perf_df: pd.DataFrame, outdir: str):
    g = perf_df.groupby("N")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["N"], g["runtime_s"], marker="o")
    plt.xlabel("System size N"); plt.ylabel("Mean runtime (s)"); plt.title("Scalability: Runtime vs N")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = perf_df.groupby("N")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g2["N"], g2["peak_mem_mb"], marker="s")
    plt.xlabel("System size N"); plt.ylabel("Peak memory (MB)"); plt.title("Scalability: Peak memory vs N")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_N.png"), **plt_kwargs); plt.close()

    g3 = perf_df.copy(); g3["rt_per_iter"] = g3["runtime_s"] / g3["steps"]; g3 = g3.groupby("N")["rt_per_iter"].mean().reset_index()
    plt.figure(); plt.plot(g3["N"], g3["rt_per_iter"], marker="^")
    plt.xlabel("System size N"); plt.ylabel("Runtime per iteration (s)"); plt.title("Scalability: Runtime/iter vs N")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_rt_per_iter_vs_N.png"), **plt_kwargs); plt.close()

    g4 = perf_df.groupby("N")["avg_grad_norm"].mean().reset_index()
    plt.figure(); plt.plot(g4["N"], g4["avg_grad_norm"], marker="d")
    plt.xlabel("System size N"); plt.ylabel("Avg grad norm"); plt.title("Scalability: Gradient norm vs N")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_gradnorm_vs_N.png"), **plt_kwargs); plt.close()


#This function generates reliability plots analyzing variability across seeds and system sizes, 
#   including boxplots, standard deviation curves, and success rates.
#It also produces a heatmap showing final energies for each (seed, N) pair, revealing patterns of stability or sensitivity.
#All plots use consistent formatting and are saved to the specified directory 
#   for comprehensive reliability evaluation.

def _boxplot_with_labels(data, labels):
    try: plt.boxplot(data, tick_labels=labels)
    except TypeError: plt.boxplot(data, labels=labels)

def plot_reliability(perf_df: pd.DataFrame, outdir: str):
    plt.figure()
    data = [perf_df[perf_df["N"]==N]["final_energy"].values for N in sorted(perf_df["N"].unique())]
    _boxplot_with_labels(data, labels=sorted(perf_df["N"].unique()))
    plt.xlabel("System size N"); plt.ylabel("Final energy distribution"); plt.title("Reliability: Final energy variability")
    plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, "rel_boxplot_final_energy.png"), **plt_kwargs); plt.close()

    g = perf_df.groupby("N")["final_energy"].std().reset_index()
    plt.figure(); plt.plot(g["N"], g["final_energy"], marker="o")
    plt.xlabel("System size N"); plt.ylabel("Std dev of final energy"); plt.title("Reliability: Std dev vs N")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_std_vs_N.png"), **plt_kwargs); plt.close()

    g2 = perf_df.groupby("N")["success"].mean().reset_index(name="success_rate")
    plt.figure(); plt.plot(g2["N"], g2["success_rate"], marker="s"); plt.ylim(0,1)
    plt.xlabel("System size N"); plt.ylabel("Success rate"); plt.title("Reliability: Success rate vs N")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_N.png"), **plt_kwargs); plt.close()

    piv = perf_df.pivot_table(index="seed", columns="N", values="final_energy")
    plt.figure(); plt.imshow(piv.values, aspect='auto'); plt.colorbar(label="Final energy")
    plt.yticks(range(len(piv.index)), piv.index); plt.xticks(range(len(piv.columns)), piv.columns)
    plt.xlabel("N"); plt.ylabel("seed"); plt.title("Reliability: seed×N final energy heatmap")
    plt.savefig(os.path.join(outdir, "rel_heatmap_seed_N.png"), **plt_kwargs); plt.close()

# ------------********------------  Main runner ---------------------------------

def ensure_dir(d: str): os.makedirs(d, exist_ok=True)
def to_excel(dfs: Dict[str,pd.DataFrame], path: str):
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name,df in dfs.items():
            df.to_excel(w, index=False, sheet_name=name[:31])

def main():
    p = argparse.ArgumentParser(description="VQE (noiseless) benchmarks: Performance, Scalability, Reliability, Carbon")
    p.add_argument("--sizes", nargs="+", type=int, default=[6,8,10,12], help="System sizes N")
    p.add_argument("--steps", type=int, default=150, help="Optimizer steps")
    p.add_argument("--layers", type=int, default=3, help="Ansatz depth (layers)")
    p.add_argument("--seeds", type=int, default=5, help="Random initializations per N")
    p.add_argument("--lr", type=float, default=0.05, help="Adam learning rate")
    p.add_argument("--J", type=float, default=1.0, help="TFIM coupling J")
    p.add_argument("--h", type=float, default=1.05, help="TFIM transverse field h")
    p.add_argument("--periodic", type=int, default=0, help="1=periodic, 0=open")

    # ------------********------------  speed/robustness

    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2), help="Parallel processes")
    p.add_argument("--fast", action="store_true", help="Quick mode: steps=60, layers=2, seeds=3, early-stop")
    p.add_argument("--early-stop", action="store_true", help="Enable early stopping")
    p.add_argument("--patience", type=int, default=10, help="Early-stop patience")
    p.add_argument("--min-delta", type=float, default=1e-4, help="Min improvement to reset patience")

    # ------------********------------  reliability threshold

    p.add_argument("--rel-tol", type=float, default=0.05, help="Relative error tolerance (default 5%)")
    p.add_argument("--abs-tol", type=float, default=1e-3, help="Absolute error floor")

    # ------------********------------  carbon

    p.add_argument("--excel", type=str, default="Filtered CO2 intensity 236 Countries.xlsx")
    p.add_argument("--device-power-watts", type=float, default=65.0)
    p.add_argument("--pue", type=float, default=1.2)
    p.add_argument("--year-select", type=str, default="latest")
    p.add_argument("--outdir", type=str, default="carbon_by_country")
    p.add_argument("--combine", action="store_true")

    args = p.parse_args()
    if args.fast:
        if args.steps==150: args.steps = 60
        if args.layers==3: args.layers = 2
        if args.seeds==5: args.seeds = 3
        args.early_stop = True
        if args.patience==10: args.patience = 6
        if args.min_delta==1e-4: args.min_delta = 5e-4

    periodic = bool(args.periodic)
    sizes = sorted(set(int(n) for n in args.sizes))

    # ------------********------------  Output folders

    perf_dir = os.path.join(os.getcwd(), "Performance")
    scal_dir = os.path.join(os.getcwd(), "Scalability")
    rel_dir  = os.path.join(os.getcwd(), "Reliability")
    carb_root = os.path.join(os.getcwd(), "Carbon footprints")
    carb_dir  = os.path.join(carb_root, args.outdir)
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]: ensure_dir(d)

    # ------------********------------  Parallel jobs

    jobs = [(N, 1000+13*N+s) for N in sizes for s in range(args.seeds)]
    all_rows = []; histories: Dict[Tuple[int,int], List[float]] = {}
    print("[run] Starting noiseless VQE experiments...")

    def _consume(row):
        all_rows.append(row)
        histories[(row["N"], row["seed"])] = row["history"]
        print(f"  - done N={row['N']}, seed={row['seed']} (runtime={row['runtime_s']:.2f}s, steps={row['steps']}, layers={row['layers']})")

    if args.workers>1 and len(jobs)>1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_single, N, seed, args.steps, args.layers, args.J, args.h, periodic,
                              args.lr, args.early_stop, args.patience, args.min_delta) for (N,seed) in jobs]
            for f in as_completed(futs): _consume(f.result())
    else:
        for (N,seed) in jobs:
            _consume(run_single(N, seed, args.steps, args.layers, args.J, args.h, periodic,
                                args.lr, args.early_stop, args.patience, args.min_delta))

    perf_df = pd.DataFrame(all_rows)

    # ------------********--------------------------- Performance ----------------

    perf_excel = os.path.join(perf_dir, "performance_results.xlsx")
    perf_agg = perf_df.groupby("N").agg(
        runtime_s=("runtime_s","mean"),
        final_energy=("final_energy","mean"),
        error=("error","mean"),
        avg_grad_norm=("avg_grad_norm","mean"),
    ).reset_index()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_excel, engine="openpyxl") as w:
            perf_df.drop(columns=["history"]).to_excel(w, index=False, sheet_name="raw_runs")
            perf_agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(perf_df, histories, perf_dir)

    # ------------********--------------------------- Scalability ----------------

    scal_excel = os.path.join(scal_dir, "scalability_results.xlsx")
    scal_agg = perf_df.groupby("N").agg(
        runtime_s=("runtime_s","mean"),
        peak_mem_mb=("peak_mem_mb","mean"),
        steps=("steps","mean"),
        layers=("layers","mean"),
        avg_grad_norm=("avg_grad_norm","mean"),
    ).reset_index()
    scal_agg["rt_per_iter"] = scal_agg["runtime_s"]/scal_agg["steps"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_excel, engine="openpyxl") as w:
            perf_df[["N","seed","runtime_s","peak_mem_mb","steps","layers","avg_grad_norm"]].to_excel(w, index=False, sheet_name="from_perf_runs")
            scal_agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(perf_df, scal_dir)

    # ------------********------------  Reliability ----------------

    rel_excel = os.path.join(rel_dir, "reliability_results.xlsx")
    rel_tol = float(args.rel_tol); abs_tol = float(args.abs_tol)
    def success(row):
        if not pd.isna(row["exact_energy"]):
            thr = max(abs_tol, rel_tol * abs(row["exact_energy"]))
            return abs(row["final_energy"] - row["exact_energy"]) <= thr
        else:

            # fallback: improved from init by ≥5%

            return (row["init_energy"] - row["final_energy"]) > 0.05 * abs(row["init_energy"])
    perf_df["success"] = perf_df.apply(success, axis=1)
    rel_agg = perf_df.groupby("N").agg(
        mean_final=("final_energy","mean"),
        std_final=("final_energy","std"),
        success_rate=("success","mean"),
    ).reset_index()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(rel_excel, engine="openpyxl") as w:
            perf_df[["N","seed","final_energy","init_energy","exact_energy","success"]].to_excel(w, index=False, sheet_name="runs_with_success")
            rel_agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_reliability(perf_df, rel_dir)

    # ------------********------------ Carbon (from PERFORMANCE runtimes only)

    excel_path = resolve_excel_path(args.excel)
    intensity_df = None
    try:
        intensity_df = load_carbon_excel(excel_path, year_select=args.year_select)
    except Exception as e:
        print(f"[carbon] ERROR reading Excel '{excel_path}': {e}")
        print("[carbon] Skipping carbon benchmark. (You can re-run once the file is available.)")

    ensure_dir(carb_dir)
    if intensity_df is not None:
        carbon_df, summary_df = compute_carbon(perf_df, intensity_df,
                                               power_watts=args.device_power_watts,
                                               pue=args.pue, combine=args.combine)
        carb_excel = os.path.join(carb_dir, "carbon_results.xlsx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pd.ExcelWriter(carb_excel, engine="openpyxl") as w:
                carbon_df.to_excel(w, index=False, sheet_name="per_country")
                summary_df.to_excel(w, index=False, sheet_name="summary")
                intensity_df.to_excel(w, index=False, sheet_name="intensity_input")
        # plots
        top = carbon_df.nlargest(15, "kgCO2e"); plt.figure(figsize=(8,5))
        plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
        plt.xlabel("kg CO2e (higher = dirtier grid)"); plt.title("Carbon: Top 15 countries (kgCO2e)")
        plt.tight_layout(); plt.savefig(os.path.join(carb_dir, "carbon_top15.png"), **plt_kwargs); plt.close()

        bot = carbon_df.nsmallest(15, "kgCO2e"); plt.figure(figsize=(8,5))
        plt.barh(bot["Country"][::-1], bot["kgCO2e"][::-1])
        plt.xlabel("kg CO2e (lower = cleaner grid)"); plt.title("Carbon: Bottom 15 countries (kgCO2e)")
        plt.tight_layout(); plt.savefig(os.path.join(carb_dir, "carbon_bottom15.png"), **plt_kwargs); plt.close()

        plt.figure(); plt.hist(carbon_df["kgCO2e"], bins=30)
        plt.xlabel("kg CO2e for this experiment (per country)"); plt.title("Carbon: Emission distribution")
        plt.grid(True, alpha=0.3); plt.savefig(os.path.join(carb_dir, "carbon_distribution.png"), **plt_kwargs); plt.close()

        xs = np.sort(carbon_df["kgCO2e"].values); ys = np.arange(1,len(xs)+1)/len(xs)
        plt.figure(); plt.plot(xs, ys); plt.xlabel("kg CO2e"); plt.ylabel("CDF"); plt.title("Carbon: CDF")
        plt.grid(True, alpha=0.3); plt.savefig(os.path.join(carb_dir, "carbon_cdf.png"), **plt_kwargs); plt.close()
        print(f"[carbon] Wrote Excel and plots to: {carb_dir}")
    else:
        carb_excel = os.path.join(carb_dir, "carbon_results.xlsx")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pd.ExcelWriter(carb_excel, engine="openpyxl") as w:
                pd.DataFrame({"message":["Carbon benchmark skipped due to missing/invalid Excel input"]}).to_excel(w, index=False, sheet_name="note")

    # ------------********------------  wrap-up
    #This block prints a concise console summary showing locations of all 
    #  generated Excel reports across performance, scalability, reliability, and carbon analyses.
#It also lists the created output folders, ensuring users know where to find all benchmark artifacts.
#The script ends by invoking `main()` when executed as a standalone program.

    print("\n=== Summary (console) ===")
    print("Performance Excel:", os.path.join(perf_dir, "performance_results.xlsx"))
    print("Scalability Excel:", os.path.join(scal_dir, "scalability_results.xlsx"))
    print("Reliability Excel:", os.path.join(rel_dir, "reliability_results.xlsx"))
    print("Carbon Excel:", os.path.join(carb_dir, "carbon_results.xlsx"))
    print("Folders created:\n -", perf_dir, "\n -", scal_dir, "\n -", rel_dir, "\n -", carb_dir)

if __name__ == "__main__":
    main()
