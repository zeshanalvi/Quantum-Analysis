#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# CLASSICAL baseline for a VQE counterpart using Variational Monte Carlo (VMC)
# on the 1D Transverse-Field Ising Model (TFIM)

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

"""I ran it with the following baseline:
  python classical_vmc_benchmarks.py ^
    --fast --workers 4 ^
    --sizes 6 8 10 ^
    --J 1.0 --h 1.05 --periodic 0 ^
    --excel "Filtered CO2 intensity 236 Countries.xlsx" ^
    --device-power-watts 65 --pue 1.2 --year-select latest ^
    --outdir carbon_by_country --combine
"""
# ------------********------------   Imports

from __future__ import annotations
import os, sys, time, math, argparse, tracemalloc, pathlib, warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict

# ------------********------------  Dependency bootstrap 

#This def _ensure() helper function checks for required Python packages and attempts to install any missing ones automatically.
#It imports each package, collects failures, and 
#invokes pip to install missing dependencies, printing helpful messages along the way.
#If installation fails, it alerts the user and 
#raises the exception so they can resolve the issue manually.

def _ensure(pkgs: List[str]):
    import importlib
    missing = []
    for p in pkgs:
        try:
            importlib.import_module(p)
        except Exception:
            missing.append(p)
    if missing:
        print(f"[setup] Installing missing packages: {missing}")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
        except Exception:
            print("[setup] Auto-install failed. Please install manually:", missing)
            raise
_ensure(["numpy", "pandas", "matplotlib", "openpyxl"])


# ------------********------------   Imports

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless savefig
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------********------------TFIM model

#The following functions return neighbor indices for open and periodic 1D lattices, handling boundaries appropriately.

def neighbors_open(i: int, N: int) -> Tuple[int|None, int|None]:
    left = i - 1 if i - 1 >= 0 else None
    right = i + 1 if i + 1 < N else None
    return left, right

def neighbors_periodic(i: int, N: int) -> Tuple[int, int]:
    return (i - 1) % N, (i + 1) % N

#The `TFIM` dataclass encodes parameters of the transverse-field Ising model 
#and provides methods to compute ZZ-interaction energy and local energy.

@dataclass
class TFIM:
    N: int
    J: float
    h: float
    periodic: bool = False

    def energy_zz(self, z: np.ndarray) -> float:
        if self.periodic:
            return -self.J * float(np.sum(z * np.roll(z, -1)))
        else:
            return -self.J * float(np.sum(z[:-1] * z[1:]))


#`local_energy` functionevaluates the variational local energy by summing interaction terms 
#and flip-ratio contributions derived from the model’s exponential wavefunction ansatz.

    def local_energy(self, z: np.ndarray, a_vec: np.ndarray, j: float) -> float:

        # E_loc(z) = -J Σ z_i z_{i+1} - h Σ_i ψ(z^flip_i)/ψ(z)

        N = self.N
        e = self.energy_zz(z)
        ratio_sum = 0.0
        for i in range(N):
            if self.periodic:
                l, r = neighbors_periodic(i, N)
                nn = z[l] + z[r]
            else:
                l, r = neighbors_open(i, N)
                nn = 0
                if l is not None: nn += z[l]
                if r is not None: nn += z[r]

            # ratio ψ(z^flip_i)/ψ(z) for ψ ∝ exp(Σ a_i z_i + j Σ z_i z_{i+1})

            ratio_sum += math.exp(-2.0 * a_vec[i]*z[i] - 2.0 * j * z[i] * nn)
        e += -self.h * ratio_sum
        return float(e)

# ------------********------------  VMC core
#This dataclass stores all configuration parameters for a Variational Monte Carlo (VMC) optimization run.
#It includes MCMC settings (steps, samples, burn-in), SPSA hyperparameters for gradient estimation, 
#and reproducibility/early-stopping controls. These values govern convergence behavior, sampling quality,
# and stability of the VMC training loop.

@dataclass
class VMCConfig:
    steps: int = 150
    samples_per_iter: int = 3000
    burn_in: int = 300
    spsa_a0: float = 0.15
    spsa_c0: float = 0.05
    spsa_alpha: float = 0.602
    spsa_gamma: float = 0.101
    seed: int = 123
    early_stop: bool = False
    patience: int = 10
    min_delta: float = 1e-3


class VMC:
#This VMC class implements a Variational Monte Carlo solver using a site-wise exponential Jastrow ansatz 
#and Metropolis sampling. Its `_ratio_flip` helper computes 
#the wavefunction amplitude ratio for flipping a single spin, incorporating local fields and neighbor interactions.
#This flip ratio is fundamental for Metropolis acceptance decisions and SPSA-based parameter updates.

    """Site-wise exp-Jastrow ansatz with Metropolis + SPSA over θ = [a_0..a_{N-1}, j]."""
    def __init__(self, model: TFIM, cfg: VMCConfig):
        self.m = model
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _ratio_flip(self, z: np.ndarray, i: int, a_vec: np.ndarray, j: float) -> float:
        N = self.m.N
        zi = z[i]
        if self.m.periodic:
            l, r = neighbors_periodic(i, N)
            nn = z[l] + z[r]
        else:
            l, r = neighbors_open(i, N)
            nn = 0
            if l is not None: nn += z[l]
            if r is not None: nn += z[r]
        return math.exp(-2.0 * a_vec[i] * zi - 2.0 * j * zi * nn)

    def metropolis(self, a_vec: np.ndarray, j: float, n_samples: int, burn_in: int) -> Tuple[np.ndarray, float]:
        z = self.rng.choice([-1, 1], size=self.m.N)
        acc = 0; prop = 0
        for _ in range(burn_in):
            i = self.rng.integers(0, self.m.N)
            r = self._ratio_flip(z, i, a_vec, j)
            A = min(1.0, r*r)
            if self.rng.random() < A: z[i] *= -1
        samples = []
        for _ in range(n_samples):
            i = self.rng.integers(0, self.m.N)
            r = self._ratio_flip(z, i, a_vec, j)
            A = min(1.0, r*r)
            prop += 1
            if self.rng.random() < A:
                z[i] *= -1
                acc += 1
            samples.append(z.copy())
        return np.array(samples, dtype=int), acc/max(1,prop)

    def energy(self, a_vec: np.ndarray, j: float, n_samples: int, burn_in: int) -> Tuple[float, float]:
        samples, acc_rate = self.metropolis(a_vec, j, n_samples, burn_in)
        e = 0.0
        for z in samples:
            e += self.m.local_energy(z, a_vec, j)
        return e/len(samples), acc_rate

    # convenience: E(θ) with θ = [a_0..a_{N-1}, j]

    def energy_theta(self, theta: np.ndarray) -> Tuple[float, float]:
        a_vec = theta[:-1]; j = float(theta[-1])
        return self.energy(a_vec, j, self.cfg.samples_per_iter, self.cfg.burn_in)

#This method performs VMC parameter optimization using SPSA, updating the parameter vector 
#via stochastic gradient estimates from symmetric perturbations.
#It tracks energy history, acceptance statistics, early stopping criteria, and runtime, producing a full diagnostic summary.
#The returned dictionary includes final energies, learned parameters, convergence trajectory, and average Metropolis acceptance rate.
    def optimize_spsa(self) -> Dict[str, object]:
        cfg = self.cfg
        theta = np.zeros(self.m.N + 1, dtype=float)  # all a_i=0, j=0
        rng = self.rng
        hist = []
        start = time.perf_counter()
        total_acc = 0.0; total_prop = 0
        best = float("inf"); no_improve = 0
        for k in range(1, cfg.steps + 1):
            ak = cfg.spsa_a0 / (k ** cfg.spsa_alpha)
            ck = cfg.spsa_c0 / (k ** cfg.spsa_gamma)
            Delta = rng.choice([-1.0, 1.0], size=theta.shape)
            e_plus, acc_plus = self.energy_theta(theta + ck*Delta)
            e_minus, acc_minus = self.energy_theta(theta - ck*Delta)
            ghat = (e_plus - e_minus) / (2.0 * ck) * Delta
            theta -= ak * ghat
            e_curr, acc_curr = self.energy_theta(theta)
            hist.append(e_curr)
            total_acc += (acc_plus + acc_minus + acc_curr); total_prop += 3
            if cfg.early_stop:
                if e_curr + cfg.min_delta < best:
                    best = e_curr; no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= cfg.patience: break
        runtime = time.perf_counter() - start
        return {
            "final_energy": float(hist[-1]),
            "theta_a": [float(x) for x in theta[:-1]],
            "theta_j": float(theta[-1]),
            "history": [float(x) for x in hist],
            "runtime_s": float(runtime),
            "avg_acceptance_rate": float(total_acc / max(1, total_prop)),
        }


# ------------********------------exact (tiny N)


#The def exact_tfim_energy () function computes the exact ground-state energy of the TFIM 
#by constructing the full Hamiltonian matrix in the computational basis.
#It handles open or periodic boundaries, applies the correct J and h couplings, 
#and performs an eigenvalue solve when the Hilbert space is small enough.
#If the system exceeds the specified dimension cap, it returns `None` to avoid excessive memory usage.

def exact_tfim_energy(N: int, J: float, h: float, periodic: bool=False, max_dim: int=8192) -> float|None:
    dim = 1 << N
    if dim > max_dim: return None
    H = np.zeros((dim, dim), dtype=float)
    for state in range(dim):
        z = 1 - 2*np.array([(state >> i) & 1 for i in range(N)], dtype=int)
        zz = int(np.sum(z[:-1]*z[1:]))
        if periodic and N>1: zz += int(z[-1]*z[0])
        H[state, state] += -zz  # J=1 temporarily, scale later
        for i in range(N):
            H[state, state ^ (1<<i)] += -1.0  # h=1 temporarily

    # scale for actual J,h

    evals = np.linalg.eigvalsh((-1.0)*H)  # I built with J=h=1 but with a sign; safer to rebuild:

    # Simpler: rebuild precisely for given J,h:

    H = np.zeros((dim, dim), dtype=float)
    for state in range(dim):
        z = 1 - 2*np.array([(state >> i) & 1 for i in range(N)], dtype=int)
        zz = int(np.sum(z[:-1]*z[1:]))
        if periodic and N>1: zz += int(z[-1]*z[0])
        H[state, state] += -J*zz
        for i in range(N):
            H[state, state ^ (1<<i)] += -h
    evals = np.linalg.eigvalsh(H)
    return float(evals[0])


# ------------********------------  Orchestration

#This dataclass defines configuration parameters for orchestrating a full TFIM VMC benchmark 
#across multiple system sizes and seeds.
#It captures model parameters (J, h, periodicity) 
#along with VMC settings such as SPSA steps, sampling budget, and early-stopping controls.
#These values allow the benchmarking driver 
#to systematically generate experiments and ensure consistent reproducibility.

@dataclass
class BenchConfig:
    sizes: List[int]; steps: int; samples_per_iter: int; seeds: int
    J: float; h: float; periodic: bool
    early_stop: bool; patience: int; min_delta: float


#This def run_single () function runs a full VMC optimization workflow for a single TFIM instance, 
# including initialization, sampling, SPSA optimization, and diagnostics.
#It measures runtime, peak memory usage, initial vs. final energies, 
# and computes exact energies when feasible to quantify accuracy.
#The returned dictionary aggregates all relevant metrics, learned parameters, 
# and convergence history for downstream benchmarking.

def run_single(N: int, seed: int, steps: int, samples: int, J: float, h: float, periodic: bool,
               early_stop: bool, patience: int, min_delta: float) -> Dict[str, object]:
    model = TFIM(N=N, J=J, h=h, periodic=periodic)
    vmc_cfg = VMCConfig(steps=steps, samples_per_iter=samples, seed=seed,
                        early_stop=early_stop, patience=patience, min_delta=min_delta)
    tracemalloc.start()
    t0 = time.perf_counter()
    vmc = VMC(model, vmc_cfg)

    # init θ=0 => a_i=0, j=0

    e_init, acc0 = vmc.energy(np.zeros(N), 0.0, samples, vmc_cfg.burn_in)
    res = vmc.optimize_spsa()
    runtime = time.perf_counter() - t0
    current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    e_exact = exact_tfim_energy(N, J, h, periodic=periodic, max_dim=8192)
    error = None if e_exact is None else float(res["final_energy"] - e_exact)
    return {
        "N": N, "seed": seed, "steps": steps, "samples_per_iter": samples,
        "runtime_s": float(runtime),
        "final_energy": float(res["final_energy"]),
        "init_energy": float(e_init),
        "exact_energy": None if e_exact is None else float(e_exact),
        "error": error,
        "avg_acceptance_rate": float(res["avg_acceptance_rate"]),
        "peak_mem_mb": float(peak/(1024*1024)),
        "theta_a_len": N,
        "theta_j": float(res["theta_j"]),
        "history": res["history"],
    }


# ------------********------------  Carbon I/O

#The def resolve_excel_path(excel_arg: str) -> str function resolves an input Excel file path 
#by first checking whether the provided string is an absolute, existing path.
#If not found, it next checks whether the file exists on the user’s Desktop and returns that path if available.
#If neither location contains the file, it simply returns the original path string as a fallback.

def resolve_excel_path(excel_arg: str) -> str:
    p = pathlib.Path(excel_arg)
    if p.is_absolute() and p.exists(): return str(p)
    desktop = pathlib.Path.home() / "Desktop" / excel_arg
    return str(desktop if desktop.exists() else p)


#The def load_carbon_excel function loads a CO₂-intensity Excel file and 
# auto-detects columns for country, year, and intensity using flexible name matching.
#It standardizes the output, selects the latest year per country if requested, 
# and heuristically converts g/kWh values to kg/kWh when magnitudes suggest unit mismatch.
#The final cleaned DataFrame is returned with consistent column names and only valid rows.
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

    # Heuristic unit fix (g → kg)

    med = float(df["Intensity"].dropna().median())
    if med > 50: df["Intensity"] = df["Intensity"]/1000.0
    return df.dropna(subset=["Country","Intensity"]).reset_index(drop=True)

def compute_carbon(perf_df: pd.DataFrame, intensity_df: pd.DataFrame,
                   power_watts: float, pue: float, combine: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_runtime_s = float(perf_df["runtime_s"].sum())
    kwh_total = power_watts * pue * total_runtime_s / 3_600_000.0
    df = intensity_df.copy()
    df["kWh"] = kwh_total
    df["kgCO2e"] = df["Intensity"] * df["kWh"]
    summary = pd.DataFrame({
        "total_runtime_s":[total_runtime_s],
        "power_watts":[power_watts], "PUE":[pue], "kWh_total":[kwh_total],
        "median_intensity_kg_per_kWh":[float(df["Intensity"].median())],
        "mean_kgCO2e_across_countries":[float(df["kgCO2e"].mean())],
    })
    return df.sort_values("kgCO2e", ascending=False), summary


# ------------********------------  plotting 

#The folllowing are for plotting purposes

plt_kwargs = dict(dpi=140, bbox_inches="tight")
def _boxplot_with_labels(data, labels):
    try: plt.boxplot(data, tick_labels=labels)
    except TypeError: plt.boxplot(data, labels=labels)

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

def plot_scalability(perf_df: pd.DataFrame, outdir: str):
    g = perf_df.groupby("N")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["N"], g["runtime_s"], marker="o")
    plt.xlabel("System size N"); plt.ylabel("Mean runtime (s)"); plt.title("Scalability: Runtime vs N")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_runtime_vs_N.png"), **plt_kwargs); plt.close()

    g2 = perf_df.groupby("N")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g2["N"], g2["peak_mem_mb"], marker="s")
    plt.xlabel("System size N"); plt.ylabel("Peak memory (MB)"); plt.title("Scalability: Peak memory vs N")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_N.png"), **plt_kwargs); plt.close()

    g3 = perf_df.copy(); g3["rt_per_iter"] = g3["runtime_s"] / g3["steps"]
    g3 = g3.groupby("N")["rt_per_iter"].mean().reset_index()
    plt.figure(); plt.plot(g3["N"], g3["rt_per_iter"], marker="^")
    plt.xlabel("System size N"); plt.ylabel("Runtime per iteration (s)"); plt.title("Scalability: Runtime/iter vs N")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_rt_per_iter_vs_N.png"), **plt_kwargs); plt.close()

    g4 = perf_df.groupby("N")["avg_acceptance_rate"].mean().reset_index()
    plt.figure(); plt.plot(g4["N"], g4["avg_acceptance_rate"], marker="d")
    plt.xlabel("System size N"); plt.ylabel("Mean acceptance rate"); plt.title("Scalability: Acceptance vs N")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_acceptance_vs_N.png"), **plt_kwargs); plt.close()

def plot_reliability(perf_df: pd.DataFrame, outdir: str):
    plt.figure()
    data = [perf_df[perf_df["N"]==N]["final_energy"].values for N in sorted(perf_df["N"].unique())]
    try: plt.boxplot(data, tick_labels=sorted(perf_df["N"].unique()))
    except TypeError: plt.boxplot(data, labels=sorted(perf_df["N"].unique()))
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


# ------------********------------  Main runner

#The following helpers in the Main runner block ensure output directories exist, provide a utility 
#  to save multiple DataFrames to one Excel file, and define the CLI for running classical VMC benchmarks.
#The argument parser configures system sizes, Monte-Carlo steps, sampling budget, model parameters, and boundary conditions.
#These inputs drive the benchmark orchestrator, 
#  enabling reproducible VMC performance, scalability, reliability, and carbon-impact studies.

def ensure_dir(d: str): os.makedirs(d, exist_ok=True)
def to_excel(dfs: Dict[str,pd.DataFrame], path: str):
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name,df in dfs.items():
            df.to_excel(w, index=False, sheet_name=name[:31])

def main():
    p = argparse.ArgumentParser(description="Classical VMC benchmarks: Performance, Scalability, Reliability, Carbon")
    p.add_argument("--sizes", nargs="+", type=int, default=[6,8,10,12])
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--samples-per-iter", type=int, default=3000)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=1.05)
    p.add_argument("--periodic", type=int, default=0)

# ------------********------------ speed/robustness

    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    p.add_argument("--fast", action="store_true")
    p.add_argument("--early-stop", action="store_true")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-3)

# ------------********------------ reliability threshold

    p.add_argument("--rel-tol", type=float, default=0.05, help="Relative error tolerance for success (default 5%)")
    p.add_argument("--abs-tol", type=float, default=1e-3, help="Absolute error floor")

# ------------********------------ carbon

    p.add_argument("--excel", type=str, default="Filtered CO2 intensity 236 Countries.xlsx")
    p.add_argument("--device-power-watts", type=float, default=65.0)
    p.add_argument("--pue", type=float, default=1.2)
    p.add_argument("--year-select", type=str, default="latest")
    p.add_argument("--outdir", type=str, default="carbon_by_country")
    p.add_argument("--combine", action="store_true")

    args = p.parse_args()
    if args.fast:
        if args.steps==150: args.steps = 60
        if args.samples_per_iter==3000: args.samples_per_iter = 800
        if args.seeds==5: args.seeds = 3
        args.early_stop = True
        if args.patience==10: args.patience = 6
        if args.min_delta==1e-3: args.min_delta = 2e-3

    periodic = bool(args.periodic)
    sizes = sorted(set(int(n) for n in args.sizes))

    perf_dir = os.path.join(os.getcwd(), "Performance")
    scal_dir = os.path.join(os.getcwd(), "Scalability")
    rel_dir  = os.path.join(os.getcwd(), "Reliability")
    carb_root = os.path.join(os.getcwd(), "Carbon footprints")
    carb_dir  = os.path.join(carb_root, args.outdir)
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]: ensure_dir(d)

# ------------********------------ parallel jobs
#This parallel jobs block constructs a list of VMC job configurations, 
#   then runs them—either in parallel using a process pool or sequentially—based on worker availability.
#Each completed result is collected, logged, and its energy history stored for later analysis and visualization.
#Finally, all experiment outputs are aggregated into a performance DataFrame for downstream reporting.

    jobs = [(N, 1000+13*N+s) for N in sizes for s in range(args.seeds)]
    all_rows = []; histories: Dict[Tuple[int,int], List[float]] = {}
    print("[run] Starting classical VMC experiments...")

    def _consume(row):
        all_rows.append(row)
        histories[(row["N"], row["seed"])] = row["history"]
        print(f"  - done N={row['N']}, seed={row['seed']} (runtime={row['runtime_s']:.2f}s, steps={row['steps']}, samples={row['samples_per_iter']})")

    if args.workers>1 and len(jobs)>1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(run_single, N, seed, args.steps, args.samples_per_iter, args.J, args.h, periodic,
                              args.early_stop, args.patience, args.min_delta) for (N,seed) in jobs]
            for f in as_completed(futs): _consume(f.result())
    else:
        for (N,seed) in jobs:
            _consume(run_single(N, seed, args.steps, args.samples_per_iter, args.J, args.h, periodic,
                                args.early_stop, args.patience, args.min_delta))

    perf_df = pd.DataFrame(all_rows)

    # ------------********------------ Performance ----------------

    perf_excel = os.path.join(perf_dir, "performance_results.xlsx")
    perf_agg = perf_df.groupby("N").agg(
        runtime_s=("runtime_s","mean"),
        final_energy=("final_energy","mean"),
        error=("error","mean"),
        avg_acceptance_rate=("avg_acceptance_rate","mean"),
    ).reset_index()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_excel, engine="openpyxl") as w:
            perf_df.drop(columns=["history"]).to_excel(w, index=False, sheet_name="raw_runs")
            perf_agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(perf_df, histories, perf_dir)

    # ------------********------------ Scalability ----------------

    scal_excel = os.path.join(scal_dir, "scalability_results.xlsx")
    scal_agg = perf_df.groupby("N").agg(
        runtime_s=("runtime_s","mean"),
        peak_mem_mb=("peak_mem_mb","mean"),
        steps=("steps","mean"),
        samples_per_iter=("samples_per_iter","mean"),
        avg_acceptance_rate=("avg_acceptance_rate","mean"),
    ).reset_index()
    scal_agg["rt_per_iter"] = scal_agg["runtime_s"]/scal_agg["steps"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_excel, engine="openpyxl") as w:
            perf_df[["N","seed","runtime_s","peak_mem_mb","steps","samples_per_iter","avg_acceptance_rate"]].to_excel(w, index=False, sheet_name="from_perf_runs")
            scal_agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(perf_df, scal_dir)

    # ------------********------------ Reliability ----------------

    rel_excel = os.path.join(rel_dir, "reliability_results.xlsx")
    rel_tol = float(args.rel_tol); abs_tol = float(args.abs_tol)
    def success(row):
        if not pd.isna(row["exact_energy"]):
            thr = max(abs_tol, rel_tol * abs(row["exact_energy"]))
            return abs(row["final_energy"] - row["exact_energy"]) <= thr
        else:
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

    # ------------********------------ Carbon (from PERFORMANCE runtimes only) ----------------

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

        # ------------********------------ plots
#This block generates carbon-impact visualizations by plotting the highest- and lowest-emission countries, 
#   a distribution histogram, and a CDF of CO₂ values.
#It saves all figures to the carbon-output directory 
#   and writes the full carbon table to Excel; if no valid input exists, it writes a placeholder note instead.
#These plots help compare grid cleanliness across countries 
#   and understand how carbon emissions vary for the experiment.

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

    # ------------********------------   wrap-up

    print("\n=== Summary (console) ===")
    print("Performance Excel:", os.path.join(perf_dir, "performance_results.xlsx"))
    print("Scalability Excel:", os.path.join(scal_dir, "scalability_results.xlsx"))
    print("Reliability Excel:", os.path.join(rel_dir, "reliability_results.xlsx"))
    print("Carbon Excel:", os.path.join(carb_dir, "carbon_results.xlsx"))
    print("Folders created:\n -", perf_dir, "\n -", scal_dir, "\n -", rel_dir, "\n -", carb_dir)

if __name__ == "__main__":
    main()