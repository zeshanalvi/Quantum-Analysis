
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# GPU-accelerated Empirical Evaluation: Quantum Pendulum Tomography (Noiseless) – Quantum Baseline

"""
GPU-accelerated Empirical Evaluation: Quantum Pendulum Tomography (Noiseless) – Quantum Baseline

This script mirrors the classical baseline's parameters, metrics, and evaluation strategy.
It produces four benchmarks:
  1) Performance     – runtime, memory, reconstruction accuracy
  2) Scalability     – runtime & memory vs problem size
  3) Reliability     – stability of results across seeds
  4) Carbon footprint– CO₂ based solely on Performance runtimes

Folders created in the current directory:
  Performance/, Scalability/, Reliability/, Carbon footprints/<outdir>/
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

import os, sys, time, tracemalloc, pathlib, warnings, math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# ------------********------------
# ------------********------------ GPU Setup and Dependencies
# ------------********------------
print("🔍 Checking GPU availability...")

HAS_GPU = False
cp = None

try:
    import cupy as cp
    # Test if GPU is available and working
    with cp.cuda.Device(0):
        # Simple test to verify GPU functionality
        test_array = cp.zeros(10, dtype=cp.float32)
        result = cp.sum(test_array)
        HAS_GPU = True
        print(f"✅ GPU detected: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        print(f"   Memory: {cp.cuda.Device().mem_info[1] / 1e9:.2f} GB")
        print(f"   Compute Capability: {cp.cuda.runtime.getDeviceProperties(0)['major']}.{cp.cuda.runtime.getDeviceProperties(0)['minor']}")
except ImportError:
    print("❌ CuPy not available, falling back to NumPy")
    import numpy as cp
except Exception as e:
    print(f"⚠️  GPU setup failed: {e}, falling back to NumPy")
    import numpy as cp

# If GPU failed, use NumPy
if not HAS_GPU:
    import numpy as cp
    print("🔄 Using NumPy (CPU) backend")

# Install required packages
def _ensure(pkgs: List[str]):
    import importlib, subprocess
    missing = []
    for p in pkgs:
        try:
            importlib.import_module(p)
        except ImportError:
            missing.append(p)
    if missing:
        print(f"[setup] Installing missing packages: {missing}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
        except Exception as e:
            print(f"[setup] Auto-install failed: {e}")

_ensure(["pandas", "matplotlib", "openpyxl", "psutil", "scikit-learn"])

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# GPU utility functions
def get_device_name() -> str:
    """Get current device name"""
    if HAS_GPU:
        try:
            return f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}"
        except:
            return "GPU: Unknown"
    return "CPU: NumPy"

def safe_gpu_cleanup():
    """Safely cleanup GPU memory"""
    if HAS_GPU:
        try:
            cp.get_default_memory_pool().free_all_blocks()
        except:
            pass

def to_numpy(x):
    """Safely convert CuPy array to NumPy array"""
    if HAS_GPU and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x

def to_cupy(x):
    """Safely convert NumPy array to CuPy array"""
    if HAS_GPU and isinstance(x, np.ndarray):
        return cp.asarray(x)
    return x

# Thread-safe GPU operations
_gpu_lock = threading.Lock()

def run_experiment_thread_safe(args):
    """
    Thread-safe GPU experiment runner.
    Uses a lock to prevent multiple threads from accessing GPU simultaneously.
    """
    grid_dim, steps, seed = args
    with _gpu_lock:  # Ensure only one thread uses GPU at a time
        return _run_single_experiment(grid_dim, steps, seed)

# ------------********------------
# ========================= Quantum Tomography Core =========================
# ------------********------------
#This helper generates S evenly spaced angles in the interval [0, π).
#These angles are typically used as quadrature measurement directions.

def make_angles(S: int) -> cp.ndarray:
    """Generate S angles in [0, π)"""
    return cp.linspace(0.0, cp.pi, S, endpoint=False)

#This function computes the mean and variance of a quadrature for a squeezed Gaussian state.
#It accounts for displacement, squeezing strength, and squeezing angle.
def quad_mean_var(theta: cp.ndarray, x0: float, p0: float, r: float, phi: float) -> Tuple[cp.ndarray, cp.ndarray]:
    """Compute quadrature mean and variance for given angles"""
    mu = x0 * cp.cos(theta) + p0 * cp.sin(theta)
    # squeezing variance along angle phi
    var = 0.5 * (cp.exp(-2*r) * cp.cos(theta - phi)**2 + cp.exp(2*r) * cp.sin(theta - phi)**2)
    return mu, var

#This function draws random quadrature samples from a Gaussian distribution.
#The distribution parameters are determined by the quadrature angle and squeezing properties.

def sample_quadrature(theta: cp.ndarray, shots: int, x0: float, p0: float, r: float, phi: float, rng) -> cp.ndarray:
    """Sample quadrature measurements for given angles"""
    mu, var = quad_mean_var(theta, x0, p0, r, phi)
    # Generate normal samples on GPU
    if HAS_GPU:
        samples = cp.random.normal(loc=mu, scale=cp.sqrt(var), size=(len(theta), shots))
    else:
        samples = rng.normal(loc=to_numpy(mu), scale=to_numpy(cp.sqrt(var)), size=(len(theta), shots))
    return samples

#This function computes the analytic Wigner function of a displaced squeezed Gaussian state.
#It evaluates the distribution on a phase-space grid and enforces proper normalization.

def wigner_truth(grid_x: cp.ndarray, grid_p: cp.ndarray, x0: float, p0: float, r: float, phi: float) -> cp.ndarray:
    """Compute analytic Wigner function for displaced squeezed state"""
    X, P = cp.meshgrid(grid_x, grid_p, indexing="xy")

    # rotate coordinates by phi to apply squeezing along x'
    c, s = cp.cos(phi), cp.sin(phi)
    Xp = c * X + s * P
    Pp = -s * X + c * P

    # squeezing parameters
    sigx2 = 0.5 * cp.exp(-2*r)
    sigp2 = 0.5 * cp.exp(2*r)

    # shift displacement (x0,p0) into rotated frame
    X0 = c * x0 + s * p0
    P0 = -s * x0 + c * p0

    # Gaussian Wigner function
    expo = -((Xp - X0)**2 / sigx2 + (Pp - P0)**2 / sigp2)
    W = (1.0 / cp.pi) * cp.exp(expo)

    # normalize
    dx = grid_x[1] - grid_x[0]
    dp = grid_p[1] - grid_p[0]
    sarea = cp.sum(W) * dx * dp
    if sarea > 0:
        W /= sarea

    return W

#This helper applies a ramp filter in the frequency domain to a 1D density.
#It is commonly used in filtered backprojection for tomographic reconstruction.

def ramp_filter_1d(dens: cp.ndarray, dx: float) -> cp.ndarray:
    """Apply ramp filter in frequency domain"""
    n = len(dens)
    F = cp.fft.rfft(dens)
    freqs = cp.fft.rfftfreq(n, d=dx)
    Ff = F * cp.abs(freqs)
    back = cp.fft.irfft(Ff, n=n)
    return back

#This function reconstructs a 2D phase-space distribution using filtered backprojection.
#It interpolates projected densities along rotated axes and normalizes the result.

def backproject(angles: cp.ndarray, proj_dens_list: List[cp.ndarray],
                t_centers: cp.ndarray, grid_x: cp.ndarray, grid_p: cp.ndarray,
                use_filter: bool = True) -> cp.ndarray:
    """Perform filtered backprojection reconstruction"""
    Gx, Gp = len(grid_x), len(grid_p)
    X, P = cp.meshgrid(grid_x, grid_p, indexing="xy")
    recon = cp.zeros((Gp, Gx), dtype=cp.float64)
    dt = t_centers[1] - t_centers[0]
    dtheta = (cp.pi / len(angles)) if len(angles) > 0 else 0.0

    for th, dens in zip(angles, proj_dens_list):
        # Apply ramp filter
        d = ramp_filter_1d(dens, dt) if use_filter else dens

        # Project coordinates
        T = X * cp.cos(th) + P * cp.sin(th)

        # Interpolate - convert to numpy for interpolation then back to cupy
        T_flat = to_numpy(T.ravel())
        d_np = to_numpy(d)
        t_centers_np = to_numpy(t_centers)

        vals_np = np.interp(T_flat, t_centers_np, d_np, left=0.0, right=0.0)
        vals = to_cupy(vals_np).reshape(T.shape)

        recon += vals * dtheta

    # normalize
    dx = grid_x[1] - grid_x[0]
    dp = grid_p[1] - grid_p[0]
    s = cp.sum(recon) * dx * dp
    if s > 0:
        recon /= s

    return recon

#This routine runs a complete tomography experiment for a single parameter setting.
#It samples quadratures, reconstructs the Wigner function, and compares it to the analytic truth.
#Error metrics such as MSE and L1 distance are returned along with reconstruction data.

def _run_single_experiment(grid_dim: int, steps: int, seed: int) -> dict:
    """
    Core experiment logic - run single Quantum Tomography experiment.
    """
    # Fixed parameters
    shots_per_angle = 1500
    x0, p0 = 0.8, -0.5  # displacement
    r, phi = 0.0, 0.0   # squeezing parameters (coherent state)
    lim = 3.0           # grid extends to ±lim in x and p

    # Set random seed
    np.random.seed(seed)
    if HAS_GPU:
        cp.random.seed(seed)

    tracemalloc.start()
    t0 = time.perf_counter()

    try:
        # Generate angles
        angles = make_angles(steps)

        # Choose t support for projections
        tmax = lim * cp.max(cp.abs(cp.cos(angles))) + lim * cp.max(cp.abs(cp.sin(angles)))
        B = max(256, 2 * grid_dim)
        t_edges = cp.linspace(-tmax, tmax, B + 1)
        t_centers = 0.5 * (t_edges[:-1] + t_edges[1:])

        # Sample quadrature measurements
        proj_dens_list = []
        for i, th in enumerate(angles):
            # Sample for this angle
            samples = sample_quadrature(cp.array([th]), shots_per_angle, x0, p0, r, phi,
                                      np.random.default_rng(seed + i))

            # Compute histogram
            hist, _ = cp.histogram(samples, bins=t_edges)
            dens = hist / (shots_per_angle * (t_edges[1] - t_edges[0]))
            proj_dens_list.append(dens)

        # Create reconstruction grid
        gx = cp.linspace(-lim, lim, grid_dim)
        gp = cp.linspace(-lim, lim, grid_dim)

        # Perform backprojection
        recon = backproject(angles, proj_dens_list, t_centers, gx, gp, use_filter=True)

        # Compute ground truth
        truth = wigner_truth(gx, gp, x0, p0, r, phi)

        # Compute error metrics
        mse = float(cp.mean((recon - truth)**2))
        dx = gx[1] - gx[0]
        dp = gp[1] - gp[0]
        l1 = float(cp.sum(cp.abs(recon - truth)) * dx * dp)

        # Additional metrics
        fidelity = float(cp.sum(cp.sqrt(recon * truth)) * dx * dp)**2
        success = True

    except Exception as e:
        print(f"❌ Experiment failed for grid_dim={grid_dim}, steps={steps}, seed={seed}: {e}")
        mse = float('inf')
        l1 = float('inf')
        fidelity = 0.0
        success = False
        recon = truth = cp.zeros((grid_dim, grid_dim))

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Cleanup GPU memory
    safe_gpu_cleanup()

    return {
        "grid_dim": int(grid_dim),
        "steps": int(steps),
        "seed": int(seed),
        "shots_per_angle": int(shots_per_angle),
        "mse": float(mse),
        "l1_distance": float(l1),
        "fidelity": float(fidelity),
        "success": bool(success),
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024**2)),
        "device": get_device_name()
    }
# ------------********------------
# ------------********------------ Carbon I/O ==============================
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
    cand_country = next((v for k, v in cols.items() if "country" in k or "nation" in k or k == "location"), None)
    if cand_country is None:
        raise ValueError("No 'Country' column found.")
    cand_year = next((v for k, v in cols.items() if "year" in k or "date" in k), None)
    cand_intensity = next((v for k, v in cols.items()
                           if "intensity" in k or ("co2" in k and "kwh" in k) or "kgco2" in k or "gco2" in k), None)
    if cand_intensity is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if cand_year in numeric_cols:
            numeric_cols.remove(cand_year)
        if not numeric_cols:
            raise ValueError("No numeric intensity column detected.")
        cand_intensity = numeric_cols[0]
    keep = [cand_country] + ([cand_year] if cand_year else []) + [cand_intensity]
    df = df[keep].copy()
    df.columns = ["Country", "Year", "Intensity"] if len(keep) == 3 else ["Country", "Intensity"]
    if "Year" in df.columns and year_select.lower() == "latest":
        df = df.sort_values(["Country", "Year"]).groupby("Country", as_index=False).tail(1)

    # Heuristic unit fix: if median looks like g/kWh, convert to kg/kWh
    med = float(df["Intensity"].dropna().median())
    if med > 50:
        df["Intensity"] = df["Intensity"] / 1000.0

    return df.dropna(subset=["Country", "Intensity"]).reset_index(drop=True)

def compute_carbon(perf_df: pd.DataFrame, intensity_df: pd.DataFrame,
                   power_watts: float, pue: float, combine: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_runtime_s = float(perf_df["runtime_s"].sum())
    kWh_total = power_watts * pue * total_runtime_s / 3_600_000.0
    df = intensity_df.copy()
    df["kWh"] = kWh_total
    df["kgCO2e"] = df["Intensity"] * df["kWh"]

    # ALWAYS return country-wise results (ignore combine parameter)
    carbon_output = df.sort_values("kgCO2e", ascending=False)

    summary = pd.DataFrame({
        "total_runtime_s": [total_runtime_s],
        "power_watts": [power_watts],
        "PUE": [pue],
        "kWh_total": [kWh_total],
        "median_intensity_kg_per_kWh": [float(df["Intensity"].median())],
        "mean_kgCO2e_across_countries": [float(df["kgCO2e"].mean())],
        "device": [perf_df["device"].iloc[0] if "device" in perf_df.columns else "Unknown"],
        "combine_mode": [combine],
        "countries_analyzed": [len(df)]
    })
    return carbon_output, summary
# ------------********------------
## ------------********------------ Plot Helpers ==============================
# ------------********------------
plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    # Matplotlib 3.9 renamed 'labels' -> 'tick_labels'
    try:
        plt.boxplot(data, tick_labels=labels)
    except TypeError:
        plt.boxplot(data, labels=labels)

def plot_performance(df: pd.DataFrame, outdir: str):
    # Runtime vs steps for different grid dimensions
    plt.figure(figsize=(10, 6))
    for grid_dim in sorted(df["grid_dim"].unique()):
        subset = df[df["grid_dim"] == grid_dim]
        avg_runtime = subset.groupby("steps")["runtime_s"].mean()
        plt.plot(avg_runtime.index, avg_runtime.values, marker="o", label=f"Grid={grid_dim}")

    plt.xscale("log", base=2)
    plt.xlabel("Number of Angles")
    plt.ylabel("Runtime (s)")
    plt.title(f"Performance: Runtime vs Angles\n{get_device_name()}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "perf_runtime_vs_angles.png"), **plt_kwargs)
    plt.close()

    # Memory usage
    plt.figure(figsize=(10, 6))
    for grid_dim in sorted(df["grid_dim"].unique()):
        subset = df[df["grid_dim"] == grid_dim]
        avg_memory = subset.groupby("steps")["peak_mem_mb"].mean()
        plt.plot(avg_memory.index, avg_memory.values, marker="s", label=f"Grid={grid_dim}")

    plt.xscale("log", base=2)
    plt.xlabel("Number of Angles")
    plt.ylabel("Memory Usage (MB)")
    plt.title(f"Performance: Memory vs Angles\n{get_device_name()}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "perf_memory_vs_angles.png"), **plt_kwargs)
    plt.close()

# ------------********------------

def plot_scalability(df: pd.DataFrame, outdir: str):
    # Runtime vs grid dimension (log-log)
    max_steps = df["steps"].max()
    subset = df[df["steps"] == max_steps]
    scalability_data = subset.groupby("grid_dim").agg({
        "runtime_s": "mean"
    }).reset_index()

    plt.figure(figsize=(8, 6))
    plt.loglog(scalability_data["grid_dim"], scalability_data["runtime_s"],
               marker="o", linewidth=2, markersize=8)
    plt.xlabel("Grid Dimension")
    plt.ylabel("Runtime (s)")
    plt.title(f"Scalability: Runtime vs Grid Size (log-log)\n{get_device_name()}")
    plt.grid(True, which="both")
    plt.savefig(os.path.join(outdir, "scal_runtime_vs_grid.png"), **plt_kwargs)
    plt.close()

    # Memory vs grid dimension
    plt.figure(figsize=(8, 6))
    memory_data = subset.groupby("grid_dim").agg({
        "peak_mem_mb": "mean"
    }).reset_index()

    plt.semilogy(memory_data["grid_dim"], memory_data["peak_mem_mb"],
                marker="^", linewidth=2, markersize=8, color='green')
    plt.xlabel("Grid Dimension")
    plt.ylabel("Memory Usage (MB)")
    plt.title(f"Scalability: Memory vs Grid Size\n{get_device_name()}")
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "scal_memory_vs_grid.png"), **plt_kwargs)
    plt.close()

# ------------********------------

def plot_reliability(df: pd.DataFrame, outdir: str):
    # Only plot successful experiments
    success_df = df[df["success"]]
    if len(success_df) == 0:
        print("⚠️ No successful experiments to plot for reliability")
        # Create a placeholder plot
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, "No successful experiments\nAll runs failed",
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
        plt.title("Reliability: No Successful Experiments")
        plt.savefig(os.path.join(outdir, "rel_error_distribution.png"), **plt_kwargs)
        plt.close()
        return

    # Error distribution by grid dimension
    data = [success_df[success_df["grid_dim"] == n]["mse"].values for n in sorted(success_df["grid_dim"].unique())]
    plt.figure(figsize=(10, 6))
    _boxplot_with_labels(data, labels=[f"Grid={n}" for n in sorted(success_df["grid_dim"].unique())])
    plt.yscale("log")
    plt.xlabel("Grid Dimension")
    plt.ylabel("MSE (log scale)")
    plt.title(f"Reliability: Error Distribution by Grid Size\n{get_device_name()}")
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "rel_error_distribution.png"), **plt_kwargs)
    plt.close()

    # Success rate
    success_rates = df.groupby("grid_dim")["success"].mean().reset_index()
    plt.figure(figsize=(8, 6))
    plt.plot(success_rates["grid_dim"], success_rates["success"],
             marker="o", linewidth=2, markersize=8)
    plt.xlabel("Grid Dimension")
    plt.ylabel("Success Rate")
    plt.title(f"Reliability: Success Rate vs Grid Size\n{get_device_name()}")
    plt.grid(True)
    plt.savefig(os.path.join(outdir, "rel_success_rate.png"), **plt_kwargs)
    plt.close()

    # MSE vs angles heatmap (only successful runs)
    if len(success_df) > 0:
        pivot_data = success_df.pivot_table(values="mse", index="grid_dim", columns="steps", aggfunc="mean")
        plt.figure(figsize=(10, 6))
        plt.imshow(np.log10(pivot_data.values + 1e-16), aspect="auto", cmap="viridis", origin="lower")
        plt.colorbar(label="log10(MSE)")
        plt.xticks(range(len(pivot_data.columns)), pivot_data.columns)
        plt.yticks(range(len(pivot_data.index)), pivot_data.index)
        plt.xlabel("Number of Angles")
        plt.ylabel("Grid Dimension")
        plt.title(f"Reliability: MSE Heatmap\n{get_device_name()}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "rel_mse_heatmap.png"), **plt_kwargs)
        plt.close()

# ------------********------------

def plot_carbon(df: pd.DataFrame, combine: bool, outdir: str):
    """Generate carbon footprint plots - ALWAYS country-wise"""
    # Top 15 highest emissions
    top = df.nlargest(15, "kgCO2e")
    plt.figure(figsize=(12, 8))
    plt.barh(top["Country"][::-1], top["kgCO2e"][::-1], color='red', alpha=0.7)
    plt.xlabel("kg CO2e (higher = dirtier grid)", fontsize=12)
    plt.title(f"Carbon: Top 15 Highest Emissions\n{get_device_name()}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "carbon_top15.png"), **plt_kwargs)
    plt.close()

    # Bottom 15 lowest emissions
    bot = df.nsmallest(15, "kgCO2e")
    plt.figure(figsize=(12, 8))
    plt.barh(bot["Country"][::-1], bot["kgCO2e"][::-1], color='green', alpha=0.7)
    plt.xlabel("kg CO2e (lower = cleaner grid)", fontsize=12)
    plt.title(f"Carbon: Bottom 15 Lowest Emissions\n{get_device_name()}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "carbon_bottom15.png"), **plt_kwargs)
    plt.close()

    # Distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df["kgCO2e"], bins=30, alpha=0.7, edgecolor='black', color='orange')
    plt.xlabel("kg CO2e for this experiment", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.title(f"Carbon: Emission Distribution\n{get_device_name()}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "carbon_distribution.png"), **plt_kwargs)
    plt.close()

    # CDF plot
    xs = np.sort(df["kgCO2e"].values)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    plt.figure(figsize=(10, 6))
    plt.plot(xs, ys, linewidth=2, color='purple')
    plt.xlabel("kg CO2e", fontsize=12)
    plt.ylabel("CDF", fontsize=12)
    plt.title(f"Carbon: Cumulative Distribution\n{get_device_name()}", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "carbon_cdf.png"), **plt_kwargs)
    plt.close()

# ------------********------------
# ------------********------------ Utility Functions ==============================
# ------------********------------

def ensure_dir(d: str):
    """Ensure directory exists"""
    os.makedirs(d, exist_ok=True)

def to_excel(dfs: Dict[str, pd.DataFrame], path: str):
    """Save multiple DataFrames to Excel with different sheets"""
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        for name, df in dfs.items():
            df.to_excel(w, index=False, sheet_name=name[:31])

# ------------********------------
# ------------********------------Main Experiment Runner ==============================
# ------------********------------

def run_quantum_tomography_experiment():
    """
    Main Quantum Tomography experiment runner with hardcoded parameters for Kaggle.
    Equivalent to: --grid 64 --steps 8,16,32 --repeats 2 --shots-per-angle 1500
                   --device-power-watts 65 --pue 1.2 --combine
    """
    # Experiment parameters
    grid_dim = 64
    steps_list = [8, 16, 32]
    trials = 2   # trials per steps configuration
    workers = 4   # Number of parallel threads

    # Carbon parameters
    device_power_watts = 65.0
    pue = 1.2
    combine = True

    print(f"🚀 Starting Quantum Tomography experiments on {get_device_name()}")
    print(f"🔧 Configuration:")
    print(f"   Grid dimension: {grid_dim}×{grid_dim}")
    print(f"   Angles (steps): {steps_list}")
    print(f"   Trials per configuration: {trials}")
    print(f"   Shots per angle: 1500")
    print(f"   State: coherent (x0=0.8, p0=-0.5)")
    print(f"   Workers: {workers}")
    print(f"   Device power: {device_power_watts}W")
    print(f"   PUE: {pue}")
    print(f"   Combine carbon: {combine}")

    # Create output folders
    perf_dir = os.path.join(os.getcwd(), "Performance")
    scal_dir = os.path.join(os.getcwd(), "Scalability")
    rel_dir  = os.path.join(os.getcwd(), "Reliability")
    carb_root = os.path.join(os.getcwd(), "Carbon footprints")
    carb_dir  = os.path.join(carb_root, "carbon_by_country")

    for d in [perf_dir, scal_dir, rel_dir, carb_dir]:
        ensure_dir(d)

    # Prepare jobs
    jobs = []
    for steps in steps_list:
        for i in range(trials):
            seed = 1000 + 17 * grid_dim + 31 * steps + i
            jobs.append((grid_dim, steps, seed))

    all_rows = []

    def _consume(row):
        """Process completed job results"""
        all_rows.append(row)
        status = "✅" if row["success"] else "❌"
        print(f"  {status} grid={row['grid_dim']}, angles={row['steps']}, seed={row['seed']} "
              f"(runtime={row['runtime_s']:.6f}s, mse={row['mse']:.2e})")

    # Execute jobs with thread-based parallelization
    print(f"🔀 Running {len(jobs)} jobs in parallel with {workers} threads...")

    if workers > 1 and len(jobs) > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all jobs
            future_to_job = {executor.submit(run_experiment_thread_safe, job): job for job in jobs}

            # Collect results as they complete
            for future in as_completed(future_to_job):
                try:
                    result = future.result()
                    _consume(result)
                except Exception as e:
                    job = future_to_job[future]
                    print(f"❌ Job failed grid_dim={job[0]}, steps={job[1]}, seed={job[2]}: {e}")
    else:
        # Fallback to sequential execution
        for job in jobs:
            result = run_experiment_thread_safe(job)
            _consume(result)

    # Check if we have any successful results
    if not all_rows:
        print("❌ All jobs failed! Running sequentially as fallback...")
        for job in jobs:
            try:
                result = _run_single_experiment(job[0], job[1], job[2])
                _consume(result)
            except Exception as e:
                print(f"❌ Sequential fallback also failed for grid_dim={job[0]}, steps={job[1]}, seed={job[2]}: {e}")

    if not all_rows:
        print("💥 CRITICAL: No experiments completed successfully!")
        return None

    perf_df = pd.DataFrame(all_rows)

    # ---------------- Performance Results ----------------
    print("📊 Generating Performance results...")
    perf_excel = os.path.join(perf_dir, "performance_results.xlsx")

    # Only aggregate successful runs
    success_df = perf_df[perf_df["success"]]
    if len(success_df) > 0:
        perf_agg = success_df.groupby(["grid_dim", "steps"]).agg({
            "runtime_s": ["mean", "std"],
            "peak_mem_mb": ["mean", "std"],
            "mse": ["mean", "std"],
            "fidelity": ["mean", "std"]
        }).reset_index()
        # Flatten column names
        perf_agg.columns = ['_'.join(col).strip('_') for col in perf_agg.columns.values]
    else:
        perf_agg = pd.DataFrame({"note": ["No successful experiments"]})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        to_excel({
            "raw_runs": perf_df,
            "aggregated": perf_agg
        }, perf_excel)

    plot_performance(perf_df, perf_dir)

    # In the run_quantum_tomography_experiment() function, replace the scalability section with this:

# ---------------- Scalability Results ----------------
    print("📈 Generating Scalability results...")
    scal_excel = os.path.join(scal_dir, "scalability_results.xlsx")

    # Compute scaling coefficients only for successful runs
    success_df = perf_df[perf_df["success"]]
    if len(success_df) > 0:
        max_steps = max(steps_list)
        scaling_subset = success_df[success_df["steps"] == max_steps]

        # Only compute scaling if we have enough data points
        if len(scaling_subset) >= 2:  # Need at least 2 points for linear fit
            log_steps = np.log(scaling_subset["steps"].values)
            log_time = np.log(scaling_subset["runtime_s"].values + 1e-12)  # Avoid log(0)

            # Check if we have enough variation in the data
            if np.std(log_steps) > 1e-6 and np.std(log_time) > 1e-6:
                try:
                    scaling_coeff = np.polyfit(log_steps, log_time, 1)[0]
                    scaling_summary = pd.DataFrame({
                        "parameter": ["scaling_exponent"],
                        "value": [scaling_coeff],
                        "description": ["Exponent in time ~ steps^exponent"]
                    })
                except np.RankWarning:
                    scaling_summary = pd.DataFrame({
                        "parameter": ["scaling_exponent"],
                        "value": [np.nan],
                        "description": ["Poorly conditioned data for scaling analysis"]
                    })
            else:
                scaling_summary = pd.DataFrame({
                    "parameter": ["scaling_exponent"],
                    "value": [np.nan],
                    "description": ["Insufficient data variation for scaling analysis"]
                })
        else:
            scaling_summary = pd.DataFrame({
                "parameter": ["scaling_exponent"],
                "value": [np.nan],
                "description": ["Insufficient data points for scaling analysis"]
            })
    else:
        scaling_summary = pd.DataFrame({
            "parameter": ["scaling_exponent"],
            "value": [np.nan],
            "description": ["No successful experiments for scaling analysis"]
        })

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        to_excel({
            "raw_data": perf_df[["grid_dim", "steps", "runtime_s", "peak_mem_mb", "success"]],
            "scaling_analysis": scaling_summary
        }, scal_excel)

    plot_scalability(perf_df, scal_dir)

    # ---------------- Reliability Results ----------------
    print("🎯 Generating Reliability results...")
    rel_excel = os.path.join(rel_dir, "reliability_results.xlsx")

    reliability_analysis = perf_df.groupby("steps").agg({
        "mse": ["mean", "std", "min", "max"],
        "fidelity": ["mean", "std"],
        "success": "mean",
        "runtime_s": ["mean", "std"]
    }).reset_index()
    reliability_analysis.columns = ['_'.join(col).strip('_') for col in reliability_analysis.columns.values]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        to_excel({
            "error_analysis": reliability_analysis,
            "raw_results": perf_df[["grid_dim", "steps", "seed", "mse", "l1_distance", "fidelity", "success"]]
        }, rel_excel)

    plot_reliability(perf_df, rel_dir)

    # ---------------- Carbon Footprint Results ----------------
    print("🌍 Generating Carbon footprint results...")

    # Load carbon data from Kaggle input directory
    excel_path = "/kaggle/input/masters/Filtered CO2 intensity 236 Countries.xlsx"
    intensity_df = None

    try:
        # Try to load from the exact path
        intensity_df = pd.read_excel(excel_path)
        print(f"✅ Loaded carbon data from: {excel_path}")
        print(f"   Columns found: {list(intensity_df.columns)}")
        print(f"   Countries loaded: {len(intensity_df)}")

        # Auto-detect and process columns
        cols = {c.lower(): c for c in intensity_df.columns}
        cand_country = next((v for k,v in cols.items() if "country" in k or "nation" in k or k=="location"), None)
        if cand_country is None:
            raise ValueError("No 'Country' column found.")

        cand_intensity = next((v for k,v in cols.items() if "intensity" in k or ("co2" in k and ("kwh" in k or "/kwh" in k)) or "kgco2" in k or "gco2" in k), None)

        if cand_intensity is None:
            numeric_cols = [c for c in intensity_df.columns if pd.api.types.is_numeric_dtype(intensity_df[c])]
            if not numeric_cols:
                raise ValueError("No numeric intensity column detected.")
            cand_intensity = numeric_cols[0]

        # Keep only necessary columns
        intensity_df = intensity_df[[cand_country, cand_intensity]].copy()
        intensity_df.columns = ["Country", "Intensity"]

        # Convert gCO2/kWh to kgCO2/kWh if needed
        med = float(intensity_df["Intensity"].dropna().median())
        if med > 50:
            intensity_df["Intensity"] = intensity_df["Intensity"] / 1000.0

        intensity_df = intensity_df.dropna(subset=["Country", "Intensity"]).reset_index(drop=True)
        print(f"✅ Processed carbon data for {len(intensity_df)} countries")

    except Exception as e:
        print(f"❌ ERROR loading carbon data: {e}")
        print("💡 Using sample carbon data as fallback...")
        # Create sample data similar to other codes
        countries = ["France", "Sweden", "Norway", "Switzerland", "Austria", "Germany", "United States", "China", "India"]
        intensities = [52, 41, 32, 45, 62, 385, 389, 537, 708]
        intensity_df = pd.DataFrame({
            "Country": countries,
            "Intensity": [i/1000.0 for i in intensities]  # Convert to kg/kWh
        })

    ensure_dir(carb_dir)
    carbon_df, summary_df = compute_carbon(perf_df, intensity_df,
                                           power_watts=device_power_watts,
                                           pue=pue, combine=combine)
    carb_excel = os.path.join(carb_dir, "carbon_results.xlsx")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        to_excel({
            "carbon_results": carbon_df,
            "summary": summary_df,
            "intensity_input": intensity_df,
            "performance_reference": perf_df[["grid_dim", "steps", "seed", "runtime_s"]]
        }, carb_excel)

    # Generate carbon plots
    plot_carbon(carbon_df, combine, carb_dir)
    print(f"✅ Carbon analysis complete: {len(carbon_df)} entries")

    # Final summary
    success_count = perf_df["success"].sum()
    total_count = len(perf_df)
    success_rate = success_count / total_count if total_count > 0 else 0

    print("\n" + "="*60)
    print("🎉 QUANTUM TOMOGRAPHY EXPERIMENT COMPLETE - RESULTS SUMMARY")
    print("="*60)
    print(f"📁 Performance:     {perf_excel}")
    print(f"📁 Scalability:     {scal_excel}")
    print(f"📁 Reliability:     {rel_excel}")
    print(f"📁 Carbon:          {carb_excel}")
    print(f"🖥️  Device:          {get_device_name()}")
    print(f"📊 Total runs:      {total_count}")
    print(f"✅ Successful:      {success_count}")
    print(f"❌ Failed:          {total_count - success_count}")
    print(f"⚡ Total runtime:   {perf_df['runtime_s'].sum():.2f}s")
    print(f"🔬 Grid dimension:  {grid_dim}×{grid_dim}")
    print(f"📐 Angles tested:   {sorted(perf_df['steps'].unique())}")
    print(f"🎯 Success rate:    {success_rate * 100:.1f}%")

    if success_count > 0:
        success_df = perf_df[perf_df["success"]]
        print(f"📏 Mean MSE:        {success_df['mse'].mean():.2e}")
        print(f"📐 Mean fidelity:   {success_df['fidelity'].mean():.3f}")
    else:
        print(f"📏 Mean MSE:        N/A (no successful runs)")
        print(f"📐 Mean fidelity:   N/A (no successful runs)")

    print(f"\n📂 All results saved to:")
    for d in [perf_dir, scal_dir, rel_dir, carb_dir]:
        print(f"   • {d}")

    return perf_df

# ------------********------------
# ------------********------------ Main execution for Kaggle
# ------------********------------
if __name__ == "__main__":
    # Clean up GPU state
    safe_gpu_cleanup()

    print("🚀 Starting Quantum Tomography Benchmark on Kaggle")
    print("=" * 50)

    # Run the complete experiment
    results_df = run_quantum_tomography_experiment()

    if results_df is not None:
        # Display final results summary
        success_count = results_df["success"].sum()
        total_count = len(results_df)

        print("\n📊 FINAL RESULTS SUMMARY")
        print("=" * 30)
        print(f"Total experiments: {total_count}")
        print(f"Successful experiments: {success_count}")
        print(f"Failed experiments: {total_count - success_count}")
        print(f"Grid dimension: {results_df['grid_dim'].iloc[0]}×{results_df['grid_dim'].iloc[0]}")
        print(f"Angles tested: {sorted(results_df['steps'].unique())}")
        print(f"Average runtime: {results_df['runtime_s'].mean():.6f}s")
        print(f"Success rate: {success_count / total_count * 100:.1f}%")

        if success_count > 0:
            success_df = results_df[results_df["success"]]
            print(f"Average MSE: {success_df['mse'].mean():.2e}")
            print(f"Average fidelity: {success_df['fidelity'].mean():.3f}")
        else:
            print(f"Average MSE: N/A")
            print(f"Average fidelity: N/A")

        print(f"Device used: {get_device_name()}")

        # Show folder structure
        print("\n📁 GENERATED OUTPUT STRUCTURE:")
        base_dir = '/kaggle/working'
        for root, dirs, files in os.walk(base_dir):
            # Only show our output directories
            if any(x in root for x in ['Performance', 'Scalability', 'Reliability', 'Carbon footprints']):
                level = root.replace(base_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f'{indent}📁 {os.path.basename(root)}/')
                sub_indent = ' ' * 2 * (level + 1)
                for file in files:
                    if file.endswith(('.xlsx', '.png')):
                        file_path = os.path.join(root, file)
                        size = os.path.getsize(file_path) / 1024  # KB
                        icon = "📊" if file.endswith('.xlsx') else "🖼️"
                        print(f'{sub_indent}{icon} {file} ({size:.1f} KB)')

        print("\n✅ All Quantum Tomography deliverables generated successfully!")
        print("   • Performance/performance_results.xlsx + perf_*.png")
        print("   • Scalability/scalability_results.xlsx + scal_*.png")
        print("   • Reliability/reliability_results.xlsx + rel_*.png")
        print("   • Carbon footprints/carbon_by_country/carbon_results.xlsx + carbon_*.png")
    else:
        print("❌ Quantum Tomography benchmark failed - no results generated")

