#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# Classical Hamiltonian Monte Carlo (HMC) — Benchmarks (with dual-averaging ε adaptation)

#Target:
#  Correlated Gaussian N(0, Σ) in dimension d, with condition number κ.
#  Exact mean/cov known → reliability metrics (ESS/steps, mean-error norm).

# Per (d, seed) batch:
#  - Warmup (optionally adapts ε toward --target-accept via dual averaging),
#    then collect --steps HMC samples with fixed (L, ε).
#  - Mass matrix options: identity, diag(Σ), or full (Σ).
#  - Metrics: runtime, peak_mem_mb, accept rate, grad evals, |ΔH|,
#            ESS/steps (mean over dims), mean-error norm, tuned ε, success flag.
#  - Folders: Performance/ Scalability/ Reliability/ Carbon footprints/<outdir>/

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

#These imports provide system utilities for timing, memory profiling, argument parsing, and filesystem handling.
#NumPy and pandas support numerical computation and tabular data analysis.
#Matplotlib is configured for non-interactive plotting in batch environments.
#Concurrent futures enable parallel execution and efficient result collection.

import os, time, math, argparse, warnings, pathlib, tracemalloc
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# ------------********------------
# ----------------------------- Linear algebra utils -----------------------------
# ------------********------------

#This function constructs a symmetric positive-definite covariance matrix with a controlled condition number.
#Eigenvalues are generated with a geometric progression and combined with a random orthogonal basis.
#The matrix is symmetrized for numerical stability and its inverse is computed for later use.
#Both the covariance, its inverse, and the orthogonal basis are returned for downstream algorithms.

def make_covariance(d, kappa, rng):
    """
    Build Σ = R^T diag(λ) R with geometric spectrum spanning condition number κ.
    Returns (Sigma, Sigma_inv, R).
    """
    if kappa < 1.0:
        kappa = 1.0
    lam_min, lam_max = 1.0, float(kappa)
    if d == 1:
        lambdas = np.array([lam_max], dtype=np.float64)
    else:
        lambdas = lam_min * (lam_max / lam_min) ** (np.linspace(0, 1, d))
    A = rng.normal(size=(d, d))
    Q, _ = np.linalg.qr(A)
    R = Q.astype(np.float64)
    Sigma = (R.T * lambdas) @ R
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sigma_inv = np.linalg.inv(Sigma)
    return Sigma, Sigma_inv, R

# ------------********------------
# ----------------------------- HMC core ----------------------------------------
# ------------********------------
#This function implements a velocity-Verlet (leapfrog) integrator for Hamiltonian dynamics with a matrix mass inverse.
#It alternates momentum updates (kicks) and position updates (drifts) to simulate one proposal trajectory.
#Gradient evaluations are carefully counted, with half-steps at the beginning and end for symmetry.
#The updated position and momentum are returned for use in sampling or acceptance tests.

def leapfrog(q, p, eps, L, gradU, Minv):
    """
    Velocity Verlet / leapfrog with *matrix* Minv.
    Counting rule: (L + 1) gradient evaluations per proposal.
    """
    qn = q.copy()
    pn = p.copy()

    # half kick
    g = gradU(qn)                  # 1st grad
    pn -= 0.5 * eps * g

    for i in range(1, L + 1):
        # full drift
        qn += eps * (Minv @ pn)
        # full kicks except after last drift
        if i != L:
            g = gradU(qn)          # middle grads
            pn -= eps * g

    # final half kick
    g = gradU(qn)                  # (L+1)-th grad
    pn -= 0.5 * eps * g

    return qn, pn


#This function evaluates the Hamiltonian energy of a system state.
#It combines the potential energy term with the kinetic energy defined by the inverse mass matrix.

def hamiltonian(q, p, U, Minv):
    # H = U(q) + 0.5 * p^T M^{-1} p
    return U(q) + 0.5 * float(p @ (Minv @ p))

# ------------********------------
# ----------------------------- ESS via FFT (per-dim) ---------------------------
# ------------********------------

#These helper functions support efficient autocovariance computation for time series analysis.
#The power-of-two utility selects an FFT-friendly length to improve computational performance.


def _next_pow2(n: int) -> int:
    return 1 << (int(np.ceil(np.log2(max(2, n)))))


#Autocovariance is computed via the Fourier domain to reduce complexity for long sequences.
#Edge cases for very short inputs are handled to ensure numerical stability.
def _autocov_fft(x: np.ndarray) -> np.ndarray:
    n = len(x)
    if n <= 1:
        return np.array([0.0], dtype=np.float64)
    nfft = _next_pow2(2 * n)
    fx = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(fx * np.conjugate(fx), n=nfft)[:n]
    ac = ac / max(1, n)
    return ac.real

def ess_per_dim(samples: np.ndarray) -> np.ndarray:
    #samples: shape (T, d)
    #Geyer's initial positive sequence on autocorrelations.
    #Returns ESS for each dimension.
    
    T, d = samples.shape
    if T < 3:
        return np.ones(d)
    ess = np.empty(d, dtype=np.float64)
    for j in range(d):
        x = samples[:, j] - np.mean(samples[:, j])
        acv = _autocov_fft(x)
        v0 = acv[0]
        if v0 <= 0:
            ess[j] = 1.0
            continue
        rho = acv / v0
        rho[0] = 1.0
        # Initial positive sequence on gamma_k = rho_{2k}+rho_{2k+1}
        s = 0.0
        k = 1
        while k + 1 < len(rho):
            gamma = rho[k] + rho[k + 1]
            if gamma <= 0:
                break
            s += gamma
            k += 2
        tau = max(1.0, 1.0 + 2.0 * s)
        ess[j] = T / tau
    return ess

# ------------********------------
# ----------------------------- Dual-averaging step-size adaptation --------------
# ------------********------------

#This function performs dual-averaging warmup to adaptively tune the HMC step size during an initial phase.
#   It repeatedly proposes leapfrog trajectories, computes Metropolis–Hastings acceptance
#   The algorithm follows Hoffman–Gelman dual averaging to target a desired acceptance probability.
#   Gradient evaluations are tracked explicitly to account for computational cost.
#   After warmup, the averaged step size and final state are returned for stable sampling.

def dual_averaging_warmup(q_init, warmup, L, eps_init, target_accept,
                          gradU, U, Minv, rng,
                          gamma=0.05, t0=10.0, kappa=0.75):
    if warmup <= 0:
        return q_init, float(eps_init), 0.0, 0

    # Initialize DA state
    eps = float(eps_init)
    mu = np.log(10.0 * eps)  # bias toward larger steps initially
    Hbar = 0.0
    epsbar = 1.0
    accept_sum = 0.0
    grad_evals = 0

    q = q_init.copy()

    for m in range(1, warmup + 1):
        # Propose with current eps
        p = rng.normal(size=q.shape[0]).astype(np.float64)
        H0 = hamiltonian(q, p, U, Minv)
        q_prop, p_prop = leapfrog(q, p, eps, L, gradU, Minv)
        p_prop = -p_prop
        H1 = hamiltonian(q_prop, p_prop, U, Minv)
        dH = H1 - H0
        # Acceptance prob (clipped for numerical stability)
        alpha = min(1.0, math.exp(-min(100.0, max(-100.0, dH))))
        accept_sum += alpha
        # MH accept
        if rng.uniform() < alpha:
            q = q_prop

        # Dual averaging updates
        eta = 1.0 / (m + t0)
        Hbar = (1 - eta) * Hbar + eta * (target_accept - alpha)
        log_eps = mu - (math.sqrt(m) / gamma) * Hbar
        eps = float(np.exp(log_eps))
        # Iterate averaged step size
        w = m ** (-kappa)
        epsbar = float(np.exp(w * log_eps + (1 - w) * np.log(epsbar)))

        grad_evals += (L + 1)

    # Use averaged epsilon after warmup
    eps_final = epsbar
    accept_mean = accept_sum / max(1, warmup)
    return q, eps_final, accept_mean, grad_evals

# ------------********------------
# ----------------------------- Batch run for (d, seed) -------------------------
# ------------********------------

#This function sets up and runs a single Hamiltonian Monte Carlo experiment with configurable parameters.
#       It constructs a Gaussian target distribution, selects a mass matrix strategy, and defines energy functions.
#An initial state is chosen, and timing and memory tracking are initialized.
#       Bookkeeping variables are prepared to track gradient evaluations, energy errors, and acceptance statistics.

def run_single(d, steps, warmup, L, eps_input,
               mass, kappa, seed,
               ess_tol, mean_tol,
               adapt_stepsize, target_accept, da_gamma, da_t0, da_kappa):

    rng = np.random.default_rng(seed)

    # Construct Σ and precision Λ
    Sigma, Lambda, _ = make_covariance(d, kappa, rng)

    # Mass matrix (Minv = M^{-1})
    if mass == "identity":
        Minv = np.eye(d, dtype=np.float64)
    elif mass == "diag":
        diagM = np.clip(np.diag(Sigma), 1e-12, None)
        Minv = np.diag(1.0 / diagM)
    elif mass == "full":
        Minv = np.linalg.inv(Sigma)  # = Λ
    else:
        raise ValueError("mass must be 'identity','diag','full'")

    # U(q) and ∇U(q)
    def U(q):        return 0.5 * float(q @ (Lambda @ q))
    def gradU(q):    return Lambda @ q

    # Start at 0
    q = np.zeros(d, dtype=np.float64)

    tracemalloc.start()
    t0 = time.perf_counter()

    total_grad_evals = 0
    energy_errs = []
    accept_warm = 0.0

# ------------********--------------- Warmup  ---
#This block handles the warmup phase of the HMC run, optionally adapting the step size.
#When enabled, dual averaging is used to tune  toward a target acceptance rate.
#Otherwise, a fixed-step warmup performs standard proposals to stabilize the chain.
#Gradient evaluations and energy errors are tracked for performance and diagnostics.

    eps = float(eps_input)
    if adapt_stepsize and warmup > 0:
        q, eps, accept_warm, ge = dual_averaging_warmup(
            q_init=q, warmup=warmup, L=L, eps_init=eps_input, target_accept=target_accept,
            gradU=gradU, U=U, Minv=Minv, rng=rng,
            gamma=da_gamma, t0=da_t0, kappa=da_kappa
        )
        total_grad_evals += ge
    else:
        # Non-adapting warmup (still run proposals to settle the chain)
        for _ in range(warmup):
            p = rng.normal(size=d).astype(np.float64)
            H0 = hamiltonian(q, p, U, Minv)
            q_prop, p_prop = leapfrog(q, p, eps, L, gradU, Minv)
            p_prop = -p_prop
            H1 = hamiltonian(q_prop, p_prop, U, Minv)
            dH = H1 - H0
            alpha = min(1.0, math.exp(-min(100.0, max(-100.0, dH))))
            if rng.uniform() < alpha:
                q = q_prop
            total_grad_evals += (L + 1)
            energy_errs.append(float(dH))

# ------------********-------------- Sampling phase (fixed tuned) ---

#This block runs the main HMC sampling phase after warmup and collects posterior samples.
#       Each iteration proposes a new state via leapfrog integration and applies a Metropolis acceptance test.
#       Runtime, memory usage, acceptance rate, and energy diagnostics are recorded.
#       Sampling quality is assessed using effective sample size and mean error criteria.
#       A comprehensive set of metrics is returned for evaluation and comparison.

    acc = 0
    samples = np.empty((steps, d), dtype=np.float64)
    for t in range(steps):
        p = rng.normal(size=d).astype(np.float64)
        H0 = hamiltonian(q, p, U, Minv)
        q_prop, p_prop = leapfrog(q, p, eps, L, gradU, Minv)
        p_prop = -p_prop
        H1 = hamiltonian(q_prop, p_prop, U, Minv)
        dH = H1 - H0
        alpha = min(1.0, math.exp(-min(100.0, max(-100.0, dH))))
        if rng.uniform() < alpha:
            q = q_prop
            acc += 1
        samples[t, :] = q
        total_grad_evals += (L + 1)
        energy_errs.append(float(dH))

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    accept_rate = acc / max(1, steps)
    mean_dH = float(np.mean(np.abs(energy_errs))) if energy_errs else 0.0

    # Reliability
    ess_vec = ess_per_dim(samples)
    ess_norm_per_dim = ess_vec / max(1, steps)
    ess_norm_mean = float(np.mean(ess_norm_per_dim))
    mean_error = float(np.linalg.norm(np.mean(samples, axis=0)))  # true mean = 0
    success = (ess_norm_mean >= ess_tol) and (mean_error <= mean_tol)

    return {
        "d": int(d),
        "steps": int(steps),
        "warmup": int(warmup),
        "seed": int(seed),
        "L": int(L),
        "step_size_init": float(eps_input),
        "step_size_final": float(eps),
        "adapt_stepsize": bool(adapt_stepsize),
        "target_accept": float(target_accept),
        "accept_warmup": float(accept_warm),
        "mass": str(mass),
        "kappa": float(kappa),
        "accept_rate": float(accept_rate),
        "mean_abs_deltaH": mean_dH,
        "grad_evals_total": int(total_grad_evals),
        "ess_norm_mean": ess_norm_mean,
        "mean_error_norm": mean_error,
        "success": bool(success),
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024**2)),
    }

# ------------********------------
# ----------------------------- Carbon accounting -------------------------------
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

# ------------********------------
# ----------------------------- Plotting helpers --------------------------------
# ------------********------------
#This section defines common plotting parameters and a compatibility helper for boxplots.
#It ensures consistent figure rendering and handles API differences across Matplotlib versions.

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    try:
        plt.boxplot(data, tick_labels=labels)  # Matplotlib ≥3.9
    except TypeError:
        plt.boxplot(data, labels=labels)

def plot_performance(df: pd.DataFrame, outdir: str):
    g = df.groupby("d")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["d"], g["runtime_s"], marker="o")
    plt.title("HMC Performance: Runtime vs dimension"); plt.xlabel("dimension d"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_d.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("d")["accept_rate"].mean().reset_index()
    plt.figure(); plt.plot(g2["d"], g2["accept_rate"], marker="s")
    plt.title("HMC Performance: Acceptance rate vs dimension"); plt.xlabel("dimension d"); plt.ylabel("Acceptance rate")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_accept_vs_d.png"), **plt_kwargs); plt.close()

def plot_scalability(df: pd.DataFrame, outdir: str):
    g = df.groupby("d")["runtime_s"].mean().reset_index()
    plt.figure(); plt.loglog(g["d"], g["runtime_s"], "o")
    plt.title("HMC Scalability: Runtime (log–log) vs dimension"); plt.xlabel("dimension d"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, which="both", alpha=0.3); plt.savefig(os.path.join(outdir, "scal_loglog_runtime_vs_d.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("d")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g2["d"], g2["peak_mem_mb"], marker="^")
    plt.title("HMC Scalability: Peak memory vs dimension"); plt.xlabel("dimension d"); plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_d.png"), **plt_kwargs); plt.close()

def plot_reliability(df: pd.DataFrame, outdir: str):
    g = df.groupby("d")["success"].mean().reset_index()
    plt.figure(); plt.plot(g["d"], g["success"], marker="o"); plt.ylim(0,1.05)
    plt.title("HMC Reliability: Success rate vs dimension"); plt.xlabel("dimension d"); plt.ylabel("Success rate")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_d.png"), **plt_kwargs); plt.close()

    data = [df[df["d"]==k]["ess_norm_mean"].values for k in sorted(df["d"].unique())]
    plt.figure(); _boxplot_with_labels(data, labels=sorted(df["d"].unique()))
    plt.title("HMC Reliability: Normalized ESS/steps by dimension"); plt.xlabel("dimension d"); plt.ylabel("ESS/steps (mean over dims)")
    plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, "rel_ess_boxplot.png"), **plt_kwargs); plt.close()

# ------------********------------
# ----------------------------- Main --------------------------------------------
# ------------********------------


def main():
    parser = argparse.ArgumentParser(description="Classical Hamiltonian Monte Carlo (HMC) — Benchmarks (with ε adaptation)")
    parser.add_argument("--sizes", nargs="+", type=int, default=[2, 8, 16, 32],
                        help="Dimensions d to evaluate.")
    parser.add_argument("--steps", type=int, default=2000, help="Collected samples per batch (post-warmup).")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup steps (discarded).")
    parser.add_argument("--trials", type=int, default=10, help="Seeds per dimension.")
    parser.add_argument("--leapfrog-L", type=int, default=20, dest="L", help="Leapfrog steps per proposal.")
    parser.add_argument("--step-size", type=float, default=0.05, dest="eps", help="Initial step size ε (used or adapted).")
    parser.add_argument("--mass", choices=["identity","diag","full"], default="diag",
                        help="Mass matrix: identity, diag(Σ), or full Σ (best).")
    parser.add_argument("--kappa", type=float, default=100.0, help="Condition number for Σ eigen-spectrum.")
    # Reliability thresholds
    parser.add_argument("--ess-tol", type=float, default=0.5, help="Require mean(ESS/steps over dims) ≥ this.")
    parser.add_argument("--mean-tol", type=float, default=0.1, help="Require ||sample mean|| ≤ this.")
    parser.add_argument("--acc-tol", type=float, default=0.8, help="Used in aggregated plots (success fraction).")
    # Step-size adaptation
    parser.add_argument("--adapt-stepsize", action="store_true", help="Enable dual-averaging adaptation of ε during warmup.")
    parser.add_argument("--target-accept", type=float, default=0.75, help="Target acceptance rate for adaptation.")
    parser.add_argument("--adapt-gamma", type=float, default=0.05, help="Dual-averaging γ (stability).")
    parser.add_argument("--adapt-t0", type=float, default=10.0, help="Dual-averaging t0 (bias early iters).")
    parser.add_argument("--adapt-kappa", type=float, default=0.75, help="Dual-averaging κ (averaging decay).")
    # Carbon/infra
    parser.add_argument("--excel", type=str, default="Filtered CO2 intensity 236 Countries.xlsx")
    parser.add_argument("--device-power-watts", type=float, default=65.0)
    parser.add_argument("--pue", type=float, default=1.2)
    parser.add_argument("--year-select", type=str, default="latest")
    parser.add_argument("--outdir", type=str, default="carbon_by_country")
    parser.add_argument("--combine", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2)//2))
    args = parser.parse_args()

    # Output folders
    cwd = os.getcwd()
    perf_dir = os.path.join(cwd, "Performance")
    scal_dir = os.path.join(cwd, "Scalability")
    rel_dir  = os.path.join(cwd, "Reliability")
    carb_root = os.path.join(cwd, "Carbon footprints")
    carb_dir  = os.path.join(carb_root, args.outdir)
    for dpath in [perf_dir, scal_dir, rel_dir, carb_dir]:
        os.makedirs(dpath, exist_ok=True)

    print("[run] Starting classical HMC benchmarks (ε adaptation: %s)..." % ("ON" if args.adapt_stepsize else "OFF"))
    rows, jobs = [], []
    for d in args.sizes:
        for i in range(args.trials):
            seed = 1000 + 41*int(d) + i
            jobs.append((d, args.steps, args.warmup, args.L, args.eps,
                         args.mass, args.kappa, seed, args.ess_tol, args.mean_tol,
                         args.adapt_stepsize, args.target_accept, args.adapt_gamma, args.adapt_t0, args.adapt_kappa))


#The above main function () defines the command-line interface and configuration for HMC benchmarking.
#       It exposes controls for model size, sampling parameters, mass matrix choice, and step-size adaptation.
#       Output directories for performance, scalability, reliability, and carbon analysis are created automatically.
#       Experiment jobs are assembled across dimensions and random seeds for parallel execution.
#       The setup prepares all metadata needed to run and aggregate large-scale HMC experiments.

    def consume(res: dict):
        rows.append(res)
        tuned = f"eps={res['step_size_final']:.4g}"
        if res["adapt_stepsize"]:
            tuned = f"eps0={res['step_size_init']:.4g}->{res['step_size_final']:.4g} (warm acc~{res['accept_warmup']:.2f})"
        print(f"  - d={res['d']} seed={res['seed']} {tuned} "
              f"acc={res['accept_rate']:.2f} ess/steps={res['ess_norm_mean']:.2f} "
              f"mean||={res['mean_error_norm']:.3f} grad_evals={res['grad_evals_total']} "
              f"runtime={res['runtime_s']:.3f}s success={int(res['success'])}")

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
    # ------------------- Performance (Excel + plots) -------------------
# ------------********------------

    perf_xlsx = os.path.join(perf_dir, "performance_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw_runs")
            agg = df.groupby(["d","mass","adapt_stepsize"]).agg(
                mean_runtime_s=("runtime_s","mean"),
                mean_peak_mem_mb=("peak_mem_mb","mean"),
                mean_accept_rate=("accept_rate","mean"),
                mean_grad_evals=("grad_evals_total","mean"),
                mean_abs_deltaH=("mean_abs_deltaH","mean"),
                mean_eps_final=("step_size_final","mean"),
            ).reset_index()
            agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(df, perf_dir)

# ------------********------------
    # ------------------- Scalability (Excel + plots) -------------------
# ------------********------------
#This block generates scalability reports from the HMC benchmark results.
#   It writes both raw measurements and aggregated statistics to an Excel workbook.
#       Warnings are suppressed during file output for cleaner execution.
#           A scalability plot is produced to visualize runtime and resource trends across dimensions.

    scal_xlsx = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_xlsx, engine="openpyxl") as w:
            df[["d","runtime_s","peak_mem_mb","grad_evals_total","mass","adapt_stepsize"]].to_excel(w, index=False, sheet_name="raw")
            df.groupby(["d","mass","adapt_stepsize"]).mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(df, scal_dir)

# ------------********------------
    # ------------------- Reliability (Excel + plots) -------------------
# ------------********------------

    rel_xlsx = os.path.join(rel_dir, "reliability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(rel_xlsx, engine="openpyxl") as w:
            df[["d","seed","accept_rate","ess_norm_mean","mean_error_norm","success","mass","adapt_stepsize"]].to_excel(w, index=False, sheet_name="runs")
            rel_agg = df.groupby(["d","mass","adapt_stepsize"]).agg(
                success_rate=("success","mean"),
                mean_norm_ess=("ess_norm_mean","mean"),
                std_norm_ess=("ess_norm_mean","std"),
                mean_accept=("accept_rate","mean"),
            ).reset_index()
            rel_agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_reliability(df, rel_dir)

# ------------********------------
    # ------------------- Carbon (Performance runtime only) --------------
# ------------********------------

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
