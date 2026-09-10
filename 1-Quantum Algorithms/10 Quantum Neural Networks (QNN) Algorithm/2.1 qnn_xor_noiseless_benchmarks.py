#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# QNN for XOR — Ideal (Noiseless) with FAST SPSA Gradients
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
QNN for XOR — Ideal (Noiseless) with FAST SPSA Gradients

Model:
  2-qubit PQC classifier. Input (x1,x2) encoded by RY(πx1) on q0 and RY(πx2) on q1.
  H = --sizes (number of layers). Each layer has 4 params: [ry0, ry1, rz0, rz1].
  Readout: P(q1=1); loss: binary cross-entropy; optimizer: Adam on SPSA gradient.

"""

# ------------********------------ Imports
#These imports provide filesystem access, timing, CLI parsing, warnings, and memory profiling for the benchmark scripts.
import os, sys, time, argparse, warnings, pathlib, tracemalloc, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# --------------------------- Data: XOR with augmentation ---------------------------
#def make_xor_dataset generates the XOR classification dataset with optional Gaussian-noise augmentation around each corner point.
#It supports reproducible sampling via a seed, clips augmented points to a bounded range, and stacks all samples into feature and label arrays.

def make_xor_dataset(augment_per_corner: int = 0, noise_std: float = 0.05, seed: int | None = None):
    rng = np.random.default_rng(seed)
    X_base = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y_base = np.array([[0],[1],[1],[0]], dtype=np.float64)
    if augment_per_corner <= 0:
        return X_base, y_base
    X_list, y_list = [], []
    for i in range(4):
        X_list.append(X_base[i:i+1]); y_list.append(y_base[i:i+1])
        noise = rng.normal(0.0, noise_std, size=(augment_per_corner, 2))
        X_aug = np.clip(X_base[i] + noise, -0.25, 1.25)
        y_aug = np.repeat(y_base[i:i+1], augment_per_corner, axis=0)
        X_list.append(X_aug); y_list.append(y_aug)
    X = np.vstack(X_list); y = np.vstack(y_list)
    return X, y

# --------------------------- 2-qubit statevector utilities ---------------------------
#I2 defines the 2×2 complex identity matrix used as a base single-qubit operator.
#def Ry and def Rz construct single-qubit rotation matrices about the Y and Z axes using half-angle trigonometric forms.
#CNOT_01 encodes a 2-qubit CNOT gate matrix with qubit 0 as control and qubit 1 as target in the computational basis.

I2 = np.eye(2, dtype=np.complex128)
def Ry(theta):
    c = math.cos(theta/2.0); s = math.sin(theta/2.0)
    return np.array([[c, -s],[s, c]], dtype=np.complex128)
def Rz(theta):
    c = math.cos(theta/2.0); s = math.sin(theta/2.0)
    return np.array([[c-1j*s, 0],[0, c+1j*s]], dtype=np.complex128)

# CNOT control=0 → target=1
CNOT_01 = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,0,1],
    [0,0,1,0]
], dtype=np.complex128)

#def layer_unitary builds a 4×4 two-qubit unitary from a parameter layer consisting of single-qubit rotations and an entangling CNOT.
#It applies RY rotations on both qubits, follows with a fixed CNOT (control 0 → target 1), and finishes with RZ rotations.
#The function returns the combined layer unitary in matrix form, suitable for variational circuit simulation.

def layer_unitary(params_layer):
    """Return 4×4 layer unitary for [ry0, ry1, rz0, rz1] with order: RY0, RY1, CNOT, RZ0, RZ1."""
    ry0, ry1, rz0, rz1 = params_layer
    U = np.kron(Ry(ry0), I2)
    U = np.kron(I2, Ry(ry1)) @ U
    U = CNOT_01 @ U
    U = np.kron(Rz(rz0), I2) @ U
    U = np.kron(I2, Rz(rz1)) @ U
    return U

#def circuit_unitary composes multiple parameterized layers into a single two-qubit unitary matrix.
#It left-multiplies each new layer so later layers act first, matching circuit execution order.
def circuit_unitary(params):
    """Compose all H layers to a single 4×4 unitary (left-multiply newest layer)."""
    U = np.eye(4, dtype=np.complex128)
    for layer in params:
        U = layer_unitary(layer) @ U
    return U

#def encode_states_matrix maps classical inputs X into quantum feature states 
#     using Y-rotations on a two-qubit |00⟩ reference state.
#For each sample, it applies RY(π·x1) on qubit 0 and RY(π·x2) on qubit 1, 
#      storing the resulting state as a column.
#The function returns a matrix whose columns are encoded input kets ready for circuit evaluation.

def encode_states_matrix(X):
    """Return matrix S_in of shape (4, N): each column is |ψ_in(x)⟩ = (I⊗RY(πx2)) (RY(πx1)⊗I) |00⟩."""
    N = X.shape[0]
    S = np.zeros((4, N), dtype=np.complex128)
    ket00 = np.zeros(4, dtype=np.complex128); ket00[0] = 1.0
    for i in range(N):
        x1, x2 = float(X[i,0]), float(X[i,1])
        Uenc = np.kron(Ry(math.pi*x1), I2)
        Uenc = np.kron(I2, Ry(math.pi*x2)) @ Uenc
        S[:, i] = Uenc @ ket00
    return S

#def probs_from_unitary applies a two-qubit unitary to all encoded input states in a vectorized manner.
#It computes output states and returns the measurement probability of qubit 1 being in state |1⟩ for each input sample.

def probs_from_unitary(U, S_in):
    """Vectorized: S_out = U @ S_in; return p (N,) with p_i = P(q1=1)."""
    S_out = U @ S_in  # (4,N)
    p = (np.abs(S_out[1,:])**2 + np.abs(S_out[3,:])**2).real
    return p

# --------------------------- Loss, accuracy ---------------------------
#def bce_loss computes the binary cross-entropy loss between predicted probabilities and true labels with numerical clipping for stability.
#It returns the mean loss value as a scalar float suitable for optimization tracking.

def bce_loss(p, y):
    p = np.clip(p, 1e-12, 1-1e-12)
    return float(-np.mean(y*np.log(p) + (1-y)*np.log(1-p)))

#def accuracy_from_probs converts probabilities to binary predictions using a 0.5 threshold 
#   and computes classification accuracy against labels.
def accuracy_from_probs(p, y):
    pred = (p >= 0.5).astype(np.int32)
    return float(np.mean(pred.reshape(-1,1) == (y.reshape(-1,1) >= 0.5)))

# --------------------------- Optimizers: Adam + SPSA ---------------------------
#class Adam implements the Adam optimizer for parameter arrays with first and second moment estimates.
#It updates running averages of gradients, applies bias correction, and performs adaptive learning-rate parameter updates.
#The step method modifies parameters in-place and advances the internal timestep for successive optimization iterations.

class Adam:
    def __init__(self, shape, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        self.m = np.zeros(shape, dtype=np.float64)
        self.v = np.zeros(shape, dtype=np.float64)
        self.lr = lr; self.b1 = beta1; self.b2 = beta2; self.eps = eps; self.t = 0
    def step(self, params, grad):
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*grad
        self.v = self.b2*self.v + (1-self.b2)*(grad*grad)
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        params -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return params


#def spsa_epoch performs a single SPSA optimization step by estimating 
#     gradients from two stochastic perturbations of the parameters.
#It evaluates the loss at θ±c_kΔ, forms a simultaneous perturbation gradient estimate, 
#     and computes the adaptive step size a_k.
#The function returns the averaged loss, gradient estimate, and 
#     current learning-rate scale for external parameter updates.

def spsa_epoch(params, S_in, y, a0, c0, k, alpha=0.602, gamma=0.101, rng=None):
    """
    One SPSA iteration:
      a_k = a0 / (k+1)^alpha, c_k = c0 / (k+1)^gamma
      ĝ = (L(θ+c_kΔ) - L(θ-c_kΔ)) / (2 c_k) * Δ^{-1}
    Returns (loss, grad_est).
    """
    if rng is None: rng = np.random.default_rng()
    ak = a0 / ((k+1)**alpha)
    ck = c0 / ((k+1)**gamma)
    delta = rng.choice([-1.0, 1.0], size=params.shape)
    thetap = params + ck*delta
    thetam = params - ck*delta
    Up = circuit_unitary(thetap)
    Um = circuit_unitary(thetam)
    p_plus  = probs_from_unitary(Up, S_in)
    p_minus = probs_from_unitary(Um, S_in)
    Lp = bce_loss(p_plus, y)
    Lm = bce_loss(p_minus, y)
    ghat = ((Lp - Lm) / (2.0*ck)) * (1.0 / delta)
    return float((Lp+Lm)/2.0), ghat, ak  # return ak to allow external scaling if wanted

# --------------------------- One training run ---------------------------

def run_single(H_layers: int, epochs: int, augment: int, noise_std: float,
               lr: float, acc_tol: float, seed: int,
               spsa_a0: float, spsa_c0: float, optimizer: str) -> dict:
    # dataset + pre-encode inputs once
    X, y = make_xor_dataset(augment_per_corner=augment, noise_std=noise_std, seed=seed)
    y = y.astype(np.float64).reshape(-1,1)
    S_in = encode_states_matrix(X)  # (4,N)

    # initialize parameters (H x 4)
    rng = np.random.default_rng(seed)
    params = rng.normal(0.0, 0.15, size=(H_layers, 4)).astype(np.float64)

    # train
    tracemalloc.start()
    t0 = time.perf_counter()

    opt = Adam(params.shape, lr=lr)

    last_loss = None
    if optimizer.lower() == "spsa":
        for k in range(epochs):
            loss, ghat, _ = spsa_epoch(params, S_in, y, a0=spsa_a0, c0=spsa_c0, k=k, rng=rng)
            params = opt.step(params, ghat)
            last_loss = loss
    else:
        # Fallback: exact parameter-shift (slower). Still accelerated by unitary composition.
        shift = math.pi/2.0
        for _ in range(epochs):
            U = circuit_unitary(params)
            p = probs_from_unitary(U, S_in); last_loss = bce_loss(p, y)
            # dL/dp
            p_clip = np.clip(p, 1e-12, 1-1e-12)
            dLdp = (p_clip - y.reshape(-1)) / (p_clip*(1-p_clip))
            # Parameter-shift gradient
            grad = np.zeros_like(params)
            for l in range(H_layers):
                for k in range(4):
                    p_plus  = probs_from_unitary(circuit_unitary(params + np.pad(np.zeros_like(params), ((0,0),(0,0))) + np.eye(H_layers,4)[l,k]*0 + np.where(np.indices(params.shape)==np.array([[l],[k]]), shift, 0)), S_in)  # unused; replaced below
            # Clean parameter-shift (readable loop)
            grad = np.zeros_like(params)
            for l in range(H_layers):
                for k in range(4):
                    params_plus  = params.copy(); params_plus[l,k]  += shift
                    params_minus = params.copy(); params_minus[l,k] -= shift
                    p_plus  = probs_from_unitary(circuit_unitary(params_plus),  S_in)
                    p_minus = probs_from_unitary(circuit_unitary(params_minus), S_in)
                    dp = 0.5*(p_plus - p_minus)
                    grad[l,k] = float(np.mean(dLdp * dp))
            params = opt.step(params, grad)

    # final metrics
    U_final = circuit_unitary(params)
    p_final = probs_from_unitary(U_final, S_in)
    acc = accuracy_from_probs(p_final, y)

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "H": int(H_layers),               
        "epochs": int(epochs),
        "augment_per_corner": int(augment),
        "seed": int(seed),
        "final_loss": float(last_loss if last_loss is not None else bce_loss(p_final, y)),
        "accuracy": float(acc),
        "success": bool(acc >= acc_tol),
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024**2)),
        "n_samples": int(X.shape[0]),
        "n_params": int(H_layers*4),
    }

# --------------------------- Carbon accounting (Performance only) ---------------------------

#def resolve_excel_path resolves a user-provided Excel filename to an absolute path, 
#   checking the current path and falling back to the Desktop if needed.

def resolve_excel_path(excel_arg: str) -> str:
    p = pathlib.Path(excel_arg)
    if p.is_absolute() and p.exists(): return str(p)
    desktop = pathlib.Path.home() / "Desktop" / excel_arg
    return str(desktop if desktop.exists() else p)

#def load_carbon_excel loads a carbon-intensity dataset, infers country/year/intensity columns, 
#    normalizes units to kgCO2/kWh, and selects the latest year if requested.

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

#def compute_carbon converts total experiment runtime into energy use (kWh) using power and PUE, 
#     then estimates per-country CO₂ emissions and summary statistics.

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

# --------------------------- Plotting helpers ---------------------------

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    try:
        plt.boxplot(data, tick_labels=labels)
    except TypeError:
        plt.boxplot(data, labels=labels)

def plot_performance(df: pd.DataFrame, outdir: str):
    g = df.groupby("H")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["H"], g["runtime_s"], marker="o")
    plt.title("Performance: Runtime vs layers H"); plt.xlabel("H (layers)"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_H.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("H")["accuracy"].mean().reset_index()
    plt.figure(); plt.plot(g2["H"], g2["accuracy"], marker="s")
    plt.title("Performance: Mean accuracy vs H"); plt.xlabel("H (layers)"); plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_accuracy_vs_H.png"), **plt_kwargs); plt.close()

def plot_scalability(df: pd.DataFrame, outdir: str):
    g = df.groupby("H")["runtime_s"].mean().reset_index()
    plt.figure(); plt.loglog(g["H"], g["runtime_s"], "o")
    plt.title("Scalability: Runtime (log–log) vs H"); plt.xlabel("H (layers)"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, which="both", alpha=0.3); plt.savefig(os.path.join(outdir, "scal_loglog_runtime_vs_H.png"), **plt_kwargs); plt.close()

    g2 = df.groupby("H")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g2["H"], g2["peak_mem_mb"], marker="^")
    plt.title("Scalability: Peak memory vs H"); plt.xlabel("H (layers)"); plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_H.png"), **plt_kwargs); plt.close()

def plot_reliability(df: pd.DataFrame, outdir: str):
    g = df.groupby("H")["success"].mean().reset_index()
    plt.figure(); plt.plot(g["H"], g["success"], marker="o"); plt.ylim(0,1.05)
    plt.title("Reliability: Success rate vs H (acc ≥ tol)"); plt.xlabel("H (layers)"); plt.ylabel("Success rate")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_H.png"), **plt_kwargs); plt.close()

    data = [df[df["H"]==k]["accuracy"].values for k in sorted(df["H"].unique())]
    plt.figure(); _boxplot_with_labels(data, labels=sorted(df["H"].unique()))
    plt.title("Reliability: Accuracy distribution by H"); plt.xlabel("H (layers)"); plt.ylabel("Accuracy")
    plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, "rel_accuracy_boxplot.png"), **plt_kwargs); plt.close()

def plot_carbon(df: pd.DataFrame, outdir: str):
    top = df.nlargest(15, "kgCO2e")
    plt.figure(figsize=(8,5)); plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
    plt.title("Carbon: Top 15 countries (kgCO2e)"); plt.xlabel("kg CO2e")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "carbon_top15.png"), **plt_kwargs); plt.close()

    plt.figure(); plt.hist(df["kgCO2e"], bins=30)
    plt.title("Carbon: Emission distribution"); plt.xlabel("kg CO2e"); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "carbon_distribution.png"), **plt_kwargs); plt.close()

# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser(description="QNN XOR — Ideal (Noiseless) with Fast SPSA")
    parser.add_argument("--sizes", nargs="+", type=int, default=[2, 4, 8, 16],
                        help="H = number of ansatz layers.")
    parser.add_argument("--steps", type=int, default=300, help="Training epochs.")
    parser.add_argument("--trials", type=int, default=10, help="Trials per H (different seeds).")
    parser.add_argument("--augment-per-corner", type=int, default=32, help="Noisy replicas per XOR corner (0=only 4 points).")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Stddev for Gaussian augmentation.")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam learning rate.")
    parser.add_argument("--acc-tol", type=float, default=0.99, help="Success if accuracy ≥ acc_tol.")
    # SPSA knobs (safe defaults)
    parser.add_argument("--optimizer", type=str, default="spsa", choices=["spsa","ps"],
                        help="spsa = fast (recommended). ps = exact parameter-shift (slow).")
    parser.add_argument("--spsa-a0", type=float, default=0.1, help="SPSA a0 (step size scale).")
    parser.add_argument("--spsa-c0", type=float, default=0.2, help="SPSA c0 (perturb scale).")
    # carbon/infra
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

    print("[run] Starting FAST QNN XOR benchmarks (SPSA, noiseless)...")
    rows = []
    jobs = []
    for H in args.sizes:
        for i in range(args.trials):
            seed = 1000 + 17*H + i
            jobs.append((H, args.steps, args.augment_per_corner, args.noise_std,
                         args.lr, args.acc_tol, seed, args.spsa_a0, args.spsa_c0, args.optimizer))

    def consume(res: dict):
        rows.append(res)
        print(f"  - H={res['H']} seed={res['seed']} acc={res['accuracy']:.3f} "
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

    # ----- Performance -----
    perf_xlsx = os.path.join(perf_dir, "performance_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(perf_xlsx, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="raw_runs")
            agg = df.groupby("H").agg(
                mean_runtime_s=("runtime_s","mean"),
                mean_peak_mem_mb=("peak_mem_mb","mean"),
                mean_accuracy=("accuracy","mean"),
                mean_loss=("final_loss","mean"),
                success_rate=("success","mean"),
                mean_params=("n_params","mean"),
                mean_samples=("n_samples","mean"),
            ).reset_index()
            agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_performance(df, perf_dir)

    # ----- Scalability -----
    scal_xlsx = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_xlsx, engine="openpyxl") as w:
            df[["H","runtime_s","peak_mem_mb","n_params"]].to_excel(w, index=False, sheet_name="raw")
            df.groupby("H").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(df, scal_dir)

    # ----- Reliability -----
    rel_xlsx = os.path.join(rel_dir, "reliability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(rel_xlsx, engine="openpyxl") as w:
            df[["H","seed","accuracy","success"]].to_excel(w, index=False, sheet_name="runs")
            rel_agg = df.groupby("H").agg(
                success_rate=("success","mean"),
                mean_acc=("accuracy","mean"),
                std_acc=("accuracy","std"),
            ).reset_index()
            rel_agg.to_excel(w, index=False, sheet_name="aggregated")
    plot_reliability(df, rel_dir)

    # ----- Carbon (PERFORMANCE runtime only) -----
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
