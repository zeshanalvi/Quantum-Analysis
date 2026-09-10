#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# Classical Neural Network for XOR
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
Classical Neural Network for XOR
Network: 2 -> H -> 1 MLP (tanh hidden, sigmoid output), Adam optimizer, BCE loss.
Dataset: XOR { (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0 }, with optional Gaussian augmentation.
Folders created in CWD:
  Performance/
  Scalability/
  Reliability/
  Carbon footprints/<outdir>/
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
# ============== Data: XOR with optional augmentation
# ------------********------------

#def make_xor_dataset constructs a classic XOR classification dataset with optional data augmentation via Gaussian noise around each corner.
#It supports deterministic generation through a random seed and can replicate each base point multiple times with controlled noise.

def make_xor_dataset(augment_per_corner: int = 0, noise_std: float = 0.05, seed: int | None = None):
    """
    Build XOR dataset. Optionally augment each corner with Gaussian noise.
    Returns (X, y) with shapes (N,2), (N,1).
    """
    rng = np.random.default_rng(seed)
    X_base = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64)
    y_base = np.array([[0],[1],[1],[0]], dtype=np.float64)

    if augment_per_corner <= 0:
        return X_base, y_base

    X_list = []
    y_list = []
    for i in range(4):
        X_list.append(X_base[i:i+1])
        y_list.append(y_base[i:i+1])
        # add noisy replicas
        if augment_per_corner > 0:
            noise = rng.normal(0.0, noise_std, size=(augment_per_corner, 2))
            X_aug = np.clip(X_base[i] + noise, -0.5, 1.5)
            y_aug = np.repeat(y_base[i:i+1], augment_per_corner, axis=0)
            X_list.append(X_aug); y_list.append(y_aug)
    X = np.vstack(X_list)
    y = np.vstack(y_list)
    return X, y

# ------------********------------
# ============== Minimal MLP with Adam (numpy)
# ------------********------------
#sigmoid, dsigmoid, and dtanh define activation functions and their derivatives used in forward and backward passes.
#class MLP_XOR initializes a simple 2→H→1 neural network with tanh hidden units and a sigmoid output for XOR classification.
#It uses scaled random initialization, L2 regularization, and maintains Adam optimizer moments for full-batch training.

def sigmoid(z): return 1.0 / (1.0 + np.exp(-z))
def dsigmoid(a): return a * (1.0 - a)           # when given a = sigmoid(z)
def dtanh(a): return 1.0 - a*a                   # when given a = tanh(z)

class MLP_XOR:
    """2 -> H -> 1 with tanh hidden, sigmoid output; BCE loss; Adam optimizer (full-batch)."""
    def __init__(self, H: int, lr: float, l2: float, seed: int):
        rng = np.random.default_rng(seed)
        self.H = H
        # He init for tanh ~ scaled
        self.W1 = rng.normal(0, 1/np.sqrt(2), size=(2, H))
        self.b1 = np.zeros((1, H))
        self.W2 = rng.normal(0, 1/np.sqrt(H), size=(H, 1))
        self.b2 = np.zeros((1, 1))
        self.lr = lr
        self.l2 = l2
        # Adam moments
        self.mW1 = np.zeros_like(self.W1); self.vW1 = np.zeros_like(self.W1)
        self.mb1 = np.zeros_like(self.b1); self.vb1 = np.zeros_like(self.b1)
        self.mW2 = np.zeros_like(self.W2); self.vW2 = np.zeros_like(self.W2)
        self.mb2 = np.zeros_like(self.b2); self.vb2 = np.zeros_like(self.b2)
        self.beta1 = 0.9; self.beta2 = 0.999; self.eps = 1e-8
        self.t = 0

    def forward(self, X):
        Z1 = X @ self.W1 + self.b1
        A1 = np.tanh(Z1)
        Z2 = A1 @ self.W2 + self.b2
        A2 = sigmoid(Z2)
        cache = (X, Z1, A1, Z2, A2)
        return A2, cache

#loss_and_grads performs a forward pass to compute binary cross-entropy loss 
#   with L2 regularization, then backpropagates gradients.
#It computes gradients for all weights and biases using the 
#   sigmoid/tanh derivatives and averages them over the batch.
#The method returns the scalar loss along with gradients needed for parameter updates via the optimizer.

    def loss_and_grads(self, X, y):
        N = X.shape[0]
        A2, (X, Z1, A1, Z2, A2) = self.forward(X)
        # BCE (stable): L = -(y log a + (1-y) log(1-a))
        eps = 1e-12
        L = -np.mean(y*np.log(A2+eps) + (1-y)*np.log(1-A2+eps))
        # L2 regularization (on weights only)
        L += 0.5*self.l2*(np.sum(self.W1*self.W1) + np.sum(self.W2*self.W2))/N

        # Gradients
        dZ2 = (A2 - y) / N                        # (N,1)
        dW2 = A1.T @ dZ2 + (self.l2/N)*self.W2    # (H,1)
        db2 = np.sum(dZ2, axis=0, keepdims=True)  # (1,1)

        dA1 = dZ2 @ self.W2.T                     # (N,H)
        dZ1 = dA1 * dtanh(A1)                     # (N,H)
        dW1 = X.T @ dZ1 + (self.l2/N)*self.W1     # (2,H)
        db1 = np.sum(dZ1, axis=0, keepdims=True)  # (1,H)
        return L, (dW1, db1, dW2, db2)

#adam_step applies one Adam optimizer update to all network parameters using the provided gradients.
#It updates biased first and second moments, performs bias correction, and scales steps by the adaptive learning rates.
#The method advances the internal timestep and updates both weights and biases in-place.

    def adam_step(self, grads):
        self.t += 1
        dW1, db1, dW2, db2 = grads
        # W1
        self.mW1 = self.beta1*self.mW1 + (1-self.beta1)*dW1
        self.vW1 = self.beta2*self.vW1 + (1-self.beta2)*(dW1*dW1)
        mW1_hat = self.mW1 / (1 - self.beta1**self.t)
        vW1_hat = self.vW1 / (1 - self.beta2**self.t)
        self.W1 -= self.lr * mW1_hat / (np.sqrt(vW1_hat) + self.eps)
        # b1
        self.mb1 = self.beta1*self.mb1 + (1-self.beta1)*db1
        self.vb1 = self.beta2*self.vb1 + (1-self.beta2)*(db1*db1)
        mb1_hat = self.mb1 / (1 - self.beta1**self.t)
        vb1_hat = self.vb1 / (1 - self.beta2**self.t)
        self.b1 -= self.lr * mb1_hat / (np.sqrt(vb1_hat) + self.eps)
        # W2
        self.mW2 = self.beta1*self.mW2 + (1-self.beta1)*dW2
        self.vW2 = self.beta2*self.vW2 + (1-self.beta2)*(dW2*dW2)
        mW2_hat = self.mW2 / (1 - self.beta1**self.t)
        vW2_hat = self.vW2 / (1 - self.beta2**self.t)
        self.W2 -= self.lr * mW2_hat / (np.sqrt(vW2_hat) + self.eps)
        # b2
        self.mb2 = self.beta1*self.mb2 + (1-self.beta1)*db2
        self.vb2 = self.beta2*self.vb2 + (1-self.beta2)*(db2*db2)
        mb2_hat = self.mb2 / (1 - self.beta1**self.t)
        vb2_hat = self.vb2 / (1 - self.beta2**self.t)
        self.b2 -= self.lr * mb2_hat / (np.sqrt(vb2_hat) + self.eps)

    def fit(self, X, y, epochs: int):
        for _ in range(epochs):
            L, grads = self.loss_and_grads(X, y)
            self.adam_step(grads)
        return L

    def predict(self, X):
        A2, _ = self.forward(X)
        return (A2 >= 0.5).astype(np.int32), A2

# ------------********------------
# ============== Metrics & one experiment
# ------------********------------

def accuracy(pred_labels, y):
    return float(np.mean(pred_labels.reshape(-1,1) == y.reshape(-1,1)))

def run_single(H: int, epochs: int, augment: int, noise_std: float,
               lr: float, l2: float, acc_tol: float, seed: int) -> dict:
    X, y = make_xor_dataset(augment_per_corner=augment, noise_std=noise_std, seed=seed)

    tracemalloc.start()
    t0 = time.perf_counter()

    model = MLP_XOR(H=H, lr=lr, l2=l2, seed=seed)
    final_loss = model.fit(X, y, epochs=epochs)
    yhat, prob = model.predict(X)
    acc = accuracy(yhat, y)

    runtime = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "H": H,
        "epochs": epochs,
        "augment_per_corner": augment,
        "seed": seed,
        "final_loss": float(final_loss),
        "accuracy": float(acc),
        "success": bool(acc >= acc_tol),
        "runtime_s": float(runtime),
        "peak_mem_mb": float(peak / (1024**2)),
        "n_samples": int(X.shape[0]),
        "n_params": int(2*H + H + H*1 + 1),  # W1(2xH)+b1(H)+W2(Hx1)+b2(1)
    }

# ------------********------------
# ============== Carbon (Performance only)
# ------------********------------
# for carbon file path

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
# ============== Plotting helpers
# ------------********------------

plt_kwargs = dict(dpi=140, bbox_inches="tight")

def _boxplot_with_labels(data, labels):
    try:
        plt.boxplot(data, tick_labels=labels)
    except TypeError:
        plt.boxplot(data, labels=labels)

def plot_performance(df: pd.DataFrame, outdir: str):
    # Runtime vs H
    g = df.groupby("H")["runtime_s"].mean().reset_index()
    plt.figure(); plt.plot(g["H"], g["runtime_s"], marker="o")
    plt.title("Performance: Runtime vs hidden units H"); plt.xlabel("H"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_runtime_vs_H.png"), **plt_kwargs); plt.close()

    # Accuracy vs H
    g2 = df.groupby("H")["accuracy"].mean().reset_index()
    plt.figure(); plt.plot(g2["H"], g2["accuracy"], marker="s")
    plt.title("Performance: Mean accuracy vs H"); plt.xlabel("H"); plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "perf_accuracy_vs_H.png"), **plt_kwargs); plt.close()

def plot_scalability(df: pd.DataFrame, outdir: str):
    # Runtime log-log vs H
    g = df.groupby("H")["runtime_s"].mean().reset_index()
    plt.figure(); plt.loglog(g["H"], g["runtime_s"], "o")
    plt.title("Scalability: Runtime (log–log) vs H"); plt.xlabel("H (hidden units)"); plt.ylabel("Mean runtime (s)")
    plt.grid(True, which="both", alpha=0.3); plt.savefig(os.path.join(outdir, "scal_loglog_runtime_vs_H.png"), **plt_kwargs); plt.close()

    # Peak memory vs H
    g2 = df.groupby("H")["peak_mem_mb"].mean().reset_index()
    plt.figure(); plt.plot(g2["H"], g2["peak_mem_mb"], marker="^")
    plt.title("Scalability: Peak memory vs H"); plt.xlabel("H (hidden units)"); plt.ylabel("Peak memory (MB)")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "scal_peakmem_vs_H.png"), **plt_kwargs); plt.close()

def plot_reliability(df: pd.DataFrame, outdir: str):
    # Success rate vs H
    g = df.groupby("H")["success"].mean().reset_index()
    plt.figure(); plt.plot(g["H"], g["success"], marker="o"); plt.ylim(0,1.05)
    plt.title("Reliability: Success rate vs H (acc ≥ tol)"); plt.xlabel("H"); plt.ylabel("Success rate")
    plt.grid(True, alpha=0.3); plt.savefig(os.path.join(outdir, "rel_success_vs_H.png"), **plt_kwargs); plt.close()

    # Accuracy distribution by H
    data = [df[df["H"]==k]["accuracy"].values for k in sorted(df["H"].unique())]
    plt.figure(); _boxplot_with_labels(data, labels=sorted(df["H"].unique()))
    plt.title("Reliability: Accuracy distribution by H"); plt.xlabel("H"); plt.ylabel("Accuracy")
    plt.grid(True, axis='y', alpha=0.3); plt.savefig(os.path.join(outdir, "rel_accuracy_boxplot.png"), **plt_kwargs); plt.close()

def plot_carbon(df: pd.DataFrame, outdir: str):
    top = df.nlargest(15, "kgCO2e")
    plt.figure(figsize=(8,5)); plt.barh(top["Country"][::-1], top["kgCO2e"][::-1])
    plt.title("Carbon: Top 15 countries (kgCO2e)"); plt.xlabel("kg CO2e")
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "carbon_top15.png"), **plt_kwargs); plt.close()

    plt.figure(); plt.hist(df["kgCO2e"], bins=30)
    plt.title("Carbon: Emission distribution"); plt.xlabel("kg CO2e"); plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(outdir, "carbon_distribution.png"), **plt_kwargs); plt.close()

# ------------********------------
# ============== Main ==============
# ------------********------------
#def main defines the CLI entry point for running classical XOR MLP benchmarks with configurable model, data, and training parameters.
#It parses command-line arguments, prepares output directories for performance, scalability, reliability, and carbon analysis, 
#   and prints a start banner.
#The function then enumerates hidden sizes and trials, assembling a list of seeded benchmark jobs to be executed.

def main():
    parser = argparse.ArgumentParser(description="Classical XOR Neural Network Benchmarks (MLP, numpy)")
    # Keep CLI consistent with previous projects
    parser.add_argument("--sizes", nargs="+", type=int, default=[2, 4, 8, 16],
                        help="Hidden layer sizes H to test.")
    parser.add_argument("--steps", type=int, default=500, help="Training epochs.")
    parser.add_argument("--trials", type=int, default=10, help="Trials per H (different seeds).")
    parser.add_argument("--augment-per-corner", type=int, default=32,
                        help="Noisy replicas per XOR corner (0 = exact 4 points).")
    parser.add_argument("--noise-std", type=float, default=0.05, help="Stddev for Gaussian augmentation.")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam learning rate.")
    parser.add_argument("--l2", type=float, default=1e-4, help="L2 regularization strength.")
    parser.add_argument("--acc-tol", type=float, default=0.99, help="Success if accuracy ≥ acc_tol.")
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

    print("[run] Starting XOR MLP benchmarks...")
    rows = []
    jobs = []
    for H in args.sizes:
        for i in range(args.trials):
            seed = 1000 + 17*H + i
            jobs.append((H, args.steps, args.augment_per_corner, args.noise_std, args.lr, args.l2, args.acc_tol, seed))

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

# ------------********------------
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

# ------------********------------
    # ----- Scalability -----
    scal_xlsx = os.path.join(scal_dir, "scalability_results.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pd.ExcelWriter(scal_xlsx, engine="openpyxl") as w:
            df[["H","runtime_s","peak_mem_mb","n_params"]].to_excel(w, index=False, sheet_name="raw")
            df.groupby("H").mean(numeric_only=True).reset_index().to_excel(w, index=False, sheet_name="aggregated")
    plot_scalability(df, scal_dir)

# ------------********------------
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

# ------------********------------
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
