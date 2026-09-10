#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
#  classical_pendulum_tomography.py
# Reconstruct a classical harmonic oscillator phase-space PDF f(x,p)
# from quadrature histograms X_theta = x cosθ + (p/(mω)) sinθ via filtered backprojection.
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

import os, sys, time, argparse, warnings, tracemalloc
import numpy as np, pandas as pd, matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)

# ------------********------------
# ------------********------------ IO helpers
# ------------********------------
#This helper ensures that a target directory exists before writing any outputs.
#It creates the directory if needed and safely ignores the case where it already exists.

def ensure_dir(p): os.makedirs(p, exist_ok=True)

#This function writes multiple pandas DataFrames into a single Excel workbook.
#If Excel export fails, it gracefully falls back to writing each table as a CSV file.

def write_excel(tables, path):
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            for name, df in tables.items():
                df.to_excel(w, index=False, sheet_name=(name[:31] or "Sheet1"))
        print(f"[OK] Wrote Excel: {path}")
    except Exception as e:
        print(f"[WARN] Excel write failed ({e}); writing CSVs instead.")
        base = os.path.splitext(path)[0]
        for name, df in tables.items():
            csvp = f"{base}__{name}.csv"; df.to_csv(csvp, index=False); print(f"[OK] {csvp}")

#This utility saves a Matplotlib figure with consistent layout and resolution.
#It closes the figure after saving to free resources and reports the output path.

def save_fig(fig, path, dpi=150):
    fig.tight_layout(); fig.savefig(path, dpi=dpi, bbox_inches="tight"); plt.close(fig)
    print(f"[OK] Saved plot: {path}")

#This function resolves a file path by checking user input and Desktop locations.
#It supports quoted paths, missing extensions, and a list of default fallback filenames.

def resolve_desktop_file(user_arg, defaults):
    desk = os.path.join(os.environ.get("USERPROFILE",""), "Desktop")
    def exists(x): return os.path.isfile(x)
    if user_arg:
        base = user_arg.strip().strip('"').strip("'")
        if exists(base): return os.path.abspath(base)
        c = os.path.join(desk, base); 
        if exists(c): return os.path.abspath(c)
        if not os.path.splitext(base)[1]:
            for ext in (".xlsx",".xls",".csv"):
                c = os.path.join(desk, base+ext)
                if exists(c): return os.path.abspath(c)
        tail = os.path.split(base)[1].lower()
        for fn in os.listdir(desk):
            if fn.lower()==tail or any((not os.path.splitext(base)[1]) and fn.lower()==tail+ext for ext in (".xlsx",".xls",".csv")):
                c = os.path.join(desk, fn)
                if exists(c): return os.path.abspath(c)
        return None
    for c in defaults:
        if exists(c): return os.path.abspath(c)
    return None

# ------------********------------
# ------------********------------ Physics: classical ensemble (Gaussian in phase-space)
# ------------********------------

#This function draws samples from a classical phase-space Gaussian ensemble.
#It generates correlated position and momentum samples using the provided mean and covariance.

def sample_classical_ensemble(n_samples, mean, cov, mass, omega, rng):
    # mean = (x0, p0), cov 2x2 on (x, p) (SI-like). We will use p/(mω) inside the projector.
    x, p = rng.multivariate_normal(mean, cov, size=n_samples).T
    return x, p

#This function computes a rotated quadrature value from position and momentum samples.
#It combines x and p according to the rotation angle and physical scaling factors.
def quadrature_from_samples(x, p, theta, mass, omega):
    return x*np.cos(theta) + (p/(mass*omega))*np.sin(theta)

#This helper generates a set of evenly spaced angles in the interval [0, π).
#These angles are typically used for quadrature or phase-space projections.
def make_angles(S):
    # S angles in [0, π)
    return np.linspace(0.0, np.pi, S, endpoint=False)

# ------------********------------
# ------------********------------ Tomography: projections & filtered backprojection (ramp filter)
# ------------********------------
#This helper applies a ramp filter to a 1D projected density in the frequency domain.
#It multiplies Fourier components by |f| to enhance high-frequency content used in tomography.

def ramp_filter_1d(dens, dx):
    # dens sampled at uniform dx on t-axis; filter in frequency |f|
    n = len(dens)
    F = np.fft.rfft(dens)
    freqs = np.fft.rfftfreq(n, d=dx)
    F_filtered = F * np.abs(freqs)
    back = np.fft.irfft(F_filtered, n=n)
    return back

#This function reconstructs a 2D phase-space distribution via filtered backprojection.
#It maps each projection onto the (x,p) grid using the physical quadrature relation and interpolates values.
#The result is accumulated over angles and normalized to yield a valid density.

def backproject(angles, proj_dens_list, t_centers, grid_x, grid_p, mass, omega, use_filter=True):
    # proj_dens_list[k] is density over t_centers for angle angles[k]
    Gx, Gp = len(grid_x), len(grid_p)
    X, P = np.meshgrid(grid_x, grid_p, indexing="xy")  # shape (Gp,Gx)
    recon = np.zeros((Gp, Gx), dtype=float)
    dt = t_centers[1]-t_centers[0]
    dtheta = (np.pi/len(angles)) if len(angles)>0 else 0.0

    for theta, dens in zip(angles, proj_dens_list):
        d = ramp_filter_1d(dens, dt) if use_filter else dens
        # For each pixel, compute t = x cosθ + (p/(mω)) sinθ
        T = X*np.cos(theta) + (P/(mass*omega))*np.sin(theta)
        # Interpolate filtered projection at T
        vals = np.interp(T.ravel(), t_centers, d, left=0.0, right=0.0).reshape(T.shape)
        recon += vals * dtheta
    # Normalize to unit integral
    dx = grid_x[1]-grid_x[0]; dp = grid_p[1]-grid_p[0]
    s = np.sum(recon)*dx*dp
    if s>0: recon /= s
    return recon

# ------------********------------
# ------------********------------ Ground-truth PDF (Gaussian)
# ------------********------------
#This function evaluates a 2D Gaussian probability density over a phase-space grid.
#It computes the exponent efficiently using the inverse covariance and Einstein summation.
#The result is a properly normalized Gaussian PDF sampled on the (x, p) grid.

def gaussian_pdf_grid(grid_x, grid_p, mean, cov):
    X, P = np.meshgrid(grid_x, grid_p, indexing="xy")
    pos = np.stack([X, P], axis=-1)  # (...,2)
    mu = np.array(mean)
    Sigma = np.array(cov)
    inv = np.linalg.inv(Sigma)
    det = np.linalg.det(Sigma)
    diff = pos - mu
    expo = -0.5*np.einsum("...i,ij,...j->...", diff, inv, diff)
    norm = 1.0/(2*np.pi*np.sqrt(det))
    return norm*np.exp(expo)

# ------------********------------
# ------------********------------ Carbon helpers 
# ------------********------------
#This function loads country-level CO₂ intensity data from a CSV or Excel file.
#It infers relevant columns, normalizes units to kg CO₂ per kWh, and optionally selects the latest year.

def load_country_co2(path, year_select="latest"):
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df.columns = [str(c).strip().lower() for c in df.columns]
    cname = next((c for c in df.columns if "country" in c or c in ("name","nation")), None)
    if not cname: raise ValueError("No country column")
    yname = next((c for c in df.columns if "year" in c), None)
    cand = [c for c in df.columns if "intensity" in c] or \
           [c for c in df.columns if ("co2" in c and "kwh" in c)] or \
           [c for c in df.columns if (("kg" in c or "g" in c) and "kwh" in c)]
    if not cand: raise ValueError("No intensity column")
    icol = cand[0]
    use = [cname, icol] + ([yname] if yname else [])
    out = df[use].dropna().copy()
    out.columns = ["country","intensity_raw"] + (["year"] if yname else [])
    out["intensity_raw"] = pd.to_numeric(out["intensity_raw"], errors="coerce")
    out = out.dropna(subset=["intensity_raw"])
    if np.nanmedian(out["intensity_raw"])>10.0:
        out["kg_per_kwh"] = out["intensity_raw"]/1000.0
    else:
        out["kg_per_kwh"] = out["intensity_raw"]
    out["country"] = out["country"].astype(str).str.strip().str.title()
    if yname and year_select=="latest":
        out = out.sort_values(["country","year"]).groupby("country", as_index=False).tail(1)
    keep = ["country","kg_per_kwh"] + (["year"] if yname else [])
    return out[keep].drop_duplicates()

#This function computes carbon emissions by combining runtime-derived energy use with CO₂ intensity data.
#It applies power and PUE scaling, then cross-joins runs with countries to estimate emissions.

def compute_emissions(perf_df, co2_df, power_watts, pue):
    perf = perf_df.copy()
    perf["energy_kwh"] = (perf["runtime_sec"]*power_watts)/3.6e6
    perf["energy_kwh"] *= pue
    co2 = co2_df.rename(columns={"kg_per_kwh":"kgco2_per_kwh"}).copy()
    perf["_k"]=1; co2["_k"]=1
    joined = perf.merge(co2, on="_k").drop(columns="_k")
    joined["emissions_kgCO2"] = joined["energy_kwh"]*joined["kgco2_per_kwh"]
    return joined

#This function builds a worst-case emissions table based on the heaviest benchmark configuration.
#It estimates energy use from the slowest setting and ranks countries by resulting CO₂ emissions.
def worstcase_country_table(perf_df, co2_df, power_watts, pue, dim_key="grid_dim", step_key="steps"):
    heavy_dim = int(perf_df[dim_key].max())
    heavy_S   = int(perf_df[step_key].max())
    rt = float(perf_df[(perf_df[dim_key]==heavy_dim)&(perf_df[step_key]==heavy_S)]["runtime_sec"].mean())
    kwh = rt*power_watts/3.6e6 * pue
    tbl = co2_df.copy()
    if "year" not in tbl.columns: tbl["year"] = pd.NA
    out = pd.DataFrame({
        "Country": tbl["country"], "Year": tbl["year"],
        "Intensity": tbl["kg_per_kwh"], "kWh": kwh
    })
    out["kgCO2e"] = out["Intensity"]*out["kWh"]
    out = out.sort_values("kgCO2e", ascending=False).reset_index(drop=True)
    meta = {"grid_dim": heavy_dim, "steps": heavy_S, "mean_runtime_s": rt, "kWh_used": kwh}
    return out, meta

# ------------********------------
# ------------********------------ Benchmark driver
# ------------********------------
def run_once(S, shots, G, mass, omega, rng, gt_mean, gt_cov, xlim, plim):
    # 1) simulate projections
    thetas = make_angles(S)
    # Choose histogram support (t) to cover the rectangle in (x,p)
    tmax = xlim*np.max(np.abs(np.cos(thetas))) + (plim/(mass*omega))*np.max(np.abs(np.sin(thetas)))
    B = max(256, 2*G)  # histogram bins
    t_edges = np.linspace(-tmax, tmax, B+1)
    t_centers = 0.5*(t_edges[:-1]+t_edges[1:])
    proj_dens = []
    for theta in thetas:
        x, p = sample_classical_ensemble(shots, gt_mean, gt_cov, mass, omega, rng)
        qt = quadrature_from_samples(x, p, theta, mass, omega)
        hist, _ = np.histogram(qt, bins=t_edges)
        dens = hist/(shots*(t_edges[1]-t_edges[0]))  # empirical density
        proj_dens.append(dens)

    # 2) reconstruct on grid
    gx = np.linspace(-xlim, xlim, G)
    gp = np.linspace(-plim, plim, G)
    recon = backproject(thetas, proj_dens, t_centers, gx, gp, mass, omega, use_filter=True)

    # 3) ground-truth on grid, reliability metrics
    truth = gaussian_pdf_grid(gx, gp, gt_mean, gt_cov)
    dx = gx[1]-gx[0]; dp = gp[1]-gp[0]
    mse = float(np.mean((recon-truth)**2))
    l1  = float(np.sum(np.abs(recon-truth))*dx*dp)
    return dict(recon=recon, truth=truth, gx=gx, gp=gp, mse=mse, l1=l1,
                S=S, shots=shots, grid=G)


# ------------********---------------------********---------------------********------------

def main():
    ap = argparse.ArgumentParser(description="Classical pendulum tomography with four benchmarks + carbon.")
    # benchmark grid controls (keep style parity)
    ap.add_argument("--n-min", type=int, default=2)       # unused; placeholder for parity
    ap.add_argument("--n-max", type=int, default=6)       # unused
    ap.add_argument("--steps", type=str, default="8,16,32", help="comma-separated number of projection angles S")
    ap.add_argument("--repeats", type=int, default=2)
    # data/model
    ap.add_argument("--shots-per-angle", type=int, default=1500)
    ap.add_argument("--grid", type=int, default=64, help="grid size G (GxG pixels)")
    ap.add_argument("--mass", type=float, default=1.0)
    ap.add_argument("--omega", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    # ground-truth Gaussian (mean & cov)
    ap.add_argument("--x0", type=float, default=0.6)
    ap.add_argument("--p0", type=float, default=-0.4)
    ap.add_argument("--sx", type=float, default=0.5)      # std in x
    ap.add_argument("--sp", type=float, default=0.7)      # std in p
    ap.add_argument("--rho", type=float, default=0.2)     # correlation between x and p
    # field-of-view
    ap.add_argument("--xlim", type=float, default=3.0)
    ap.add_argument("--plim", type=float, default=3.0)
    # carbon + IO parity
    ap.add_argument("--excel", type=str, default=None)
    ap.add_argument("--countries-co2-file", type=str, default=None)
    ap.add_argument("--device-power-watts", type=float, default=None)
    ap.add_argument("--power-watts", type=float, default=None)
    ap.add_argument("--pue", type=float, default=1.2)
    ap.add_argument("--year-select", choices=["latest","all"], default="latest")
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--combine", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    S_list = sorted(set(int(s) for s in args.steps.split(",") if s.strip()))
    shots = args.shots_per_angle; G = args.grid
    mass, omega = args.mass, args.omega
    # GT Gaussian params
    cov = np.array([[args.sx**2, args.rho*args.sx*args.sp],
                    [args.rho*args.sx*args.sp, args.sp**2]])
    mean = np.array([args.x0, args.p0])

    # outputs
    root = os.path.abspath(args.outdir)
    perf_dir = os.path.join(root, "Performance")
    scal_dir = os.path.join(root, "Scalability")
    reli_dir = os.path.join(root, "Reliability")
    carb_dir = os.path.join(root, "Carbon")
    for d in (perf_dir, scal_dir, reli_dir, carb_dir): ensure_dir(d)

    # run grid (vary S = steps), repeats
    rows_perf, rows_rel = [], []
    t0_all = time.perf_counter()
    for S in S_list:
        for r in range(args.repeats):
            tracemalloc.start(); t0 = time.perf_counter()
            out = run_once(S, shots, G, mass, omega, rng, mean, cov, args.xlim, args.plim)
            runtime = time.perf_counter()-t0
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

            # save representative images for first repeat of each S
            if r==0:
                fig1, ax1 = plt.subplots(1,3, figsize=(13,4))
                im0=ax1[0].imshow(out["truth"], extent=[-args.xlim,args.xlim,-args.plim,args.plim],
                                  origin="lower", aspect="auto"); ax1[0].set_title("Ground truth f(x,p)"); plt.colorbar(im0, ax=ax1[0])
                im1=ax1[1].imshow(out["recon"], extent=[-args.xlim,args.xlim,-args.plim,args.plim],
                                  origin="lower", aspect="auto"); ax1[1].set_title(f"Reconstruction (S={S})"); plt.colorbar(im1, ax=ax1[1])
                im2=ax1[2].imshow(out["recon"]-out["truth"], extent=[-args.xlim,args.xlim,-args.plim,args.plim],
                                  origin="lower", aspect="auto"); ax1[2].set_title("Difference"); plt.colorbar(im2, ax=ax1[2])
                for a in ax1: a.set_xlabel("x"); a.set_ylabel("p")
                save_fig(fig1, os.path.join(reli_dir, f"tomography_maps_S{S}.png"))

            rows_perf.append({
                "grid_dim": G, "steps": S, "repeat": r,
                "runtime_sec": runtime, "peak_mem_mb": peak/(1024**2)
            })
            rows_rel.append({
                "grid_dim": G, "steps": S, "repeat": r,
                "mse": out["mse"], "l1_distance": out["l1"]
            })

    perf_df = pd.DataFrame(rows_perf)
    rel_df  = pd.DataFrame(rows_rel)

    # write excel & plots
    perf_sum = perf_df.groupby(["grid_dim","steps"], as_index=False).agg(
        mean_runtime_sec=("runtime_sec","mean"),
        std_runtime_sec=("runtime_sec","std"),
        mean_peak_mem_mb=("peak_mem_mb","mean"),
        std_peak_mem_mb=("peak_mem_mb","std"),
    )
    write_excel({"raw":perf_df, "summary":perf_sum}, os.path.join(perf_dir,"performance.xlsx"))

# ------------********------------ performance plots
    fig, ax = plt.subplots(figsize=(7,5))
    for S in S_list:
        sub = perf_df[perf_df["steps"]==S]
        ax.scatter([G]*len(sub), sub["runtime_sec"], label=f"S={S}")
    ax.set_xlabel("Grid dimension G (fixed)"); ax.set_ylabel("Runtime (s)"); ax.set_title("Runtime scatter by steps at fixed grid")
    ax.legend(); save_fig(fig, os.path.join(perf_dir,"runtime_scatter.png"))

# ------------********------------ scalability fits: runtime vs steps (S) at fixed grid
    sfit = perf_df.groupby("steps", as_index=False)["runtime_sec"].mean()
    xb = np.log(sfit["steps"].values); yb = np.log(sfit["runtime_sec"].values+1e-12)
    beta, a0 = np.polyfit(xb, yb, 1)
    scal_table = pd.DataFrame({"fit_parameter":["beta_step_scaling"], "estimate":[beta], "note":[f"G={G}"]})
    write_excel({"scaling_fit":scal_table}, os.path.join(scal_dir,"scalability.xlsx"))

    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(sfit["steps"], sfit["runtime_sec"], marker="o")
    ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("Steps S (angles)"); ax.set_ylabel("Runtime (s)")
    ax.set_title("Scalability: runtime vs S"); save_fig(fig, os.path.join(scal_dir,"runtime_vs_steps.png"))

# ------------********------------ reliability
    rel_sum = rel_df.groupby(["grid_dim","steps"], as_index=False).agg(
        mean_mse=("mse","mean"), std_mse=("mse","std"),
        mean_l1=("l1_distance","mean"), std_l1=("l1_distance","std"),
    )
    write_excel({"raw":rel_df, "summary":rel_sum}, os.path.join(reli_dir,"reliability.xlsx"))

    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(rel_sum["steps"], rel_sum["mean_mse"], marker="o"); ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Steps S"); ax.set_ylabel("Mean MSE (log)"); ax.set_title("Reliability: reconstruction error vs S")
    save_fig(fig, os.path.join(reli_dir,"mse_vs_steps.png"))

    # carbon (performance only)
    desk = os.path.join(os.environ.get("USERPROFILE",""), "Desktop")
    defaults = [os.path.join(desk,"Filtered CO2 intensity 236 Countries.xlsx"),
                os.path.join(desk,"Filtered CO2 intensity 236 Countries.csv")]
    co2_path = resolve_desktop_file(args.excel or args.countries_co2_file, defaults)
    power = args.device_power_watts if args.device_power_watts is not None else (args.power_watts if args.power_watts is not None else 45.0)
    if co2_path and os.path.isfile(co2_path):
        co2 = load_country_co2(co2_path, year_select=args.year_select)
        carb_all = compute_emissions(perf_df, co2, power, args.pue)
        wc, meta = worstcase_country_table(perf_df, co2, power, args.pue, dim_key="grid_dim", step_key="steps")
        write_excel({"emissions_all":carb_all, "worst_case_table":wc, "meta":pd.DataFrame([meta])},
                    os.path.join(carb_dir,"carbon_footprint.xlsx"))
        # plots
        fig, ax = plt.subplots(figsize=(7,5)); ax.hist(wc["kgCO2e"], bins=30)
        ax.set_xlabel("kg CO2e"); ax.set_title("Carbon distribution (worst-case)"); 
        save_fig(fig, os.path.join(carb_dir,"carbon_distribution.png"))
    else:
        write_excel({"info":pd.DataFrame([{"note":"CO2 file not provided; carbon skipped."}])},
                    os.path.join(carb_dir,"carbon_placeholder.xlsx"))

    print(f"[DONE] All done in {time.perf_counter()-t0_all:.2f}s")

if __name__=="__main__":
    main()
