#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------------------------------------------
# quantum_pendulum_tomography_noiseless.py
# Homodyne-style (noiseless) quantum tomography of a single bosonic mode.
# Reconstruct Wigner W(x,p) from quadrature histograms via filtered backprojection.
# States: coherent (x0,p0) and single-mode squeezed (r, phi) displacement.
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
# -------------- shared helpers --------------
# ------------********------------

#This helper ensures that a target directory exists before writing any outputs.
#It creates the directory if needed and safely ignores the case where it already exists.

def ensure_dir(p): os.makedirs(p, exist_ok=True)

#This function writes multiple pandas DataFrames into a single Excel workbook.
#If Excel export fails, it gracefully falls back to writing each table as a CSV file.
def write_excel(tables, path):
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as w:
            for k,df in tables.items(): df.to_excel(w, index=False, sheet_name=(k[:31] or "Sheet1"))
        print(f"[OK] Wrote Excel: {path}")
    except Exception as e:
        print(f"[WARN] Excel write failed ({e}); CSV fallback.")
        base = os.path.splitext(path)[0]
        for k,df in tables.items():
            csvp = f"{base}__{k}.csv"; df.to_csv(csvp, index=False); print(f"[OK] {csvp}")
            
            #This utility saves a Matplotlib figure with consistent layout and resolution.
#It closes the figure after saving to free resources and reports the output path.


def save_fig(fig, path, dpi=150): fig.tight_layout(); fig.savefig(path,dpi=dpi,bbox_inches="tight"); plt.close(fig); print(f"[OK] Saved plot: {path}")


#This function resolves a file path by checking user input and Desktop locations.
#It supports quoted paths, missing extensions, and a list of default fallback filenames.

def resolve_desktop_file(arg, defaults):
    desk = os.path.join(os.environ.get("USERPROFILE",""), "Desktop")
    def exists(x): return os.path.isfile(x)
    if arg:
        base = arg.strip().strip('"').strip("'")
        if exists(base): return os.path.abspath(base)
        cand = os.path.join(desk, base); 
        if exists(cand): return os.path.abspath(cand)
        if not os.path.splitext(base)[1]:
            for ext in (".xlsx",".xls",".csv"):
                cand = os.path.join(desk, base+ext)
                if exists(cand): return os.path.abspath(cand)
        tail = os.path.split(base)[1].lower()
        for fn in os.listdir(desk):
            if fn.lower()==tail or any((not os.path.splitext(base)[1]) and fn.lower()==tail+e for e in (".xlsx",".xls",".csv")):
                cand = os.path.join(desk, fn)
                if exists(cand): return os.path.abspath(cand)
        return None
    for c in defaults:
        if exists(c): return os.path.abspath(c)
    return None

# ------------********------------ carbon helpers --------------
#This function loads country-level CO₂ intensity data from a CSV or Excel file.
#It infers relevant columns, normalizes units to kg CO₂ per kWh, and optionally selects the latest year.

def load_country_co2(path, year_select="latest"):
    if path.lower().endswith(".csv"): df = pd.read_csv(path)
    else: df = pd.read_excel(path)
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
    if np.nanmedian(out["intensity_raw"])>10.0: out["kg_per_kwh"]=out["intensity_raw"]/1000.0
    else: out["kg_per_kwh"]=out["intensity_raw"]
    out["country"]=out["country"].astype(str).str.strip().str.title()
    if yname and year_select=="latest":
        out = out.sort_values(["country","year"]).groupby("country", as_index=False).tail(1)
    return out[["country","kg_per_kwh"]+(["year"] if yname else [])].drop_duplicates()

#This function computes carbon emissions by combining runtime-derived energy use with CO₂ intensity data.
#It applies power and PUE scaling, then cross-joins runs with countries to estimate emissions.


def compute_emissions(perf_df, co2_df, power_watts, pue):
    perf = perf_df.copy()
    perf["energy_kwh"] = (perf["runtime_sec"]*power_watts)/3.6e6
    perf["energy_kwh"] *= pue
    co2 = co2_df.rename(columns={"kg_per_kwh":"kgco2_per_kwh"}).copy()
    perf["_k"]=1; co2["_k"]=1
    j = perf.merge(co2, on="_k").drop(columns="_k")
    j["emissions_kgCO2"] = j["energy_kwh"]*j["kgco2_per_kwh"]
    return j
    
#This function builds a worst-case emissions table based on the heaviest benchmark configuration.
#It estimates energy use from the slowest setting and ranks countries by resulting CO₂ emissions.
def worstcase_country_table(perf_df, co2_df, power_watts, pue, step_key="steps", dim_key="grid_dim"):
    heavy_dim = int(perf_df[dim_key].max()); heavy_S = int(perf_df[step_key].max())
    rt = float(perf_df[(perf_df[dim_key]==heavy_dim)&(perf_df[step_key]==heavy_S)]["runtime_sec"].mean())
    kwh = rt*power_watts/3.6e6 * pue
    tbl = co2_df.copy()
    if "year" not in tbl.columns: tbl["year"]=pd.NA
    out = pd.DataFrame({"Country":tbl["country"], "Year":tbl["year"], "Intensity":tbl["kg_per_kwh"], "kWh":kwh})
    out["kgCO2e"]=out["Intensity"]*out["kWh"]
    out = out.sort_values("kgCO2e", ascending=False).reset_index(drop=True)
    return out, {"grid_dim":heavy_dim, "steps":heavy_S, "mean_runtime_s":rt, "kWh_used":kwh}

# -------------- tomography core --------------
#This helper generates S evenly spaced angles in the interval [0, π).
#These angles are typically used as quadrature measurement directions.

def make_angles(S): return np.linspace(0.0, np.pi, S, endpoint=False)
# units: vacuum Var(X)=0.5

#This function computes the mean and variance of a quadrature for a squeezed Gaussian state.
#It accounts for displacement, squeezing strength, and squeezing angle.

def quad_mean_var(theta, x0, p0, r, phi):
    mu = x0*np.cos(theta) + p0*np.sin(theta)
    # squeezing variance along angle phi
    var = 0.5*( np.exp(-2*r)*np.cos(theta - phi)**2 + np.exp(2*r)*np.sin(theta - phi)**2 )
    return mu, var

#This function draws random quadrature samples from a Gaussian distribution.
#The distribution parameters are determined by the quadrature angle and squeezing properties.

def sample_quadrature(theta, shots, x0, p0, r, phi, rng):
    mu, var = quad_mean_var(theta, x0, p0, r, phi)
    return rng.normal(loc=mu, scale=np.sqrt(var), size=shots)

#This function computes the analytic Wigner function of a displaced squeezed Gaussian state.
#It evaluates the distribution on a phase-space grid and enforces proper normalization.

def wigner_truth(grid_x, grid_p, x0, p0, r, phi):
    # analytic Wigner for displaced squeezed Gaussian (no rotation of displacement)
    X, P = np.meshgrid(grid_x, grid_p, indexing="xy")
    # rotate coordinates by phi to apply squeezing along x'
    c, s = np.cos(phi), np.sin(phi)
    Xp =  c*X + s*P
    Pp = -s*X + c*P
    sigx2 = 0.5*np.exp(-2*r); sigp2 = 0.5*np.exp(2*r)
    # shift displacement (x0,p0) into rotated frame
    X0 =  c*x0 + s*p0
    P0 = -s*x0 + c*p0
    expo = -((Xp-X0)**2/sigx2 + (Pp-P0)**2/sigp2)
    W = (1.0/np.pi)*np.exp(expo)
    # ensure normalization ∫ W dx dp = 1 (should hold)
    dx = grid_x[1]-grid_x[0]; dp = grid_p[1]-grid_p[0]
    sarea = np.sum(W)*dx*dp
    if sarea>0: W /= sarea
    return W

#This helper applies a ramp filter in the frequency domain to a 1D density.
#It is commonly used in filtered backprojection for tomographic reconstruction.

def ramp_filter_1d(dens, dx):
    n = len(dens); F = np.fft.rfft(dens)
    freqs = np.fft.rfftfreq(n, d=dx)
    Ff = F*np.abs(freqs)
    back = np.fft.irfft(Ff, n=n)
    return back

#This function reconstructs a 2D phase-space distribution using filtered backprojection.
#It interpolates projected densities along rotated axes and normalizes the result.

def backproject(angles, proj_dens_list, t_centers, grid_x, grid_p, use_filter=True):
    Gx, Gp = len(grid_x), len(grid_p)
    X, P = np.meshgrid(grid_x, grid_p, indexing="xy")
    recon = np.zeros((Gp,Gx), dtype=float)
    dt = t_centers[1]-t_centers[0]
    dtheta = (np.pi/len(angles)) if len(angles)>0 else 0.0
    for th, dens in zip(angles, proj_dens_list):
        d = ramp_filter_1d(dens, dt) if use_filter else dens
        T = X*np.cos(th) + P*np.sin(th)  # optical scaling (m=ω=1)
        vals = np.interp(T.ravel(), t_centers, d, left=0.0, right=0.0).reshape(T.shape)
        recon += vals*dtheta
    # normalize
    dx = grid_x[1]-grid_x[0]; dp = grid_p[1]-grid_p[0]
    s = np.sum(recon)*dx*dp
    if s>0: recon /= s
    return recon

#This routine runs a complete tomography experiment for a single parameter setting.
#It samples quadratures, reconstructs the Wigner function, and compares it to the analytic truth.
#Error metrics such as MSE and L1 distance are returned along with reconstruction data.

def run_once(S, shots, G, x0, p0, r, phi, rng, lim):
    angles = make_angles(S)
    # choose t support
    tmax = lim*np.max(np.abs(np.cos(angles))) + lim*np.max(np.abs(np.sin(angles)))
    B = max(256, 2*G)
    t_edges = np.linspace(-tmax, tmax, B+1); t_centers = 0.5*(t_edges[:-1]+t_edges[1:])
    proj = []
    for th in angles:
        x = sample_quadrature(th, shots, x0, p0, r, phi, rng)
        hist, _ = np.histogram(x, bins=t_edges)
        dens = hist/(shots*(t_edges[1]-t_edges[0]))
        proj.append(dens)
    gx = np.linspace(-lim, lim, G); gp = np.linspace(-lim, lim, G)
    recon = backproject(angles, proj, t_centers, gx, gp, use_filter=True)
    truth = wigner_truth(gx, gp, x0, p0, r, phi)
    mse = float(np.mean((recon-truth)**2))
    l1  = float(np.sum(np.abs(recon-truth))*(gx[1]-gx[0])*(gp[1]-gp[0]))
    return dict(recon=recon, truth=truth, gx=gx, gp=gp, mse=mse, l1=l1, S=S, shots=shots, grid=G)

# ------------********------------
def main():
    ap = argparse.ArgumentParser(description="Quantum (noiseless) pendulum/optical tomography with four benchmarks + carbon.")
    ap.add_argument("--n-min", type=int, default=2)   # parity placeholders
    ap.add_argument("--n-max", type=int, default=6)
    ap.add_argument("--steps", type=str, default="8,16,32", help="angles S in [0,π)")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--shots-per-angle", type=int, default=1500)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--state", choices=["coherent","squeezed"], default="coherent")
    ap.add_argument("--x0", type=float, default=0.8)   # displacement (x0,p0)
    ap.add_argument("--p0", type=float, default=-0.5)
    ap.add_argument("--squeeze-r", type=float, default=0.0)
    ap.add_argument("--squeeze-phi", type=float, default=0.0)
    ap.add_argument("--lim", type=float, default=3.0, help="grid extends to ±lim in x and p")
    # carbon + IO
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
    r = args.squeeze_r if args.state=="squeezed" else 0.0
    phi = args.squeeze_phi if args.state=="squeezed" else 0.0
    x0, p0 = args.x0, args.p0

    root = os.path.abspath(args.outdir)
    perf_dir = os.path.join(root, "Performance")
    scal_dir = os.path.join(root, "Scalability")
    reli_dir = os.path.join(root, "Reliability")
    carb_dir = os.path.join(root, "Carbon")
    for d in (perf_dir, scal_dir, reli_dir, carb_dir): ensure_dir(d)

    rows_p, rows_r = [], []
    t0_all = time.perf_counter()
    for S in S_list:
        for rep in range(args.repeats):
            tracemalloc.start(); t0 = time.perf_counter()
            out = run_once(S, shots, G, x0, p0, r, phi, rng, args.lim)
            runtime = time.perf_counter()-t0
            _, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()

            if rep==0:
                fig, ax = plt.subplots(1,3, figsize=(13,4))
                im0=ax[0].imshow(out["truth"], extent=[-args.lim,args.lim,-args.lim,args.lim], origin="lower"); ax[0].set_title("Truth Wigner")
                im1=ax[1].imshow(out["recon"], extent=[-args.lim,args.lim,-args.lim,args.lim], origin="lower"); ax[1].set_title(f"Reconstruction (S={S})")
                im2=ax[2].imshow(out["recon"]-out["truth"], extent=[-args.lim,args.lim,-args.lim,args.lim], origin="lower"); ax[2].set_title("Difference")
                for a in ax: a.set_xlabel("x"); a.set_ylabel("p")
                for im, a in zip([im0,im1,im2], ax): plt.colorbar(im, ax=a)
                save_fig(fig, os.path.join(reli_dir, f"wigner_maps_S{S}.png"))

            rows_p.append({"grid_dim":G, "steps":S, "repeat":rep,
                           "runtime_sec":runtime, "peak_mem_mb":peak/(1024**2)})
            rows_r.append({"grid_dim":G, "steps":S, "repeat":rep,
                           "mse":out["mse"], "l1_distance":out["l1"]})

    perf_df = pd.DataFrame(rows_p); rel_df = pd.DataFrame(rows_r)

    perf_sum = perf_df.groupby(["grid_dim","steps"], as_index=False).agg(
        mean_runtime_sec=("runtime_sec","mean"),
        std_runtime_sec=("runtime_sec","std"),
        mean_peak_mem_mb=("peak_mem_mb","mean"),
        std_peak_mem_mb=("peak_mem_mb","std")
    )
    write_excel({"raw":perf_df, "summary":perf_sum}, os.path.join(perf_dir,"performance.xlsx"))

# ------------********------------ perf plots
    fig, ax = plt.subplots(figsize=(7,5))
    sfit = perf_df.groupby("steps", as_index=False)["runtime_sec"].mean()
    ax.plot(sfit["steps"], sfit["runtime_sec"], marker="o")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Steps S"); ax.set_ylabel("Runtime (s)"); ax.set_title("Scalability: runtime vs S")
    save_fig(fig, os.path.join(scal_dir,"runtime_vs_steps.png"))

# ------------********------------ scal fit table
    xb = np.log(sfit["steps"].values); yb = np.log(sfit["runtime_sec"].values+1e-12)
    beta, a0 = np.polyfit(xb, yb, 1)
    write_excel({"scaling_fit":pd.DataFrame({"fit_parameter":["beta_step_scaling"],"estimate":[beta],"note":[f"G={args.grid}"]})},
                os.path.join(scal_dir,"scalability.xlsx"))

# ------------********------------ reliability
    rel_sum = rel_df.groupby(["grid_dim","steps"], as_index=False).agg(
        mean_mse=("mse","mean"), std_mse=("mse","std"),
        mean_l1=("l1_distance","mean"), std_l1=("l1_distance","std")
    )
    write_excel({"raw":rel_df, "summary":rel_sum}, os.path.join(reli_dir,"reliability.xlsx"))

    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(rel_sum["steps"], rel_sum["mean_mse"], marker="o")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Steps S"); ax.set_ylabel("Mean MSE (log)"); ax.set_title("Reliability: Wigner error vs S")
    save_fig(fig, os.path.join(reli_dir,"mse_vs_steps.png"))

 # ------------********------------ carbon
    desk = os.path.join(os.environ.get("USERPROFILE",""), "Desktop")
    defaults = [os.path.join(desk,"Filtered CO2 intensity 236 Countries.xlsx"),
                os.path.join(desk,"Filtered CO2 intensity 236 Countries.csv")]
    co2_path = resolve_desktop_file(args.excel or args.countries_co2_file, defaults)
    power = args.device_power_watts if args.device_power_watts is not None else (args.power_watts if args.power_watts is not None else 45.0)
    if co2_path and os.path.isfile(co2_path):
        co2 = load_country_co2(co2_path, year_select=args.year_select)
        carb_all, (wc, meta) = compute_emissions(perf_df, co2, power, args.pue), worstcase_country_table(perf_df, co2, power, args.pue)
        write_excel({"emissions_all":carb_all, "worst_case_table":wc, "meta":pd.DataFrame([meta])},
                    os.path.join(carb_dir,"carbon_footprint.xlsx"))
        fig, ax = plt.subplots(figsize=(7,5)); ax.hist(wc["kgCO2e"], bins=30)
        ax.set_xlabel("kg CO2e"); ax.set_title("Carbon distribution (worst-case)")
        save_fig(fig, os.path.join(carb_dir,"carbon_distribution.png"))
    else:
        write_excel({"info":pd.DataFrame([{"note":"CO2 file not provided; carbon skipped."}])},
                    os.path.join(carb_dir,"carbon_placeholder.xlsx"))

    print(f"[DONE] Ideal quantum tomography completed in {time.perf_counter()-t0_all:.2f}s")

if __name__=="__main__":
    main()
