#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# co2_from_bench_cli.py
# Empirical evaluation of Shor's algorithm
# ----------------------------------------------------------------------------------------------------------------
# Compute per-country CO2 emissions from a benchmark CSV + grid-intensity Excel.
# CLI aligned with the following setup:
# --device-power-watts, 
#--pue, 
#--year-select, 
#--outdir, 
#--combine

# Code focuses on Carbon Footprints/Emissions

# ----------------------------------------------------------------------------------------------------------------



# ------------********------------ Imports

import argparse
import os
import sys
from datetime import datetime, timezone
import pandas as pd

TRY_PLOT = True
try:
    import matplotlib.pyplot as plt
except Exception:
    TRY_PLOT = False                     # Disable plotting if matplotlib is unavailable

COUNTRY_CANDIDATES = [
    "Country", "country", "Nation", "nation", "Region", "region",
    "NAME", "Country Name", "Entity"
]                                        # Possible column names for country identification

INTENSITY_CANDIDATES = [
    "kgCO2/kWh", "kgCO2_per_kWh", "kgCO2 per kWh", "Intensity_kgCO2_per_kWh",
    "gCO2/kWh",  "gCO2_per_kWh",  "gCO2 per kWh",  "Intensity_gCO2_per_kWh",
    "tCO2/MWh",  "tCO2_per_MWh",  "tCO2 per MWh",  "Intensity_tCO2_per_MWh",
]                                        # Possible column names for carbon intensity

YEAR_COL_CANDIDATES = ["Year", "year", "YEAR"]   # Possible year column labels

def _sanitize_label(s: str) -> str:
    return "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in s.strip())[:60] or "run"                    
       # Clean text for safe filenames/labels


def _find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c                          # Direct match found in column list

    lowmap = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lowmap:
            return lowmap[c.lower()]          # Case-insensitive match

    for col in df.columns:
        lc = col.lower().replace(" ", "").replace("_", "")
        for cand in candidates:
            if cand.lower().replace(" ", "").replace("_", "") in lc:
                return col                    # Fuzzy match ignoring spacing/underscores

    return None                               # No suitable column found


def _normalize_intensity_to_kg_per_kwh(series, unit_hint=None):
    s = pd.to_numeric(series, errors="coerce")   # Convert to numeric; invalid values -> NaN
    name = (series.name or "").lower().replace(" ", "").replace("_", "")
    def looks(x): return x in name               # Helper for substring detection in column name

    if looks("kgco2") and looks("kwh"): return s          # Already in kgCO2/kWh
    if looks("gco2")  and looks("kwh"): return s / 1000.0 # Convert g→kg
    if (looks("tco2") and looks("mwh")) or ("ton" in name and "mwh" in name):
        return s * 1.0                                   # tCO2/MWh ≈ kgCO2/kWh numerically

    # Fallback: infer from provided unit hint
    if unit_hint:
        u = unit_hint.lower()
        if "kg" in u and "kwh" in u: return s
        if "g"  in u and "kwh" in u: return s / 1000.0
        if ("t" in u and "mwh" in u) or ("ton" in u and "mwh" in u): return s * 1.0

    return s  # If unsure, leave unchanged

def _select_year_from_wide(df, prefer="latest"):
    country_col = _find_col(df, COUNTRY_CANDIDATES)
    if country_col is None:
        return None
    year_cols = []
    for c in df.columns:
        if c == country_col: continue
        try:
            y = int(str(c).strip())
            if 1800 <= y <= 2100:
                year_cols.append((y, c))
        except Exception:
            pass
    if not year_cols:
        return None
    year_cols.sort(key=lambda t: t[0])
    all_years = [y for y, _ in year_cols]
    if str(prefer).lower() == "latest":
        chosen_year = all_years[-1]
    else:
        try:
            req = int(str(prefer))
            chosen_year = req if req in all_years else all_years[-1]
        except Exception:
            chosen_year = all_years[-1]
    colname = [c for y, c in year_cols if y == chosen_year][0]
    narrow = df[[country_col, colname]].copy()
    narrow.columns = [country_col, str(chosen_year)]
    return narrow, chosen_year, colname, country_col

def _select_year_from_long(df, prefer="latest"):
    country_col = _find_col(df, COUNTRY_CANDIDATES)
    year_col = _find_col(df, YEAR_COL_CANDIDATES)
    if country_col is None or year_col is None:
        return None                                      # Cannot proceed without country/year info

    intensity_col = _find_col(df, INTENSITY_CANDIDATES)
    if intensity_col is None:
        # Fallback: auto-detect first numeric column (other than country/year)
        num_cols = [c for c in df.columns if c not in (country_col, year_col)]
        numeric = [c for c in num_cols if pd.api.types.is_numeric_dtype(df[c])]
        if not numeric:
            return None
        intensity_col = numeric[0]

    years = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)  # Clean numeric year values
    if years.empty:
        return None

    # Select desired year (explicit or latest available)
    if str(prefer).lower() == "latest":
        chosen_year = years.max()
    else:
        try:
            req = int(str(prefer))
            chosen_year = req if req in set(years) else years.max()
        except Exception:
            chosen_year = years.max()

    sub = df[df[year_col].astype(str) == str(chosen_year)][[country_col, intensity_col]].copy()
    sub.columns = [country_col, str(chosen_year)]          # Rename intensity column to chosen year label
    return sub, chosen_year, intensity_col, country_col


def _load_excel_sheet(path, sheet_name=None):
    
    try:
        xf = pd.ExcelFile(path)               # Attempt to open the Excel workbook
    except Exception as e:
        print(f"[ERROR] Unable to open Excel: {e}")
        sys.exit(1)                           # Exit if file cannot be opened

    chosen = sheet_name if sheet_name is not None else xf.sheet_names[0]  # Default to first sheet
    if chosen not in xf.sheet_names:
        print(f"[ERROR] Sheet '{chosen}' not found. Available sheets: {', '.join(xf.sheet_names)}")
        sys.exit(1)                           # Exit if requested sheet does not exist

    df = xf.parse(chosen)                     # Load the chosen sheet into a DataFrame
    return df, chosen

# ------------********------------ Main

def main():
    ap = argparse.ArgumentParser(description="Compute CO2 emissions per country from benchmark results.")
    ap.add_argument("--results-csv", default="gnfs_benchmark_results.csv")
    ap.add_argument("--countries-xlsx", required=True)
    ap.add_argument("--sheet", default=None, help="Worksheet name (defaults to first sheet).")
    ap.add_argument("--year-select", default="latest")
    ap.add_argument("--device-power-watts", type=float, default=25.0)   # Power draw used for energy→CO2 estimation
    ap.add_argument("--pue", type=float, default=1.0)                   # Power Usage Effectiveness (datacenter overhead)
    ap.add_argument("--outdir", default="carbon_by_country")
    ap.add_argument("--combine", action="store_true")                   # Aggregate CO₂ over all runs
    ap.add_argument("--label", default="Classical GNFS (YAFU)")
    ap.add_argument("--country-col", dest="country_col", default=None)
    ap.add_argument("--intensity-col", dest="intensity_col", default=None)
    ap.add_argument("--unit-hint", default=None)
    ap.add_argument("--no-plots", action="store_true")                  # Disable plot generation
    args = ap.parse_args()

    label_sane = _sanitize_label(args.label)                            # Safe label for filenames
    os.makedirs(args.outdir, exist_ok=True)                             # Ensure output directory exists


    # ------------********------------ Load benchmark
    if not os.path.isfile(args.results_csv):
        print(f"[ERROR] results CSV not found: {args.results_csv}")
        sys.exit(1)

    df = pd.read_csv(args.results_csv)                 # Load benchmark results

    if "Status" in df.columns:
        df = df[df["Status"] == "ok"].copy()           # Keep only successful runs

    time_col = next((c for c in ["Wall_Time_s", "wall_time_s", "Elapsed_s"]
                     if c in df.columns), None)        # Detect wall-time column
    if time_col is None:
        print("[ERROR] Could not find wall-time column (expected 'Wall_Time_s').")
        sys.exit(1)

    total_wall_s = pd.to_numeric(df[time_col], errors="coerce").fillna(0).sum()  # Total wall-clock time (OK runs)

    total_kwh = (args.device_power_watts * total_wall_s) / 3_600_000.0           # Convert W·s → kWh
    total_kwh *= args.pue                                                         # Apply PUE factor

    # ------------********------------ Load Excel (always one DataFrame)
    if not os.path.isfile(args.countries_xlsx):
        print(f"[ERROR] countries Excel not found: {args.countries_xlsx}")
        sys.exit(1)

    cdf, actual_sheet = _load_excel_sheet(args.countries_xlsx, args.sheet)   # Load CO₂ intensity data
    print(f"[INFO] Using sheet: {actual_sheet}")

    chosen_year = None          # Will store the selected intensity year
    chosen_country_col = None   # Will store the detected country column name


    # ------------********------------ Explicit overrides
    if args.country_col or args.intensity_col:
        country_col = args.country_col or _find_col(cdf, COUNTRY_CANDIDATES)
        intensity_col = args.intensity_col or _find_col(cdf, INTENSITY_CANDIDATES)
        if not country_col or not intensity_col:
            print("[ERROR] Could not resolve country/intensity columns with overrides.")
            sys.exit(1)
        chosen_country_col = country_col
        narrow = cdf[[country_col, intensity_col]].copy()
        narrow.columns = [country_col, "INTENSITY"]
    else:
        # Try "long" format first
        long_sel = _select_year_from_long(cdf, prefer=args.year_select)
        if long_sel is not None:
            narrow, chosen_year, intensity_orig_col, chosen_country_col = long_sel
            narrow.columns = [chosen_country_col, "INTENSITY"]
        else:
            # Try "wide" format
            wide_sel = _select_year_from_wide(cdf, prefer=args.year_select)
            if wide_sel is None:
                # Fallback: country + intensity column candidates
                country_col = _find_col(cdf, COUNTRY_CANDIDATES)
                intensity_col = _find_col(cdf, INTENSITY_CANDIDATES)
                if country_col and intensity_col:
                    chosen_country_col = country_col
                    narrow = cdf[[country_col, intensity_col]].copy()
                    narrow.columns = [country_col, "INTENSITY"]
                else:
                    print("[ERROR] Could not detect layout (no Year column and no year-named columns).")
                    sys.exit(1)
            else:
                narrow_wide, chosen_year, intensity_orig_col, chosen_country_col = wide_sel
                narrow = narrow_wide.copy()
                narrow.columns = [chosen_country_col, "INTENSITY"]

    # Normalize units and compute emissions
    narrow["INTENSITY"] = _normalize_intensity_to_kg_per_kwh(
        narrow["INTENSITY"], unit_hint=args.unit_hint
    )   # Convert intensity column to kgCO₂/kWh if needed

    out = pd.DataFrame({
        "Timestamp": datetime.now(timezone.utc).isoformat(),
        "Label": args.label,
        "Assumed_Power_W": args.device_power_watts,
        "PUE": args.pue,
        "Year": chosen_year if chosen_year is not None else "",
        "Total_Wall_Time_s": total_wall_s,            # Total time across all valid runs
        "Total_Energy_kWh": total_kwh,                # Energy consumption after PUE applied
        "Country": narrow[chosen_country_col].astype(str),
        "Intensity_kgCO2_per_kWh": pd.to_numeric(narrow["INTENSITY"], errors="coerce"),
    })

    out["Emissions_kgCO2"] = (out["Intensity_kgCO2_per_kWh"] * out["Total_Energy_kWh"])   # Calculate emissions per country
    out.sort_values("Emissions_kgCO2", ascending=False, inplace=True)  # Rank countries by total emissions

    # Write per-run CSV
    out_csv = os.path.join(args.outdir, f"co2_by_country_{label_sane}.csv")
    out.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved → {out_csv}")
    print(f"Total energy (kWh) with PUE={args.pue}: {total_kwh:.6f} kWh")

    # Append into combined file if requested
    if args.combine:
        comb_path = os.path.join(args.outdir, "combined_emissions.csv")
        if os.path.exists(comb_path):
            prev = pd.read_csv(comb_path)
            comb = pd.concat([prev, out], ignore_index=True)
        else:
            comb = out.copy()
        comb.to_csv(comb_path, index=False, encoding="utf-8")
        print(f"Updated combined file → {comb_path}")

    # ------------********------------ Plot (optional)
    if (not args.no_plots) and TRY_PLOT:
        try:
            top = out.head(20)                          # Select top 20 highest-emission countries
            plt.figure()
            plt.barh(top["Country"], top["Emissions_kgCO2"])   # Horizontal bar chart of emissions
            plt.xlabel("Emissions (kg CO₂)")
            plt.title(f"CO₂ by Country — {args.label}")
            plt.gca().invert_yaxis()                   # Largest emitter on top
            plt.tight_layout()
            fig_path = os.path.join(
                args.outdir,
                f"co2_by_country_top20_{label_sane}.png"
            )
            plt.savefig(fig_path, dpi=200)              # Save plot to output directory
            print(f"Saved → {fig_path}")
        except Exception as e:
            print(f"[WARN] Plotting failed: {e}")

if __name__ == "__main__":
    main()                                              # Script entry point
