#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import re

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely import affinity

# =====================================================
# Paths (relative to repository root)
# =====================================================
ROOT = Path(__file__).resolve().parents[1]

SHP_PATH = ROOT / "data" / "LSOA_joined_remodeling_noShapeArea.shp"
CSV_PATH = ROOT / "data" / "LSOA_joined_remodeling_rate.csv"
COL_TXT_PATH = ROOT / "data" / "satscan_input_IA" / "IA.col.txt"

# Output directory specifically for OCI results
OUT_DIR = ROOT / "outputs" / "OCI" / "IA"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "OCI_results.csv"

# =====================================================
# Settings
# =====================================================
GROUP_VARS = [
    "indicator_sex_female",
    "indicator_Ethnic_nonwhite",
    "indicator_education_noqualification",
    "indicator_age_elder",
    "indicator_age_children",
    "indicator_NS_SeC_unemployed",
    "indicator_NS_SeC_students",
]

JOIN_KEY = "LSOA21CD"
POP_COL = "POPULATION"

# =====================================================
# Helpers
# =====================================================
def create_circle(x: float, y: float, radius: float, resolution: int = 100):
    """
    Create a shapely Polygon representing a circle from SaTScan output.
    """
    return affinity.scale(Point(x, y).buffer(1, resolution=resolution), radius, radius)

def normalize_population_column(gdf: gpd.GeoDataFrame, pop_col: str) -> gpd.GeoDataFrame:
    """
    Fix duplicate population columns (e.g., POPULATION_x, POPULATION_y) resulting from merges.
    """
    x, y = f"{pop_col}_x", f"{pop_col}_y"
    if x in gdf.columns and y in gdf.columns:
        # Check if columns are identical (filling NaNs with -1 for comparison)
        same = (gdf[x].fillna(-1) == gdf[y].fillna(-1)).all()
        if same:
            gdf = gdf.rename(columns={x: pop_col}).drop(columns=[y])
        else:
            # If not same, default to keeping 'y' (usually from the CSV)
            gdf = gdf.drop(columns=[x]).rename(columns={y: pop_col})
    elif x in gdf.columns:
        gdf = gdf.rename(columns={x: pop_col})
    elif y in gdf.columns:
        gdf = gdf.rename(columns={y: pop_col})
    return gdf

# =====================================================
# Main Execution
# =====================================================
def main():
    print(f"Running OCI calculation for IP...")
    print(f"Reading shapefile: {SHP_PATH}")
    gdf = gpd.read_file(SHP_PATH)
    
    print(f"Reading CSV: {CSV_PATH}")
    df_csv = pd.read_csv(CSV_PATH)

    # Merge CSV data onto Shapefile
    cols_to_merge = [JOIN_KEY, POP_COL] + GROUP_VARS
    gdf = gdf.merge(df_csv[cols_to_merge], on=JOIN_KEY, how="left")
    gdf = normalize_population_column(gdf, POP_COL)

    # Fill NaNs for calculation safety
    gdf[POP_COL] = gdf[POP_COL].fillna(0)
    for v in GROUP_VARS:
        gdf[v] = gdf[v].fillna(0)

    # ------------------------------------------------
    # Read and parse SaTScan cluster file (.col.txt)
    # ------------------------------------------------
    print(f"Parsing SaTScan clusters from: {COL_TXT_PATH}")
    cluster_rows = []
    with open(COL_TXT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            # Skip empty lines or headers/comments if necessary, 
            # here we split by multiple spaces
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) < 15:
                continue
            
            # Indices based on standard SaTScan .col.txt format
            # ID=0, Lat=1, Long=2, X=?, Y=?... Assuming parts match your logic:
            # ID(0), ..., X(2), Y(3), Radius(4), ..., RR(14)
            try:
                cluster_rows.append(
                    {
                        "ClusterID": int(parts[0]),
                        "X": float(parts[2]),
                        "Y": float(parts[3]),
                        "Radius": float(parts[4]),
                        "RelativeRisk": float(parts[14]),
                    }
                )
            except (ValueError, IndexError):
                continue

    df_cluster = pd.DataFrame(cluster_rows)
    if df_cluster.empty:
        print("No clusters found or file format issue.")
        return

    # Create geometry for clusters
    df_cluster["geometry"] = df_cluster.apply(
        lambda r: create_circle(r["X"], r["Y"], r["Radius"]),
        axis=1,
    )
    gdf_cluster = gpd.GeoDataFrame(df_cluster, geometry="geometry", crs=gdf.crs)

    # ------------------------------------------------
    # Identify Low-Risk Union
    # ------------------------------------------------
    low_risk = gdf_cluster[gdf_cluster["RelativeRisk"] <= 1]
    
    # Use union_all() for modern shapely/geopandas, or unary_union for older
    if not low_risk.empty:
        try:
            low_union = low_risk.geometry.union_all()
        except AttributeError:
            low_union = low_risk.geometry.unary_union
        
        # Flag LSOAs that intersect with the low-risk union
        gdf["is_low_rep"] = gdf.geometry.intersects(low_union).astype(int)
    else:
        print("Warning: No low-risk clusters (RR <= 1) found.")
        gdf["is_low_rep"] = 0
        low_union = None

    # ------------------------------------------------
    # Compute OCI (Over-representation Composition Index)
    # ------------------------------------------------
    # OCI = (Count_in_Low / Pop_in_Low) / (Count_Total / Pop_Total)
    
    ori_results = []
    low_mask = gdf["is_low_rep"] == 1
    
    # Total population in the Low-Risk areas vs Global
    low_pop_sum = gdf.loc[low_mask, POP_COL].sum()
    total_pop_sum = gdf[POP_COL].sum()

    print(f"Low-risk area population: {low_pop_sum}")
    print(f"Total area population:    {total_pop_sum}")

    for var in GROUP_VARS:
        group_name = var.replace("indicator_", "").replace("_", " ").title()

        # Calculate absolute count of people in this group per LSOA
        # (Indicator is likely a rate/percentage [0-1] or [0-100]? 
        #  If indicator is 0-1 rate: count = rate * pop)
        # Assuming indicator is 0-1 rate based on typical usage. 
        # If it is percentage 0-100, ensure consistency.
        
        count_col = f"{var}_count"
        gdf[count_col] = gdf[var] * gdf[POP_COL]

        # Prevalence in Low Risk Area
        if low_pop_sum > 0:
            p_low = gdf.loc[low_mask, count_col].sum() / low_pop_sum
        else:
            p_low = np.nan

        # Prevalence in Total Area
        if total_pop_sum > 0:
            p_total = gdf[count_col].sum() / total_pop_sum
        else:
            p_total = np.nan

        # Calculate OCI
        if pd.notna(p_low) and pd.notna(p_total) and p_total > 0:
            oci = p_low / p_total
        else:
            oci = np.nan
        
        ori_results.append((group_name, oci))

    # ------------------------------------------------
    # Output
    # ------------------------------------------------
    print("\nOCI Results (Group Prevalence in Low-Risk / Global Prevalence):")
    print("-" * 60)
    for group_name, val in ori_results:
        print(f"{group_name: <30}: {val:.4f}" if pd.notna(val) else f"{group_name: <30}: NaN")

    # Save to CSV
    ori_df = pd.DataFrame(ori_results, columns=["Group", "OCI"])
    ori_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\nSaved OCI results to: {OUT_PATH}")

if __name__ == "__main__":
    main()