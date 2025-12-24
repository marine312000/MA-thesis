# %% [markdown]
# this is my new workbook for calculations, functions

# %%
# Import required libraries
import pymrio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
import pickle
from datetime import datetime
import os

# Set display options for better readability
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', '{:.2f}'.format)

print("Libraries imported successfully")
print(f"pymrio version: {pymrio.__version__}")

# %%
# Define data paths
BASE_PATH = Path(r"C:\Users\Marine Riesterer\OneDrive\Desktop\MA Arbeit")
EXIOBASE_PATH = BASE_PATH / "Exiobase3_ixi_latest"
RESULTS_PATH = BASE_PATH / "Results"
POPULATION_FILE = BASE_PATH / "code" / "producerappraoch" / "clean code" / "exiobase3_population_2022_official.csv"

# Create results directory 
RESULTS_PATH.mkdir(exist_ok=True)

# Define analysis parameters
SINGLE_YEAR = 2019  # For detailed single-year analysis
YEARS_TIME_SERIES = list(range(1995, 2023))  # 1995-2022

print(f"Base path: {BASE_PATH}")
print(f"EXIOBASE data path: {EXIOBASE_PATH}")
print(f"Results will be saved to: {RESULTS_PATH}")
print(f"Years for time series: {YEARS_TIME_SERIES[0]}-{YEARS_TIME_SERIES[-1]}")

# %% [markdown]
# DATA LOADING FUNCTIONS

# %%
def load_exiobase_year(year):
    """
    Load EXIOBASE data for a specific year.
    
    Parameters:
    -----------
    year : int
        Year to load (e.g., 2019)
    
    Returns:
    --------
    ixi_data : pymrio object
        Loaded EXIOBASE data with calculated system
    """
    print(f"Loading EXIOBASE data for year {year}...")
    
    # Load the data
    ixi_data = pymrio.parse_exiobase3(
        str(EXIOBASE_PATH / f"IOT_{year}_ixi.zip")
    )
    
    # Calculate the system (Leontief inverse, etc.)
    ixi_data.calc_system()
    
    print(f"✓ Year {year} loaded successfully")
    print(f"  Regions: {len(ixi_data.get_regions())}")
    print(f"  Sectors: {len(ixi_data.get_sectors())}")
    
    return ixi_data



# %% [markdown]
# Population data 

# %%
def load_population_data():
    """
    Load population data for 2022.
    
    Returns:
    --------
    population : pd.Series
        Population by country (EXIOBASE3 codes as index)
    """
    print(f"Loading population data from: {POPULATION_FILE}")
    
    if not POPULATION_FILE.exists():
        print(f"ERROR: Population file not found!")
        return None
    
    # Load CSV
    pop_df = pd.read_csv(POPULATION_FILE)
    print(f"Population data shape: {pop_df.shape}")
    print(f"Columns: {pop_df.columns.tolist()}")
    
    # Display first few rows to understand structure
    print("\nFirst 5 rows:")
    print(pop_df.head())
    
    # Create Series with country codes as index
    if 'EXIOBASE3_Code' in pop_df.columns and 'Population_2022' in pop_df.columns:
        population = pd.Series(
            pop_df['Population_2022'].values,
            index=pop_df['EXIOBASE3_Code']
        )
        return population
    else:
        print("Please check column names and adjust the code!")
        return None

# Load population
population_2022 = load_population_data()

# ===================================================================
# REGION GROUPS
# ===================================================================

EU_COUNTRIES = [ 
    'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 
    'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT', 
    'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK', 'GB'
]
def aggregate_region_in_series(series, group_name, members, drop_members=True):
    """
    Aggregate a list of regions in a Series into a single group row.

    Parameters
    ----------
    series : pd.Series
        Indexed by EXIOBASE region codes (e.g. 'DE', 'FR', ...)
    group_name : str
        Name of the aggregated region (e.g. 'EU')
    members : list of str
        Region codes to aggregate
    drop_members : bool
        If True, drop individual members after aggregation.

    Returns
    -------
    pd.Series
    """
    series = series.copy()
    common = series.index.intersection(members)
    if len(common) == 0:
        return series

    agg_value = series.loc[common].sum()
    if drop_members:
        series = series.drop(common)
    series.loc[group_name] = agg_value
    return series


def aggregate_region_in_df(df, group_name, members, drop_members=True):
    """
    Aggregate a list of regions in a DataFrame (rows) into a single group row.

    Parameters
    ----------
    df : pd.DataFrame
        Rows indexed by region codes.
    group_name : str
        Name of aggregated region.
    members : list of str
        Region codes to aggregate.
    drop_members : bool
        If True, drop individual members after aggregation.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    common = df.index.intersection(members)
    if len(common) == 0:
        return df

    agg_row = df.loc[common].sum(axis=0)
    if drop_members:
        df = df.drop(common)
    df.loc[group_name] = agg_row
    return df

# %% [markdown]
# GHG EMISSIONS

# %%
# Define greenhouse gas categories with IPCC AR6 GWP100 factors
EMISSION_CATEGORIES = {
    # CO2 emissions (GWP = 1.0)
    'CO2 - combustion - air': 1.0,
    'CO2 - waste - fossil - air': 1.0,
    'CO2 - agriculture - peat decay - air': 1.0,
    'CO2 - non combustion - Cement production - air': 1.0,
    'CO2 - non combustion - Lime production - air': 1.0,
    
    # CH4 emissions - Combustion (GWP = 27.0)
    'CH4 - combustion - air': 27.0,
    
    # CH4 emissions - Fugitive/Process (GWP = 29.8)
    'CH4 - non combustion - Extraction/production of (natural) gas - air': 29.8,
    'CH4 - non combustion - Extraction/production of crude oil - air': 29.8,
    'CH4 - non combustion - Oil refinery - air': 29.8,
    'CH4 - non combustion - Mining of bituminous coal - air': 29.8,
    'CH4 - non combustion - Mining of coking coal - air': 29.8,
    'CH4 - non combustion - Mining of lignite (brown coal) - air': 29.8,
    'CH4 - non combustion - Mining of sub-bituminous coal - air': 29.8,
    
    # CH4 emissions - Biogenic (GWP = 27.0)
    'CH4 - agriculture - air': 27.0,
    'CH4 - waste - air': 27.0,
    
    # N2O emissions (GWP = 273.0)
    'N2O - combustion - air': 273.0,
    'N2O - agriculture - air': 273.0,
    
    # Industrial gases
    'SF6 - air': 25184.0,
    'HFC - air': 1.0,  # Already in CO2-eq
    'PFC - air': 1.0   # Already in CO2-eq
}

ALL_VALUE_ADDED_COMPONENTS = [
    "Taxes less subsidies on products purchased: Total",     # For sensitivity only!
    "Other net taxes on production",
    "Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled",
    "Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled",
    "Compensation of employees; wages, salaries, & employers' social contributions: High-skilled",
    "Operating surplus: Consumption of fixed capital",    
    "Operating surplus: Rents on land",
    "Operating surplus: Royalties on resources",
    "Operating surplus: Remaining net operating surplus"    
]
# Define value-added components
VALUE_ADDED_COMPONENTS = [
    "Other net taxes on production",
    "Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled",
    "Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled",
    "Compensation of employees; wages, salaries, & employers' social contributions: High-skilled",
    "Operating surplus: Consumption of fixed capital",    
    "Operating surplus: Rents on land",
    "Operating surplus: Royalties on resources",
    "Operating surplus: Remaining net operating surplus"    
]
     
# Extract only profit components (all operating surplus)
PROFIT_COMPONENTS = [
    "Operating surplus: Consumption of fixed capital",    
    "Operating surplus: Rents on land",
    "Operating surplus: Royalties on resources",
    "Operating surplus: Remaining net operating surplus"
]     

TAX_COMPONENTS = [
    "Taxes less subsidies on products purchased: Total",
    "Other net taxes on production"
]

WAGE_COMPONENTS = [
    "Compensation of employees; wages, salaries, & employers' social contributions: Low-skilled",
    "Compensation of employees; wages, salaries, & employers' social contributions: Medium-skilled",
    "Compensation of employees; wages, salaries, & employers' social contributions: High-skilled"
]

print(f"Defined {len(EMISSION_CATEGORIES)} emission categories")
print(f"Defined {len(VALUE_ADDED_COMPONENTS)} total value-added components")
print(f"  - {len(TAX_COMPONENTS)} tax components")
print(f"  - {len(PROFIT_COMPONENTS)} profit components")
print(f"  - {len(WAGE_COMPONENTS)} wage components")


# %% [markdown]
# DATA CLEANING FUNCTIONS

# %%
def calculate_clean_va_coefficients(ixi_data, components_to_use):
    """
    Calculate and clean value-added coefficients.
    
    Parameters:
    -----------
    ixi_data : pymrio object
        Loaded EXIOBASE data
    components_to_use : list
        Which value-added components to include (e.g., VALUE_ADDED_COMPONENTS, PROFIT_COMPONENTS)
    
    Returns:
    --------
    v_clean : np.array
        Cleaned value-added coefficients (one per sector-region)
    v_raw : np.array
        Raw coefficients before cleaning
    """
    print(f"Calculating VA coefficients using {len(components_to_use)} components...")
    
    # Extract factor inputs
    factor_inputs = ixi_data.factor_inputs.F
    
    # Sum the selected components
    VA = factor_inputs.loc[components_to_use].sum(axis=0).values
    
    # Fix 1: Set negative VA to 0 to avoid instability
    neg_count = np.sum(VA < 0)
    VA[VA < 0] = 0
    print(f"  Set {neg_count} negative VA values to 0")
    
    # Calculate total output
    total_output = ixi_data.x.values.flatten()
    
    # Calculate raw coefficients (v = VA / output)
    v_raw = np.divide(VA, total_output, out=np.zeros_like(VA), where=(total_output != 0))
    
    # Copy for cleaning
    v_clean = v_raw.copy()
    
    # Fix 2: Cap coefficients > 1 (prevents over-allocation)
    over_one = np.sum(v_clean > 1)
    v_clean[v_clean > 1] = 1
    print(f"  Capped {over_one} coefficients > 1")
    
    # Fix 3: Set any remaining negative coefficients to 0
    negatives = np.sum(v_clean < 0)
    v_clean[v_clean < 0] = 0
    if negatives > 0:
        print(f"  Set {negatives} negative coefficients to 0")
    
    # Summary statistics
    print(f"  BEFORE cleaning: Min={v_raw.min():.4f}, Max={v_raw.max():.4f}")
    print(f"  AFTER cleaning:  Min={v_clean.min():.4f}, Max={v_clean.max():.4f}")
    
    return v_clean, v_raw

# %% [markdown]
# MAIN RESPONSIBILITY CALCULATION FUNCTIONS

# %%

def calculate_producer_responsibility(ixi_data, emission_categories):
    """
    Calculate producer-based responsibility.
    
    Returns:
    --------
    producer_emissions : np.array
        Emissions in tonnes CO2-eq per sector-region
    missing_emissions : list
        Any emission categories not found in data
    """
    # Get emissions (EXIOBASE in kg, convert to tonnes)
    air_emissions_tonnes = ixi_data.air_emissions.F / 1000
    
    # Check for missing categories
    missing_emissions = [em for em in emission_categories.keys() 
                        if em not in air_emissions_tonnes.index]
    if missing_emissions:
        print(f"Warning: {len(missing_emissions)} emission categories not found")
    
    # Calculate total GHG emissions in tonnes CO2-eq
    n_sectors = len(air_emissions_tonnes.columns)
    producer_emissions = np.zeros(n_sectors)
    
    for emission_type, gwp_factor in emission_categories.items():
        if emission_type in air_emissions_tonnes.index:
            emission_data = air_emissions_tonnes.loc[emission_type].values
            producer_emissions += emission_data * gwp_factor
    
    print(f"Total producer emissions: {producer_emissions.sum()/1e9:.3f} Gt CO2-eq")
    return producer_emissions, missing_emissions

# %%
def calculate_consumer_responsibility(ixi_data, producer_emissions, 
                                     return_category_details=False):
    """
    Calculate consumer-based responsibility.
    
    Parameters:
    -----------
    return_category_details : bool
        If True, also returns breakdown by final demand category
        (household, government, investment, etc.)
    
    Returns:
    --------
    consumer_by_country : dict
    consumer_by_sector_region : dict
    consumer_by_category : dict (only if return_category_details=True)
        Breakdown by final demand category for each country
    """
    regions = ixi_data.get_regions()
    
    # Calculate emission intensities
    total_output = ixi_data.x.values.flatten()
    emission_intensity = np.divide(
        producer_emissions,
        total_output,
        out=np.zeros_like(producer_emissions),
        where=(total_output != 0)
    )
    
    B = ixi_data.L.values
    Y_full = ixi_data.Y
    
    consumer_by_country = {}
    consumer_by_sector_region = {}
    consumer_by_category = {} if return_category_details else None
    
    for region in regions:
        region_mask = Y_full.columns.get_level_values(0) == region
        Y_region = Y_full.loc[:, region_mask]
        
        # Total for the country
        y_region_total = Y_region.sum(axis=1).values
        emissions_vector = emission_intensity * (B @ y_region_total)
        
        consumer_by_country[region] = emissions_vector.sum()
        consumer_by_sector_region[region] = pd.Series(
            emissions_vector,
            index=ixi_data.x.index
        )
        
        # Optional: category breakdown
        if return_category_details:
            fd_categories = Y_region.columns.get_level_values(1).unique()
            consumer_by_category[region] = {}
            
            for category in fd_categories:
                category_mask = Y_region.columns.get_level_values(1) == category
                y_category = Y_region.loc[:, category_mask].sum(axis=1).values
                emissions_category = emission_intensity * (B @ y_category)
                consumer_by_category[region][category] = emissions_category.sum()
    
    total_consumer = sum(consumer_by_country.values())
    print(f"Total consumer emissions: {total_consumer/1e9:.3f} Gt CO2-eq")

    # Convert to Series before returning
    consumer_series = pd.Series(consumer_by_country)
    
    if return_category_details:
        return consumer_series, consumer_by_sector_region, consumer_by_category
    else:
        return consumer_series, consumer_by_sector_region
    

# %%
def calculate_vabr(ixi_data, producer_emissions, v_clean, 
                   return_allocation_details=False):
    """
    Calculate VABR (Value-Added-Based Responsibility) — literal Piñero-style,
    but with per-consuming-country scaling so it stays mass-conserving even
    with cleaned/capped v_clean.

    Returns (same as old function):
      vabr_totals (producer-country totals),
      vabr_by_sector_region (dict by producer country),
      consumer_totals (CBA by consuming country),
      optional allocation_matrix, allocation_df
    """

    print(f"\n=== VABR CALCULATION (Literal Piñero, mass-conserving) ===")
    print(f"Return allocation details: {return_allocation_details}")

    regions = ixi_data.get_regions()
    idx = ixi_data.x.index
    n = len(idx)

    # ----------------------------
    # 1) Emission intensity f (t/€)
    # ----------------------------
    x = ixi_data.x.values.flatten()
    e = np.asarray(producer_emissions).flatten()

    f = np.divide(
        e, x,
        out=np.zeros_like(e, dtype=float),
        where=(x != 0)
    )

    # ----------------------------
    # 2) Leontief inverse and Y
    # ----------------------------
    L = ixi_data.L.values
    Y_full = ixi_data.Y  # DataFrame with MultiIndex columns (region, fd_cat)

    # ----------------------------
    # 3) Consumer baseline (CBA) by consuming region
    # ----------------------------
    consumer_by_country = {}
    for r in regions:
        mask_fd = (Y_full.columns.get_level_values(0) == r)
        y_r = Y_full.loc[:, mask_fd].sum(axis=1).values  # (n,)
        cba_vec = f * (L @ y_r)                          # (n,)
        consumer_by_country[r] = float(cba_vec.sum())

    consumer_totals = pd.Series(consumer_by_country)
    total_cba = float(consumer_totals.sum())
    print(f"Total consumer emissions (CBA): {total_cba/1e9:.3f} Gt")

    # ----------------------------
    # 4) Piñero allocation aggregated over all consuming regions
    #    We build ONE producer-side allocation vector (length n)
    # ----------------------------
    vabr_allocation = np.zeros(n, dtype=float)

    allocation_flows = [] if return_allocation_details else None

    for consuming_region in regions:
        mask_fd = (Y_full.columns.get_level_values(0) == consuming_region)
        y_r = Y_full.loc[:, mask_fd].sum(axis=1).values  # (n,)

        total_emissions_region = float(consumer_totals[consuming_region])
        if total_emissions_region == 0:
            continue

        # Piñero operator steps:
        req1 = L @ y_r          # total output requirements for this final demand
        emis_vec = f * req1     # emissions by producer sector-region
        req2 = L @ emis_vec     # propagate emissions through supply chains again

        raw_alloc = v_clean * req2
        s = raw_alloc.sum()

        # >>> THIS is the block you were asking about <<<
        # rescale so this consuming region's allocated sum == its CBA total
        if s > 0:
            allocated = raw_alloc * (total_emissions_region / s)
        else:
            allocated = np.ones_like(raw_alloc) * (total_emissions_region / len(raw_alloc))

        vabr_allocation += allocated

        if return_allocation_details:
            # consuming_region → producing sector-region flows
            for i, (prod_country, prod_sector) in enumerate(idx):
                val = allocated[i]
                if val > 0:
                    allocation_flows.append({
                        'consuming_region': consuming_region,
                        'producing_country': prod_country,
                        'producing_sector': prod_sector,
                        'allocated_emissions': val,
                        'value_creation': np.nan,
                        'allocation_share': np.nan
                    })

    # ----------------------------
    # 5) Aggregate producer-side totals by producing country (LIKE OLD FUNCTION)
    # ----------------------------
    vabr_by_country = {}
    vabr_by_sector_region = {}

    prod_regions = idx.get_level_values(0).to_numpy()

    for r in regions:
        mask_r = (prod_regions == r)
        region_indices = np.where(mask_r)[0]

        vabr_by_country[r] = float(vabr_allocation[region_indices].sum())
        vabr_by_sector_region[r] = pd.Series(
            vabr_allocation[region_indices],
            index=idx[mask_r]
        )

    vabr_totals = pd.Series(vabr_by_country)

    # ----------------------------
    # 6) Validation: global mass conservation vs CBA
    # ----------------------------
    total_vabr = float(vabr_totals.sum())
    error = abs(total_vabr - total_cba) / total_cba * 100 if total_cba > 0 else 0.0
    print(f"Total VABR (Piñero): {total_vabr/1e9:.3f} Gt, Error vs CBA: {error:.6f}%")

    # ----------------------------
    # 7) Return (same format as old function)
    # ----------------------------
    if return_allocation_details:
        allocation_df = pd.DataFrame(allocation_flows)
        allocation_matrix = allocation_df.pivot_table(
            index=['producing_country', 'producing_sector'],
            columns='consuming_region',
            values='allocated_emissions',
            fill_value=0
        )
        print(f"Allocation matrix: {allocation_matrix.shape}, {len(allocation_df):,} flows")
        return vabr_totals, vabr_by_sector_region, consumer_totals, allocation_matrix, allocation_df
    else:
        return vabr_totals, vabr_by_sector_region, consumer_totals


import numpy as np
import pandas as pd

def calculate_vabr_tech_adjusted(
    ixi_data,
    producer_emissions,
    v_clean,
    conserve_global=True,
    return_allocation_details=False,
    batch_cols=8,
    export_benchmark_mode="export_weighted",  # "export_weighted" (Kander “world market”) or "output_weighted"
):
    """
    KANDER-STYLE TCBA (Option A) + (literal Piñero) allocation.

    Implements Kander's definitions using:
      x_i^{sr} = (L @ y^r)_i    (output of producer row i in country s for final demand in r)
      world-market benchmark (per sector i):
        qdot_i = sum_{s,r≠s} q_i^s * x_i^{sr} / sum_{s,r≠s} x_i^{sr}
      EEE_s (actual)  = sum_{rows in s} q_row * x_foreign_row
      EEE*_s (bench)  = sum_{rows in s} qdot_sector(row) * x_foreign_row
      TCBA_s = CBA_s - (EEE*_s - EEE_s)

    Then allocates each consuming country's TCBA total with Piñero operator and rescales to TCBA_c.
    """

    print("\n=== TECHNOLOGY-ADJUSTED VABR (Kander-style TCBA, embodied exports) + Piñero allocation ===")

    regions = list(ixi_data.get_regions())
    idx = ixi_data.x.index
    n = len(idx)
    R = len(regions)
    region_to_j = {r: j for j, r in enumerate(regions)}

    # --- align inputs ---
    x = ixi_data.x.values.reshape(-1).astype(float)

    if hasattr(producer_emissions, "reindex"):
        e = producer_emissions.reindex(idx).values.reshape(-1).astype(float)
    else:
        e = np.asarray(producer_emissions, dtype=float).reshape(-1)

    if hasattr(v_clean, "reindex"):
        v_clean_vec = v_clean.reindex(idx).values.reshape(-1).astype(float)
    else:
        v_clean_vec = np.asarray(v_clean, dtype=float).reshape(-1)

    # domestic intensity (Kander's q_row)
    f_dom = np.divide(e, x, out=np.zeros_like(e, dtype=float), where=(x > 0))

    L = ixi_data.L.values  # (n x n)
    Y_df = ixi_data.Y      # DataFrame with MultiIndex columns (region, fd_cat)

    prod_regions = idx.get_level_values(0).to_numpy()
    sectors = idx.get_level_values(1).to_numpy()
    unique_sectors = pd.unique(sectors)

    # ------------------------------------------------------------------
    # STEP A: Build Y_by_country (n x R): y^r (sum over final demand categories)
    # ------------------------------------------------------------------
    print("\nBuilding final demand matrix Y_by_country...")
    Y_by_country = np.zeros((n, R), dtype=float)
    for j, r in enumerate(regions):
        mask_fd_r = (Y_df.columns.get_level_values(0) == r)
        Y_by_country[:, j] = Y_df.loc[:, mask_fd_r].sum(axis=1).values

    # ------------------------------------------------------------------
    # STEP B: Compute X = L @ Y_by_country in batches
    #         Column j is x^{·r} = L @ y^r (producer outputs for final demand of r)
    # ------------------------------------------------------------------
    print("Computing X = L @ Y_by_country (batched)...")
    X = np.zeros_like(Y_by_country)
    for start in range(0, R, batch_cols):
        end = min(R, start + batch_cols)
        X[:, start:end] = L @ Y_by_country[:, start:end]

    # ------------------------------------------------------------------
    # STEP 1: Standard CBA totals: CBA_r = sum_i q_i * x_i^{·r}
    # ------------------------------------------------------------------
    consumer_cba_totals = pd.Series(
        {regions[j]: float((f_dom * X[:, j]).sum()) for j in range(R)}
    )
    total_cba = float(consumer_cba_totals.sum())
    print(f"Standard CBA sum: {total_cba/1e9:.3f} Gt")

    # ------------------------------------------------------------------
    # STEP 2: Foreign-final-demand output per producer row (Kander weights for 'world market')
    #         x_foreign_row = sum_{r≠s(row)} x_row^{s(row), r}
    # ------------------------------------------------------------------
    print("\nComputing foreign-final-demand output per producer row (x_foreign)...")
    x_total_by_row = X.sum(axis=1)  # sum over all final demand regions

    # domestic final demand column for each row depends on its producer region
    j_by_row = np.array([region_to_j[r] for r in prod_regions], dtype=int)
    x_domestic_by_row = X[np.arange(n), j_by_row]

    x_foreign = x_total_by_row - x_domestic_by_row
    x_foreign = np.maximum(x_foreign, 0.0)  # guard tiny negatives from numerics

    # ------------------------------------------------------------------
    # STEP 3: Compute qdot (world market avg) per SECTOR, then map to rows
    #         qdot_sec = sum_{rows in sec} q_row * x_foreign_row / sum x_foreign_row
    # ------------------------------------------------------------------
    print(f"\nComputing benchmark intensities per sector ({export_benchmark_mode})...")

    q_world = np.zeros_like(f_dom, dtype=float)

    if export_benchmark_mode == "export_weighted":
        # Kander "world market": weighted by production that ends up in foreign final consumption
        for sec in unique_sectors:
            sec_mask = (sectors == sec)
            w = x_foreign[sec_mask]
            q = f_dom[sec_mask]
            wsum = np.nansum(w)
            if wsum > 0:
                qdot = np.nansum(q * w) / wsum
            else:
                # fallback: output-weighted if sector has no foreign-final-demand output
                x_sec = x[sec_mask]
                e_sec = e[sec_mask]
                qdot = (e_sec.sum() / x_sec.sum()) if x_sec.sum() > 0 else 0.0
            q_world[sec_mask] = qdot

    elif export_benchmark_mode == "output_weighted":
        # global output-weighted benchmark (NOT “world market” in Kander wording, but useful sensitivity)
        for sec in unique_sectors:
            sec_mask = (sectors == sec)
            x_sec = x[sec_mask]
            e_sec = e[sec_mask]
            qdot = (e_sec.sum() / x_sec.sum()) if x_sec.sum() > 0 else 0.0
            q_world[sec_mask] = qdot
    else:
        raise ValueError("export_benchmark_mode must be 'export_weighted' or 'output_weighted'")

    print(f"  Domestic q:   min={np.nanmin(f_dom):.2e}, max={np.nanmax(f_dom):.2e}")
    print(f"  Bench qdot:   min={np.nanmin(q_world):.2e}, max={np.nanmax(q_world):.2e}")

    # ------------------------------------------------------------------
    # STEP 4: Compute EEE_s (actual) and EEE*_s (benchmark) using x_foreign weights
    # ------------------------------------------------------------------
    print("\nComputing embodied export emissions (EEE) per exporter country...")
    EEE_actual = {}
    EEE_world = {}
    for r in regions:
        mask_r = (prod_regions == r)
        EEE_actual[r] = float((f_dom[mask_r]   * x_foreign[mask_r]).sum())
        EEE_world[r]  = float((q_world[mask_r] * x_foreign[mask_r]).sum())

    # ------------------------------------------------------------------
    # STEP 5: Build TCBA totals (Kander Option A)
    #         TCBA_s = CBA_s - (EEE*_s - EEE_s)
    # ------------------------------------------------------------------
    consumer_tcba_totals = pd.Series(
        {r: float(consumer_cba_totals[r] - (EEE_world[r] - EEE_actual[r])) for r in regions}
    )
    total_tcba = float(consumer_tcba_totals.sum())
    print(f"TCBA sum (before scaling): {total_tcba/1e9:.3f} Gt")

    if conserve_global and total_tcba > 0:
        scale = total_cba / total_tcba
        consumer_tcba_totals *= scale
        print(f"  Scaling TCBA by factor {scale:.6f} to match global CBA.")
    else:
        scale = 1.0

    print(f"TCBA sum (after scaling): {consumer_tcba_totals.sum()/1e9:.3f} Gt")

    # ------------------------------------------------------------------
    # STEP 6: Allocate TCBA via (literal) Piñero operator, then rescale to TCBA_c
    # ------------------------------------------------------------------
    print("\nAllocating TCBA via Piñero operator (then rescaling to TCBA_c)...")

    vabr_allocation = np.zeros(n, dtype=float)
    allocation_rows = [] if return_allocation_details else None

    for j, consuming_region in enumerate(regions):
        y_c = Y_by_country[:, j]
        tcba_c = float(consumer_tcba_totals[consuming_region])
        if tcba_c == 0:
            continue

        # reuse req1 = L @ y_c from X (already computed)
        req1 = X[:, j]
        emis_vec = f_dom * req1
        req2 = L @ emis_vec
        raw_alloc = v_clean_vec * req2

        s = float(raw_alloc.sum())
        if s > 0:
            allocated = raw_alloc * (tcba_c / s)
        else:
            allocated = np.ones_like(raw_alloc) * (tcba_c / len(raw_alloc))

        vabr_allocation += allocated

        if return_allocation_details:
            # WARNING: can be huge
            nz = np.where(allocated > 0)[0]
            for i in nz:
                prod_country, prod_sector = idx[i]
                allocation_rows.append(
                    {
                        "consuming_region": consuming_region,
                        "producing_country": prod_country,
                        "producing_sector": prod_sector,
                        "allocated_emissions": float(allocated[i]),
                    }
                )

    # ------------------------------------------------------------------
    # STEP 7: Aggregate by producing country
    # ------------------------------------------------------------------
    vabr_by_country = {}
    vabr_by_sector_region = {}

    for r in regions:
        mask_r = (prod_regions == r)
        vabr_by_country[r] = float(vabr_allocation[mask_r].sum())
        vabr_by_sector_region[r] = pd.Series(vabr_allocation[mask_r], index=idx[mask_r])

    vabr_t_techA = pd.Series(vabr_by_country)

    # ------------------------------------------------------------------
    # STEP 8: Validation / diagnostics
    # ------------------------------------------------------------------
    total_out = float(vabr_t_techA.sum())
    denom = float(consumer_tcba_totals.sum())
    err = (abs(total_out - denom) / denom * 100) if denom > 0 else 0.0

    print(f"\nTotal T-VABR: {total_out/1e9:.3f} Gt, Error vs TCBA sum: {err:.6f}%")
    print("Sanity (EEE* - EEE) for first 5 regions:")
    for r in regions[:5]:
        print(f"  {r}: {(EEE_world[r] - EEE_actual[r]) / 1e9:+.3f} Gt")

    if return_allocation_details:
        allocation_df = pd.DataFrame(allocation_rows)
        allocation_matrix = allocation_df.pivot_table(
            index=["producing_country", "producing_sector"],
            columns="consuming_region",
            values="allocated_emissions",
            fill_value=0.0,
            aggfunc="sum",
        )
        return vabr_t_techA, vabr_by_sector_region, consumer_tcba_totals, allocation_matrix, allocation_df
    else:
        return vabr_t_techA, vabr_by_sector_region, consumer_tcba_totals


def calculate_tech_gap_penalty(
    ixi_data,
    producer_emissions,
    benchmark_mode="world_avg"
):
    """
    Producer-side technology-gap penalty (sector-region level).

    For each sector-region (row i):
      - actual intensity f_i = e_i / x_i
      - benchmark intensity f*_sec(i) computed across ALL regions in that sector
      - absolute gap_i = max(0, f_i - f*_sec(i))
      - avoidable emissions (absolute) = gap_i * x_i

    Returns
    -------
    excess_emissions : pd.Series indexed like ixi_data.x.index
    sector_gaps : pd.DataFrame with intensities, gaps, excess emissions, output
    """
    print(f"\n=== TECHNOLOGY-GAP (Producer-side), benchmark={benchmark_mode} ===")

    idx = ixi_data.x.index
    x = ixi_data.x.values.reshape(-1).astype(float)

    if hasattr(producer_emissions, "reindex"):
        e = producer_emissions.reindex(idx).values.reshape(-1).astype(float)
    else:
        e = np.asarray(producer_emissions, dtype=float).reshape(-1)

    # Actual intensity (t/€)
    f_actual = np.divide(e, x, out=np.zeros_like(e, dtype=float), where=(x > 0))

    sectors = idx.get_level_values(1).to_numpy()
    unique_sectors = pd.unique(sectors)

    f_bench = np.zeros_like(f_actual, dtype=float)

    print("Calculating sectoral benchmarks...")
    for sec in unique_sectors:
        mask = (sectors == sec)
        f_sec = f_actual[mask]
        x_sec = x[mask]

        if benchmark_mode == "world_avg":
            # output-weighted world average intensity for that sector
            if np.sum(x_sec) > 0:
                bench = np.average(f_sec, weights=x_sec)
            else:
                bench = float(np.nanmean(f_sec))
        elif benchmark_mode == "best_quartile":
            vals = f_sec[f_sec > 0]
            if len(vals) > 0:
                vals_sorted = np.sort(vals)
                cutoff = max(1, int(len(vals_sorted) * 0.25))
                bench = float(np.mean(vals_sorted[:cutoff]))
            else:
                bench = 0.0
        elif benchmark_mode == "best":
            vals = f_sec[f_sec > 0]
            bench = float(np.min(vals)) if len(vals) > 0 else 0.0
        else:
            raise ValueError("benchmark_mode must be 'world_avg', 'best_quartile', or 'best'")

        f_bench[mask] = bench

    # Absolute gap (t/€) and "avoidable" emissions (t)
    gap_abs = np.maximum(0.0, f_actual - f_bench)
    excess_emissions_array = gap_abs * x

    excess_emissions = pd.Series(excess_emissions_array, index=idx, name="excess_emissions")

    sector_gaps = pd.DataFrame(
        {
            "output": x,
            "actual_intensity": f_actual,
            "benchmark_intensity": f_bench,
            "gap_abs": gap_abs,
            "excess_emissions": excess_emissions_array,
        },
        index=idx
    )

    total_actual = float(np.nansum(e))
    total_excess = float(np.nansum(excess_emissions_array))
    share = 100 * total_excess / total_actual if total_actual > 0 else 0.0
    print(f"Total actual emissions:    {total_actual/1e9:.2f} Gt")
    print(f"Total 'avoidable' (abs):   {total_excess/1e9:.2f} Gt ({share:.1f}% of global)")

    return excess_emissions, sector_gaps


def calculate_vabr_with_tech_penalty(
    ixi_data,
    producer_emissions,
    v_clean,
    benchmark_mode="world_avg",
    alpha=1.0,
    rel_gap_cap=5.0,
    bench_floor=1e-12,
    penalty_only=True,
):
    """
    Annex exploration: VABR + producer-side technology-gap adjustment.

    IMPORTANT DESIGN CHOICE (for comparability with plots):
    - This function RETURNS producer-indexed totals (like Producer and TechA),
      so you can put it in the same bar chart without mixing "consumer vs producer".

    How it works:
    1) Compute base (literal Piñero) allocation *details* by consuming country:
         vabr_details[c] : Series over producing (region,sector) rows (tonnes)
       (This is existing calculate_vabr output.)
    2) Compute producer-side relative intensity gap per producing row:
         rel_gap_i = (f_i - f*_sec(i)) / f*_sec(i)
       with safe floors/caps; optionally penalty-only (no credits).
    3) For each consuming country c, compute a scalar penalty term:
         tech_penalty_cons[c] = sum_i vabr_details[c][i] * rel_gap_i
    4) Build consumer-indexed adjusted totals and rescale to conserve global mass.
    5) Convert to producer-indexed totals by aggregating adjusted allocations:
         adjusted_alloc_c[i] = vabr_details[c][i] * (adj_total_c / base_total_c)
         producer_total[r] = sum_{i in r} sum_c adjusted_alloc_c[i]

    Returns (same signature as existing code expects):
      responsibility_total : pd.Series (PRODUCER-indexed adjusted totals)
      vabr_totals          : pd.Series (base Piñero totals by consuming country)
      tech_penalty         : pd.Series (penalty term by consuming country, pre-scaling)
      sector_gaps          : pd.DataFrame (producer-side gaps + rel_gap)
    """
    print(f"\n=== VABR + TECH-GAP (Annex exploration) ===")
    print(f"Benchmark={benchmark_mode}, alpha={alpha}, rel_gap_cap={rel_gap_cap}, penalty_only={penalty_only}")

    idx = ixi_data.x.index
    prod_regions = idx.get_level_values(0).to_numpy()
    regions = list(ixi_data.get_regions())

    # --------------------------------------------------------------
    # 1) Base Piñero VABR totals (consumers) + allocation vectors
    # --------------------------------------------------------------
    vabr_totals_cons, vabr_details, _ = calculate_vabr(
        ixi_data, producer_emissions, v_clean, return_allocation_details=False
    )
    base_total_global = float(vabr_totals_cons.sum())
    print(f"Base Piñero VABR global (consumer-index): {base_total_global/1e9:.3f} Gt")

    # --------------------------------------------------------------
    # 2) Producer-side gaps
    # --------------------------------------------------------------
    _, sector_gaps = calculate_tech_gap_penalty(ixi_data, producer_emissions, benchmark_mode)

    f_act = sector_gaps["actual_intensity"].astype(float)
    f_bench = sector_gaps["benchmark_intensity"].astype(float)

    denom = f_bench.copy()
    denom[denom.abs() < bench_floor] = np.nan

    rel_gap = (f_act - f_bench) / denom
    rel_gap = rel_gap.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # cap to avoid blow-ups
    rel_gap = rel_gap.clip(lower=-rel_gap_cap, upper=rel_gap_cap)

    if penalty_only:
        rel_gap = rel_gap.clip(lower=0.0)

    sector_gaps["rel_gap"] = rel_gap
    rel_gap_vec = rel_gap.reindex(idx).fillna(0.0).values  # aligned vector

    # --------------------------------------------------------------
    # 3) Tech penalty term by CONSUMING country (scalar)
    # --------------------------------------------------------------
    tech_penalty_cons = {}
    for c in regions:
        if c not in vabr_details:
            tech_penalty_cons[c] = 0.0
            continue
        alloc_c = vabr_details[c].reindex(idx).fillna(0.0).values  # tonnes on producing rows
        tech_penalty_cons[c] = float(np.sum(alloc_c * rel_gap_vec))

    tech_penalty_cons = pd.Series(tech_penalty_cons)
    print(f"Net penalty sum (consumer-index, pre-scaling): {tech_penalty_cons.sum()/1e9:.3f} Gt")

    # --------------------------------------------------------------
    # 4) Consumer-index adjusted totals + global mass rescale
    # --------------------------------------------------------------
    cons_adjusted_raw = vabr_totals_cons + alpha * tech_penalty_cons
    raw_total = float(cons_adjusted_raw.sum())
    print(f"Raw adjusted global (consumer-index): {raw_total/1e9:.3f} Gt")

    scale = (base_total_global / raw_total) if raw_total != 0 else 1.0
    cons_adjusted = cons_adjusted_raw * scale

    print(f"Scale applied for mass conservation: {scale:.6f}")
    print(f"Final adjusted global (consumer-index): {cons_adjusted.sum()/1e9:.3f} Gt")

    # --------------------------------------------------------------
    # 5) Convert to PRODUCER-index totals for comparability with TechA/Producer plots
    #    We keep the allocation *pattern* within each consumer c the same,
    #    only scaling its allocation vector by (adjusted_total_c / base_total_c).
    # --------------------------------------------------------------
    producer_total = pd.Series(0.0, index=pd.Index(regions, name="region"))

    for c in regions:
        base_c = float(vabr_totals_cons.get(c, 0.0))
        adj_c = float(cons_adjusted.get(c, 0.0))

        if base_c == 0:
            continue

        factor_c = adj_c / base_c

        alloc_c = vabr_details[c].reindex(idx).fillna(0.0).values  # tonnes on producing rows
        alloc_c_adj = alloc_c * factor_c

        # aggregate to producing country
        for r in regions:
            mask_r = (prod_regions == r)
            producer_total[r] += float(np.sum(alloc_c_adj[mask_r]))

    # Diagnostics: producer totals should sum to global (same as consumer global)
    print(f"Producer-index adjusted global: {producer_total.sum()/1e9:.3f} Gt")

    # IMPORTANT: return signature unchanged for existing plotting pipeline
    # responsibility_total (producer-index), vabr_totals (consumer-index base), tech_penalty (consumer-index), sector_gaps
    return producer_total, vabr_totals_cons, tech_penalty_cons, sector_gaps




# %%
    
def calculate_pcpr(
    ixi_data, 
    producer_emissions, 
    profit_components=None,
    method='inverse',
    max_layers=50,
    x_floor=1e3,
    return_flows=False,
    min_flow=0.0,
):
    """
    Calculate Producer-Centric Profit-Based Responsibility (PCPR).
    
    Allocates producer emissions based on downstream profit capture
    using forward value-flow tracing through supply chains.
    
    Parameters
    ----------
    ixi_data : pymrio object
        Loaded EXIOBASE data.
    producer_emissions : np.array
        Producer emissions in tonnes CO2-eq per sector (same order as x).
    profit_components : list, optional
        Profit/operating surplus components to use.
        If None, uses standard operating surplus components.
    method : {'inverse', 'layered'}, default 'inverse'
        'inverse' = full matrix inversion (fast, exact).
        'layered' = layer-by-layer expansion (slow, but shows convergence).
    max_layers : int, default 50
        Maximum layers for Taylor expansion (only used if method='layered').
    x_floor : float, default 1e3
        Minimum output value (€) to avoid division by (near) zero.
    return_flows : bool, default False
        If True, additionally returns a DataFrame of detailed flows:
        producing_region, producing_sector, beneficiary_region,
        beneficiary_sector, allocated_emissions.
    min_flow : float, default 0.0
        Minimum allocated emission (in tonnes CO2-eq) for a flow to be
        recorded in the flows DataFrame when return_flows=True.
    
    Returns
    -------
    If method == 'inverse' and return_flows == False:
        pcpr_totals : pd.Series
            Responsibility by country in tonnes CO2-eq.
        pcpr_by_sector_region : dict
            Responsibility by sector-region for each country.
    
    If method == 'inverse' and return_flows == True:
        pcpr_totals : pd.Series
        pcpr_by_sector_region : dict
        flows_df : pd.DataFrame
            Columns: producing_region, producing_sector,
                     beneficiary_region, beneficiary_sector,
                     allocated_emissions.
    
    If method == 'layered' and return_flows == False:
        pcpr_totals : pd.Series
        pcpr_by_sector_region : dict
        layer_convergence : list of dict
            Convergence diagnostics per layer.
    
    If method == 'layered' and return_flows == True:
        pcpr_totals : pd.Series
        pcpr_by_sector_region : dict
        layer_convergence : list of dict
        flows_df : pd.DataFrame
    """
    
    print(f"\n=== PRODUCER-CENTRIC PROFIT-BASED RESPONSIBILITY ({method.upper()}) ===")
    
    regions = ixi_data.get_regions()
    n = len(ixi_data.x)
    
    # Core IO data
    Z = ixi_data.Z.values
    x = ixi_data.x.values.flatten()
    Y = ixi_data.Y.values
    FD = Y.sum(axis=1)
    
    producer_emissions = producer_emissions.flatten()
    
    # Floor output to avoid extreme coefficients
    x_safe = np.maximum(x, x_floor)
    floored_count = (x < x_floor).sum()
    
    print(f"Sectors: {n}, Regions: {len(regions)}")
    print(f"Total emissions: {producer_emissions.sum()/1e9:.3f} Gt CO2-eq")
    print(f"Floored sectors: {floored_count} ({floored_count/n*100:.1f}%)")
    
    # Build S matrix with a final-demand column
    S = np.zeros((n, n+1))
    S[:, :n] = Z / x_safe[:, None]
    S[:, n] = FD / x_safe
    
    row_sums = S.sum(axis=1)
    print(f"S row sums: mean={row_sums.mean():.3f}, max={row_sums.max():.3f}")
    
    # Profit components
    if profit_components is None:
        profit_components = [
            "Operating surplus: Consumption of fixed capital",
            "Operating surplus: Rents on land",
            "Operating surplus: Royalties on resources",
            "Operating surplus: Remaining net operating surplus"
        ]
    
    VA_profit = ixi_data.factor_inputs.F.loc[profit_components].sum(axis=0).values
    v_profit = np.divide(
        VA_profit, x_safe,
        out=np.zeros_like(VA_profit),
        where=(x_safe > 0)
    )
    v_profit = np.clip(v_profit, 0, 1)
    
    print(f"Total profit VA: {VA_profit.sum()/1e9:.1f} B€")
    print(f"Profit coefficients: mean={v_profit.mean():.3f}")
    
    # Compute D matrix (square part)
    I = np.eye(n)
    S_square = S[:, :n]
    
    layer_convergence = None
    
    if method == 'inverse':
        print("Computing D = (I - S)^(-1)...")
        cond = np.linalg.cond(I - S_square)
        print(f"  Condition number: {cond:.2e}")
        
        try:
            D_square = np.linalg.inv(I - S_square)
        except np.linalg.LinAlgError:
            print("  Warning: Using regularization")
            D_square = np.linalg.inv(I - 0.9999 * S_square)
    
    elif method == 'layered':
        print(f"Computing D via Taylor expansion (max {max_layers} layers)...")
        
        D_square = np.zeros((n, n))
        S_power = np.eye(n)
        layer_convergence = []
        
        for layer in range(max_layers):
            # Add current term
            D_square += S_power
            
            term_norm = np.linalg.norm(S_power)
            cumulative = D_square.sum()
            layer_convergence.append({
                'layer': layer,
                'term_norm': term_norm,
                'cumulative': cumulative
            })
            
            if layer < 5 or layer % 10 == 0:
                print(f"    Layer {layer}: norm={term_norm:.2e}, cumulative={cumulative:.2e}")
            
            S_power = S_power @ S_square
            
            if term_norm < 1e-12:
                print(f"  ✓ Converged at layer {layer}")
                break
        else:
            print(f"  ⚠ Reached max_layers ({max_layers})")
    else:
        raise ValueError("method must be 'inverse' or 'layered'")
    
    # Extend D with a final-demand column
    D = np.zeros((n, n+1))
    D[:, :n] = D_square
    D[:, n] = D_square @ S[:, n]
    
    direct_FD = S[:, n].mean()
    total_FD = D[:, n].mean()
    print(f"FD flows: direct={direct_FD:.3f}, total={total_FD:.3f} (ratio: {total_FD/direct_FD:.2f}x)")
    
    # Per-producer allocation
    print("Per-producer allocation...")
    responsibility = np.zeros(n)
    zero_profit_count = 0
    
    # optional: detailed flows for Sankeys
    flows_list = [] if return_flows else None
    idx = ixi_data.x.index  # MultiIndex (region, sector)
    
    for p in range(n):
        if producer_emissions[p] <= 0:
            continue
        
        # Profit capture pattern of this producer p across all sectors
        d_p = D[p, :n]
        profit_capture = v_profit * d_p
        total_profit = profit_capture.sum()
        
        if total_profit <= 0:
            # No profit captured anywhere → keep responsibility with the producer itself
            responsibility[p] += producer_emissions[p]
            zero_profit_count += 1
        else:
            shares = profit_capture / total_profit  # how much each sector benefits
            alloc = producer_emissions[p] * shares  # allocated emissions
            
            responsibility += alloc
            
            if return_flows:
                prod_region, prod_sector = idx[p]
                
                # record only non-zero / above threshold flows
                nonzero = np.where(alloc > min_flow)[0]
                for j in nonzero:
                    ben_region, ben_sector = idx[j]
                    flows_list.append({
                        "producing_region": prod_region,
                        "producing_sector": prod_sector,
                        "beneficiary_region": ben_region,
                        "beneficiary_sector": ben_sector,
                        "allocated_emissions": alloc[j]
                    })
    
    print(f"  Producers processed: {(producer_emissions > 0).sum()}")
    print(f"  Zero profit capture: {zero_profit_count}")
    
    # Validation
    total_in = producer_emissions.sum()
    total_out = responsibility.sum()
    error = abs(total_out - total_in) / total_in * 100 if total_in > 0 else 0.0
    
    print(f"Conservation: {total_out/1e9:.4f} Gt (error: {error:.6f}%)")
    
    # Aggregate by country + sector-region
    pcpr_by_country = {}
    pcpr_by_sector_region = {}
    
    for region in regions:
        mask = idx.get_level_values(0) == region
        region_indices = np.where(mask)[0]
        
        pcpr_by_country[region] = responsibility[region_indices].sum()
        pcpr_by_sector_region[region] = pd.Series(
            responsibility[region_indices],
            index=idx[mask]
        )
    
    pcpr_totals = pd.Series(pcpr_by_country)
    
    print(f"\nTop 5 countries:")
    for country, value in pcpr_totals.nlargest(5).items():
        denom = producer_emissions[idx.get_level_values(0) == country].sum()
        mult = value / denom if denom > 0 else np.nan
        print(f"  {country}: {value/1e9:.3f} Gt ({mult:.2f}x)")
    
    # Build flows DataFrame if requested
    flows_df = None
    if return_flows and flows_list:
        flows_df = pd.DataFrame(flows_list)
    
    # Return according to method + return_flows
    if method == 'layered':
        if return_flows:
            return pcpr_totals, pcpr_by_sector_region, layer_convergence, flows_df
        else:
            return pcpr_totals, pcpr_by_sector_region, layer_convergence
    else:  # inverse
        if return_flows:
            return pcpr_totals, pcpr_by_sector_region, flows_df
        else:
            return pcpr_totals, pcpr_by_sector_region
   

# %% [markdown]
# SECTOR CLASSIFICATION

# %%
import pandas as pd

# ----------------------------
# 1. Sector classification
# ----------------------------
sector_classification = {

    # Fossil fuels
    "Coal": [
        "Mining of coal and lignite; extraction of peat (10)",
        "Production of electricity by coal",
        "Manufacture of coke oven products",
    ],

    "Oil": [
       "Extraction of crude petroleum and services related to crude oil extraction, excluding surveying",
        "Petroleum Refinery",
        "Production of electricity by petroleum and other oil derivatives",
        "Retail sale of automotive fuel", # at the end of Oil supply chain
    ],

    "Gas": [
        "Extraction of natural gas and services related to natural gas extraction, excluding surveying",
        "Extraction, liquefaction, and regasification of other petroleum and gaseous materials",
        "Manufacture of gas; distribution of gaseous fuels through mains",
        "Production of electricity by gas",
        "Transport via pipelines",
    
    ],

    "Energy & Heat Infrastructure": [
        "Distribution and trade of electricity",
        "Transmission of electricity",
        "Steam and hot water supply",
        "Processing of nuclear fuel",
        "Production of electricity nec",
        "Production of electricity by nuclear",
        "Collection, purification and distribution of water (41)",
    ],

    # Renewable electricity
    "Renewables": [
        "Production of electricity by Geothermal",
        "Production of electricity by biomass and waste",
        "Production of electricity by hydro",
        "Production of electricity by solar photovoltaic",
        "Production of electricity by solar thermal",
        "Production of electricity by tide, wave, ocean",
        "Production of electricity by wind",
    ],

    # Metals
    "Metals": [
        "Manufacture of basic iron and steel and of ferro-alloys and first products thereof",
        "Aluminium production",
        "Copper production",
        "Lead, zinc and tin production",
        "Other non-ferrous metal production",
        "Casting of metals",
        "Mining of aluminium ores and concentrates",
        "Mining of copper ores and concentrates",
        "Mining of iron ores",
        "Mining of lead, zinc and tin ores and concentrates",
        "Mining of nickel ores and concentrates",
        "Mining of other non-ferrous metal ores and concentrates",
        "Mining of precious metal ores and concentrates",
        "Mining of uranium and thorium ores (12)",
        "Precious metals production",
    ],

    # Minerals
    "Non-metallic Minerals": [
        "Manufacture of cement, lime and plaster",
        "Manufacture of other non-metallic mineral products n.e.c.",
        "Manufacture of bricks, tiles and construction products, in baked clay",
        "Manufacture of ceramic goods",
        "Manufacture of glass and glass products",
        "Quarrying of sand and clay",
        "Quarrying of stone",
        "Re-processing of ash into clinker",
        "Mining of chemical and fertilizer minerals, production of salt, other mining and quarrying n.e.c.",
    ],

    # Chemicals
    "Chemicals & Plastics": [
        "Chemicals nec",
        "Manufacture of rubber and plastic products (25)",
        "Paper",
        "Pulp",
        "N-fertiliser", 
        "P- and other fertiliser",
        "Plastics, basic",
    ],

    # Light / consumer manufacturing
   "Manufacturing (Food & Beverages)": [
        "Processing of Food products nec", "Processing of dairy products", "Processed rice",
        "Processing of meat cattle", "Processing of meat pigs", "Processing of meat poultry",
        "Processing vegetable oils and fats", "Production of meat products nec",
        "Manufacture of beverages", "Sugar refining", "Manufacture of tobacco products (16)",
        "Manufacture of fish products",

],
    "Manufacturing (Textiles, Leather & Wood)": [
        "Manufacture of textiles (17)", "Manufacture of furniture; manufacturing n.e.c. (36)",
        "Manufacture of wearing apparel; dressing and dyeing of fur (18)",
        "Tanning and dressing of leather; manufacture of luggage, handbags, saddlery, harness and footwear (19)",
        "Manufacture of wood and of products of wood and cork, except furniture; manufacture of articles of straw and plaiting materials (20)",
    ],
    "Manufacturing (Machinery & Equipment)": [
        "Manufacture of fabricated metal products, except machinery and equipment (28)",
        "Manufacture of machinery and equipment n.e.c. (29)",
        "Manufacture of office machinery and computers (30)",
        "Manufacture of electrical machinery and apparatus n.e.c. (31)",
        "Manufacture of radio, television and communication equipment and apparatus (32)",
        "Manufacture of medical, precision and optical instruments, watches and clocks (33)",
        "Manufacture of motor vehicles, trailers and semi-trailers (34)",
        "Manufacture of other transport equipment (35)",
    ],
    
        
      # Agriculture, forestry, fishing
    "Agriculture": [
        "Cattle farming", "Poultry farming", "Pigs farming", "Meat animals nec",
        "Raw milk", "Cultivation of cereal grains nec", "Cultivation of crops nec",
        "Cultivation of oil seeds", "Cultivation of paddy rice",
        "Cultivation of vegetables, fruit, nuts", "Cultivation of sugar cane, sugar beet",
        "Cultivation of wheat", "Cultivation of plant-based fibers", "Animal products nec",
        "Forestry, logging and related service activities (02)",
        "Fishing, operating of fish hatcheries and fish farms; service activities incidental to fishing (05)",
        "Wool, silk-worm cocoons"
    ],

    # Transport
    "Transport": [
        "Air transport (62)",
        "Other land transport",
        "Sea and coastal water transport",
        "Inland water transport",
        "Supporting and auxiliary transport activities; activities of travel agencies (63)",
        "Water transport", "Transport via railways", "Sale, maintenance, repair of motor vehicles, motor vehicles parts, motorcycles, motor cycles parts and accessoiries",
    ],

    # Waste & recycling
    "Waste & Recycling": [
        "Incineration of waste: Food", "Incineration of waste: Metals and Inert materials",
        "Incineration of waste: Paper", "Incineration of waste: Plastic",
        "Incineration of waste: Textiles", "Incineration of waste: Wood", "Incineration of waste: Oil/Hazardous waste",
        "Recycling of waste and scrap",
        "Landfill of waste: Food", "Landfill of waste: Inert/metal/hazardous",
        "Landfill of waste: Paper", "Landfill of waste: Plastic", "Landfill of waste: Textiles",
        "Landfill of waste: Wood",
        "Composting of food waste, incl. land application",
        "Composting of paper and wood, incl. land application",
        "Manure treatment (biogas), storage and land application",
        "Manure treatment (conventional), storage and land application",
        "Biogasification of food waste, incl. land application",
        "Biogasification of sewage slugde, incl. land application",
        "Biogasification of paper, incl. land application",
        "Waste water treatment, other", "Waste water treatment, food",
        "Recycling of bottles by direct reuse",

        "Re-processing of secondary aluminium into new aluminium",
        "Re-processing of secondary construction material into aggregates",
        "Re-processing of secondary copper into new copper",
        "Re-processing of secondary glass into new glass",
        "Re-processing of secondary lead into new lead, zinc and tin",
        "Re-processing of secondary other non-ferrous metals into new other non-ferrous metals",
        "Re-processing of secondary paper into new pulp",
        "Re-processing of secondary plastic into new plastic",
        "Re-processing of secondary preciuos metals into new preciuos metals",
        "Re-processing of secondary steel into new steel",
        "Re-processing of secondary wood material into new wood material",
         
    ],

    # Services
    "Services": [
        "Post and telecommunications (64)",
        "Financial intermediation, except insurance and pension funding (65)",
        "Activities auxiliary to financial intermediation (67)",
        "Insurance and pension funding, except compulsory social security (66)",
        "Computer and related activities (72)",
        "Research and development (73)",
        "Education (80)", "Health and social work (85)", "Hotels and restaurants (55)",
        "Other business activities (74)", "Real estate activities (70)",
        "Public administration and defence; compulsory social security (75)",
        "Activities of membership organisation n.e.c. (91)",
        "Extra-territorial organizations and bodies",
        "Recreational, cultural and sporting activities (92)",
        "Other service activities (93)",
        "Private households with employed persons (95)",
        "Insurance and pension funding",
        "Public administration and defence",
        "Renting of machinery and equipment without operator and of personal and household goods (71)",
        "Community, social and personal services nec","Households as employers", "Publishing, printing and reproduction of recorded media (22)",
        
    ],

    # Construction + Trade (kept together to reduce tiny colours)
    "Construction & Trade": [
        "Construction (45)",
        "Wholesale trade and commission trade, except of motor vehicles and motorcycles (51)",
        "Retail trade, except of motor vehicles and motorcycles; repair of personal and household goods (52)",
    ],
     # Empty catch-all
    "other": [],
}


# 2. Simple colour palette (hex)
# ----------------------------
sector_colours = {
    "Coal": "#7f7f7f",
    "Oil": "#d62728",
    "Gas": "#1f77b4",
    "Electricity & Heat Infrastructure": "#9467bd",
    "Renewables": "#2ca02c",
    "Metals": "#bcbd22",
    "Non-metallic Minerals": "#8c564b",
    "Chemicals & Plastics": "#e377c2",
    "Manufacturing (Food & Beverages)": "#ff7f0e",
    "Manufacturing (Textiles, Leather & Wood)": "#17becf",
    "Manufacturing (Machinery & Equipment)":  "#6b6ecf",
    "Agriculture": "#a6d854",
    "Construction & Trade": "#8dd3c7",
    "Transport": "#c49c94",
    "Waste & Recycling": "#f7b6d2",
    "Services": "#c7c7c7",
    "other": "#9edae5",
}

# ----------------------------
# 3. Flat mapping (lower-case keys avoid typos)
# ----------------------------
sector_to_category = {
    s.lower(): cat
    for cat, lst in sector_classification.items()
    for s in lst
}
def category_of(sec):
    """Get category for a sector name."""
    return sector_to_category.get(sec.strip().lower(), "other")

def colour_of(sec):
    """Get colour for a sector name."""
    return sector_colours.get(category_of(sec), "#9edae5")

def get_sectors_by_categories(category_list):
    """Get all sectors belonging to specified categories."""
    return [s for cat in category_list 
            for s in sector_classification[cat]]


# %%



