# %% [markdown]
# this is just for functions

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

# Define value-added components
VALUE_ADDED_COMPONENTS = [
    "Taxes less subsidies on products purchased: Total",     # taxes to be exculded later
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
    VA[VA < 0] = 0
    print(f"  Set {np.sum(VA < 0)} negative VA values to 0")
    
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
    Calculate VABR (Value-Added-Based Responsibility).
    Mass-conserving variant that reallocates consumer emissions based on value creation.
    
    Parameters:
    -----------
    ixi_data : pymrio object
        Loaded EXIOBASE data
    producer_emissions : np.array
        Producer emissions in tonnes CO2-eq
    v_clean : np.array
        Clean value-added coefficients (use calculate_clean_va_coefficients)
    return_allocation_details : bool
        If True, returns detailed allocation matrix for Sankey diagrams
    
    Returns:
    --------
    vabr_by_country : pd.Series
        Total VABR per country
    vabr_by_sector_region : dict
        Detailed VABR by sector for each country
    consumer_by_country : pd.Series
        Consumer baseline for comparison
    allocation_matrix : pd.DataFrame (only if return_allocation_details=True)
        Detailed flows: consuming region → producing sector
    allocation_df : pd.DataFrame (only if return_allocation_details=True)
        Detailed flow records
    """
    print(f"\n=== VABR CALCULATION ===")
    print(f"Return allocation details: {return_allocation_details}")
    
    regions = ixi_data.get_regions()
    n_sectors = len(ixi_data.x)
    
    # STEP 1: Calculate emission intensity
    total_output = ixi_data.x.values.flatten()
    emission_intensity = np.divide(
        producer_emissions,
        total_output,
        out=np.zeros_like(producer_emissions),
        where=(total_output != 0)
    )
    
    # STEP 2: Get matrices
    B = ixi_data.L.values
    Y_full = ixi_data.Y
    
    # STEP 3: Calculate consumer baseline
    consumer_by_country = {}
    for region in regions:
        region_mask = Y_full.columns.get_level_values(0) == region
        y_region = Y_full.loc[:, region_mask].sum(axis=1).values
        emissions_vector = emission_intensity * (B @ y_region)
        consumer_by_country[region] = emissions_vector.sum()
    
    total_consumer = sum(consumer_by_country.values())
    print(f"Total consumer emissions: {total_consumer/1e9:.3f} Gt")
    
    # STEP 4: VABR reallocation
    vabr_allocation = np.zeros(n_sectors)
    
    # Optional: store detailed flows
    allocation_flows = [] if return_allocation_details else None
    
    for consuming_region in regions:
        # Get final demand for this consumer
        region_mask = Y_full.columns.get_level_values(0) == consuming_region
        y_region = Y_full.loc[:, region_mask].sum(axis=1).values
        
        # Total emissions to reallocate
        total_emissions = consumer_by_country[consuming_region]
        if total_emissions == 0:
            continue
        
        # Calculate value creation: v * (B @ y)
        value_creation = v_clean * (B @ y_region)
        total_value = value_creation.sum()
        
        if total_value > 0:
            # Allocate proportionally to value creation
            allocation_shares = value_creation / total_value
            allocated = total_emissions * allocation_shares
            vabr_allocation += allocated
            
            # Store details if requested
            if return_allocation_details:
                for i, (prod_country, prod_sector) in enumerate(ixi_data.x.index):
                    if allocated[i] > 0:
                        allocation_flows.append({
                            'consuming_region': consuming_region,
                            'producing_country': prod_country,
                            'producing_sector': prod_sector,
                            'allocated_emissions': allocated[i],
                            'value_creation': value_creation[i],
                            'allocation_share': allocation_shares[i]
                        })
        else:
            print(f"Warning: No value for {consuming_region}, uniform allocation")
            uniform = total_emissions / n_sectors
            vabr_allocation += uniform
    
    # STEP 5: Aggregate by country
    vabr_by_country = {}
    vabr_by_sector_region = {}
    
    for region in regions:
        region_mask = ixi_data.x.index.get_level_values(0) == region
        region_indices = np.where(region_mask)[0]
        
        vabr_by_country[region] = vabr_allocation[region_indices].sum()
        vabr_by_sector_region[region] = pd.Series(
            vabr_allocation[region_indices],
            index=ixi_data.x.index[region_mask]
        )
    
    # STEP 6: Validation
    total_vabr = sum(vabr_by_country.values())
    error = abs(total_vabr - total_consumer) / total_consumer * 100
    print(f"Total VABR: {total_vabr/1e9:.3f} Gt, Error: {error:.4f}%")
    
    # Convert to Series
    vabr_totals = pd.Series(vabr_by_country)
    consumer_totals = pd.Series(consumer_by_country)
    
    # Return with or without details
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

# %%
# ===================================================================
# PRODUCER-CENTRIC (BOTTOM-UP) VABR
# ===================================================================

def calculate_producer_centric_vabr(ixi_data, producer_emissions, v_clean,
                                     method='leontief', max_layers=100):
    """
    Producer-Centric (Bottom-Up) VABR.
    
    Starts from producer emissions and traces forward through supply chains
    to allocate responsibility based on value-added capture.
    
    Parameters:
    -----------
    ixi_data : pymrio object
        Loaded EXIOBASE data
    producer_emissions : np.array
        Producer emissions in tonnes CO2-eq
    v_clean : np.array
        Clean value-added coefficients
    method : str
        'leontief' = matrix inversion (fast, exact)
        'taylor' = layer-by-layer expansion (slow, shows convergence)
    max_layers : int
        Maximum layers for Taylor expansion (default 100)
    
    Returns:
    --------
    results : pd.Series
        Responsibility by country in tonnes CO2-eq
    """
    regions = ixi_data.get_regions()
    n_sectors = len(ixi_data.x)
    
    print(f"\n=== PRODUCER-CENTRIC VABR ({method.upper()}) ===")
    print(f"Mean VA coefficient: {v_clean.mean():.4f}")
    
    # Calculate standard A-matrix (no scaling)
    Z = ixi_data.Z.values
    total_output = ixi_data.x.values.flatten()
    A = np.divide(Z, total_output, 
                  out=np.zeros_like(Z, dtype=float), 
                  where=(total_output != 0))
    
    # Calculate value capture based on method
    if method == 'leontief':
        # Matrix inversion: (I - A^T)^-1
        I = np.eye(n_sectors)
        L = np.linalg.inv(I - A.T)
        
        # Calculate flows and value capture
        flows = L @ producer_emissions
        value_capture = v_clean * flows
        
        print(f"Total value captured: {value_capture.sum()/1e9:.6f} Gt")
    
    elif method == 'taylor':
        # Layer-by-layer expansion: sum(A^T)^k
        value_capture = np.zeros(n_sectors)
        A_power = np.eye(n_sectors)
        
        for layer in range(max_layers):
            # Calculate flows at this layer
            flows = A_power @ producer_emissions
            
            # Value retained at this layer
            retained = v_clean * flows
            value_capture += retained
            
            # Progress reporting
            if layer < 5 or layer % 20 == 0:
                print(f"Layer {layer}: cumulative = {value_capture.sum()/1e9:.6f} Gt")
            
            # Move to next layer
            A_power = A.T @ A_power
            
            # Check convergence
            if np.abs(A_power).max() < 1e-15:
                print(f"Converged at layer {layer+1}")
                break
        
        if layer == max_layers - 1:
            print(f"Warning: Reached max_layers ({max_layers})")
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'leontief' or 'taylor'")
    
    # Normalize to conserve total emissions
    total_emissions = producer_emissions.sum()
    shares = value_capture / value_capture.sum()
    responsibility = total_emissions * shares
    
    # Aggregate by country
    results = {}
    for region in regions:
        mask = ixi_data.x.index.get_level_values(0) == region
        results[region] = responsibility[mask].sum()
    
    # Validation
    total_out = sum(results.values())
    error = abs(total_out - total_emissions) / total_emissions * 100
    
    print(f"\nInput:  {total_emissions/1e9:.3f} Gt")
    print(f"Output: {total_out/1e9:.3f} Gt")
    print(f"Error:  {error:.6f}%")
    
    return pd.Series(results)

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
        "Retail sale of automotive fuel",
        "Incineration of waste: Oil/Hazardous waste",
    ],

    "Gas": [
        "Extraction of natural gas and services related to natural gas extraction, excluding surveying",
        "Extraction, liquefaction, and regasification of other petroleum and gaseous materials",
        "Manufacture of gas; distribution of gaseous fuels through mains",
        "Production of electricity by gas",
        "Transport via pipelines",
        "N-fertiliser",
    ],

    # Electricity & heat infrastructure (system-level)
    "Electricity & Heat Infrastructure": [
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
        "Mining of chemical and fertiliser minerals, production of salt, other mining and quarrying n.e.c.",
    ],

    # Chemicals
    "Chemicals & Plastics": [
        "Chemicals nec",
        "Manufacture of rubber and plastic products (25)",
        "Paper",
        "Pulp",
        "N-fertiliser",  # already included, keep
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
        "Manufacture of textiles (17)",
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
    "Manufacturing (Other)": [
        "Manufacture of furniture; manufacturing n.e.c. (36)",
        "Publishing, printing and reproduction of recorded media (22)",
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
        "Water transport", "Transport via railways", "Sale, maintenance, repair of motor vehicles, motor vehicles parts and motorcycles",
    ],

    # Waste & recycling
    "Waste & Recycling": [
        "Incineration of waste: Food", "Incineration of waste: Metals and Inert materials",
        "Incineration of waste: Paper", "Incineration of waste: Plastic",
        "Incineration of waste: Textiles", "Incineration of waste: Wood",
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
        "Community, social and personal services nec","Households as employers",
        
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
    "Manufacturing (Machinery & Equipment)": "#1f77b4",
    "Manufacturing (Other)": "#aec7e8",
    "Agriculture": "#ffbb78",
    "Construction & Trade": "#98df8a",
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



