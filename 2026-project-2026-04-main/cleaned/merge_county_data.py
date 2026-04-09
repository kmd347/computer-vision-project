import pandas as pd

# Helpers
# Normalize to 'name county' / 'name city' (lowercase)
def norm(s):
    s = str(s).strip()
    s = s.replace(', VA', '').replace(', Virginia', '').strip()
    return s.lower()

def norm_slope(s):
    # SLOPE files use bare names -> Add ' county' to everything and then fix true cities
    # Note 'James City' & 'Charles City' are counties NOT cities 
    s = str(s).strip().lower()
    # Only these 2 are explicitly marked as independent cities in the SLOPE files
    if s in ('fairfax city', 'roanoke city'):
        return s
    return s + ' county'

# Chesapeake and Fredericksburg appear as cities in VA_DataCenters but w/o 'City' suffix in SLOPE files
SLOPE_CITY_FIXES = {
    'chesapeake county':     'chesapeake city',
    'fredericksburg county': 'fredericksburg city',
}

# Convert mixed-unit generation strings (MWh/GWh/TWh) to float GWh
def to_gwh(s):
    s = str(s).strip()
    if 'TWh' in s:
        return float(s.replace(' TWh', '')) * 1000
    elif 'GWh' in s:
        return float(s.replace(' GWh', ''))
    elif 'MWh' in s:
        return float(s.replace(' MWh', '')) / 1000
    return None

# Random typos/missing suffixes in VA_DataCenters County col
DC_FIXES = {
    'clark county': 'clarke county',
    'loudoun': 'loudoun county',
}

# Load & key
dc = pd.read_csv('VA_DataCenters.csv')
dc['_key'] = dc['County'].apply(norm).replace(DC_FIXES)

pop = pd.read_csv('va_county_population.csv')
pop['_key'] = pop['county'].apply(norm)

density = pd.read_csv('va_county_population_density.csv')
density['_key'] = density['county'].apply(norm)

income = pd.read_csv('va_county_median_household_income.csv')
income['_key'] = income['county'].apply(norm)

gen = pd.read_csv('va_county_electricity_generation.csv')
gen['_key'] = gen['County'].apply(norm)
gen['net_gen_gwh'] = gen['Net Generation  (Past Year)'].apply(to_gwh)

elec = pd.read_csv('va_county_commercial_electricity_consumption_2025.csv')
elec['_key'] = elec['County Name'].apply(norm_slope).replace(SLOPE_CITY_FIXES)
elec = elec.groupby('_key', as_index=False)['Consumption MMBtu'].sum()

gas = pd.read_csv('va_county_commercial_natural_gas_consumption_2025.csv')
gas['_key'] = gas['County Name'].apply(norm_slope).replace(SLOPE_CITY_FIXES)
gas = gas.groupby('_key', as_index=False)['Consumption MMBtu'].sum()

# Merge everything into VA_DataCenters
merged = dc.copy()
merged = merged.merge(pop[['_key', 'population']],                   on='_key', how='left')
merged = merged.merge(density[['_key', 'densityMi']],               on='_key', how='left')
merged = merged.merge(income[['_key', 'median_household_income']],   on='_key', how='left')
merged = merged.merge(gen[['_key', 'net_gen_gwh']],                  on='_key', how='left')
merged = merged.merge(elec[['_key', 'Consumption MMBtu']].rename(columns={'Consumption MMBtu': 'elec_mmbtu'}),
                      on='_key', how='left')
merged = merged.merge(gas[['_key', 'Consumption MMBtu']].rename(columns={'Consumption MMBtu': 'gas_mmbtu'}),
                      on='_key', how='left')

merged = merged.drop(columns=['_key'])
merged = merged.rename(columns={
    'population':              'county_population',
    'densityMi':               'county_population_density',
    'median_household_income': 'county_median_household_income',
    'net_gen_gwh':             'county_electricity_generation_GWh_04_25_to_04_26',
    'elec_mmbtu':              'county_commercial_electricity_consumption_mmbtu',
    'gas_mmbtu':               'county_commercial_natural_gas_consumption_mmbtu',
})

print(merged.head())
print(f"\nShape: {merged.shape}")
print(f"\nNull counts:\n{merged.isnull().sum()}")
merged.to_csv('../data_centers_all.csv', index=False)
