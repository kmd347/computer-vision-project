import pandas as pd

T7_PATH = "../raw-eia-data/EIA_Electric_Sales_Revenue_Average_Price_table_7.csv"
SERVICE_TERRITORY_PATH = "../raw-eia-data/EIA_Service_Territory_2024.csv"

t7 = pd.read_csv(T7_PATH, skiprows=2, thousands=',')
svc = pd.read_csv(SERVICE_TERRITORY_PATH)

# Normalize col names
t7.columns = t7.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)
svc.columns = svc.columns.str.strip().str.lower().str.replace(r'\s+', '_', regex=True)

# Coerce sales col to numeric /some rows have '.' as placeholder
t7['sales_(megawatthours)'] = pd.to_numeric(t7['sales_(megawatthours)'], errors='coerce')

# Filter to VA
t7_va = t7[t7['state'] == 'VA'].copy()
svc_va = svc[svc['state'] == 'VA'].copy()

# Join on utility name (SERVICE_TERRITORY) = entity (T7)
merged = svc_va.merge(t7_va[['entity', 'sales_(megawatthours)']],
                      left_on='utility_name', right_on='entity', how='left')

# Sum commercial sales across all utilities serving each county
county_sales = (
    merged
    .groupby('county')['sales_(megawatthours)']
    .sum()
    .reset_index()
    .rename(columns={'sales_(megawatthours)': 'commercial_sales_mwh'})
    .sort_values('county')
)

print(county_sales)
county_sales.to_csv("va_county_commercial_sales_mwh.csv", index=False)