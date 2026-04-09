import csv
import pandas as pd

INPUT = "../raw-census-data/raw_va_census_county_income.csv"
OUTPUT = "va_county_median_household_income.csv"

records = []
current_county = None
in_households = False

with open(INPUT, newline='', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    for row in reader:
        label = row[0].strip()
        value = row[1].strip() if len(row) > 1 else ''

        if label.endswith(', Virginia'):
            current_county = label.replace(', Virginia', '')
            in_households = False
        elif label == 'Virginia':
            current_county = None  # skip state total
        elif label == 'Households':
            in_households = True
        elif label == 'Estimate' and in_households and current_county:
            income = int(value.replace(',', ''))
            records.append({'county': current_county, 'median_household_income': income})
            in_households = False

df = pd.DataFrame(records).sort_values('county').reset_index(drop=True)
print(df)
df.to_csv(OUTPUT, index=False)
