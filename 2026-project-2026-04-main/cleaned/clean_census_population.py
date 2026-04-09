import csv
import pandas as pd

INPUT = "../raw-census-data/raw_va_census_county_population.csv"
OUTPUT = "va_county_population.csv"

records = []
current_county = None

with open(INPUT, newline='', encoding='utf-8-sig') as f:
    reader = csv.reader(f)
    next(reader) # skip header
    for row in reader:
        label = row[0].strip()
        value = row[1].strip() if len(row) > 1 else ''

        if label.endswith(', Virginia'):
            current_county = label.replace(', Virginia', '')
        elif label == 'Virginia':
            current_county = None  # skip state total
        elif label == 'Estimate' and current_county:
            records.append({'county': current_county, 'population': int(value.replace(',', ''))})
            current_county = None  # one estimate per county, skip remaining rows

df = pd.DataFrame(records).sort_values('county').reset_index(drop=True)
print(df)
df.to_csv(OUTPUT, index=False)
