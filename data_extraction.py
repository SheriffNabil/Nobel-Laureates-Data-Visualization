import json
from datetime import datetime
import pandas as pd



with open('laureates.json') as f:
    data = json.load(f)

# Flatten the JSON data
flat_data = []
for d in data['laureates']:
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                flat_dict[f"{key}_{sub_key}"] = sub_value
        else:
            flat_dict[key] = value
    flat_data.append(flat_dict)


df = pd.DataFrame(flat_data)


def extract_nested_columns():
    nested_cols = []
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, dict) or isinstance(x, list)).any():
            nested_cols.append(col)
    return nested_cols


def expand_nested_columns():
    for col in extract_nested_columns():
        sample_value = df[col][0]
        if isinstance(sample_value, dict):
            for key in sample_value.keys():
                new_col_name = f"{col}_{key}"
                df[new_col_name] = df[col].apply(lambda x: x.get(key) if isinstance(x, dict) else None)
        elif isinstance(sample_value, list):
            for i, _ in enumerate(sample_value):
                new_col_name = f"{col}_{i}"
                df[new_col_name] = df[col].apply(lambda x: x[i] if isinstance(x, list) and len(x) > i else None)


for _ in range(200):
    expand_nested_columns()


def calculate_age(row):
    birth_date = str(row['birth_date'])
    death_date = str(row['death_date'])
    try:
        birth_date = datetime.strptime(birth_date, '%Y-%m-%d')
        death_date = datetime.strptime(death_date, '%Y-%m-%d')
        return (death_date - birth_date).days / 365.25
    except ValueError:
        return None

df['age'] = df.apply(calculate_age, axis=1)


df = df.filter(regex='^(?!.*(_se|_no)$).*$')

