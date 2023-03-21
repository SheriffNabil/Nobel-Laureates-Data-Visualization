import pandas as pd
from datetime import datetime
from pandas.io.json import json_normalize
import json


with open('laureates.json') as f:
    data = json.load(f)

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



def lists_or_dict():
    dict_list_cols = []
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, dict) or isinstance(x, list)).any():
            dict_list_cols.append(col)
    return dict_list_cols


lists_or_dict()


def extract_lists_dicts():
    dict_list_cols = []
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, dict) or isinstance(x, list)).any():
            if isinstance(df[col][0], dict):
                for key in df[col][0].keys():
                    new_col_name = col + '_' + key
                    df[new_col_name] = df[col].apply(lambda x: x.get(key) if isinstance(x, dict) else None)
                    dict_list_cols.append(new_col_name)
            elif isinstance(df[col][0], list):
                for i in range(len(df[col][0])):
                    new_col_name = col + '_' + str(i)
                    df[new_col_name] = df[col].apply(lambda x: x[i] if isinstance(x, list) and len(x) > i else None)
                    dict_list_cols.append(new_col_name)
                    
                    
                    
for i in range(1,200):
    extract_lists_dicts()
    
    
    
    
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
