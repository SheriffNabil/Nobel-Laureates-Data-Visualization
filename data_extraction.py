#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[3]:


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


def extract_affiliation(row):
    affiliation = None
    affiliation_country = None
    if 'nobelPrizes' in row:
        for prize_info in row['nobelPrizes']:
            if 'affiliations' in prize_info:
                for affiliation_info in prize_info['affiliations']:
                    affiliation = affiliation_info['nameNow']['en']
                    if 'countryNow' in affiliation_info:
                        affiliation_country = affiliation_info['countryNow']['en']
                    else:
                        affiliation_country = None
    return (affiliation, affiliation_country)

df[['affiliation', 'affiliation_country']] = pd.DataFrame(df.apply(extract_affiliation, axis=1).tolist(), index=df.index)


normalized_df = json_normalize(df['nobelPrizes'].explode())


merged_df = df.join(normalized_df, rsuffix='_normalized')



