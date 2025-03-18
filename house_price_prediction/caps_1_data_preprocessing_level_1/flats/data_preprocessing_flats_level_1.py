import numpy as np
import pandas as pd

pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)

df = pd.read_csv('flats.csv')
df.sample(5)

# shape
df.shape

# info
df.info()

# check for duplicates
df.duplicated().sum()

# check for missing values
df.isnull().sum()

# coiumns to drop - (link, property-id)
df.drop(columns = ['link', 'property_id'], inplace = True)

df.head()

# renames columns
df.rename(columns = {'area':'price_per_sqft'}, inplace = True)

df.head()

# society
df['society'].value_counts()

df['society'].value_counts().shape

import re
df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★','', str(name)).strip()).str.lower()

df['society'].value_counts().shape

df['society'].value_counts()

df.head()

# price
df['price'].value_counts()

df[df['price'] == 'Price on Request']

df = df[df['price'] != 'Price on Request']

df = df[df['price'] != 'price']

df.head()

def treat_price(x):
    if type(x) == float:
        return x
    else:
        if x[1] == 'Lac':
            return round(float(x[0])/100,2)
        else:
            return round(float(x[0]),2)


df['price'] = df['price'].str.split(' ').apply(treat_price)

df.head(5)

# price_per_sqft
df['price_per_sqft'].value_counts()

df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹','').str.replace(',','').str.strip().astype('float')

df.head()

# bedrooms
df['bedRoom'].value_counts()

df[df['bedRoom'].isnull()]

df = df[~df['bedRoom'].isnull()]

df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')

df.head()

# bathroom
df['bathroom'].value_counts()

df['bathroom'].isnull().sum()

df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')

df.head()

# balcony
df['balcony'].value_counts()

df['balcony'].isnull().sum()

df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No','0')

df.head()

# additionalRoom
df['additionalRoom'].value_counts()

df['additionalRoom'].value_counts().shape

df['additionalRoom'].isnull().sum()

df['additionalRoom'].fillna('not available',inplace=True)

df['additionalRoom'] = df['additionalRoom'].str.lower()

df.head()

# floor num
df['floorNum']

df['floorNum'].isnull().sum()

df[df['floorNum'].isnull()]

df['floorNum'] = df['floorNum'].str.split(' ').str.get(0).replace('Ground','0').str.replace('Basement','-1').str.replace('Lower','0').str.extract(r'(\d+)')

df.head()

# facing
df['facing'].value_counts()

df['facing'].isnull().sum()

df['facing'].fillna('NA',inplace=True)

df['facing'].value_counts()

df.insert(loc=4,column='area',value=round((df['price']*10000000)/df['price_per_sqft']))

df.insert(loc=1,column='property_type',value='flat')

df.head()

df.info()

df.shape

df.to_csv('flats_cleaned.csv',index=False)
