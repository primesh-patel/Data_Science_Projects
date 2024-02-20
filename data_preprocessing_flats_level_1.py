#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[19]:


pd.set_option('display.max_rows', None)

pd.set_option('display.max_columns', None)


# In[3]:


df = pd.read_csv('flats.csv')
df.sample(5)


# In[4]:


# shape
df.shape


# In[5]:


# info
df.info()


# In[6]:


# check for duplicates
df.duplicated().sum()


# In[8]:


# check for missing values
df.isnull().sum()


# In[11]:


# coiumns to drop - (link, property-id)
df.drop(columns = ['link', 'property_id'], inplace = True)


# In[13]:


df.head()


# In[16]:


# renames columns
df.rename(columns = {'area':'price_per_sqft'}, inplace = True)


# In[17]:


df.head()


# In[20]:


# society
df['society'].value_counts()


# In[22]:


df['society'].value_counts().shape


# In[23]:


import re
df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★','', str(name)).strip()).str.lower()


# In[24]:


df['society'].value_counts().shape


# In[25]:


df['society'].value_counts()


# In[26]:


df.head()


# In[28]:


# price
df['price'].value_counts()


# In[30]:


df[df['price'] == 'Price on Request']


# In[31]:


df = df[df['price'] != 'Price on Request']


# In[40]:


df = df[df['price'] != 'price']


# In[32]:


df.head()


# In[42]:


def treat_price(x):
    if type(x) == float:
        return x
    else:
        if x[1] == 'Lac':
            return round(float(x[0])/100,2)
        else:
            return round(float(x[0]),2)


# In[43]:


df['price'] = df['price'].str.split(' ').apply(treat_price)


# In[44]:


df.head(5)


# In[45]:


# price_per_sqft
df['price_per_sqft'].value_counts()


# In[53]:


df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹','').str.replace(',','').str.strip().astype('float')


# In[54]:


df.head()


# In[55]:


# bedrooms
df['bedRoom'].value_counts()


# In[58]:


df[df['bedRoom'].isnull()]


# In[60]:


df = df[~df['bedRoom'].isnull()]


# In[62]:


df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')


# In[63]:


df.head()


# In[64]:


# bathroom
df['bathroom'].value_counts()


# In[65]:


df['bathroom'].isnull().sum()


# In[66]:


df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')


# In[67]:


df.head()


# In[68]:


# balcony
df['balcony'].value_counts()


# In[70]:


df['balcony'].isnull().sum()


# In[71]:


df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No','0')


# In[73]:


df.head()


# In[74]:


# additionalRoom
df['additionalRoom'].value_counts()


# In[75]:


df['additionalRoom'].value_counts().shape


# In[76]:


df['additionalRoom'].isnull().sum()


# In[77]:


df['additionalRoom'].fillna('not available',inplace=True)


# In[78]:


df['additionalRoom'] = df['additionalRoom'].str.lower()


# In[79]:


df.head()


# In[80]:


# floor num
df['floorNum']


# In[81]:


df['floorNum'].isnull().sum()


# In[82]:


df[df['floorNum'].isnull()]


# In[83]:


df['floorNum'] = df['floorNum'].str.split(' ').str.get(0).replace('Ground','0').str.replace('Basement','-1').str.replace('Lower','0').str.extract(r'(\d+)')


# In[84]:


df.head()


# In[85]:


# facing
df['facing'].value_counts()


# In[87]:


df['facing'].isnull().sum()


# In[88]:


df['facing'].fillna('NA',inplace=True)


# In[89]:


df['facing'].value_counts()


# In[91]:


df.insert(loc=4,column='area',value=round((df['price']*10000000)/df['price_per_sqft']))


# In[92]:


df.insert(loc=1,column='property_type',value='flat')


# In[93]:


df.head()


# In[94]:


df.info()


# In[95]:


df.shape


# In[ ]:


df.to_csv('flats_cleaned.csv',index=False)

