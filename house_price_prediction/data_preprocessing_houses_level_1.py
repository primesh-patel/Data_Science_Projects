#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[ ]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[3]:


df = pd.read_csv('house.csv')
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


# In[7]:


df = df.drop_duplicates()


# In[8]:


df.shape


# In[9]:


# check for missing values
df.isnull().sum()


# In[10]:


# Columns to drop -> property_name, link, property_id
df.drop(columns=['link','property_id'], inplace=True)


# In[11]:


df.head()


# In[12]:


# rename columns
df.rename(columns={'rate':'price_per_sqft'},inplace=True)
df.head()


# In[13]:


# society
df['society'].value_counts()


# In[14]:


df['society'].value_counts().shape


# In[15]:


import re
df['society'] = df['society'].apply(lambda name: re.sub(r'\d+(\.\d+)?\s?★', '', str(name)).strip()).str.lower()


# In[16]:


df['society'].value_counts().shape


# In[17]:


df['society'] = df['society'].str.replace('nan','independent')


# In[18]:


df.head()


# In[19]:


# price
df['price'].value_counts()


# In[20]:


df = df[df['price'] != 'Price on Request']


# In[21]:


df.head()


# In[22]:


def treat_price(x):
    if type(x) == float:
        return x
    else:
        if x[1] == 'Lac':
            return round(float(x[0])/100,2)
        else:
            return round(float(x[0]),2)


# In[23]:


df['price'] = df['price'].str.split(' ').apply(treat_price)


# In[24]:


df.head()


# In[25]:


# price_per_sqft
df['price_per_sqft'].value_counts()


# In[26]:


df['price_per_sqft'] = df['price_per_sqft'].str.split('/').str.get(0).str.replace('₹','').str.replace(',','').str.strip().astype('float')


# In[27]:


df.head()


# In[28]:


# bedrooms
df['bedRoom'].value_counts()


# In[29]:


df[df['bedRoom'].isnull()]


# In[30]:


df = df[~df['bedRoom'].isnull()]


# In[31]:


df.shape


# In[32]:


df['bedRoom'] = df['bedRoom'].str.split(' ').str.get(0).astype('int')


# In[33]:


df.head()


# In[34]:


# bathroom
df['bathroom'].value_counts()


# In[35]:


df['bathroom'].isnull().sum()


# In[36]:


df['bathroom'] = df['bathroom'].str.split(' ').str.get(0).astype('int')


# In[37]:


df.head()


# In[38]:


# balcony
df['balcony'].value_counts()


# In[39]:


df['balcony'].isnull().sum()


# In[40]:


df['balcony'] = df['balcony'].str.split(' ').str.get(0).str.replace('No','0')


# In[41]:


df.head()


# In[42]:


# additionalRoom
df['additionalRoom'].value_counts()


# In[43]:


df['additionalRoom'].fillna('not available',inplace=True)


# In[44]:


df['additionalRoom'] = df['additionalRoom'].str.lower()


# In[45]:


df.head()


# In[46]:


# floors
df['noOfFloor'].value_counts()


# In[47]:


df['noOfFloor'].isnull().sum()


# In[48]:


df['noOfFloor'] = df['noOfFloor'].str.split(' ').str.get(0)


# In[49]:


df.head()


# In[50]:


df.rename(columns={'noOfFloor':'floorNum'},inplace=True)


# In[51]:


df.head()


# In[52]:


df['facing'].fillna('NA',inplace=True)


# In[53]:


df['area'] = round((df['price']*10000000)/df['price_per_sqft'])


# In[54]:


df.insert(loc=1,column='property_type',value='house')


# In[55]:


df.head()


# In[56]:


df.shape


# In[57]:


df.info()


# In[ ]:


df.to_csv('house_cleaned.csv',index=False)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




