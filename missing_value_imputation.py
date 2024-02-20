#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


pd.set_option('display.max_columns', None)


# In[2]:


df = pd.read_csv('gurgaon_properties_outlier_treated.csv')


# In[3]:


df.head()


# In[4]:


df.isnull().sum()


# ### Built up area

# In[6]:


sns.scatterplot(x=df['built_up_area'], y=df['super_built_up_area'])
plt.show()


# In[7]:


sns.scatterplot(x=df['built_up_area'], y=df['carpet_area'])


# In[8]:


((df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull()))


# In[9]:


all_present_df = df[~((df['super_built_up_area'].isnull()) | (df['built_up_area'].isnull()) | (df['carpet_area'].isnull()))]


# In[10]:


all_present_df.shape


# In[11]:


super_to_built_up_ratio = (all_present_df['super_built_up_area']/all_present_df['built_up_area']).median()


# In[12]:


carpet_to_built_up_ratio = (all_present_df['carpet_area']/all_present_df['built_up_area']).median()


# In[13]:


print(super_to_built_up_ratio, carpet_to_built_up_ratio)


# In[14]:


# both present built up null
sbc_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]


# In[15]:


sbc_df.head()


# In[16]:


sbc_df['built_up_area'].fillna(round(((sbc_df['super_built_up_area']/1.105) + (sbc_df['carpet_area']/0.9))/2),inplace=True)


# In[17]:


df.update(sbc_df)


# In[18]:


df.isnull().sum()


# In[19]:


# sb present c is null built up null
sb_df = df[~(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & (df['carpet_area'].isnull())]


# In[20]:


sb_df.head()


# In[21]:


sb_df['built_up_area'].fillna(round(sb_df['super_built_up_area']/1.105),inplace=True)


# In[22]:


df.update(sb_df)


# In[23]:


df.isnull().sum()


# In[24]:


# sb null c is present built up null
c_df = df[(df['super_built_up_area'].isnull()) & (df['built_up_area'].isnull()) & ~(df['carpet_area'].isnull())]


# In[25]:


c_df.head()


# In[26]:


c_df['built_up_area'].fillna(round(c_df['carpet_area']/0.9),inplace=True)


# In[27]:


df.update(c_df)


# In[28]:


df.isnull().sum()


# In[30]:


sns.scatterplot(x=df['built_up_area'], y=df['price'])
plt.show()


# In[31]:


anamoly_df = df[(df['built_up_area'] < 2000) & (df['price'] > 2.5)][['price','area','built_up_area']]


# In[32]:


anamoly_df.sample(5)


# In[33]:


anamoly_df['built_up_area'] = anamoly_df['area']


# In[34]:


df.update(anamoly_df)


# In[36]:


sns.scatterplot(x=df['built_up_area'], y=df['price'])
plt.show()


# In[37]:


df.drop(columns=['area','areaWithType','super_built_up_area','carpet_area','area_room_ratio'],inplace=True)


# In[38]:


df.head()


# In[39]:


df.isnull().sum()


# ### floorNum

# In[40]:


df[df['floorNum'].isnull()]


# In[41]:


df[df['property_type'] == 'house']['floorNum'].median()


# In[42]:


df['floorNum'].fillna(2.0,inplace=True)


# In[43]:


df.isnull().sum()


# In[44]:


1011/df.shape[0]


# ### facing

# In[46]:


df['facing'].value_counts().plot(kind='pie',autopct='%0.2f%%')
plt.show()


# In[47]:


df.drop(columns=['facing'],inplace=True)


# In[48]:


df.sample(5)


# In[49]:


df.isnull().sum()


# In[50]:


df.drop(index=[2536],inplace=True)


# In[51]:


df.isnull().sum()


# ### agePossession

# In[52]:


df['agePossession'].value_counts()


# In[53]:


df[df['agePossession'] == 'Undefined']


# In[54]:


def mode_based_imputation(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector']) & (df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']


# In[55]:


df['agePossession'] = df.apply(mode_based_imputation,axis=1)


# In[56]:


df['agePossession'].value_counts()


# In[57]:


def mode_based_imputation2(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['sector'] == row['sector'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']


# In[58]:


df['agePossession'] = df.apply(mode_based_imputation2,axis=1)


# In[59]:


df['agePossession'].value_counts()


# In[60]:


def mode_based_imputation3(row):
    if row['agePossession'] == 'Undefined':
        mode_value = df[(df['property_type'] == row['property_type'])]['agePossession'].mode()
        # If mode_value is empty (no mode found), return NaN, otherwise return the mode
        if not mode_value.empty:
            return mode_value.iloc[0] 
        else:
            return np.nan
    else:
        return row['agePossession']


# In[61]:


df['agePossession'] = df.apply(mode_based_imputation3,axis=1)


# In[62]:


df['agePossession'].value_counts()


# In[63]:


df.isnull().sum()


# In[64]:


df.to_csv('gurgaon_properties_missing_value_imputation.csv',index=False)


# In[65]:


df.shape

