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


df = pd.read_csv('gurgaon_properties_cleaned_v2.csv').drop_duplicates()


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.columns


# In[6]:


# outliers on the basis of price column
sns.distplot(df['price'])


# In[7]:


sns.boxplot(x=df['price'])


# In[8]:


# Calculate the IQR for the 'price' column
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]

# Displaying the number of outliers and some statistics
num_outliers = outliers.shape[0]
outliers_price_stats = outliers['price'].describe()

num_outliers, outliers_price_stats


# In[9]:


outliers.sort_values('price',ascending=False).head(20)


# In[ ]:


# on the basis of price col we can say that there are some genuine outliers but there are some data erros as well


# ### Price_per_sqft

# In[10]:


sns.distplot(df['price_per_sqft'])


# In[11]:


sns.boxplot(x=df['price_per_sqft'])


# In[12]:


# Calculate the IQR for the 'price' column
Q1 = df['price_per_sqft'].quantile(0.25)
Q3 = df['price_per_sqft'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers_sqft = df[(df['price_per_sqft'] < lower_bound) | (df['price_per_sqft'] > upper_bound)]

# Displaying the number of outliers and some statistics
num_outliers = outliers_sqft.shape[0]
outliers_sqft_stats = outliers_sqft['price_per_sqft'].describe()

num_outliers, outliers_sqft_stats


# In[13]:


outliers_sqft['area'] = outliers_sqft['area'].apply(lambda x:x*9 if x<1000 else x)


# In[14]:


outliers_sqft['price_per_sqft'] = round((outliers_sqft['price']*10000000)/outliers_sqft['area'])


# In[15]:


outliers_sqft['price_per_sqft'].describe()


# In[16]:


df.update(outliers_sqft)


# In[17]:


sns.distplot(df['price_per_sqft'])


# In[18]:


sns.boxplot(x=df['price_per_sqft'])


# In[19]:


df[df['price_per_sqft']>50000]


# In[20]:


df = df[df['price_per_sqft'] <= 50000]


# In[21]:


sns.boxplot(x=df['price_per_sqft'])


# ### Area

# In[22]:


sns.distplot(df['area'])


# In[23]:


sns.boxplot(x=df['area'])


# In[24]:


df['area'].describe()


# In[25]:


df[df['area'] > 100000]


# In[26]:


df = df[df['area'] < 100000]


# In[27]:


sns.distplot(df['area'])


# In[28]:


sns.boxplot(x=df['area'])


# In[29]:


df[df['area'] > 10000].sort_values('area',ascending=False)

# 818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471


# In[30]:


df.drop(index=[818, 1796, 1123, 2, 2356, 115, 3649, 2503, 1471], inplace=True)


# In[31]:


df[df['area'] > 10000].sort_values('area',ascending=False)


# In[32]:


df.loc[48,'area'] = 115*9
df.loc[300,'area'] = 7250
df.loc[2666,'area'] = 5800
df.loc[1358,'area'] = 2660
df.loc[3195,'area'] = 2850
df.loc[2131,'area'] = 1812
df.loc[3088,'area'] = 2160
df.loc[3444,'area'] = 1175


# In[33]:


sns.distplot(df['area'])


# In[34]:


sns.boxplot(x=df['area'])


# In[35]:


df['area'].describe()


# ### Bedroom

# In[36]:


sns.distplot(df['bedRoom'])


# In[37]:


sns.boxplot(x=df['bedRoom'])


# In[38]:


df['bedRoom'].describe()


# In[39]:


df[df['bedRoom'] > 10].sort_values('bedRoom',ascending=False)


# In[40]:


df = df[df['bedRoom'] <= 10]


# In[41]:


df.shape


# In[42]:


sns.distplot(df['bedRoom'])


# In[43]:


sns.boxplot(x=df['bedRoom'])


# In[44]:


df['bedRoom'].describe()


# ### Bathroom

# In[45]:


sns.distplot(df['bathroom'])


# In[46]:


sns.boxplot(x=df['bathroom'])


# In[47]:


df[df['bathroom'] > 10].sort_values('bathroom',ascending=False)


# In[48]:


df.head()


# ### super built up area

# In[49]:


sns.distplot(df['super_built_up_area'])


# In[50]:


sns.boxplot(x=df['super_built_up_area'])


# In[51]:


df['super_built_up_area'].describe()


# In[52]:


df[df['super_built_up_area'] > 6000]


# ### built up area

# In[53]:


sns.distplot(df['built_up_area'])


# In[54]:


sns.boxplot(x=df['built_up_area'])


# In[55]:


df[df['built_up_area'] > 10000]


# ### carpet area

# In[56]:


sns.distplot(df['carpet_area'])


# In[57]:


sns.boxplot(x=df['carpet_area'])


# In[58]:


df[df['carpet_area'] > 10000]


# In[59]:


df.loc[2131,'carpet_area'] = 1812


# In[60]:


df[df['carpet_area'] > 10000]


# In[61]:


df.head()


# In[62]:


sns.distplot(df['luxury_score'])


# In[63]:


sns.boxplot(df['luxury_score'])


# In[64]:


df.shape


# In[65]:


df['price_per_sqft'] = round((df['price']*10000000)/df['area'])


# In[66]:


df.head()


# In[67]:


sns.distplot(df['price_per_sqft'])


# In[68]:


sns.boxplot(df['price_per_sqft'])


# In[69]:


df[df['price_per_sqft'] > 42000]


# In[70]:


x = df[df['price_per_sqft'] <= 20000]
(x['area']/x['bedRoom']).quantile(0.02)


# In[71]:


df[(df['area']/df['bedRoom'])<183]


# In[ ]:





# In[ ]:




