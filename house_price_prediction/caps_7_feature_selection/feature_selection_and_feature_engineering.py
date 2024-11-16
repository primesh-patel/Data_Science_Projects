#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:


df = pd.read_csv('gurgaon_properties_missing_value_imputation.csv')


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


train_df = df.drop(columns=['society','price_per_sqft'])


# In[7]:


train_df.head()


# In[8]:


sns.heatmap(train_df.corr())


# In[9]:


train_df.corr()['price'].sort_values(ascending=False)


# In[ ]:


# cols in question

# numerical -> luxury_score, others, floorNum
# categorical -> property_type, sector, agePossession

