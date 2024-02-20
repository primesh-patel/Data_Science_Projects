#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[3]:


flats = pd.read_csv('flats_cleaned.csv')
houses = pd.read_csv('house_cleaned.csv')


# In[4]:


df = pd.concat([flats,houses],ignore_index=True)


# In[6]:


df = df.sample(df.shape[0],ignore_index=True)


# In[9]:


df.head()


# In[ ]:


df.to_csv('gurgaon_properties.csv',index=False)


# In[ ]:




