#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install ydata-profiling')


# In[2]:


import pandas as pd
from ydata_profiling import ProfileReport

df = pd.read_csv('gurgaon_properties_cleaned_v2.csv').drop_duplicates()
profile = ProfileReport(df, title="Profiling Report", explorative=True)


# In[3]:


profile.to_file("output_report.html")

