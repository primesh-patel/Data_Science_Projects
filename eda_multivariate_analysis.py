#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


pd.set_option('display.max_columns', None)


# In[2]:


df = pd.read_csv('gurgaon_properties_cleaned_v2.csv').drop_duplicates()


# In[3]:


df.head()


# ### property_type vs price

# In[17]:


sns.barplot(x=df['property_type'], y=df['price'], estimator='median')


# In[6]:


sns.boxplot(x=df['property_type'], y=df['price'])


# ### property_type vs area

# In[19]:


sns.barplot(x=df['property_type'], y=df['built_up_area'], estimator='median')


# In[20]:


sns.boxplot(x=df['property_type'], y=df['built_up_area'])


# In[21]:


# removing that crazy outlier
df = df[df['built_up_area'] != 737147]


# In[22]:


sns.boxplot(x=df['property_type'], y=df['built_up_area'])


# ### property_type vs price_per_sqft

# In[23]:


sns.barplot(x=df['property_type'], y=df['price_per_sqft'], estimator='median')


# In[24]:


sns.boxplot(x=df['property_type'], y=df['price_per_sqft'])


# In[25]:


# check outliers
df[df['price_per_sqft'] > 100000][['property_type','society','sector','price','price_per_sqft','area','areaWithType', 'super_built_up_area', 'built_up_area', 'carpet_area']]


# In[26]:


df.head()


# In[27]:


sns.heatmap(pd.crosstab(df['property_type'],df['bedRoom']))


# In[28]:


# checking outliers
df[df['bedRoom'] >= 10]


# In[29]:


sns.barplot(x=df['property_type'],y=df['floorNum'])


# In[30]:


sns.boxplot(x=df['property_type'],y=df['floorNum'])


# In[31]:


# checking for outliers
df[(df['property_type'] == 'house') & (df['floorNum'] > 10)]


# In[ ]:


# conclusion houses(villa) but in appartments


# In[32]:


df.head()


# In[33]:


sns.heatmap(pd.crosstab(df['property_type'],df['agePossession']))


# In[34]:


sns.heatmap(pd.pivot_table(df,index='property_type',columns='agePossession',values='price',aggfunc='mean'),annot=True)


# In[35]:


plt.figure(figsize=(15,4))
sns.heatmap(pd.pivot_table(df,index='property_type',columns='bedRoom',values='price',aggfunc='mean'),annot=True)


# In[36]:


sns.heatmap(pd.crosstab(df['property_type'],df['furnishing_type']))


# In[37]:


sns.heatmap(pd.pivot_table(df,index='property_type',columns='furnishing_type',values='price',aggfunc='mean'),annot=True)


# In[38]:


sns.barplot(x=df['property_type'],y=df['luxury_score'])


# In[39]:


sns.boxplot(x=df['property_type'],y=df['luxury_score'])


# In[40]:


df.head()


# In[41]:


# sector analysis
plt.figure(figsize=(15,6))
sns.heatmap(pd.crosstab(df['property_type'],df['sector'].sort_index()))


# In[42]:


# sector analysis
import re
# Group by 'sector' and calculate the average price
avg_price_per_sector = df.groupby('sector')['price'].mean().reset_index()

# Function to extract sector numbers
def extract_sector_number(sector_name):
    match = re.search(r'\d+', sector_name)
    if match:
        return int(match.group())
    else:
        return float('inf')  # Return a large number for non-numbered sectors

avg_price_per_sector['sector_number'] = avg_price_per_sector['sector'].apply(extract_sector_number)

# Sort by sector number
avg_price_per_sector_sorted_by_sector = avg_price_per_sector.sort_values(by='sector_number')

# Plot the heatmap
plt.figure(figsize=(5, 25))
sns.heatmap(avg_price_per_sector_sorted_by_sector.set_index('sector')[['price']], annot=True, fmt=".2f", linewidths=.5)
plt.title('Average Price per Sector (Sorted by Sector Number)')
plt.xlabel('Average Price')
plt.ylabel('Sector')
plt.show()


# In[43]:


avg_price_per_sqft_sector = df.groupby('sector')['price_per_sqft'].mean().reset_index()

avg_price_per_sqft_sector['sector_number'] = avg_price_per_sqft_sector['sector'].apply(extract_sector_number)

# Sort by sector number
avg_price_per_sqft_sector_sorted_by_sector = avg_price_per_sqft_sector.sort_values(by='sector_number')

# Plot the heatmap
plt.figure(figsize=(5, 25))
sns.heatmap(avg_price_per_sqft_sector_sorted_by_sector.set_index('sector')[['price_per_sqft']], annot=True, fmt=".2f", linewidths=.5)
plt.title('Sector (Sorted by Sector Number)')
plt.xlabel('Average Price per sqft')
plt.ylabel('Sector')
plt.show()


# In[44]:


luxury_score = df.groupby('sector')['luxury_score'].mean().reset_index()

luxury_score['sector_number'] = luxury_score['sector'].apply(extract_sector_number)

# Sort by sector number
luxury_score_sector = luxury_score.sort_values(by='sector_number')

# Plot the heatmap
plt.figure(figsize=(5, 25))
sns.heatmap(luxury_score_sector.set_index('sector')[['luxury_score']], annot=True, fmt=".2f", linewidths=.5)
plt.title('Sector (Sorted by Sector Number)')
plt.xlabel('Average Price per sqft')
plt.ylabel('Sector')
plt.show()


# In[45]:


df.head()


# ### price

# In[47]:


plt.figure(figsize=(12, 8))
sns.scatterplot(data=df[df['area'] < 10000], x='area', y='price', hue='bedRoom')
plt.show()


# In[49]:


plt.figure(figsize=(12, 8))
sns.scatterplot(data=df[df['area'] < 10000], x='area', y='price', hue='agePossession')
plt.show()


# In[51]:


plt.figure(figsize=(12, 8))
sns.scatterplot(data=df[df['area'] < 10000], x='area', y='price', hue='furnishing_type')
plt.show()


# In[53]:


sns.barplot(x=df['bedRoom'],y=df['price'],estimator='median')


# In[55]:


sns.barplot(x=df['agePossession'],y=df['price'],estimator='median')
plt.xticks(rotation='vertical')
plt.show()


# In[56]:


sns.barplot(x=df['agePossession'],y=df['area'],estimator='median')
plt.xticks(rotation='vertical')
plt.show()


# In[57]:


sns.barplot(x=df['furnishing_type'],y=df['price'],estimator='median')


# In[60]:


sns.scatterplot(x=df['luxury_score'], y=df['price'])


# ### correlation

# In[61]:


plt.figure(figsize=(8,8))
sns.heatmap(df.corr())


# In[62]:


df.corr()['price'].sort_values(ascending=False)


# In[63]:


df.head()


# In[64]:


sns.pairplot(df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




