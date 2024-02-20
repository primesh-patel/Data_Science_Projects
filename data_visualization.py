#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[49]:


df = pd.read_csv('gurgaon_properties_missing_value_imputation.csv')


# In[50]:


df.shape


# In[51]:


df.head()


# In[5]:


latlong = pd.read_csv('latlong.csv')


# In[6]:


latlong


# In[7]:


latlong['latitude'] = latlong['coordinates'].str.split(',').str.get(0).str.split('°').str.get(0).astype('float')


# In[8]:


latlong['longitude'] = latlong['coordinates'].str.split(',').str.get(1).str.split('°').str.get(0).astype('float')


# In[9]:


latlong.head()


# In[10]:


new_df = df.merge(latlong, on='sector')


# In[11]:


new_df.columns


# In[12]:


group_df = new_df.groupby('sector').mean()[['price','price_per_sqft','built_up_area','latitude','longitude']]


# In[13]:


group_df


# In[14]:


fig = px.scatter_mapbox(group_df, lat="latitude", lon="longitude", color="price_per_sqft", size='built_up_area',
                  color_continuous_scale=px.colors.cyclical.IceFire, zoom=10,
                  mapbox_style="open-street-map",text=group_df.index)
fig.show()


# In[15]:


new_df.to_csv('data_viz1.csv',index=False)


# In[16]:


df1 = pd.read_csv('gurgaon_properties.csv')


# In[17]:


df1.head()


# In[18]:


wordcloud_df = df1.merge(df, left_index=True, right_index=True)[['features','sector']]


# In[19]:


wordcloud_df.head()


# In[20]:


import ast
main = []
for item in wordcloud_df['features'].dropna().apply(ast.literal_eval):
    main.extend(item)


# In[21]:


main


# In[22]:


from wordcloud import WordCloud


# In[23]:


feature_text = ' '.join(main)


# In[24]:


import pickle
pickle.dump(feature_text, open('feature_text.pkl','wb'))


# In[25]:


feature_text


# In[26]:


plt.rcParams["font.family"] = "Arial"

wordcloud = WordCloud(width = 800, height = 800, 
                      background_color ='white', 
                      stopwords = set(['s']),  # Any stopwords you'd like to exclude
                      min_font_size = 10).generate(feature_text)

plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud, interpolation='bilinear') 
plt.axis("off") 
plt.tight_layout(pad = 0) 
plt.show() # st.pyplot()


# In[29]:


data = dict(
    names=["A", "B", "C", "D", "E", "F"],
    parents=["", "", "", "A", "A", "C"],
    values=[10, 20, 30, 40, 50, 60],
)

df = pd.DataFrame(data)

fig = px.sunburst(
    df,
    names='names',
    parents='parents',
    values='values',
    title="Sample Sunburst Chart"
)

fig.show()


# In[52]:


fig = px.scatter(df, x="built_up_area", y="price", color="bedRoom", title="Area Vs Price")

# Show the plot
fig.show()


# In[53]:


fig = px.pie(df, names='bedRoom', title='Total Bill Amount by Day')

# Show the plot
fig.show()


# In[54]:


temp_df = df[df['bedRoom'] <= 4]
# Create side-by-side boxplots of the total bill amounts by day
fig = px.box(temp_df, x='bedRoom', y='price', title='BHK Price Range')

# Show the plot
fig.show()


# In[55]:


sns.distplot(df[df['property_type'] == 'house']['price'])
sns.distplot(df[df['property_type'] == 'flat']['price'])


# In[56]:


new_df['sector'].unique().tolist().insert(0,'overall')


# In[57]:


new_df


# In[ ]:





# In[ ]:




