#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from sklearn.decomposition import PCA


# In[2]:


df = pd.read_csv('gurgaon_properties_post_feature_selection_v2.csv').drop(columns=['store room','floor_category','balcony'])


# In[3]:


df.head()


# In[4]:


# 0 -> unfurnished
# 1 -> semifurnished
# 2 -> furnished


# In[5]:


# Numerical = bedRoom, bathroom, built_up_area, servant room
# Ordinal = property_type, furnishing_type, luxury_category 
# OHE = sector, agePossession


# In[7]:


df['agePossession'] = df['agePossession'].replace(
    {
        'Relatively New':'new',
        'Moderately Old':'old',
        'New Property' : 'new',
        'Old Property' : 'old',
        'Under Construction' : 'under construction'
    }
)


# In[8]:


df.head()


# In[9]:


df['property_type'] = df['property_type'].replace({'flat':0,'house':1})


# In[10]:


df.head()


# In[11]:


df['luxury_category'] = df['luxury_category'].replace({'Low':0,'Medium':1,'High':2})


# In[12]:


df.head()


# In[13]:


new_df = pd.get_dummies(df,columns=['sector','agePossession'],drop_first=True)


# In[14]:


X = new_df.drop(columns=['price'])
y = new_df['price']


# In[15]:


y_log = np.log1p(y)


# In[16]:


y_log


# In[17]:


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# In[18]:


X_scaled = pd.DataFrame(X_scaled,columns=X.columns)


# In[19]:


X_scaled


# In[20]:


kfold = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(LinearRegression(), X_scaled, y_log, cv=kfold, scoring='r2')


# In[21]:


scores.mean(),scores.std()


# In[22]:


lr = LinearRegression()
ridge = Ridge(alpha=0.0001)


# In[23]:


lr.fit(X_scaled,y_log)


# In[24]:


ridge.fit(X_scaled,y_log)


# In[25]:


coef_df = pd.DataFrame(ridge.coef_.reshape(1,112),columns=X.columns).stack().reset_index().drop(columns=['level_0']).rename(columns={'level_1':'feature',0:'coef'})


# In[26]:


coef_df


# In[27]:


# 1. Import necessary libraries
import statsmodels.api as sm

# 2. Add a constant to X
X_with_const = sm.add_constant(X_scaled)

# 3. Fit the model
model = sm.OLS(y_log, X_with_const).fit()

# 4. Obtain summary statistics
print(model.summary())


# In[28]:


y_log.std()


# In[29]:


X_scaled['bedRoom'].std()


# In[30]:


0.21 * (0.557/1)


# In[31]:


np.expm1(0.030)


# In[32]:


2.4726962617564903e-05 * 100


# In[ ]:




