#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('gurgaon_properties_post_feature_selection.csv')


# In[3]:


df.head()


# In[ ]:


# one hot encode -> sector, balcony, agePossession, furnishing type, luxury category, floor category


# In[4]:


X = df.drop(columns=['price'])
y = df['price']


# In[5]:


from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVR


# In[6]:


columns_to_encode = ['sector', 'balcony', 'agePossession', 'furnishing_type', 'luxury_category', 'floor_category']


# In[7]:


# Applying the log1p transformation to the target variable
y_transformed = np.log1p(y)


# In[9]:


# Creating a column transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['property_type', 'bedRoom', 'bathroom', 'built_up_area', 'servant room', 'store room']),
        ('cat', OneHotEncoder(drop='first'), columns_to_encode)
    ], 
    remainder='passthrough'
)


# In[10]:


# Creating a pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='rbf'))
])


# In[11]:


# K-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(pipeline, X, y_transformed, cv=kfold, scoring='r2')


# In[12]:


scores.mean()


# In[13]:


scores.std()


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y_transformed,test_size=0.2,random_state=42)


# In[15]:


pipeline.fit(X_train,y_train)


# In[16]:


y_pred = pipeline.predict(X_test)


# In[18]:


y_pred = np.expm1(y_pred)


# In[19]:


from sklearn.metrics import mean_absolute_error
mean_absolute_error(np.expm1(y_test),y_pred)


# In[ ]:





# In[ ]:





# In[ ]:




