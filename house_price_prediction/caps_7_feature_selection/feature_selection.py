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


df = pd.read_csv('gurgaon_properties_missing_value_imputation.csv')


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


train_df = df.drop(columns=['society','price_per_sqft'])


# In[6]:


train_df.head()


# ### luxury score

# In[10]:


sns.boxplot(df['luxury_score'])


# In[11]:


def categorize_luxury(score):
    if 0 <= score < 50:
        return "Low"
    elif 50 <= score < 150:
        return "Medium"
    elif 150 <= score <= 175:
        return "High"
    else:
        return None  # or "Undefined" or any other label for scores outside the defined bins


# In[12]:


train_df['luxury_category'] = train_df['luxury_score'].apply(categorize_luxury)


# In[13]:


train_df.head()


# ### floorNum

# In[14]:


sns.boxplot(df['floorNum'])


# In[15]:


def categorize_floor(floor):
    if 0 <= floor <= 2:
        return "Low Floor"
    elif 3 <= floor <= 10:
        return "Mid Floor"
    elif 11 <= floor <= 51:
        return "High Floor"
    else:
        return None  # or "Undefined" or any other label for floors outside the defined bins


# In[16]:


train_df['floor_category'] = train_df['floorNum'].apply(categorize_floor)


# In[17]:


train_df.head()


# In[18]:


train_df.drop(columns=['floorNum','luxury_score'],inplace=True)


# In[19]:


train_df.head()


# In[20]:


from sklearn.preprocessing import OrdinalEncoder

# Create a copy of the original data for label encoding
data_label_encoded = train_df.copy()

categorical_cols = train_df.select_dtypes(include=['object']).columns

# Apply label encoding to categorical columns
for col in categorical_cols:
    oe = OrdinalEncoder()
    data_label_encoded[col] = oe.fit_transform(data_label_encoded[[col]])
    print(oe.categories_)

# Splitting the dataset into training and testing sets
X_label = data_label_encoded.drop('price', axis=1)
y_label = data_label_encoded['price']


# In[21]:


X_label


# In[22]:


y_label


# ### Technique 1 - Correlation Analysis

# In[23]:


sns.heatmap(data_label_encoded.corr())


# In[24]:


fi_df1 = data_label_encoded.corr()['price'].iloc[1:].to_frame().reset_index().rename(columns={'index':'feature','price':'corr_coeff'})
fi_df1


# ### Technique 2 - Random Forest Feature Importance

# In[25]:


from sklearn.ensemble import RandomForestRegressor

# Train a Random Forest regressor on label encoded data
rf_label = RandomForestRegressor(n_estimators=100, random_state=42)
rf_label.fit(X_label, y_label)

# Extract feature importance scores for label encoded data
fi_df2 = pd.DataFrame({
    'feature': X_label.columns,
    'rf_importance': rf_label.feature_importances_
}).sort_values(by='rf_importance', ascending=False)

fi_df2


# ### Technique 3 - Gradient Boosting Feature importances
# 

# In[26]:


from sklearn.ensemble import GradientBoostingRegressor

# Train a Random Forest regressor on label encoded data
gb_label = GradientBoostingRegressor()
gb_label.fit(X_label, y_label)

# Extract feature importance scores for label encoded data
fi_df3 = pd.DataFrame({
    'feature': X_label.columns,
    'gb_importance': gb_label.feature_importances_
}).sort_values(by='gb_importance', ascending=False)

fi_df3


# ### Technique 4 - Permutation Importance
# 

# In[27]:


from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

X_train_label, X_test_label, y_train_label, y_test_label = train_test_split(X_label, y_label, test_size=0.2, random_state=42)

# Train a Random Forest regressor on label encoded data
rf_label = RandomForestRegressor(n_estimators=100, random_state=42)
rf_label.fit(X_train_label, y_train_label)

# Calculate Permutation Importance
perm_importance = permutation_importance(rf_label, X_test_label, y_test_label, n_repeats=30, random_state=42)

# Organize results into a DataFrame
fi_df4 = pd.DataFrame({
    'feature': X_label.columns,
    'permutation_importance': perm_importance.importances_mean
}).sort_values(by='permutation_importance', ascending=False)

fi_df4


# ### Technique 5 - LASSO

# In[28]:


from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_label)

# Train a LASSO regression model
# We'll use a relatively small value for alpha (the regularization strength) for demonstration purposes
lasso = Lasso(alpha=0.01, random_state=42)
lasso.fit(X_scaled, y_label)

# Extract coefficients
fi_df5 = pd.DataFrame({
    'feature': X_label.columns,
    'lasso_coeff': lasso.coef_
}).sort_values(by='lasso_coeff', ascending=False)

fi_df5


# ### Technique 6 - RFE

# In[29]:


from sklearn.feature_selection import RFE

# Initialize the base estimator
estimator = RandomForestRegressor()

# Apply RFE on the label-encoded and standardized training data
selector_label = RFE(estimator, n_features_to_select=X_label.shape[1], step=1)
selector_label = selector_label.fit(X_label, y_label)

# Get the selected features based on RFE
selected_features = X_label.columns[selector_label.support_]

# Extract the coefficients for the selected features from the underlying linear regression model
selected_coefficients = selector_label.estimator_.feature_importances_

# Organize the results into a DataFrame
fi_df6 = pd.DataFrame({
    'feature': selected_features,
    'rfe_score': selected_coefficients
}).sort_values(by='rfe_score', ascending=False)

fi_df6


# ### Technique 7 - Linear Regression Weights

# In[31]:


from sklearn.linear_model import LinearRegression
import pandas as pd

# Assuming X_label contains your features, y_label contains your target variable, and X_scaled is the standardized feature matrix
# Train a linear regression model on the label-encoded and standardized training data
lin_reg = LinearRegression()
lin_reg.fit(X_scaled, y_label)

# Extract coefficients and create a DataFrame
fi_df7 = pd.DataFrame({
    'feature': X_label.columns,
    'reg_coeffs': lin_reg.coef_
})

# Sort the DataFrame by regression coefficients in descending order
fi_df7 = fi_df7.sort_values(by='reg_coeffs', ascending=False)

# Display the DataFrame with feature names and corresponding regression coefficients
print(fi_df7)


# ### Technique 8 - SHAP

# In[32]:


get_ipython().system('pip install shap')


# In[33]:


import shap

# Compute SHAP values using the trained Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_label, y_label)

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_label)

# Summing the absolute SHAP values across all samples to get an overall measure of feature importance
shap_sum = np.abs(shap_values).mean(axis=0)

shap_values


# In[34]:


fi_df8 = pd.DataFrame({
    'feature': X_label.columns,
    'SHAP_score': np.abs(shap_values).mean(axis=0)
}).sort_values(by='SHAP_score', ascending=False)

fi_df8


# In[35]:


final_fi_df = fi_df1.merge(fi_df2,on='feature').merge(fi_df3,on='feature').merge(fi_df4,on='feature').merge(fi_df5,on='feature').merge(fi_df6,on='feature').merge(fi_df7,on='feature').merge(fi_df8,on='feature').set_index('feature')


# In[36]:


final_fi_df


# In[37]:


# normalize the score
final_fi_df = final_fi_df.divide(final_fi_df.sum(axis=0), axis=1)


# In[38]:


final_fi_df[['rf_importance','gb_importance','permutation_importance','rfe_score','SHAP_score']].mean(axis=1).sort_values(ascending=False)


# In[39]:


# to drop pooja room, study room, others
X_label


# In[40]:


# with all the cols
from sklearn.model_selection import cross_val_score

rf = RandomForestRegressor(n_estimators=100, random_state=42)

scores = cross_val_score(rf, X_label, y_label, cv=5, scoring='r2')


# In[41]:


scores.mean()


# In[42]:


rf = RandomForestRegressor(n_estimators=100, random_state=42)

scores = cross_val_score(rf, X_label.drop(columns=['pooja room', 'study room', 'others']), y_label, cv=5, scoring='r2')


# In[43]:


scores.mean()


# In[44]:


export_df = X_label.drop(columns=['pooja room', 'study room', 'others'])
export_df['price'] = y_label


# In[45]:


export_df.to_csv('gurgaon_properties_post_feature_selection.csv', index=False)


# In[46]:


export_df

