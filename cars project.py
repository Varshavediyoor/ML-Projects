#!/usr/bin/env python
# coding: utf-8

# In[82]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# In[3]:


raw_data = pd.read_csv('cars.csv')
raw_data.head()


# In[4]:


raw_data.describe(include='all')


# In[6]:


data=raw_data.drop(["Model"],axis=1)
data


# In[10]:


data.isnull().sum()


# In[12]:


data_nomv=data.dropna(axis=0)
data_nomv


# In[13]:


data_nomv.isnull().sum()


# In[14]:


sns.distplot(data_nomv["Price"])


# In[32]:


q=data_nomv["Price"].quantile(0.99)
data1=data_nomv[data_nomv["Price"]<q]
data1.describe(include="all")


# In[33]:


sns.distplot(data1['Price'])


# In[35]:


sns.distplot(data_nomv['Mileage'])


# In[39]:


q = data1['Mileage'].quantile(0.99)
data2 = data1[data1['Mileage']<q]


# In[40]:


sns.distplot(data2["Mileage"])


# In[43]:


sns.distplot(data_nomv['EngineV'])


# In[44]:


data3 = data2[data2['EngineV']<6.5]


# In[46]:


sns.distplot(data3['EngineV'])


# In[47]:


sns.distplot(data_nomv['Year'])


# In[49]:


q = data3['Year'].quantile(0.01)
data4 = data3[data3['Year']>q]


# In[51]:


sns.distplot(data4['Year'])


# In[53]:


data_cleaned = data4.reset_index(drop=True)


# In[54]:


data_cleaned.describe(include='all')


# In[55]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')

plt.show()


# In[56]:


sns.distplot(data_cleaned['Price'])


# In[57]:


log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
data_cleaned


# In[58]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')


plt.show()


# In[59]:


data_cleaned = data_cleaned.drop(['Price'],axis=1)


# In[60]:


data_cleaned.columns.values


# In[61]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns


# In[62]:


vif


# In[63]:


data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)


# In[64]:


data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)


# In[86]:


data_with_dummies.head()


# In[84]:


data_with_dummies = data_with_dummies.astype(int)


# In[85]:


data_with_dummies.columns.values


# In[87]:


cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']


# In[88]:


data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()


# In[89]:


variables = data_preprocessed.drop(['log_price'],axis=1)
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
vif


# In[90]:


data_with_dummies_new = pd.get_dummies(data_no_multicollinearity)#, drop_first=True)
data_with_dummies_new.head()


# In[ ]:




