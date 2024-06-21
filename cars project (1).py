#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()


# In[25]:


raw_data = pd.read_csv('cars.csv')
raw_data.head()


# In[26]:


raw_data.describe(include='all')


# In[27]:


data=raw_data.drop(["Model"],axis=1)
data


# In[28]:


data.isnull().sum()


# In[29]:


data_nomv=data.dropna(axis=0)
data_nomv


# In[30]:


data_nomv.isnull().sum()


# In[31]:


sns.distplot(data_nomv["Price"])


# In[32]:


q=data_nomv["Price"].quantile(0.99)
data1=data_nomv[data_nomv["Price"]<q]
data1.describe(include="all")


# In[33]:


sns.distplot(data1['Price'])


# In[34]:


sns.distplot(data_nomv['Mileage'])


# In[35]:


q = data1['Mileage'].quantile(0.99)
data2 = data1[data1['Mileage']<q]


# In[36]:


sns.distplot(data2["Mileage"])


# In[37]:


sns.distplot(data_nomv['EngineV'])


# In[38]:


data3 = data2[data2['EngineV']<6.5]


# In[39]:


sns.distplot(data3['EngineV'])


# In[40]:


sns.distplot(data_nomv['Year'])


# In[41]:


q = data3['Year'].quantile(0.01)
data4 = data3[data3['Year']>q]


# In[42]:


sns.distplot(data4['Year'])


# In[43]:


data_cleaned = data4.reset_index(drop=True)


# In[44]:


data_cleaned.describe(include='all')


# In[45]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['Price'])
ax3.set_title('Price and Mileage')

plt.show()


# In[46]:


sns.distplot(data_cleaned['Price'])


# In[47]:


log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price'] = log_price
data_cleaned


# In[48]:


f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize =(15,3))
ax1.scatter(data_cleaned['Year'],data_cleaned['log_price'])
ax1.set_title('Log Price and Year')
ax2.scatter(data_cleaned['EngineV'],data_cleaned['log_price'])
ax2.set_title('Log Price and EngineV')
ax3.scatter(data_cleaned['Mileage'],data_cleaned['log_price'])
ax3.set_title('Log Price and Mileage')


plt.show()


# In[49]:


data_cleaned = data_cleaned.drop(['Price'],axis=1)


# In[50]:


data_cleaned.columns.values


# In[51]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = data_cleaned[['Mileage','Year','EngineV']]
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns


# In[52]:


vif


# In[53]:


data_no_multicollinearity = data_cleaned.drop(['Year'],axis=1)


# In[54]:


data_with_dummies = pd.get_dummies(data_no_multicollinearity, drop_first=True)


# In[55]:


data_with_dummies.head()


# In[56]:


data_with_dummies = data_with_dummies.astype(int)


# In[57]:


data_with_dummies.columns.values


# In[58]:


cols = ['log_price', 'Mileage', 'EngineV', 'Brand_BMW',
       'Brand_Mercedes-Benz', 'Brand_Mitsubishi', 'Brand_Renault',
       'Brand_Toyota', 'Brand_Volkswagen', 'Body_hatch', 'Body_other',
       'Body_sedan', 'Body_vagon', 'Body_van', 'Engine Type_Gas',
       'Engine Type_Other', 'Engine Type_Petrol', 'Registration_yes']


# In[59]:


data_preprocessed = data_with_dummies[cols]
data_preprocessed.head()


# In[60]:


variables = data_preprocessed.drop(['log_price'],axis=1)
vif = pd.DataFrame()
vif["VIF"] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif["features"] = variables.columns
vif


# In[61]:


data_with_dummies_new = pd.get_dummies(data_no_multicollinearity)#, drop_first=True)
data_with_dummies_new.head()


# In[62]:


targets = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'],axis=1)


# In[63]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(inputs)


# In[64]:


inputs_scaled = scaler.transform(inputs)


# In[65]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=365)


# In[66]:


reg = LinearRegression()
reg.fit(x_train,y_train)


# In[67]:


y_hat = reg.predict(x_train)


# In[68]:


plt.scatter(y_train, y_hat)
plt.xlabel('Targets (y_train)',size=18)
plt.ylabel('Predictions (y_hat)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[69]:


sns.distplot(y_train - y_hat)
plt.title("Residuals PDF", size=18)


# In[70]:


reg.score(x_train,y_train)


# In[71]:


reg.intercept_


# In[72]:


reg.coef_


# In[73]:


reg_summary = pd.DataFrame(inputs.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary


# In[74]:


data_cleaned['Brand'].unique()


# In[75]:


y_hat_test = reg.predict(x_test)


# In[76]:


plt.scatter(y_test, y_hat_test, alpha=0.2)
plt.xlabel('Targets (y_test)',size=18)
plt.ylabel('Predictions (y_hat_test)',size=18)
plt.xlim(6,13)
plt.ylim(6,13)
plt.show()


# In[77]:


df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
df_pf.head()


# In[78]:


df_pf['Target'] = np.exp(y_test)
df_pf


# In[79]:


y_test = y_test.reset_index(drop=True)
y_test.head()


# In[80]:


df_pf['Target'] = np.exp(y_test)
df_pf


# In[81]:


df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']


# In[82]:


df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf


# In[83]:


df_pf.describe()


# In[84]:


pd.options.display.max_rows = 999
pd.set_option('display.float_format', lambda x: '%.2f' % x)
df_pf.sort_values(by=['Difference%'])


# In[ ]:




