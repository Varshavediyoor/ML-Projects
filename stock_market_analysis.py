#!/usr/bin/env python
# coding: utf-8

# <h1> Stock Market Analysis

# <h3> Importing necessary libraries

# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')


# <h3> Loading and reading the data

# In[64]:


df=pd.read_csv("tesla-stock-price.csv")
df.head()


# <h3> Summary of data

# In[18]:


df.shape


# In[19]:


df.info()


# In[22]:


df.describe()


# <h3> Explanatory data analysis

# In[29]:


plt.figure(figsize=(15,5))
plt.plot(df['close'],c="blue")
plt.title('Tesla Close price.', fontsize=15,c="red")
plt.ylabel('Price in dollars.')
plt.show()


# In[30]:


df.head()


# In[35]:


df.isnull().sum()


# In[47]:


for col in features:
    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')


# In[48]:


features = ['open', 'high', 'low', 'close', 'volume']

plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sns.distplot(df[col])
plt.show()


# In[49]:


plt.subplots(figsize=(20,10))

for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sns.boxplot(df[col])
plt.show()


# In[76]:


splitted = df['date'].str.split('/', expand=True)

df['day'] = pd.to_numeric(splitted[2], errors='coerce').fillna(0).astype(int)
df['month'] = pd.to_numeric(splitted[1], errors='coerce').fillna(0).astype(int)
df['year'] = pd.to_numeric(splitted[0], errors='coerce').fillna(0).astype(int)
df.head()


# In[68]:


df.head()


# In[70]:


df=df.drop(index=0)


# In[77]:


df["is_quarter_end"]=np.where(df["month"]%3==0,1,0)
df.head()


# In[78]:


df.tail()


# In[82]:


df['open-close']  = df['open'] - df['close']
df['low-high']  = df['low'] - df['high']
df['target'] = np.where(df['close'].shift(-1) > df['close'], 1, 0)


# In[83]:


df.head()


# In[84]:


plt.pie(df['target'].value_counts().values, 
        labels=[0, 1], autopct='%1.1f%%')
plt.show()


# In[92]:


plt.figure(figsize=(10,10))
sns.heatmap(df.drop('date', axis=1).corr() > 0.9, annot=True, cbar=False)
plt.show()


# In[93]:


features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_test, Y_train, Y_test = train_test_split(
    features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_test.shape)


# In[98]:


models = [LogisticRegression(), SVC(
  kernel='poly', probability=True)]

for i in range(2):
  models[i].fit(X_train, Y_train)

  print(f'{models[i]} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(
    Y_train, models[i].predict_proba(X_train)[:,1]))
  print('Validation Accuracy : ', metrics.roc_auc_score(
    Y_test, models[i].predict_proba(X_test)[:,1]))
  print()


# In[ ]:




