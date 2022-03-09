#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[6]:


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]


# In[9]:


states = pd.get_dummies(X['State'],drop_first=True)


# In[10]:


X = X.drop('State',axis=1)
X = pd.concat([X,states], axis=1)


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)


# In[17]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score
score = r2_score(y_test, y_pred)

score


# In[ ]:




