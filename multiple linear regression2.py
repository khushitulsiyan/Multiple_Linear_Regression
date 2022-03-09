#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[16]:


#Reading the data

df = pd.read_csv('50_Startups.csv')
df.head()

cdf = df[['R&D Spend', 'Administration', 'Marketing Spend', 'State', 'Profit']]
cdf.head(9)


plt.scatter(cdf.Profit, cdf.Administration, color='blue')
plt.xlabel('Profit')
plt.ylabel('Administration')
plt.show()


# In[7]:


#Creating train and test dataset

msk = np.random.rand(len(df))<0.8
train = cdf[msk]
test = cdf[~msk]


# In[9]:


#Train data distribution

plt.scatter(train.Profit, train.Administration, color='green')
plt.xlabel("Profit")
plt.ylabel("Administration")
plt.show()


# In[15]:


#Multiple Regression Model

from sklearn import linear_model

regr = linear_model.LinearRegression()
x = np.asanyarray(train[["R&D Spend", "Administration", "Marketing Spend"]])
y = np.asanyarray(train[["Profit"]])

regr.fit(x,y)
print('coefficients:', regr.coef_)


# In[20]:


#Prediction using ordinary least squares (OLS)

y_hat = regr.predict(test[['R&D Spend', 'Administration', 'Marketing Spend']])
x = np.asanyarray(test[['R&D Spend', 'Administration', 'Marketing Spend']])
y = np.asanyarray(test[['Profit']])
print("Residual sum of squares: %2f" %np.mean(y_hat-y)**2)
print("Variance score: %2f" %regr.score(x,y))


# In[ ]:




