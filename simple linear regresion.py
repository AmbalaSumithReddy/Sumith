#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

data1 = pd.read_csv("NewspaperData.csv")
data1


# In[7]:


data1.info()


# OBSERVATIONS
# 
# NO NULL VALUES ARE FOUND IN DATA

# In[8]:


data1.describe()


# In[17]:


data1['daily'] = pd.to_numeric(data1['daily'])
data2 = data1['daily']
data2


# In[21]:


plt.scatter(data1['daily'],data1['sunday'])


# In[22]:


data1["daily"].corr(data1["sunday"])


# # OBSERVATIONS

# 
# DAILY AND SUNDAY HAS HIGH POSITIVE CORRELATION STRENGTH

# In[33]:


import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()
model.summary()


# In[35]:


x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x,y,color="m",marker = "o",s = 30)
b0 = 13.84
b1 =1.33
y_hat = b0 +b1*x
plt.plot(x,y_hat,color = "g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

