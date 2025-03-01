#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[2]:


data = pd.read_csv("iris.csv")


# In[3]:


data


# In[7]:


import seaborn as sns
counts = data["variety"].value_counts()
sns.barplot(data = counts)


# In[8]:


data.info()


# In[9]:


data[data.duplicated(keep=False)]


# In[11]:


data[data.duplicated()]


# ### Observations
# - There are 150 rows and 5 columns
# - There are no null values 
# - There are 1 duplicated value 
# -The x-columns are  sepal.length,sepal.width,petal.length,petal.width
# - The y-column is variety
# - All x values are continous
# - y column is catgorical
# - There are three flower categories

# In[12]:


data.drop_duplicates(keep='first', inplace = True)


# In[13]:


data[data.duplicated()]


# In[17]:


data = data.reset_index(drop=True)


# In[15]:


data


# In[18]:


labelencoder = LabelEncoder()
data.iloc[:,-1] = labelencoder.fit_transform(data.iloc[:,-1])
data.head()


# In[19]:


data.tail()


# In[20]:


data.info()       


# In[31]:


data['variety'] = pd.to_numeric(labelencoder.fit_transform(data['variety']))


# In[32]:


data.info()


# In[33]:


data.head(4)


# In[34]:


X = data.iloc[:,0:4]
Y = data['variety']


# In[35]:


X


# In[36]:


Y


# In[42]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
x_train


# In[ ]:




