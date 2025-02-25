#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
data = pd.read_csv("iris.csv")
data


# In[5]:


data.info()


# In[7]:


data[data.duplicated()]


# In[8]:


data.isnull().sum()


# In[9]:


data.describe()


# In[10]:


data[data.duplicated(keep=False)]


# In[11]:


data.drop_duplicates(keep='first', inplace = True)


# In[12]:


data[data.duplicated()]


# In[13]:


data


# In[15]:


data.reset_index(drop = True)


# ### Observations
# - There are 150 rows and 5 columns
# - There are no null values 
# - There are 1 duplicated value 
# -The x-columns are  sepal.length,sepal.width,petal.length,petal.width
# - The y-column is variety
# - All x values are continous
# - y column is catgorical
# - There are three categories

# In[ ]:




