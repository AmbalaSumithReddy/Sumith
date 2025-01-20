#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np


# In[8]:


df = pd.read_csv("universities.csv")

df


# In[9]:


sal = pd.read_csv("salaries.csv")
sal


# In[14]:


sal[["salary","phd","service"]].groupby(sal["rank"]).median()


# In[16]:


# Create a sample table with movie ratings
data = {
    'User ID': [1, 1, 2, 2, 3, 3, 4, 3, 7],
    'Movie Name': [
        'Inception', 'Titanic', 'Inception', 'Avatar', 
        'Titanic', 'Avatar', 'Lion King', 'Inter Stellar', 'Bahubali'
    ],
    'Rating': [9, 8, 7, 18, 9, 8, 7, 8, 9]
}


df = pd.DataFrame(data)

pivot_table = df.pivot(index='User ID', columns='Movie Name', values='Rating')

print(pivot_table)


# In[ ]:




