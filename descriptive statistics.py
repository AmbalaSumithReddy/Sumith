#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
df = pd.read_csv("Universities.csv")
df


# In[4]:


np.mean(df["SAT"])


# In[5]:


np.median(df["SAT"])


# In[6]:


np.std(df["SFRatio"])


# In[7]:


df.describe()


# In[ ]:




