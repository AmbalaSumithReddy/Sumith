#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
df = pd.read_csv("Universities.csv")
df


# In[2]:


np.mean(df["SAT"])


# In[3]:


np.median(df["SAT"])


# In[4]:


np.std(df["SFRatio"])


# In[5]:


df.describe()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,3))
plt.title("Acceptance Ratio")
plt.hist(df["Accept"])


# In[13]:


sns.histplot(df["Accept"], kde =True)


# In[ ]:




