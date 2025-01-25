#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("data_clean.csv")
data


# In[2]:


data.info()


# In[3]:


#dataframe attributes
print(type(data))
print(data.shape)
print(data.size)


# In[4]:


data1 =data.drop(['Unnamed: 0',"Temp C"],axis = 1)
data1


# In[5]:


data1.info()


# In[6]:


data1['Month'] = pd.to_numeric(data['Month'],errors = 'coerce')
data1.info()


# In[7]:


data1[data1.duplicated()]


# In[10]:


data1[data1.duplicated(keep = False)]


# In[9]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[12]:


data1.rename({'Solar.R':'Solar'}, axis = 1,inplace = True)
data1


# In[13]:


data1.isnull().sum()


# In[22]:


cols = data1.columns
colors = ['black' , 'blue']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar= True)


# In[23]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[24]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[25]:


median_solar = data1["Solar"].median()
mean_solar = data1["Solar"].mean()
print("Median of Solar: ", median_solar)
print("Mean of Solar: ",mean_solar)                 


# In[26]:


data1['Solar'] = data1['Solar'].fillna(median_solar)
data1.isnull().sum()


# In[27]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[28]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




