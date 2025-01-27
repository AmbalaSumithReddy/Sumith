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


# In[8]:


data1[data1.duplicated(keep = False)]


# In[9]:


data1.drop_duplicates(keep='first', inplace = True)
data1


# In[10]:


data1.rename({'Solar.R':'Solar'}, axis = 1,inplace = True)
data1


# In[11]:


data1.isnull().sum()


# In[12]:


cols = data1.columns
colors = ['black' , 'blue']
sns.heatmap(data1[cols].isnull(),cmap=sns.color_palette(colors),cbar= True)


# In[13]:


median_ozone = data1["Ozone"].median()
mean_ozone = data1["Ozone"].mean()
print("Median of Ozone: ", median_ozone)
print("Mean of Ozone: ",mean_ozone)


# In[14]:


data1['Ozone'] = data1['Ozone'].fillna(median_ozone)
data1.isnull().sum()


# In[15]:


median_solar = data1["Solar"].median()
mean_solar = data1["Solar"].mean()
print("Median of Solar: ", median_solar)
print("Mean of Solar: ",mean_solar)                 


# In[16]:


data1['Solar'] = data1['Solar'].fillna(median_solar)
data1.isnull().sum()


# In[17]:


print(data1["Weather"].value_counts())
mode_weather = data1["Weather"].mode()[0]
print(mode_weather)


# In[18]:


data1["Weather"] = data1["Weather"].fillna(mode_weather)
data1.isnull().sum()


# In[19]:


print(data1["Month"].value_counts())
mode_month = data1["Month"].mode()[0]
print(mode_month)


# In[20]:


data1["Month"] = data1["Month"].fillna(mode_month)
data1.isnull().sum()


# In[21]:


data1.tail()


# In[22]:


# reset the yndex column
data1.reset_index(drop = True)


# In[26]:




fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

sns.boxplot(data=data1["Ozone"], ax=axes[0], color='skyblue', width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Ozone Levels")
sns.histplot(data1["Ozone"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Ozone Levels")
axes[1].set_ylabel("Frequency")
plt.tight_layout()
plt.show()


# In[ ]:


"""" observations 
The ozone columns has extreme values beyond 81 as seen from box plot
The same is confirmed from the below right = skewed histogram"""


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Assuming 'data1' is your DataFrame
# data1 = pd.read_csv("your_data.csv") 

fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={'height_ratios': [1, 3]})

sns.boxplot(data=data1["Solar"], ax=axes[0], color='skyblue', width=0.5, orient='h')
axes[0].set_title("Boxplot")
axes[0].set_xlabel("Solar Levels")

sns.histplot(data1["Solar"], kde=True, ax=axes[1], color='purple', bins=30)
axes[1].set_title("Histogram with KDE")
axes[1].set_xlabel("Solar Levels")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


# In[ ]:


"""" observations 
The ozone columns has extreme values between 120 and 270 as seen from box plot
The same is confirmed from the below left = skewed histogram"""


# In[ ]:




