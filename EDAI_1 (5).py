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


# In[23]:




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


# In[24]:


"""" observations 
The ozone columns has extreme values beyond 81 as seen from box plot
The same is confirmed from the below right = skewed histogram"""


# In[25]:


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


# In[26]:


"""" observations 
The ozone columns has extreme values between 120 and 270 as seen from box plot
The same is confirmed from the below left = skewed histogram"""


# In[27]:


sns.violinplot(data=data1["Ozone"],  color='skyblue')
plt.title("Violon Plot")
plt.show()


# In[28]:


boxplot_data = plt.boxplot(data1["Ozone"],vert = False)


# In[29]:


plt.figure(figsize=(6,2))
boxplot_data = plt.boxplot(data1["Ozone"],vert = False)
[item.get_xdata() for item in boxplot_data['fliers']]


# In[30]:


data1["Ozone"].describe()


# In[31]:


mu = data1["Ozone"].describe()[1]  # Mean
sigma = data1["Ozone"].describe()[2]  # Standard Deviation

for x in data1["Ozone"]:
    if (x < (mu - 3*sigma)) or (x > (mu + 3*sigma)):
        print(x)


# In[32]:


import scipy.stats as stats

plt.figure(figsize=(8,6))
stats.probplot(data1["Ozone"],dist = "norm",plot = plt)
plt.title("Q-Q Plot for Outlier Detection",fontsize = 14)
plt.xlabel("Theoretical Quantiles", fontsize=12)


# In[33]:


sns.violinplot(data=data1["Ozone"],colors='green')
plt.title("Violin Plot")
plt.show()


# In[34]:


sns.violinplot(data=data1,x = "Weather",y="Ozone",palette="Set2")


# In[35]:


sns.swarmplot(data = data1,x = "Weather",y = "Ozone",color = "orange",palette ="Set2",size=6)


# In[36]:


sns.stripplot(data = data1,x = "Weather",y = "Ozone",color = "orange",palette ="Set2",size=6,jitter=True)


# In[37]:


sns.kdeplot(data=data1["Ozone"],fill=True,color="darkblue")
sns.rugplot(data=data1["Ozone"],color="red")


# In[38]:


sns.boxplot(data = data1,x = "Weather",y ="Ozone")


# In[39]:


plt.scatter(data1["Wind"],data1["Temp"])


# In[40]:


data1["Wind"].corr(data1["Temp"])


# In[41]:


data1_numeric = data1.iloc[:,[0,1,2,6]]
data1_numeric


# Observation
#  >The correlation between wind and temp is observed to be negatively correlated with mild strength

# In[42]:


data1.info()


# In[44]:


data1_numaeric = data1.iloc[:,[0,1,2,6,]]
data1_numeric


# In[48]:


data1_numeric.corr()


# OBSERVATIONS
# 
# 1)higest correlation strength is 0.597087 in temp,ozone column
# 2)second higest correlation strength is -0.523738 in wind,ozone column
# 3)Third higest correlation strength is -0.441228 in temp,wind column
# 4)lowest higest correlation strength is -0.055874 in Solar,Wind column

# In[56]:


# plot a pair plot between all numeric colmns using seaborn 
sns.pairplot(data1_numeric)


# In[58]:


data2 = pd.get_dummies(data1,columns=['Month','Weather'])
data2


# In[ ]:




