#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
Univ = pd.read_csv("Universities.csv")
Univ


# In[2]:


Univ.info()


# In[3]:


Univ.isna().sum()


# In[4]:


Univ.describe()


# In[5]:


Univ1=Univ.iloc[:,1:]
Univ1


# In[18]:


cols = Univ1.columns
from sklearn.preprocessing import StandardScaler 
scaler=StandardScaler() 
scaled_Univ_df= pd.DataFrame (scaler.fit_transform (Univ1), columns = cols) 
scaled_Univ_df 


# In[19]:


from sklearn.cluster import KMeans
clusters_new = KMeans(3,random_state=0)
clusters_new.fit(scaled_Univ_df)


# In[20]:


clusters_new.labels_


# In[21]:


set(clusters_new.labels_)


# In[25]:


Univ['clusterid_new']=clusters_new.labels_
Univ


# In[24]:


Univ[Univ['clusterid_new']==1]


# In[27]:


Univ.iloc[:,1:].groupby("clusterid_new").mean()


# In[28]:


Univ[Univ['clusterid_new']==0]


# ### Observations:
# - Cluster 2 appears to be the top raed universities cluster as the cut off score,Top 10,SF Ratio parameter mean values are highest
# - Cluster 1 appears to occupy the middle level rated u niversities
# - Cluster 0 comes as the lower rated universities

# In[30]:


wcss = []
for i in range(1,20):
    kmeans =KMeans(n_clusters=i,random_state=0)
    kmeans.fit(scaled_Univ_df)
    wcss.append(kmeans.inertia_)
    
print(wcss)
plt.plot(range(1,20),wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel("WCSS")
plt.show()


# In[ ]:




