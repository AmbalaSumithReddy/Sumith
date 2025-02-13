#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

from sklearn.linear_model import LinearRegression
titanic= pd.read_csv("Titanic.csv")
titanic


# In[2]:


pip install mlxtend


# In[3]:


import pandas as pd
import mlxtend
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt


# In[4]:


titanic.info()


# ### Observations 
# - All columns are object data type and categorical in nature
# - There are no null values 
# - As the columns are categorical , we can adopt one-hot-encoding

# In[5]:


import matplotlib.pyplot as plt

counts = titanic['Class'].value_counts()
plt.bar(counts.index,counts.values)


# In[6]:



counts = titanic['Gender'].value_counts()
plt.bar(counts.index,counts.values)


# In[7]:



counts = titanic['Age'].value_counts()
plt.bar(counts.index,counts.values)


# In[8]:


titanic = pd.get_dummies(titanic,dtype=int)
titanic.head()


# In[9]:


titanic.info()


# In[10]:


frquent_itemsets = apriori(titanic,min_support = 0.5,use_colnames=True,max_len=None)
frquent_itemsets


# In[ ]:





# In[13]:


rules = association_rules(frquent_itemsets, metric="lift", min_threshold=1.0)
rules.sort_values(by='lift',ascending = False)


# In[14]:


import matplotlib.pyplot as plt
rules[['support','confidence','lift']].hist(figsize=(15,7))
plt.show()


# In[15]:


plt.scatter(rules['support'],rules['confidence'])
plt.show()


# In[16]:


rules[rules["consequents"]==({"Survived_Yes"})]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




