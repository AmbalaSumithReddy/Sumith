#!/usr/bin/env python
# coding: utf-8

# In[29]:


from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold


# In[25]:


data = pd.read_csv('diabetes1.csv')
data


# In[42]:


from sklearn.ensemble import RandomForestClassifier
X = data.iloc[:,0:8]
Y = data.iloc[:,8]
kfold = StratifiedKFold(n_splits=20,random_state=203,shuffle=True)
model = RandomForestClassifier(n_estimators=200,random_state=20,max_depth=None)
results = cross_val_score(model,X,Y,cv=kfold)
print(results)
print(results.mean())


# In[ ]:




