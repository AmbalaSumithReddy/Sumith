#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold


# In[2]:


data = pd.read_csv('diabetes1.csv')
data


# In[3]:


from sklearn.ensemble import RandomForestClassifier
X = data.iloc[:,0:8]
Y = data.iloc[:,8]
kfold = StratifiedKFold(n_splits=20,random_state=203,shuffle=True)
model = RandomForestClassifier(n_estimators=200,random_state=20,max_depth=None)
results = cross_val_score(model,X,Y,cv=kfold)
print(results)
print(results.mean())


# In[18]:


#Using Grid search CV to find best parameters (Hyper parameter tuning)
from sklearn.model_selection import GridSearchCV
rf = RandomForestClassifier(random_state=42,n_jobs=-1)
params={
    'max_depth':[2,3,5,None],
    'min_samples_leaf':[5,10,20],
    'n_estimators':[50,100,200,500],
    'max_features':["sqrt","log2",None],
    'criterion':["gini","entropy"]
}
grid_search = GridSearchCV(estimator=rf,
                          param_grid=params,
                          cv=5,
                          n_jobs=-1,verbose=10,scoring="accuracy")
grid_search.fit(X,Y)


# In[28]:


print(grid_search.best_params_)
print(grid_search.best_score_)


# In[24]:


grid_search.best_estimator_


# In[30]:


# Use best estimators hyper parameter tuning obtainrd above to select important features

RandomForestClassifier(criterion='entropy', max_depth=5, max_features=None,
                       min_samples_leaf=5, n_jobs=-1, random_state=42)
model.fit(X,Y)
model.feature_importances_


# In[32]:


X = data.iloc[:,0:8]
X.columns


# In[34]:


df = pd.DataFrame(model.feature_importances_,columns = ["Importance score"],index = X.columns)
df.sort_values(by = "Importance score",inplace = True,ascending = False)


# In[38]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.bar(df.index,df["Importance score"])


# In[ ]:




