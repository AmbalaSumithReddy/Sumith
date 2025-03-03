#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder


# In[2]:


data = pd.read_csv("iris.csv")


# In[3]:


data


# In[4]:


import seaborn as sns
counts = data["variety"].value_counts()
sns.barplot(data = counts)


# In[5]:


data.info()


# In[6]:


data[data.duplicated(keep=False)]


# In[7]:


data[data.duplicated()]


# ### Observations
# - There are 150 rows and 5 columns
# - There are no null values 
# - There are 1 duplicated value 
# -The x-columns are  sepal.length,sepal.width,petal.length,petal.width
# - The y-column is variety
# - All x values are continous
# - y column is catgorical
# - There are three flower categories

# In[30]:


data.drop_duplicates(keep='first', inplace = True)


# In[31]:


data[data.duplicated()]


# In[32]:


data = data.reset_index(drop=True)


# In[33]:


data


# In[34]:


labelencoder = LabelEncoder()
data.iloc[:,-1] = labelencoder.fit_transform(data.iloc[:,-1])
data.head()


# In[35]:


data.tail()


# In[36]:


data.info()       


# In[37]:


data['variety'] = pd.to_numeric(labelencoder.fit_transform(data['variety']))


# In[38]:


data.info()


# In[39]:


data.head(4)


# In[40]:


X = data.iloc[:,0:4]
Y = data['variety']


# In[41]:


X


# In[42]:


Y


# In[51]:


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=1)
x_train


# In[52]:


model = DecisionTreeClassifier(criterion = 'entropy',max_depth=None)
model.fit(x_train,y_train)


# In[53]:


plt.figure(dpi=1200)
tree.plot_tree(model);


# In[54]:



fn = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
cn = ['setosa', 'versicolor', 'virginica']
plt.figure(dpi=1200)
plot_tree(model, feature_names=fn, class_names=cn, filled=True)
plt.show()


# In[55]:


preds = model.predict(x_test)
preds


# In[56]:


print(classification_report(y_test,preds))


# In[50]:


pred_train = model.predict(x_train)
print(classification_report(y_train,pred_train))


# In[ ]:





# In[ ]:




