#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

data1 = pd.read_csv("NewspaperData.csv")
data1


# In[2]:


data1.info()


# OBSERVATIONS
# 
# NO NULL VALUES ARE FOUND IN DATA

# In[3]:


data1.describe()


# In[4]:


data1['daily'] = pd.to_numeric(data1['daily'])
data2 = data1['daily']
data2


# In[5]:


plt.scatter(data1['daily'],data1['sunday'])


# In[6]:


data1["daily"].corr(data1["sunday"])


# # OBSERVATIONS

# 
# DAILY AND SUNDAY HAS HIGH POSITIVE CORRELATION STRENGTH

# In[7]:


import statsmodels.formula.api as smf
model = smf.ols("sunday~daily",data = data1).fit()
model.summary()


# In[8]:


x = data1["daily"].values
y = data1["sunday"].values
plt.scatter(x,y,color="m",marker = "o",s = 30)
b0 = 13.84
b1 = 1.33
y_hat = b0 +b1*x
plt.plot(x,y_hat,color = "g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# In[10]:


model.params


# In[11]:


print(f'model t-values:\n{model.tvalues}\n-----------\n model p-values: \n{model.pvalues}')


# In[12]:


(model.rsquared,model.rsquared_adj)


# In[13]:


newdata=pd.Series([200,300,1500])
data_pred=pd.DataFrame(newdata,columns = ['daily'])
data_pred


# In[14]:


model.predict(data_pred)


# In[16]:


#predict on all given training data
pred = model.predict(data1['daily'])
pred


# In[23]:


#Add predict values as a column
data1['Y_hat'] = pred
data1


# In[25]:


data1['residuals']= data1['sunday']-data1['Y_hat']
data1


# In[26]:


mse = np.mean((data1['daily']-data1['Y_hat'])**2)
rmse = np.sqrt(mse)
print("MSE: ",mse)
print("RMSE: ",rmse)


# In[27]:


plt.scatter(data1['Y_hat'],data1['residuals'])


# # OBSERVATIONS

# -THE RESIDUAL DATA POINTS ARE RANDOMLY SCATTERED AROUND THE ZERO ERROR  LINE
# 
# -HENCE THE ASSUMOTION OF HOMOSCEDASTICTY IS SATISFIED(CONSTANT VARIANCE IN RESIDUALS)

# In[28]:


import statsmodels.api as sm
sm.qqplot(data1['residuals'],line='45',fit= True)
plt.show


# In[ ]:




