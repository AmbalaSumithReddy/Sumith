#!/usr/bin/env python
# coding: utf-8

# # Assumptions in multi linear regression
# 
# -Linearity: The relationship between the predictors and the response is linear
# 
# -Independence: Observation are independent of each other
# 
# -Homoscedasticity: The residuals (difference between observed and predicted values ) exhibit constant variance at all levels of the predictor.
# 
# -Normal Distribution of Errors : The residuals of the model are normally distribbuted.
# 
# -No multicollinearity: The independent variables should not be too highly correlated with each other

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.graphics.regressionplots import influence_plot
cars = pd.read_csv("Cars.csv")
cars.head()


# In[2]:


cars = pd.DataFrame(cars, columns=['HP','VOL','SP','WT','MPG'])
cars.head()


# # Descriptiom of columns
# 
# - MPG : Milage of the car (Mile per Gallon)
# - VOL : Volume of the car (Size)
# - HP  : Horse power of the car
# - SP  : Top speed of the car(Miles per hour)
# - WT  : Weight of the car(Pounds)

# # EDA

# In[3]:


cars.info()


# In[4]:


cars.isna().sum()


# # Observations 
# - No missing values are found
# - There are 81 observations
# - Datatypes of the columns are relevent and valid

# In[7]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[8]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[9]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[10]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=cars, x='WT', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='WT', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# # Observations from boxplot and histogram
# - There are some extreme values(outliners) observed in towards the right tail of SP and HP distributors. 
# - In VOL and WT columns , a few outliners ar observed in both tails of their distributors.
# - The extreme values of cars data may have come from the specially designed nature of cars
# - As this is multi-dimensional data,the outliners with respect to spatial dimensions may have to be considered while buliding the regression model.

# # checking the duplicated rows

# In[11]:


cars[cars.duplicated()]


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt



sns.pairplot(cars)

plt.show()


# In[12]:


cars.corr()


# # observations
# - between VOL and WT has the highest co relation variable(0.999203)
# - between SP and HP has second hihgest co relation (0.973848)
# - between WT and HP has lowest co relation(0.076513)
# - The high corelation among x columns is not desirable as it might lesd to multicollinearity problem

# In[19]:


model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
model1.summary()


# In[ ]:




