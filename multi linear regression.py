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
# -No multicollinearity: The independent variables should not be too highly correlated with

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

# In[8]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.015, .85)})
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()

