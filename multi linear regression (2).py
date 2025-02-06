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

# In[5]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=cars, x='HP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='HP', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[6]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=cars, x='SP', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='SP', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[7]:


fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': (.15, .85)})
sns.boxplot(data=cars, x='VOL', ax=ax_box, orient='h')
ax_box.set(xlabel='')
sns.histplot(data=cars, x='VOL', ax=ax_hist, bins=30, kde=True, stat='density')
ax_hist.set(ylabel='Density')
plt.tight_layout()
plt.show()


# In[8]:


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

# In[9]:


cars[cars.duplicated()]


# In[10]:


import seaborn as sns
import matplotlib.pyplot as plt



sns.pairplot(cars)

plt.show()


# In[11]:


cars.corr()


# # observations
# - between VOL and WT has the highest co relation variable(0.999203)
# - between SP and HP has second hihgest co relation (0.973848)
# - between WT and HP has lowest co relation(0.076513)
# - The high corelation among x columns is not desirable as it might lesd to multicollinearity problem

# In[12]:


model1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
model1.summary()


# # observations from model summary
# - The R-squared and adjusted R-squared vaues are good and about 75% of variability in Y is explained by X columns
# - The probability value with respect to F-statistic is close to zero ,indicating that all or some of X columns are significant
# - The p-values for VOL and WT are higher than 5% indicating some interaction issue among themselves , which need to be further explored

# # Performance metrics for model1

# In[13]:


df1= pd.DataFrame()
df1["actual_y1"] = cars["MPG"]
df1.head()


# In[14]:


pred_y1 = model1.predict(cars.iloc[:,0:4])
df1["pred_y1"] = pred_y1
df1.head()


# In[15]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df1["actual_y1"], df1['pred_y1'])
print('MSE :', mse)
print('RMSE :', np.sqrt(mse))


# # checking for multicollinearity among X-columns using VIF method 

# In[17]:


cars.head()


# In[18]:


# Compute VIF values
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)

rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared  
vif_wt = 1/(1-rsq_wt) 

rsq_vol = smf.ols('VOL~WT+SP+HP',data=cars).fit().rsquared  
vif_vol = 1/(1-rsq_vol) 

rsq_sp = smf.ols('SP~WT+VOL+HP',data=cars).fit().rsquared  
vif_sp = 1/(1-rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Hp','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame


# In[ ]:




