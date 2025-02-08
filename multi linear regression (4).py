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

# In[16]:


cars.head()


# In[17]:


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


# # observations
# - The ideal range of VIF values shall be between 0 to 10.However slightly higher values can be tolerated.
# - As seen from the very high ViF values for WT and VOL,it is clear that they are prone to multicollinaerity prone
# - Hence it is decided to drop one of the column (either VOL or WT) to overcome the multicollinarity.
# - It is decided to drop WT and retain VOL columns for further models

# In[18]:


cars1 = cars.drop('WT',axis=1)
cars1.head()


# In[19]:


#build model2 on cars1 dataset
import statsmodels.formula.api as smf
model2= smf.ols('MPG~VOL+SP+HP',data=cars1).fit()
model2.summary()


# # performance metrics for model2

# In[20]:


df2 = pd.DataFrame()
df2['actual_y2']= cars['MPG']
df2.head()


# In[21]:


pred_y2 = model2.predict(cars.iloc[:,0:4])
df2["pred_y2"]=pred_y2
df2.head()


# In[22]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df2["actual_y2"], df2['pred_y2'])
print('MSE :', mse)
print('RMSE :', np.sqrt(mse))


# # observations from model2 summary()
# 
# - The adjusted R-squared value improved slightly to 0.76
# - All the p-values for model parameters are less than 5% hence they are significant
# - Therefore the HP,VOL,SP columns are finalized as the significant predictor for the MPG
# - There is no improvement in MSE values

# In[23]:


cars1.shape


# In[24]:


k = 3
n = 81
leverage_cutoff = 3*((k+1)/n)
leverage_cutoff


# In[25]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model1,alpha=.05)
y = [i for i in range(-2,8)]
x = [leverage_cutoff for i in range(10) ]
plt.plot(x,y,'r+')
plt.show()


# # observations
# - From the above plot ,it is evident that data points 65,70,76,78,79,80 are the influencers.
# - as their H Leverage values are higher and size is higher

# In[26]:


cars1[cars1.index.isin([65,70,76,78,79,80])]


# In[34]:


cars2 = cars1.drop(cars.index[[65,70,76,78,79,80]],axis=0).reset_index(drop=True)
cars2


# In[28]:


model3 = smf.ols('MPG~VOL+SP+HP',data = cars2).fit()
model3.summary()


# In[29]:


df3= pd.DataFrame()
df3['actual_y3']= cars2['MPG']
df3.head()


# In[32]:


pred_y3 = model3.predict(cars2.iloc[:,0:3])
df3["pred_y3"]=pred_y3
df3.head()


# In[33]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(df3["actual_y3"], df3['pred_y3'])
print('MSE :', mse)
print('RMSE :', np.sqrt(mse))


# #### Comparison of model
# | Metric         | Model1  | Model2  | Model3  |
# |----------------|---------|---------|---------|
# |R-Squared       | 0.771   | 0.770   | 0.885   |
# | Adj. R-squared | 0.758   | 0.761   | 0.880   |
# | MSE            | 18.89   | 18.91   | 8.68    |
# | RMSE           | 4.34    | 4.34    | 2.94    |
# 
# 
# - **From the above comparison table it is observed that model3 is the best among all with superior performance metrics**

# In[36]:


model3.resid


# In[37]:


model3.fittedvalues


# In[38]:


import statsmodels.api as sm
qqplot=sm.qqplot(model3.resid,line='q')
plt.title("Normal Q-Q plot of residuals")
plt.show()


# In[39]:


sns.displot(model3.resid,kde=True)


# In[40]:


def get_standardized_values( vals ):
    return (vals - vals.mean())/vals.std()


# In[41]:



plt.figure(figsize=(6,4))
plt.scatter(get_standardized_values(model3.fittedvalues),
            get_standardized_values(model3.resid))

plt.title('Residual Plot')
plt.xlabel('Standardized Fitted values')
plt.ylabel('Standardized residual values')
plt.show()


# In[ ]:





# In[ ]:




