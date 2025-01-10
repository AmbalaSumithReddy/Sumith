#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import numpy library
import numpy as np
# Create 1D numpy array
x = np.array([45,67,57,60])
print(x)
print(type(x))
print(x.dtype)


# In[2]:


x = np.array(['A',67,57,6.4])
print(x)
print(type(x))
print(x.dtype)


# In[3]:


a2 = np.array([[20,40],[30,60]])
print(a2)
print(type(a2))
print(a2.shape)


# In[4]:


#Reshaping an array
a = np.array([10,20,30,40])
b = a.reshape(2,2)
print(b)
print(b.shape)


# 

# In[5]:


# Use of np.sqrt()
d = np.array([1.654,3.76543,4.4324])
print(d)
print(np.around(np.sqrt(d),2))


# In[8]:


a1 = np.array([[3,4,5,8],[7,2,8,np.NAN]])
print(a1)
print(a1.shape)
a1.dtype


# In[11]:


#mathematical operations on rows and columns
a2 = np.array([[3,4,6],[7,9,10],[4,6,12]])
a2


# In[13]:


print(a2.sum(axis = 1))
print(a2.sum(axis = 0))


# In[15]:


# matrix operations
a3 = np.array([[3,4,5],[7,2,8],[9,1,6]])
print(a3)
np.fill_diagonal(a3,0)
print(a3)


# In[19]:


A = np.array([[1,3],[2,4]])
B = np.array([[5,6],[7,8]])
print(A)
print(B)
print(A.T)
print(B.T)


# In[23]:


a4 = np.array([[3,4,6],[7,9,10],[4,6,12],[10,9,18]])
print(a4)
print(a4[3])


# In[ ]:




