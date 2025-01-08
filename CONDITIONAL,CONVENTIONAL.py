#!/usr/bin/env python
# coding: utf-8

# In[10]:


num = 6
if num%2 == 0:
    print("even")
else:
    print("odd")


# In[11]:


print("even") if num%2 == 0 else print("odd")


# In[12]:


x = 10
result = "positive" if x>0 else "Negative"
print(result)


# In[13]:


num  = int(input("enter a number: "))
result = "postive" if num > 0 else ("negative" if num < 0 else "zero")
print(result)


# In[14]:


L = [1,9,2,10,56,89]
[2*x for x in L]


# In[15]:


[x for x in L if x%2 == 0]


# In[16]:


[x for x in L if x%2 != 0]


# In[8]:


salary = [45000,60000,80000,40000]
increment = x*20/100
for x in salary:
    if x < 50000:
        print(x+increment)


# In[9]:


[(x+increment if x <= 50000 else x) for x in salary]


# In[ ]:


d1 = {'Ram':[70,71,98,100],'jhon':[56,98,67,65]}
{k:sum(v)/len(v) for k,v in d1.items()}
    


# In[26]:


A = [2,4,4,6,3,74]
def mean_value(A):
    total = sum(A)
    average_value = total/len(A)
    return average_value
mean_value(A)
    


# In[23]:


def greet(name = "Sumith"):
    print(f"Good Morning {name}!")
    
greet()
    
  


# In[28]:


# function with variable number of arguments
def avg_value(*n):
    l = len(n)
    average = sum(n)/l
    return average
avg_value(100,200,400,400,100)


# In[ ]:




