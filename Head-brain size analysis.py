#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(20.0,10.0)

#reading data
data=pd.read_csv('headbrain.csv')
print(data.shape)
data.head()


# In[3]:


#collecting x and y
X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values


# In[5]:


#mean x and y
mean_x=np.mean(X)
mean_y=np.mean(Y)

#Total number of values
n=len(X)

#calculating b1(m ) and b0(c)
numer=0
denom=0
for i in range(n):
    numer+=(X[i]-mean_x)*(Y[i]-mean_y)
    denom+=(X[i]-mean_x)**2
b1=numer/denom
b0=mean_y-(b1*mean_x)

print(b1,b0)


# In[6]:


#plotting values and regression line

max_x=np.max(X)+100
min_x=np.min(X)-100

#Calculating line values X and Y
x=np.linspace(min_x,max_x,1000)
y=b0+b1*x

#plotting line
plt.plot(x,y,color='#58b970',label='Regression line')
#plotting scatter points
plt.scatter(X,Y,color='#ef5423',label='Scatter Plot')

plt.xlabel('Head size(cm^3)')
plt.ylabel('Brain Weight(grams)')
plt.legend()
plt.show()


# In[7]:


#Accuracy check by R square method
ss_t=0           #numerar
ss_r=0           #denom
for i in range(n):
    y_pred=b0+b1*X[i]
    ss_t+=(y_pred-mean_y)**2
    ss_r+=(Y[i]-mean_y)**2
r2=ss_t/ss_r
print(r2)


# In[ ]:




