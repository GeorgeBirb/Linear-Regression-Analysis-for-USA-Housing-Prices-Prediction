#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('USA_Housing.csv')


# In[5]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# In[9]:


df.columns


# In[10]:


sns.pairplot(df)


# In[12]:


sns.distplot(df['Price'])


# In[14]:


sns.heatmap(df.corr(),annot=True)


# In[15]:


df.columns


# In[16]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[17]:


y = df['Price'] 


# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[21]:


from sklearn.linear_model import LinearRegression


# In[22]:


lm = LinearRegression()


# In[24]:


lm.fit(X_train,y_train)


# In[25]:


print(lm.intercept_)


# In[26]:


lm.coef_


# In[27]:


X.columns


# In[29]:


cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[30]:


cdf


# In[36]:


predictions = lm.predict(X_test)


# In[37]:


predictions


# In[38]:


plt.scatter(y_test,predictions)  


# In[40]:


sns.distplot((y_test-predictions))


# In[41]:


from sklearn import metrics


# In[42]:


metrics.mean_absolute_error(y_test,predictions)


# In[43]:


metrics.mean_squared_error(y_test,predictions)

