#!/usr/bin/env python
# coding: utf-8

# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np   
from sklearn.linear_model import LinearRegression
import pandas as pd    
import matplotlib.pyplot as plt   
import matplotlib.style
plt.style.use('classic')
import seaborn as sns
from scipy import stats


# In[40]:


df = pd.read_csv("LinearRegression.csv")
df


# In[42]:


df.head(50)


# In[43]:


df.describe().transpose()


# In[44]:


df.dtypes


# In[47]:


df[df.isnull().any(axis=1)]


# In[48]:


df.describe()


# In[49]:


df.groupby('Designation')['Pay'].median()


# In[50]:


df.hist(figsize = (20,30))


# In[51]:


sns.pairplot(df, diag_kind='kde')


# In[52]:


fig = sns.boxplot(x='Designation', y="Pay", data=df)


# In[53]:


fig = sns.boxplot(x='Years of experience', y="Pay", data=df)


# In[54]:


df.boxplot(figsize=(15, 10))


# In[55]:


import matplotlib.pylab as plt
corr = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, annot=True, linewidths=1)


# In[57]:


# Copy all the predictor variables into X dataframe
X = df.drop('Pay', axis=1)

# Copy the 'Pay' column alone into the y dataframe. This is the dependent variable
y = df[['Pay']]


# In[59]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , random_state=1)


# In[61]:


regression_model = LinearRegression()
regression_model.fit(X_train, y_train)


# In[62]:


for idx, col_name in enumerate(X_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[0][idx]))


# In[63]:


intercept = regression_model.intercept_[0]

print("The intercept for our model is {}".format(intercept))


# In[64]:


regression_model.score(X_train, y_train)


# In[ ]:


# So the model explains 93.5% of the variability in Y using X

