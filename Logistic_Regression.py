#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics 
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[5]:


biopslides=pd.read_csv('Training.csv')
X = biopslides.drop('y', axis=1)  
y = biopslides['y']


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.80) 
training_accuracy = []
test_accuracy = []
logreg = LogisticRegression(C=100).fit(X_train, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train, y_train)))
print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))


# In[21]:


y_pred = logreg.predict(X_test)


# In[22]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred))  
print(classification_report(y_test, y_pred))


# In[38]:


arr1=[]
testdata = pd.read_csv('TestImage3.csv')
for index, row in testdata.iterrows():
     # access data using column names
     arr1.append(row)


# In[39]:


result = []
for i in range(len(arr1)):
    result.append(logreg.predict([arr1[i]]))
    print(logreg.predict([arr1[i]]))


# In[40]:


print(result)


# In[41]:


print(result.count([0]))
print(result.count([1]))


# In[ ]:




