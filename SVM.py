import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import itertools
import seaborn as sns
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics 


biopslides=pd.read_csv('Train.csv')
biopslides.shape



X = biopslides.drop('y', axis=1)  
y = biopslides['y']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.60)



svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)



y_pred = svclassifier.predict(X_test)


# In[7]:


from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))  


# In[8]:


training_accuracy = []
test_accuracy = []


# In[9]:


print("Accuracy on training set: {:.2f}".format(svclassifier.score(X_train, y_train)))
print("Accuracy on test set: {:.2f}".format(svclassifier.score(X_test, y_test)))


# In[10]:


arr1=[]
testdata = pd.read_csv('TestImage3.csv')
for index, row in testdata.iterrows():
     # access data using column names
     arr1.append(row)


# In[11]:


result = []
for i in range(len(arr1)):
    result.append(svclassifier.predict([arr1[i]]))
    print(svclassifier.predict([arr1[i]]))


# In[12]:


print(result.count([0]))
print(result.count([1]))

