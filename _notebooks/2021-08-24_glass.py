#!/usr/bin/env python
# coding: utf-8

# # Machine learning to identify types of glass.  

# For this project I will determine the optimal machine learning model for us in glassifying glass as one of 7 types based on its refractive index and composition.  I will test logistic regression, k-nearest neighbors, decision tree, random forest, and support vector classifier models.  The data for this project comes from the glass identification dataset from kaggle (https://www.kaggle.com/uciml/glass?select=glass.csv)

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('glass.csv')


# In[3]:


df.info()


# In[4]:


df.head()


# In[37]:


#check for any null values
sns.heatmap(df.isnull())


# In[ ]:





# ### Preprocessing

# The features need to be scaled so the importance of large-scale features is not overestimated.

# In[6]:


from sklearn.preprocessing import StandardScaler


# In[11]:


sc=StandardScaler()


# In[12]:


sc.fit(df.drop('Type', axis=1))


# In[13]:


sc_feats=sc.transform(df.drop('Type', axis=1))
sc_df=pd.DataFrame(sc_feats, columns=df.columns[:-1])


# In[106]:


#Create dataframe of scaled features
sc_df.head()


# ## Machine Learning Models

# In order to determine an approriate model to use in glassifying glasss types I will test several different machine learning techniques and compare their accuracy.

# In[ ]:


#separate the data into training and testing sets
X=sc_df
y=df['Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ### Logistic Regression

# In[60]:



from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


# In[ ]:





# In[161]:


#create the model, fit to the data, and test
logreg=LogisticRegression()
logreg.fit(X_train,y_train)
lr_pred=logreg.predict(X_test)
lr_score=accuracy_score(y_test,pred)


# In[109]:


lrcm=(confusion_matrix(y_test, lr_pred))


# In[162]:



print(classification_report(y_test,lr_pred))
print('\n')
print(lr_score)


# ### K-Nearest Neighbors

# In[20]:


#import KNN
from sklearn.neighbors import  KNeighborsClassifier


# First, we need to determine the appropriate number of neighbors to use in the model.  To do this, I check the error rate of k values between 1 and 20.

# In[ ]:


#use for loop to check each k value and save its error-rate
e_rate=[]
for i in range(1,20):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i=knn.predict(X_test)
    e_rate.append(np.mean(pred_i!=y_test))
    
    


# In[36]:


#plot the error-rate for each k value to determine an appropriate choice
plt.figure(figsize=(10,6))
plt.plot(range(1,20), e_rate, color='b', linestyle='--', marker='o', markerfacecolor='r', markersize=10)
plt.title('Error-rate vs K value')
plt.xlabel('K')
plt.ylabel('Error-rate')


# Based on the error rates I chose to use 5 as the k value.  At k=5, the error-rate is low and the and the error-rate curve does not show any extreme jumps.

# In[158]:


#use the chosen k value to create the model then fit and make predictions
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
predK=knn.predict(X_test)


# In[159]:


kcm=confusion_matrix(y_test, predK)
k_score=accuracy_score(y_test, predK)
k_score


# In[160]:



print(classification_report(y_test,predK))
print('\n')
print()


# In[ ]:





# ### Decision-Tree

# In[70]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[115]:


dtree=DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predT=dtree.predict(X_test)


# In[117]:


dtcm=confusion_matrix(y_test,predT)


# In[118]:


print(classification_report(y_test,predT))
print(accuracy_score(y_test,predT))


# ### Random Forest

# In[120]:


rfc=RandomForestClassifier(n_estimators=1000)
rfc.fit(X_train, y_train)
predR=rfc.predict(X_test)


# In[123]:


rfcm=confusion_matrix(y_test,predR)


# In[124]:


print(classification_report(y_test,predR))
print(accuracy_score(y_test,predR))


# ### Support Vector Classifier

# In[95]:


from sklearn.svm import SVC


# In[125]:


sMod=SVC()
sMod.fit(X_train, y_train)
predS=sMod.predict(X_test)


# In[126]:


svccm=confusion_matrix(y_test,predS)


# In[127]:


print(classification_report(y_test,predS))
print(accuracy_score(y_test,predS))


# ## Conclusion

# To evaluate the models we can examine the confusion matrices for each one.

# In[ ]:


#arrange the confusion matrices so we can loop through them and create a list of the titles for each one
cm_all=[lrcm, kcm, dtcm,rfcm,svccm]
titles=['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'SVC']
#set up a counter for the subplots and titles
count=1
#plot and label each confusion matrix
plt.subplots(figsize=(12, 12))
plt.suptitle('Confusion Matrices')  
for i in cm_all:
    plt.subplot(3,2,count)
    sns.heatmap(i, annot=True,cbar=False, cmap='viridis')
    plt.title(titles[count-1])
    count+=1


# In[165]:


scores=[accuracy_score(y_test,lr_pred),accuracy_score(y_test,predK), accuracy_score(y_test,predT), accuracy_score(y_test,predR),accuracy_score(y_test,predS)]


# In[279]:


scoresDict = dict(zip(titles, scores))
sd=pd.DataFrame(scoresDict, index=['Accuracy']).round(3).transpose()

sd['Accuracy']=sd['Accuracy'].apply(lambda x:str(x*100)+'%')


# In[280]:


sd


# The random forest method gives the highest degree of accuracy for this problem.  It correctly predicts the glass type 73.8% of the time.  The logistic regression model is the worst of the group with an accuracy of only 49.2%.

# In[ ]:




