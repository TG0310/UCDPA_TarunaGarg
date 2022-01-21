#!/usr/bin/env python
# coding: utf-8

# In[60]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


# In[61]:


#reading csvs into pandas dataframes
heartdf = pd.read_csv("C:/Users/TG/Documents/input/heart.csv")
O2Saturationdf = pd.read_csv("C:/Users/TG/Documents/input/o2Saturation.csv")
                 


# In[62]:


#check the details about the file
print(heartdf.info())
print(O2Saturationdf.info())


# In[63]:


#check the first 5 records for both the dataframes
heartdf.head()
O2Saturationdf.head()

#Using merge to perform an outer join on the dataframes
pd.merge(heartdf, O2Saturationdf, left_index=True, right_index=True) 

#Checking for missing values
heartdf.isnull().sum()


# In[64]:


#get statistical values of the data
heartdf.describe()


# In[65]:


O2Saturationdf.describe()


# In[66]:


# For data visualization
categorical = ['sex','exng','caa','cp','fbs','restecg','slp','thall']
numerical = ["age","trtbps","chol","thalachh","oldpeak"]
f = plt.figure(figsize=(18, 10))
plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=16)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=16)
plt.title('Correlation Matrix', fontsize=16);


# In[78]:


# We have 9 collon so we create 9 figures.(ax0, ax1, ..., ax8)

fig = plt.figure(figsize=(20,20))

gr = fig.add_gridspec(3,3)

ax0 = fig.add_subplot(gr[0,0])
ax1 = fig.add_subplot(gr[0,1])
ax2 = fig.add_subplot(gr[0,2])
ax3 = fig.add_subplot(gr[1,0])
ax4 = fig.add_subplot(gr[1,1])
ax5 = fig.add_subplot(gr[1,2])
ax6 = fig.add_subplot(gr[2,0])
ax7 = fig.add_subplot(gr[2,1])
ax8 = fig.add_subplot(gr[2,2])

axxes = [ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7]

for i in range(0,len(categorical)):
    axxes[i].grid(color='#000000', linestyle=':', axis='y')
    sns.countplot(ax=axxes[i],data=df,x=df[categorical[i]],palette=["#"+str(i)+str(i)+"5353"]) #so that the figures change colors in each cycle
#NOT: if str(i) is greater than 9 it will throw an error, be careful
    
#ax8 is target colon 
ax8.grid(color='#000000', linestyle=':', axis='y')
sns.countplot(ax=ax8,data=df,x=df["output"],palette=["#53535353"])
ax8.set_xticklabels(["Low chances of attack(0)","High chances of attack(1)"])
ax8.set_facecolor("#FF5353") 


# In[73]:


#Modeling and Prediction, get all but the output column
X = df.iloc[:,:-1] 
#get the output column
y = pd.DataFrame(df["output"]) # We get output column

print("Shape of X:",X.shape,"\nShape of Y:",y.shape)


# In[74]:


#Splitting of data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size = 0.15, random_state = 53)


# In[75]:


print("Shape of x train:",x_train.shape,"\nShape of y train:", y_train.shape)
print("Shape of x test:",x_test.shape,"\nShape of y test:", y_test.shape)


# In[25]:





# In[77]:


acc_df = pd.DataFrame(columns=["Name", "Accuracy_score", "AUC_score", "F1_score"])

# Importing metrics libraries of sklearn
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression

model = LogisticRegression().fit(x_train, y_train)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("The test accuracy score is ", acc)
print("The test AUC score is", auc)
print("The test F1 score is", f1)

a_series = pd.Series(["LogisticRegression", acc, auc, f1], index = acc_df.columns)
acc_df = acc_df.append(a_series, ignore_index=True)


# In[59]:


#using Gradient boosting for classification
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=1).fit(x_train, y_train) 
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("The test accuracy score is ", acc)
print("The test AUC score is", auc)
print("The test F1 score is", f1)

a_series = pd.Series(["GradientBoosting", acc, auc, f1], index = acc_df.columns)
acc_df = acc_df.append(a_series, ignore_index=True)


# In[ ]:




