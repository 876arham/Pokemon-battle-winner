# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 21:23:53 2019

@author: Arham Jain
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

fight=pd.read_csv('combats.csv')
dataset2=pd.read_csv('pokemon.csv')

temp=dataset2.iloc[:,4:12].values
#from numpy import array
#data=array([temp])
#data=data.reshape(800,8)

from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
temp[:,7]=labelencoder.fit_transform(temp[:,7])

from numpy import array
data=array(temp,dtype=int)
data=data.reshape(800,8)
#dataset2=np.append(temp,)

fight=array(fight,dtype=int)
fight=fight.reshape(50000,3)

nds=np.zeros((50000,8),dtype=int)
out=np.zeros((50000,1),dtype=int)
for i in range(50000):
    for j in range(8):
        nds[i][j]=data[fight[i][0]-1][j]-data[fight[i][1]-1][j]
        if(fight[i][2]==fight[i][0]):
            out[i][0]=1
    
X=nds
y=out
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, y_train)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)


y_pred1 = classifier1.predict(X_test)
y_pred=classifier.predict(X_test)
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm1 = confusion_matrix(y_test, y_pred1)
cm=confusion_matrix(y_test,y_pred)

fighttest=pd.read_csv('tests.csv')
fighttest=array(fighttest,dtype=int)
fighttest=fighttest.reshape(10000,2)
X_test=np.zeros((10000,8),dtype=int)
#out=np.zeros((10000,1),dtype=int)
for i in range(10000):
    for j in range(8):
        X_test[i][j]=data[fighttest[i][0]-1][j]-data[fighttest[i][1]-1][j]
 #       if(fight[i][2]==fighttest[i][0]):
  #          out[i][0]=1

answer=classifier.predict(X_test)
for i in range(10000):
    if(answer[i]):
        answer[i]=fighttest[i][0]
    else :
        answer[i]=fighttest[i][1]
    
    
