# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 02:20:22 2018

@author: MOBASSIR
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
 
 
 
#Importing the dataset
dataset = pd.read_csv('appendix_for_ml.csv')
X =  dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values
#dummy_x = dataset.iloc[:, [0,6,7,8]].values
 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

label_x1 = LabelEncoder()
X[:,0] = label_x1.fit_transform(X[:,0] ) #Menarche start early
 
 
 
label_x2 = LabelEncoder()
X[:,1] = label_x2.fit_transform(X[:,1] )
 
label_x3 = LabelEncoder()
X[:,2] = label_x3.fit_transform(X[:,2] )
 
 
label_x4 = LabelEncoder()
X[:,3] = label_x4.fit_transform(X[:,3] )
 
 
label_x5 = LabelEncoder()
X[:,4] = label_x5.fit_transform(X[:,4] )
 
 
label_x6 = LabelEncoder()
X[:,5] = label_x6.fit_transform(X[:,5] )
 
 
label_x7 = LabelEncoder()
X[:,6] = label_x7.fit_transform(X[:,6] ) #Education
 
 
 
 
label_x8 = LabelEncoder()
X[:,7] = label_x8.fit_transform(X[:,7] ) #Age of Husband
 
 
 
 

 
onehotencoder = OneHotEncoder(categorical_features = [1,4,6])
X = onehotencoder.fit_transform(X).toarray()
 
#X = X[: ,[13,16,18,19]] #9, removed


# Encoding the Dependent Variable
 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif

test = SelectKBest(score_func=chi2, k=2)
test.fit(X, y)

num_features = len(dataset.columns)

scores = []
for i in range(num_features):
    score = test.scores_[i]
    scores.append((score, dataset.columns[i]))
        
print (sorted(scores, reverse = True))


def print_best_worst (scores):
    scores = sorted(scores, reverse = True)
    
    print("The 5 best features selected by SelectKBest method :")
    for i in range(4):
        print(scores[i][1])
    
    print ("The 5 worst features selected by SelectKBest method :")
    for i in range(4):
        print(scores[len(scores)-1-i][1])
        

print_best_worst(scores)

test = SelectKBest(score_func = mutual_info_classif, k=2)
test.fit(X, y)





#rfe

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


rfe = RFE(LogisticRegression(), n_features_to_select=1)
rfe.fit(X,y)

scores = []
for i in range(4):
    scores.append((rfe.ranking_[i],dataset.columns[i]))
    
print_best_worst(scores)

from sklearn.ensemble import RandomForestClassifier

rfe = RFE(RandomForestClassifier(), n_features_to_select = 1)
rfe.fit(X,y)

scores = []
for i in range(num_features):
    scores.append((rfe.ranking_[i],dataset.columns[i]))
    
print_best_worst(scores)

