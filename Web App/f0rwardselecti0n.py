# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:37:58 2018

@author: MOBASSIR
"""


import time
start = time.time()

import matplotlib.pyplot as plt
# Importing the libraries
import numpy as np
import pandas as pd
 


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as acc
from mlxtend.feature_selection import SequentialFeatureSelector as sfs




 
  
#Importing the dataset
dataset = pd.read_csv('appendix_for_ml.csv')

X =  dataset.iloc[:, :-1].values
y = dataset.iloc[:, 8].values

 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
 
label_x1 = LabelEncoder()
X[:,0] = label_x1.fit_transform(X[:,0] )
 
 
 
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
X[:,6] = label_x7.fit_transform(X[:,6] ) 
 

label_x8 = LabelEncoder()
X[:,7] = label_x8.fit_transform(X[:,7] ) 
 


 
onehotencoder = OneHotEncoder(categorical_features = [1,4,6])
X = onehotencoder.fit_transform(X).toarray()
 






#1,7,13 columns are removed

X = X[: ,[0,2,3,4,5,6,8,9,10,11,12,14,15,16,17]]


# Encoding the Dependent Variable
 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


 
 
# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)



 
# Build RF classifier to use in feature selection
clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

# Build step forward feature selection
sfs1 = sfs(clf,
           k_features=1,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=2)

# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train)


# Which features?
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)
#sfs1.subsets_

# Build full model with selected features
clf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=4)
clf.fit(X_train[:, feat_cols], y_train)

y_train_pred = clf.predict(X_train[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))



# Build full model on ALL features, for comparison
clf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=4)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
print('Training accuracy on all features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test)
print('Testing accuracy on all features: %.3f' % acc(y_test, y_test_pred))


end = time.time()

print(end - start)


