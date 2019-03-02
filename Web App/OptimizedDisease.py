# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 14:54:29 2018

@author: MOBASSIR
"""

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

from sklearn.svm import SVC
 
  
#Importing the dataset
dataset = pd.read_csv('disease.csv')

X =  dataset.iloc[:, :-1].values
y = dataset.iloc[:,3].values
#X[199] = ["a","b","v"]
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
 
label_x1 = LabelEncoder()
X[:,0] = label_x1.fit_transform(X[:,0] )
 
 
 
label_x2 = LabelEncoder()
X[:,1] = label_x2.fit_transform(X[:,1] )
 
label_x3 = LabelEncoder()
X[:,2] = label_x3.fit_transform(X[:,2] )
 
 
onehotencoder = OneHotEncoder(categorical_features = [0,1,2])
X = onehotencoder.fit_transform(X).toarray()
 






#1,4,8 columns are removed

X = X[: ,[0,2,3,5,6,7,9,10,11,12]]


# Encoding the Dependent Variable
 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

 
 
# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)



 
# Build RF classifier to use in feature selection
clf = SVC(kernel = 'linear', random_state = 0, probability=True)

# Build step forward feature selection
sfs1 = sfs(clf,
           k_features=5,
           forward=True,
           floating=False,
           verbose=2,
           scoring='accuracy',
           cv=10)

# Perform SFFS
sfs1 = sfs1.fit(X_train, y_train)


# Which features?
feat_cols = list(sfs1.k_feature_idx_)
print(feat_cols)
sfs1.subsets_

# Build full model with selected features
clf = SVC(kernel = 'linear', random_state = 0, probability=True)
clf.fit(X_train[:, feat_cols], y_train)

y_train_pred = clf.predict(X_train[:, feat_cols])
print('Training accuracy on selected features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test[:, feat_cols])
print('Testing accuracy on selected features: %.3f' % acc(y_test, y_test_pred))



# Build full model on ALL features, for comparison
clf = SVC(kernel = 'linear', random_state = 0, probability=True)
clf.fit(X_train, y_train)

y_train_pred = clf.predict(X_train)
print('Training accuracy on all features: %.3f' % acc(y_train, y_train_pred))

y_test_pred = clf.predict(X_test)
print('Testing accuracy on all features: %.3f' % acc(y_test, y_test_pred))




from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')

plt.ylim([0.8, 1])
plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()




# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cm = confusion_matrix(y_test, y_test_pred)

end = time.time()

print(end - start)


