# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 15:21:49 2018

@author: MOBASSIR
"""


import time
start = time.time()

import matplotlib.pyplot as plt
# Importing the libraries
import numpy as np
import pandas as pd
 
 
 
  
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
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# Fitting SVM to the Training set
from sklearn.svm import SVC
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
classifier = SVC(kernel = 'linear', random_state = 0, probability=True)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

cm = confusion_matrix(y_test, y_pred)



 
print(accuracy_score(y_test,y_pred)) 

print(classification_report(y_test, y_pred))




y_pred_prb = classifier.predict_proba(X_test)

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt



estimator = SVC()


title = "Learning Curves (SVM ALGORITHM)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.25, random_state=0)

plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1)

plt.show()



from sklearn.cross_validation import cross_val_score

ck =  SVC()
scores = cross_val_score(ck,X,y,cv=10, scoring='accuracy')
print (scores)

print (scores.mean())

# finding P value from statsmodels
 
import statsmodels.formula.api as sm
 
regressor_OLS = sm.OLS(endog=y,exog = X).fit()
 
regressor_OLS.summary()


 

 

plt.rcParams['font.size'] = 14
 
plt.hist(y_pred_prb, bins = 5)

plt.xlim(0, 1)

plt.title('Appendix')
plt.xlabel('surgery? (predicted yes or no)')
plt.ylabel('frequency')




from sklearn.metrics import recall_score,precision_score

print(recall_score(y_test,y_pred,average='macro'))

print(precision_score(y_test, y_pred, average='micro'))





from sklearn import metrics
y_pred_proba = classifier.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

end = time.time()
print(end - start)