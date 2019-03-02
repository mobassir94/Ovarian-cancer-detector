# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 22:05:30 2018

@author: MOBASSIR
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
 
 
 
#Importing the dataset
dataset = pd.read_csv('ovarian.csv')
X =  dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values
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
 
 
 
 
label_x9 = LabelEncoder()
X[:,8] = label_x9.fit_transform(X[:,8] ) #Menopause End age?
 
 
 
 
label_x10 = LabelEncoder()
X[:,9] = label_x10.fit_transform(X[:,9] )
 
 
 
 
label_x11 = LabelEncoder()
X[:,10] = label_x11.fit_transform(X[:,10] )
 
 
 

onehotencoder = OneHotEncoder(categorical_features = [0,6,7,8])
X = onehotencoder.fit_transform(X).toarray()
 
X = X[: ,[13,16,18,19]] #9, removed


# Encoding the Dependent Variable
 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


 
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)




# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



"""
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
"""

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)





#Applying naive bayes classifier
 
from sklearn.naive_bayes import GaussianNB,BernoulliNB,MultinomialNB
 
classifier = BernoulliNB()
classifier.fit(X_train, y_train)
 
print(classifier)
 
y_expect = y_test
 
 
 
#predicting the test set result
 
y_pred = classifier.predict(X_test)
 
#Making the Confusion Matrix
 
from sklearn.metrics import confusion_matrix,accuracy_score
 
cm = confusion_matrix (y_test, y_pred)
 
 
print(accuracy_score(y_expect,y_pred))




 
# finding P value from statsmodels
 
import statsmodels.formula.api as sm
 
regressor_OLS = sm.OLS(endog=y,exog = X).fit()
 
regressor_OLS.summary()
 
 


from sklearn.cross_validation import cross_val_score

ck =  BernoulliNB()
scores = cross_val_score(ck,X,y,cv=10, scoring='accuracy')
print (scores)

print (scores.mean())




from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


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



estimator = BernoulliNB()


title = "Learning Curves (Naive Bayes classifier ALGORITHM)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.25, random_state=0)

plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=1)

plt.show()

#End of Bayes theorem


plt.rcParams['font.size'] = 14
 
plt.hist(y_pred, bins = 8)

plt.xlim(0, 1)

plt.title('Predicted probabilities')
plt.xlabel('Affected by ovarian cancer?(predicted)')
plt.ylabel('frequency')




from sklearn.metrics import recall_score,precision_score

recall_score(y_test,y_pred,average='macro')

precision_score(y_test, y_pred, average='micro')








# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Training set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Naive Bayes (Test set)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
