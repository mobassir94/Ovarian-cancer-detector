# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 00:34:47 2018

@author: MOBASSIR
"""

from sklearn.linear_model import RandomizedLasso


from catboost import Pool, CatBoostClassifier, cv

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import shap

 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
 
rnd_state = 0
 
strinp =['normal',	'yes',	'no',	'yes',	'yes',	'yes'	,'primary level',	'46-60',	'40-51',	'yes',	'no',	'no']


#Importing the dataset
dataset = pd.read_csv('ovarian.csv')
X =  dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#X_test[104][10] = "yes"


cat_featuresind=list(range(0, 11))

clf = CatBoostClassifier (iterations=1000,random_seed=rnd_state, custom_metric='Accuracy')

clf.fit(X_train, y_train, cat_features=cat_featuresind,plot = True)


clf.score(X_test, y_test)





from sklearn.metrics import confusion_matrix,accuracy_score
y_pred = clf.predict(X_test)

print(clf.predict(X_test[104]))
cm = confusion_matrix (y_test, y_pred)




importances = clf.feature_importances_
print(clf.feature_importances_)
plt.title('Feature Importances (ovarian cancer)')
plt.barh(range(len(cat_featuresind)), importances[cat_featuresind], color='b', align='center')
#plt.yticks(dataset[i][0] for i in cat_featuresind)
plt.xlabel('Relative Importance')
plt.show()
 





plt.bar(range(len(clf.feature_importances_)))
plt.show()

from sklearn import metrics
y_pred_proba = clf.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.xlabel("False Positive Rate (FPR)", fontsize=14)
plt.ylabel("True Positive Rate (TPR)", fontsize=14)
plt.title("ROC Curve (ovarian cancer)", fontsize=14)

plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()



from sklearn.metrics import recall_score,precision_score

recall_score(y_test,y_pred,average='macro')

precision_score(y_test, y_pred, average='micro')

plt(clf)

print(accuracy_score(y_test,y_pred))




pool1 = Pool(data=X, label=y, cat_features=cat_featuresind)


importances = clf.get_feature_importance(prettified=True)
print (importances)

shap_info = clf.get_feature_importance(data=pool1, fstr_type='ShapValues', verbose=10000)
shap_values = shap_info[:,:-1]
base_values = shap_info[:,-1]
print(shap_values.shape)
X =  dataset.iloc[:, :-1].values
shap.initjs()
shap.force_plot(base_value=base_values[0], shap_values=shap_values[0], features=X.iloc[0])
shap.summary_plot(shap_values, X)
 