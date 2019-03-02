# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 20:09:03 2018

@author: MOBASSIR
"""

from flask import Flask, render_template, request



app = Flask(__name__)


from catboost import CatBoostClassifier

# Importing the libraries
import pandas as pd


 
from sklearn.preprocessing import LabelEncoder
 
rnd_state = 0
 
#Importing the dataset
dataset = pd.read_csv('ovarian.csv')
X =  dataset.iloc[:, :-1].values
y = dataset.iloc[:, 11].values

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


cat_featuresind=list(range(0, 11))

clf = CatBoostClassifier (iterations=100,random_seed=rnd_state, custom_metric='Accuracy')

clf.fit(X_train, y_train, cat_features=cat_featuresind,plot = True)


clf.score(X_test, y_test)

from sklearn.metrics import confusion_matrix
y_pred = clf.predict(X_test)
 
cm = confusion_matrix (y_test, y_pred)



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/feedback', methods=['POST'])
def feedback():
  

    if request.method == 'POST':
        
        list = []
        comment = request.form['menarchestarts1']
        data = int(comment)
        list.append(data)
        comment = request.form['oralc1']
        data = int(comment)
        list.append(data)
        comment = request.form['Diet']
        data = int(comment)
        list.append(data)
        comment = request.form['brcancer']
        data = int(comment)
        list.append(data)
        comment = request.form['carcancer']
        data = int(comment)
        list.append(data)
        comment = request.form['canhist']
        data = int(comment)
        list.append(data)
        comment = request.form['edulev']
        data = int(comment)
        list.append(data)
        comment = request.form['husage']
        data = int(comment)
        list.append(data)
        comment = request.form['menenage']
        data = int(comment)
        list.append(data)
        comment = request.form['highfat']
        data = int(comment)
        list.append(data)
        comment = request.form['abrt']
        data = int(comment)
        list.append(data)
        
        #user input
        
        
        score = 0
        #"Menarche start early" 
       

        if(list[0] is 0):
            X_test[104][0] = "early"
            score+=2
        elif (list[0] is 1):
            X_test[104][0] = "late"
            score+=3
        else:
            X_test[104][0] = "normal"
            score+=0
        
       
        
         #"Oral Contraception" 
         


        if(list[1] is 0):
            X_test[104][1] = "yes"
            score+=0
        else:
            X_test[104][1] = "no"
            score+=3
        
        
       #"Diet Maintain" 
      
        
        if(list[2] is 0):
            X_test[104][2] = "yes"
            score+=2
        else:
            X_test[104][2] = "no"
            score+=3
        
        
        # "Affected By Breast Cancer" 

        if(list[3] is 0):
            X_test[104][3] = "yes"
            score+=3
        else:
            X_test[104][3] = "no"
            score+=0
        
        
        
       #"Affected By cervical Cancer?" 
     
        if(list[4] is 0):
            X_test[104][4] = "yes"
            score+=3
        else:
            X_test[104][4] = "no"
            score+=0
        
       
     #"Cancer History In family?" 
     

        
        if(list[5] is 0):
            X_test[104][5] = "yes"
            score+=2
        else:
            X_test[104][5] = "no"
            score+=0
        
        
       #"Education?" 
     
        
        if(list[6] is 0):
            X_test[104][6] = "primary level"
            score+=2
        elif (list[6] is 1):
            X_test[104][6] = "illitarate"
            score+=3
        else:
            X_test[104][6] = "graduated"
            score+=0
        
        
        # "Age of Husband" 

        if(list[7] is 0):
            X_test[104][7] = "46-60"
            score+=3
        elif (list[7] is 1):
            X_test[104][7] = "31-45"
            score+=0
        elif (list[7] is 2):
            X_test[104][7] = "below 30"
            score+=1
        else:
            X_test[104][7] = "above 60"
            score+=2
        
        
       # "Menopause End age?"
      
        if(list[8] is 0):
            X_test[104][8] = "40-51"
            score+=0
        elif (list[8] is 1):
            X_test[104][8] = "before 40"
            score+=2
        else:
            X_test[104][8] = "after 52"
            score+=3
        
        
        
       #"Food contains high fat?"
 
        
        if(list[9] is 0):
            X_test[104][9] = "no"
            score+=2
        else:
            X_test[104][9] = "yes"
            score+=3
        
        
        
        
      # "Abortion?" 

        if(list[10] is 0):
            X_test[104][10] = "no"
            score+=0
        else:
            X_test[104][10] = "yes"
            score+=3
        
        
        
        
        #str =['normal',	'yes',	'no',	'yes',	'yes',	'yes'	,'primary level',	'46-60',	'40-51',	'yes',	'no',	'no']

        check = clf.predict(X_test)
        check = check[104]
        #print(objs.iloc[:, :-1].values)
        print('ans = ',check, list,score)
        #return render_template('index.html')
       
        #comment = request.form['rating']
        #data = [comment]
    return render_template('result.html', prediction=check,score=score)

if __name__ == '__main__':
    app.run()