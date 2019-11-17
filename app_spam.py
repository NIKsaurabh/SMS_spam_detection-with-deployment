#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 11:26:59 2019

@author: saurabh
"""

from flask import Flask, request,render_template
import pandas as pd
from nltk.stem import WordNetLemmatizer
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('input_spam.html')

def preProcess(data):
    #data = re.sub('[^a-zA-Z]',' ',data)
    data = data.lower()
    word = data.split()
    word = [WordNetLemmatizer().lemmatize(w) for w in word if w not in set(stopwords.words('english'))]
    review = ' '.join(word)
    return review
@app.route('/predict',methods = ['POST'])
def predict():
    dataset = pd.read_csv('spam.csv', encoding = 'latin-1')
    dataset['v1'] = dataset['v1'].map({'ham':0, 'spam':1})
    x = dataset['v2']
    y = dataset['v1']
    corpus=[]
    for i in range(len(dataset)):
        corpus.append(preProcess(x[i]))
        
    cv = CountVectorizer()
    X = cv.fit_transform(x)
     
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
    
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    filename = 'app_spam.pkl'
    pickle.dump(clf,open(filename , 'wb'))
    
    if request.method == 'POST':
        message = request.form['name']
        
        data = [preProcess(message)]
        vect = cv.transform(data).toarray()
        prediction = clf.predict(vect)
        if prediction  == 0:
            output = 'Its not a spam'
        else:
            output = 'SPAM'
    
    return render_template('input_spam.html', final_output = output)

if __name__ == '__main__':
    app.run(debug = True)
