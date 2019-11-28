#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import *
from sklearn.ensemble import *
from sklearn.linear_model import *
from sklearn.multiclass import *


from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
#nltk.download('wordnet')
#nltk.download('stopwords')


# In[2]:


dataset_path = 'C:\\Users\\LENOVO\\Desktop\\spam.csv'
model_pkl_path = 'C:\\Users\\LENOVO\\Desktop\\ASSGNMNT\\PA_model.pickle'
vectorizer_pkl_path = 'C:\\Users\\LENOVO\\Desktop\\ASSGNMNT\\tfidf.pickle'
labelencoder_pkl_path = 'C:\\Users\\LENOVO\\Desktop\\ASSGNMNT\\le.pickle'


# In[3]:


#Read the data 
df = pd.read_csv(dataset_path, encoding='latin-1')
df.head()


# In[4]:


# Remove null columns

df = df.dropna(axis=1)
df.rename(columns = {'v1':'Labels', 'v2':'Messages'}, inplace = True)
df.head()


# In[5]:



labels= df['Labels']
message = df['Messages']


# Apply labelencoder to convert non-numeric labels to numerical labels
encoder = LabelEncoder()
y = encoder.fit_transform(labels)

# Count the frequency of class labels
print(labels.value_counts())

# Calculate the frequency of each message
df['Count']=0
for i in np.arange(0,len(df)):
    df.loc[i,'Count'] = len(df.loc[i,'Messages'])
    
df.head()


# In[9]:


class spam_predictor(object):
    
    def __init__(self,spam_data):
        self.data = spam_data
        
    def preprocess(self):
        cls = self.data['Labels']
        #Apply label encoder to convert to numeric labels
        encoder = LabelEncoder()
        self.y = encoder.fit_transform(cls)
        with open(labelencoder_pkl_path, 'wb') as picklefile:
            pickle.dump(encoder,picklefile)
        message = self.data['Messages'] 
        
        # Remove email address, web address, money symbols, single digit numbers, contact number from the text
        self.filtered_message = message.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$','')
        self.filtered_message = self.filtered_message.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2,3}(/\S*)?$','')
        self.filtered_message = self.filtered_message.str.replace(r'£|\$', '')
        self.filtered_message = self.filtered_message.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{4}$', '')
        self.filtered_message = self.filtered_message.str.replace(r'\d+(\.\d+)?', '')
        
        # set the stop words
        stop_words= set(stopwords.words('english')) 
        
        # Lemmatize and remove the trailing spaces form the text
        def lemmatize_text(text):
            return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]
        
        w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        lemmatizer = nltk.stem.WordNetLemmatizer()
        
        # Convert text to lowercase
        self.filtered_message = self.filtered_message.str.lower()
        
        self.filtered_message = self.filtered_message.apply(lemmatize_text)
        self.filtered_message = self.filtered_message.apply(lambda x: ' '.join([word for word in x if word not in (stop_words)]))
        
    def classifier(self):
        self.perform(
                        [
                            BernoulliNB(),
                            RandomForestClassifier(n_estimators=100, n_jobs=-1),
                            DecisionTreeClassifier(),
                            PassiveAggressiveClassifier(),
                            OneVsRestClassifier(LogisticRegression())
                        ],
                        [
                            CountVectorizer(),
                            TfidfVectorizer(),
                        ]
                    )
    def perform(self, classifiers, vectorizers):
        for classifier in classifiers:
            for vectorizer in vectorizers:
                string = ''
                string += classifier.__class__.__name__ + ' with ' + vectorizer.__class__.__name__

                # train
                vectorize_text = vectorizer.fit_transform(self.filtered_message)
                self.x = vectorize_text
                skf = StratifiedKFold(n_splits=5, random_state=None)
                for train_index, val_index in skf.split(self.x,self.y):
                    xtrain, xtest = self.x[train_index], self.x[val_index] 
                    ytrain, ytest = self.y[train_index], self.y[val_index]
                
                # Fit the training data 
                classifier.fit(xtrain,ytrain)
                if classifier.__class__.__name__ == 'PassiveAggressiveClassifier' and vectorizer.__class__.__name__ == 'TfidfVectorizer':
                    with open(model_pkl_path, 'wb') as picklefile:
                        pickle.dump(classifier,picklefile)
                    with open(vectorizer_pkl_path, 'wb') as picklefile:
                        pickle.dump(vectorizer,picklefile)
                # Predict the class label for test data
                y_pred = classifier.predict(xtest)
                
                # Compute the metrics
                score = accuracy_score(ytest, y_pred)
                f1score = f1_score(ytest, y_pred, average= 'binary')
                string += 'has Accuracy score: ' + str("{0:.4f}".format(score)) +' F1 score: ' +str("{0:.4f}".format(f1score))
                print(string)

  


# In[10]:


predict_spam = spam_predictor(df)
predict_spam.preprocess()
predict_spam.classifier()


# In[8]:


import os
#from werkzeug.wrappers import Request, Response
from flask import Flask,jsonify,request

app = Flask(__name__)

global is_spam_flag

with open(model_pkl_path, 'rb') as training_model:
    Classifier = pickle.load(training_model)
with open(vectorizer_pkl_path, 'rb') as training_model:
    vectorizer = pickle.load(training_model)
with open(labelencoder_pkl_path, 'rb') as training_model:
    encoder = pickle.load(training_model)

def predict_proba(vectorize_message):
    def softmax(X):
        max_prob = np.max(X, axis=1).reshape((-1, 1))
        X -= max_prob
        np.exp(X, X)
        sum_prob = np.sum(X, axis=1).reshape((-1, 1))
        X /= sum_prob
        return X

    decision = Classifier.decision_function(vectorize_message)
    if decision.ndim == 1:
        decision_2d = np.c_[-decision, decision]
    else:
        decision_2d = decision
    print("decisuion||||", decision_2d)
    return (softmax(decision_2d))


@app.route("/predict_spam",methods=['GET','POST'])
def get_predictions():
    if request.content_type == 'application/json':
        req = request.get_json()
        sender = req['data']['sender']
        text = req['data']['text']
        is_spam = ''
        confidence = ''
        try:
            if len(text) > 0:
                vectorize_message = vectorizer.transform([text])
                prediction = Classifier.predict(vectorize_message)[0]
                prediction = encoder.inverse_transform([prediction])
                if prediction == 'ham':
                    is_spam_flag = 'False'
                else:
                    is_spam_flag = 'True'
                prob = predict_proba(vectorize_message)
                prob = prob.tolist()
                print(is_spam_flag)
                print("prob",prob)
            else:
                return "Message not specified", 400
        except BaseException as inst:
            error = str(type(inst).__name__) + ' ' + str(inst)
            print(error)
        
        #jsonify({'prediction': prediction})
        return jsonify(is_spam = is_spam_flag, confidence=prob)
    else:
        return "Request was not JSON", 400
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run()


# In[ ]:




