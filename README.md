# SMS_CLASSIFICATION
## Overview
#### Context
The SMS Spam Collection is a set of SMS tagged messages that have been collected for SMS Spam research. It contains one set of SMS messages
in English of 5,574 messages, tagged acording being ham (legitimate) or spam.

#### Content
The files contain one message per line. Each line is composed by two columns: v1 contains the label (ham or spam) and v2 contains the 
message. The originate dataset can be found [here.]( https://www.kaggle.com/uciml/sms-spam-collection-dataset)

## Approach
* Load the Data
* Apply Regular Expression
* Convert each word into lower case
* Split words to Tokenize
* Remove Stop words and Lemmatize
* Prepare messages with remaining tokens
* Apply Vectorization and Classification models
* Predict the results

### Models used
* Naive Bayes
* Logistic Regression
* Decision Trees
* Ensemble methods(Random Forest)
* Passive Aggressive

The trained model is deployed by creating API for the model, using Flask, the Python micro framework for building web applications.To convert
the .ipynb files to .py files nbcovert is used.

```python
jupyter nbconvert --to script *.ipynb
```

## Results
With Count Vectorizer

| Model      | Score           | 
| ------------- |:-------------:| 
| Bernoulli NB      | 0.9793 | 
| Logistic Regression      | 0.9793 |  
| Decision Trees | 0.9676     |
| Random Forest     | 0.9757 | 
| Passive Aggressive      | 0.9811     |  


With Tf-idf Vectorizer

| Model      | Score           | 
| ------------- |:-------------:| 
| Bernoulli NB      | 0.9793 | 
| Logistic Regression      | 0.9640 |  
| Decision Trees | 0.9667    |
| Random Forest     | 0.9775 | 
| Passive Aggressive      | 0.9802  | 

JSON Input to API 
![api](https://user-images.githubusercontent.com/58266508/69785749-dc05c900-11de-11ea-995e-8a2540f727a7.JPG)

Response of API
![api1](https://user-images.githubusercontent.com/58266508/69786094-ba591180-11df-11ea-875f-dbbfd2b5eb38.JPG)
