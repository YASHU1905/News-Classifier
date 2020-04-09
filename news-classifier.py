#Fetching Dataset(20newsgroup) 
# import all dependencies
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.datasets import fetch_20newsgroups
import pandas as pd

# fetch the scikit learn dataset for news classification and store into `data` variable
data = fetch_20newsgroups()

print(type(data))

print(data.keys())

#Print number of categories in dataset
categories = data.target_names
print("There are {} Categories in this dataset".format(len(categories)))

#Print samples of dataset
train = fetch_20newsgroups(subset='train')
test  = fetch_20newsgroups(subset='test')
print(train.data[1])

#Printing score
# We need Vectorizer to split the words from article and assign weight
from sklearn.feature_extraction.text import TfidfVectorizer

#Import Naive Bayes Multinomial classifier
from sklearn.naive_bayes import MultinomialNB

#Import pipeline for processing data through vectorizer and naive bayes classifier
from sklearn.pipeline import make_pipeline

# initialize the navie bayes model 
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# train the model with fix number of epochs default = 1
model.fit(train.data, train.target)

# test the model on testing dataset
labels = model.predict(test.data)

score = model.score(test.data,test.target)
print("Score: ", score)

#Evaluating the model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.data,data.target)
y_predict = model.predict(x_test)
print("y_predict:\n", y_predict)

score = model.score(x_test, y_test)
print("score：\n", score)

#Creating confusion matrix
from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target, labels)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', xticklabels=train.target_names, yticklabels=train.target_names)
plt.xlabel('Correct Label')
plt.ylabel('Predicted Label')
plt.rcParams['figure.figsize'] = [50, 40]

#Function that predicts the category
def predict_category(s, train=train, model=model):
    label = model.predict([s])
    return train.target_names[label[0]]
#example-1
predict_category('Solar flares burst could potentially impact all electronic equipment')

#example-2
predict_category("The Senate prepared Wednesday to pass a short-term spending bill that would keep the government open through the New Year")
