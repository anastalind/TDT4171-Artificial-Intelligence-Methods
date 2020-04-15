
# TDT4171 Assignment 5
# @author Anastasia Lindbäck
#
# Part 1
#
import pickle
import numpy as np

from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("Loading data...")
data = pickle.load(open("sklearn-data.pickle", "rb"))

x_train, y_train = data.get('x_train'), data.get('y_train')
x_test, y_test = data.get('x_test'), data.get('y_test')

# Recoding the reviews from a list of strings to a sparse matrix
print("Transforming data...")
vectorizer = HashingVectorizer(stop_words='english', n_features=2**18, binary=True)
X_train = vectorizer.transform(x_train)
X_test = vectorizer.transform(x_test)

# Naive Bayes classifier
print("Defining Naive Bayes classifier...")
classifier_NB = BernoulliNB()

print("Training classifier..")
classifier_NB.fit(X=X_train, y=y_train)

print("Predicting test set...")
y_NB = classifier_NB.predict(X=X_test)

print("\n")

# Decision tree classifier
print("Defining Decision Tree classifier...")
classifier_DT = DecisionTreeClassifier(max_depth=2)

print("Training classifier...")
classifier_DT.fit(X=X_train, y=y_train)

print("Predicting test set...")
y_DT = classifier_DT.predict(X=X_test)

# Measure quality of the predictions by comparing to correct classes
accuracy_NB = accuracy_score(y_test, y_NB)
accuracy_DT = accuracy_score(y_test, y_DT)

print("\n")

print("The accuracy of the Naive Bayes classifier is ", accuracy_NB)
print("The accuracy of the decision tree classifier is ", accuracy_DT)
