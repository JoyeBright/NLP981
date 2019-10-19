import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# importing the dataset
dataset = pd.read_csv('Iris.csv')
# print(dataset.head())

# drop the id column
dataset = dataset.drop('Id', axis=1)
# print(dataset.head())

# Summary of dataset
# print(dataset.shape)

# more information
# print(dataset.info())
# print(dataset.describe())
# print(dataset.groupby('Species').size())

# Visualization
# Box plot or whisker plot
dataset.plot(kind='box', sharex=False, sharey=False)
# plt.show()

# Data preparation
# X = dataset.iloc[:, :-1].values
# Target = dataset.iloc[:, -1].values

X = dataset.iloc[:, :4].values
Target = dataset.iloc[:, 4]
# by using values each row separated into dependent variables
# print(Target)

# Data splitting
X_train, X_test, Target_train, Target_test = train_test_split(X, Target, test_size=0.2, random_state=0)
# print(X_train.shape)

# Define a model

# Logistic Regression Classifier
classifier = LogisticRegression()
classifier.fit(X_train, Target_train)

Target_predict = classifier.predict(X_test)

print(classification_report(Target_test, Target_predict))
print(confusion_matrix(Target_test, Target_predict))
print('Accuracy of regression classifier::', accuracy_score(Target_test, Target_predict))

# Naive Bayes Classifier
# https://www.saedsayad.com/naive_bayesian.htm
classifier = GaussianNB()
classifier.fit(X_train, Target_train)

Target_predict = classifier.predict(X_test)

# Summary of the predictions made by the classifier
print(classification_report(Target_test, Target_predict))
print(confusion_matrix(Target_test, Target_predict))
# Accuracy score
print('accuracy is of NB', accuracy_score(Target_test, Target_predict))

# Decision Tree Classifier
classifier = DecisionTreeClassifier()

classifier.fit(X_train, Target_train)

Target_predict = classifier.predict(X_test)
print(classification_report(Target_test, Target_predict))
print(confusion_matrix(Target_test, Target_predict))
print('Accuracy of Decision Tree is:', accuracy_score(Target_test, Target_predict))

# SVM classifier
classifier = SVC()
classifier.fit(X_train, Target_train)

Target_predict = classifier.predict(X_test)

print(classification_report(Target_test, Target_predict))
print(confusion_matrix(Target_test, Target_predict))
print('Accuracy of SVM model is: ', accuracy_score(Target_test, Target_predict))