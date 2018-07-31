#############################################################################
#           Import packages
#
#############################################################################

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

#############################################################################
#           Read CSV File Data
#
#############################################################################
df = pd.read_csv("C:/dev/MachineLearning/Starter_python/data/pima-data.csv")
# data frame shape
print(df.shape)
#head 5 data
print(df.head(5))
#Tail 5 data
print(df.tail(5))

#############################################################################
#           Analyze Data
#
#############################################################################

print(df.isnull().values.any())

print(df.corr()) # Coorelation matrix to see data / column correlation

def plot_corr(df, size = 11):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size,size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)),corr.columns)
    plt.yticks(range(len(corr.columns)),corr.columns)

plot_corr(df)  # plot the correlation graph

#############################################################################
#           Data preparation
#
#############################################################################
# Delete the data which is otherwise represented by another column
del df['skin'] 

print(df.head(5))

diabetes_map = {True : 1, False : 0}

df['diabetes'] = df['diabetes'].map(diabetes_map)

print(df.head(5))

#############################################################################
#          Check data distribution
#
#############################################################################
# Result data distribution
num_true = len(df.loc[df['diabetes'] == True])
num_false = len(df.loc[df['diabetes'] == False])
total = num_true + num_false
true_pct = (num_true/total)*100
false_pct = (num_false/total)*100
print("Number of True Cases: {0} ({1:.2f}%)".format(num_true, true_pct))
print("Number of False Cases: {0} ({1:.2f}%)".format(num_false, false_pct))

#############################################################################
#           Split the data
#
# SKLearn  - Training data split - 70% for training and 30% for testing
#############################################################################
from sklearn.cross_validation import train_test_split

feature_col_names = ['num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age']
predicted_class_name = ['diabetes']
x = df[feature_col_names].values
y = df[predicted_class_name].values
split_test_size = 0.3

# train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=split_test_size,random_state = 42)

# Check data distribution after split
x_train_set_pct = (len(X_train)/len(df.index))*100
x_test_set_pct = (len(X_test)/len(df.index))*100

y_train_set_pct = (len(Y_train)/len(df.index))*100
y_test_set_pct = (len(Y_test)/len(df.index))*100

print("{0:0.2f}% in X training set ".format(x_train_set_pct))
print("{0:0.2f}% in X testing set ".format(x_test_set_pct))
print("{0:0.2f}% in Y training set ".format(y_train_set_pct))
print("{0:0.2f}% in Y testing set ".format(y_test_set_pct))

# prediction class data distribution in training and testing data
print("")
print("Original True Cases: {0} ({1:.2f}%)".format(num_true, true_pct))
print("Original False Cases: {0} ({1:.2f}%)".format(num_false, false_pct))
print("")
print("Training True: {0} {1:.2f}% ".format(len(Y_train[Y_train[:] == 1]), len(Y_train[Y_train[:] == 1])/len(Y_train)*100))
print("Training False: {0} {1:.2f}% ".format(len(Y_train[Y_train[:] == 0]), len(Y_train[Y_train[:] == 0])/len(Y_train)*100))
print("")
print("Testing True: {0} {1:.2f}% ".format(len(Y_test[Y_test[:] == 1]), len(Y_test[Y_test[:] == 1])/len(Y_test)*100))
print("Testing False: {0} {1:.2f}% ".format(len(Y_test[Y_test[:] == 0]), len(Y_test[Y_test[:] == 0])/len(Y_test)*100))
#############################################################################
#           Data imputation
#
# Missing data post split
#############################################################################
# 'num_preg','glucose_conc','diastolic_bp','thickness','insulin','bmi','diab_pred','age'
print("# rows in dataframe: {0}".format(len(df)))
print("# rows missing glucose_conc: {0}".format(len(df.loc[df['glucose_conc']==0])))
print("# rows missing num_preg: {0}".format(len(df.loc[df['num_preg']==0])))
print("# rows missing diastolic_bp: {0}".format(len(df.loc[df['diastolic_bp']==0])))
print("# rows missing thickness: {0}".format(len(df.loc[df['thickness']==0])))
print("# rows missing insulin: {0}".format(len(df.loc[df['insulin']==0])))
print("# rows missing bmi: {0}".format(len(df.loc[df['bmi']==0])))
print("# rows missing diab_pred: {0}".format(len(df.loc[df['diab_pred']==0])))
print("# rows missing age: {0}".format(len(df.loc[df['age']==0])))

# Imput the missing data

from sklearn.preprocessing import Imputer

fill_0 = Imputer(missing_values=0, strategy="mean", axis=0)
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)
#############################################################################
#           Create and Train naive_bayes model
# Training Init Algorith - Naive Bayes
#############################################################################
from sklearn.naive_bayes import GaussianNB

# create a model
nb_model = GaussianNB()
nb_model.fit(X_train, Y_train.ravel())

# Performance on training data
nb_predict_train = nb_model.predict(X_train)

#############################################################################
#           Analyze the metrics post prediction
#
#############################################################################
# import metrics
from sklearn import metrics
print("Accuracy : {0:.2f}%".format(metrics.accuracy_score(Y_train, nb_predict_train)*100))

# Test data accuracy

nb_predict_test = nb_model.predict(X_test)
# import metrics
print("Accuracy : {0:.2f}%".format(metrics.accuracy_score(Y_test, nb_predict_test)*100))
print("")

# Metrics analysis

print("Confusion Metrics")
print("{0}".format(metrics.confusion_matrix(Y_test, nb_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(Y_test, nb_predict_test))

#############################################################################
#           Create and Train Random Forest model
# Try Random Forest classifier
#############################################################################
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train.ravel())

# predict training data
rf_predict_train = rf_model.predict(X_train)
print("Accuracy : {0:.2f}%".format(metrics.accuracy_score(Y_train, rf_predict_train)*100))
print("")

# Predict on Testing data
rf_predict_test = rf_model.predict(X_test)
print("Accuracy : {0:.2f}%".format(metrics.accuracy_score(Y_test, rf_predict_test)*100))
print("")

# Metrics reports
print("Confusion Metrics")
print("{0}".format(metrics.confusion_matrix(Y_test, rf_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(Y_test, rf_predict_test))
#############################################################################
#           Create and Train LogisticRegression model
# Address Overfitting problem by choosing another algorithm - LogisticRegression
#############################################################################

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(C=0.7, random_state=42)
lr_model.fit(X_train, Y_train.ravel())

lr_predict_train = lr_model.predict(X_train)
print("Accuracy : {0:.2f}%".format(metrics.accuracy_score(Y_train, lr_predict_train)*100))
print("")

# modified Logistic Regression algorithm with regularization to Address Overfitting problem

from sklearn.linear_model import LogisticRegression
lr_model_2 = LogisticRegression(C=0.3, random_state=42, class_weight="balanced") # C - Inverse of regularization strength
lr_model_2.fit(X_train, Y_train.ravel())

lr_predict_train = lr_model_2.predict(X_train)
print("Train data Accuracy : {0:.2f}%".format(metrics.accuracy_score(Y_train, lr_predict_train)*100))
print("")
lr_predict_test = lr_model_2.predict(X_test)
print("Test data Accuracy : {0:.2f}%".format(metrics.accuracy_score(Y_test, lr_predict_test)*100))
print("")
print("Confusion Metrics")
print("{0}".format(metrics.confusion_matrix(Y_test, lr_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(Y_test, lr_predict_test))
#############################################################################
#           Create and Train LogisticRegressionCV model
# Address Overfitting problem by choosing another algorithm - LogisticRegressionCV model
#############################################################################

from sklearn.linear_model import LogisticRegressionCV

lr_cv_model = LogisticRegressionCV(n_jobs=-1, random_state=42, class_weight="balanced",Cs=3,cv=10,refit=False)
lr_cv_model.fit(X_train,Y_train.ravel())

lr_cv_predict_train = lr_cv_model.predict(X_train)
print("Train data Accuracy : {0:.2f}%".format(metrics.accuracy_score(Y_train, lr_cv_predict_train)*100))
print("")
lr_cv_predict_test = lr_cv_model.predict(X_test)
print("Test data Accuracy : {0:.2f}%".format(metrics.accuracy_score(Y_test, lr_cv_predict_test)*100))
print("")
print("Confusion Metrics")
print("{0}".format(metrics.confusion_matrix(Y_test, lr_cv_predict_test)))
print("")

print("Classification Report")
print(metrics.classification_report(Y_test, lr_cv_predict_test))