import pandas as pd
import numpy as py
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv("glass.csv")
Y = df["Type"]
Y.head()
df = df.drop("Type",axis=1)
df.head()

# Preprocessing
x = df.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

# Data Split
x_train, x_test, y_train, y_test = train_test_split(df,Y, test_size=0.25)
print(x_train)
print(x_test)


# Neural Network
mlpc = MLPClassifier()
mlpc.fit(x_train,y_train)
y_predict_mlpc = mlpc.predict(x_test)
accuracyMLPC = accuracy_score(y_test,y_predict_mlpc)
print(confusion_matrix(y_test,y_predict_mlpc))
print(classification_report(y_test,y_predict_mlpc))
print(accuracyMLPC)



# Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_predict_gnb = gnb.predict(x_test)
accuracyGB = accuracy_score(y_test,y_predict_gnb)
print(confusion_matrix(y_test,y_predict_gnb))
print(classification_report(y_test,y_predict_gnb))
print(accuracyGB)


# Decision Tree
clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y_predict_tree = clf.predict(x_test)
accuracyDT = accuracy_score(y_test,y_predict_tree)
print(confusion_matrix(y_test,y_predict_tree))
print(classification_report(y_test,y_predict_tree))
print(accuracyDT)

