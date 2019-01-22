import base64
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from Datum import Datum
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np

def getData(data):

    random.shuffle(data)
    norm_data = []
    for d in data:
        norm_data.append(d[0])

    data = norm_data
    half = int(len(data)/2)
    normal = data[0:half]
    base = data[-half:]
    dataset = []

    for n in normal:
        dataset.append(Datum(n, "normal"))

    for b in base:
        b = base64.b64encode(bytes(b, 'utf-8'))
        dataset.append(Datum(b, "base64"))

    random.shuffle(dataset)

    X = []
    Y = []

    for d in dataset:
        X.append(d.st)
        Y.append(d.type)

    return X, Y


def prepareData(X, y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)

    X_train = tvect.fit_transform(X_train).toarray()
    X_test = tvect.transform(X_test).toarray()

    return X_train, X_test, Y_train, Y_test


def trainNetwork(X_train, X_test, y_train, y_test):

    logreg.fit(X_train, y_train)
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
          .format(logreg.score(X_train, y_train)))
    print('Accuracy of Logistic regression classifier on test set: {:.2f} \n'
          .format(logreg.score(X_test, y_test)))

    gnb.fit(X_train, y_train)
    print('Accuracy of GNB classifier on training set: {:.2f}'
          .format(gnb.score(X_train, y_train)))
    print('Accuracy of GNB classifier on test set: {:.2f} \n'
          .format(gnb.score(X_test, y_test)))



def predict(str):
    str = [str]
    str = tvect.transform(str).toarray()
    return gnb.predict(str)[0]

def processText(text):
    text = text.split()
    for t in text:
        r = random.randint(0,1)
        if r == 0:
            t = str(base64.b64encode(bytes(t, 'utf-8')))
        print(t + " : " + predict(t))

data = pd.read_table("data.txt").values

logreg = LogisticRegression()
gnb = GaussianNB()

cv = CountVectorizer()
tvect = TfidfVectorizer(min_df=1, max_df=1)

X, y = getData(data)

X_train, X_test, Y_train, Y_test = prepareData(X, y)

trainNetwork(X_train, X_test, Y_train, Y_test)

text = str(pd.read_table("text2.txt").columns[0])

processText(text)

