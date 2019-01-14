import base64
import random
import pandas as pd
from Datum import Datum
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer

data = pd.read_table("data.txt").values

random.shuffle(data)

norm_data = []

for d in data:
    norm_data.append(d[0])

data = norm_data


half = int(len(data)/2)

normal = data[0:half]
base = data[-half:]

normal_class = []
base_class = []
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


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

cv = CountVectorizer()

cv.fit_transform(X_train)
cv.fit_transform(X_test)

X_train = cv.transform(X_train).toarray()
X_test = cv.transform(X_test).toarray()

logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'
     .format(logreg.score(X_train, y_train)))
print('Accuracy of Logistic regression classifier on test set: {:.2f} \n'
     .format(logreg.score(X_test, y_test)))

clf = DecisionTreeClassifier().fit(X_train, y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(X_train, y_train)))
print('Accuracy of Decision Tree classifier on test set: {:.2f} \n'
     .format(clf.score(X_test, y_test)))

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(X_train, y_train)))
print('Accuracy of K-NN classifier on test set: {:.2f} \n'
     .format(knn.score(X_test, y_test)))

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(X_train, y_train)))
print('Accuracy of LDA classifier on test set: {:.2f} \n'
     .format(lda.score(X_test, y_test)))

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(X_train, y_train)))
print('Accuracy of GNB classifier on test set: {:.2f} \n'
     .format(gnb.score(X_test, y_test)))

svm = SVC()
svm.fit(X_train, y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f} \n'
     .format(svm.score(X_test, y_test)))

