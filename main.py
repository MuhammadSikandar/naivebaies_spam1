import numpy as np
import pandas as pd

import urllib
import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.cross_validation import train_test_split

from sklearn import metrics
from sklearn.metrics import accuracy_score


### from url download data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/spambase//spambase.data"

raw_data = urllib.urlopen(url)

dataset = np.loadtxt(raw_data, delimiter=',')
print dataset[2000]
print (len(dataset[0]))


### from the file download data
raw_data_2 = open("C:/Users/musi0010/Desktop/course/PycharmProjects/naivebaise_spam1/spambase.data","r")
dataset_2 = np.loadtxt(raw_data_2,delimiter=',')
print dataset_2[2000]
print (len(dataset_2))

X = dataset_2[:,0:48]
y = dataset_2[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state= 17)
print(len(y_test))


## bernoulli NB which is good if we convert data into binary

BernNB = BernoulliNB(binarize= True)
BernNB.fit(X_train,y_train)
print (BernNB)

y_expect = y_test
y_pred = BernNB.predict(X_test)
print y_pred
print accuracy_score(y_pred,y_expect)

### multinomial naive bayes classifier, i believe here we are counting freq
# of word occur so multinomial should be the best

MultiNB = MultinomialNB()
MultiNB.fit(X_train,y_train)
print (MultiNB)

y_pred = MultiNB.predict(X_test)
print y_pred
print y_expect
print accuracy_score(y_pred,y_expect)

### gaussian NB, if the 48 predictor are normally distributed (in column)
# then gaussian can also give good results

GausNB = GaussianNB()
GausNB.fit(X_train,y_train)
print (GausNB)

y_pred = GausNB.predict(X_test)
print y_pred
print y_expect
print accuracy_score(y_pred,y_expect) # gaussian does not give good results



## bernoulli NB which is good if we convert data into binary, now playing with
# parameters give good results

BernNB = BernoulliNB(binarize= 0.1)
BernNB.fit(X_train,y_train)
print (BernNB)

y_expect = y_test
y_pred = BernNB.predict(X_test)
print y_pred
print accuracy_score(y_pred,y_expect)