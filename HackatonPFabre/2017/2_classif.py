import mxnet as mx
import pandas as pd
import logging
import time
import numpy as np
import os
src = 'D:\\Data\\pFabre\\data\\'

feats = pd.read_csv(src+'outfileTr.csv')


train_df = pd.read_csv(data_root+"train_biz_fc7features.csv")
test_df  = pd.read_csv(data_root+"test_biz_fc7features.csv")

y_whole_train = train_df['label'].values
X_whole_train = train_df['feature vector'].values
#X_whole_test = test_df['feature vector'].values

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
#print("X_test: ", X_test.shape)




# todo classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.cross_validation import train_test_split

'''
from sklearn.datasets import make_classification
X_whole_train, y_whole_train = make_classification(n_samples=1000, n_features=4,
                           n_informative=2, n_redundant=0,
                           random_state=0, shuffle=False)
'''

X_train, X_test, y_train, y_test = train_test_split(
    X_whole_train,y_whole_train , test_size=0.3, random_state=42)


clfRFC = RandomForestClassifier(n_jobs=2, random_state=0)
clfLR = LogisticRegression()
score = {}

for BinModel, name in zip([clfRFC, clfLR],['RandomForestClassifier', 'LogisticRegression']):
    BinModel.fit(X_train, y_train)
    score[name] = cross_val_score(BinModel, X, y, scoring='f1')





'''
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=2, oob_score=False, random_state=0,
            verbose=0, warm_start=False)
'''