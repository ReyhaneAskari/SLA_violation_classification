# -*- coding: utf-8 -*-

# In this script we use a simple classifer called naive bayes and try to predict the violations. But before that we use
# some methods to tackle the problem of our skewed dataset. :) 

# 11 May 2016
# @author: reyhane_askari
# Universite de Montreal, DIRO

import csv
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import pandas as pd
from os import chdir, listdir
from pandas import read_csv
from os import path
from random import randint, sample, seed
from collections import OrderedDict
from pandas import DataFrame, Series
import numpy as np 
import csv
import codecs

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set()
import itertools
from sklearn.decomposition import PCA
from unbalanced_dataset import UnderSampler, NearMiss, CondensedNearestNeighbour, OneSidedSelection,\
NeighbourhoodCleaningRule, TomekLinks, ClusterCentroids, OverSampler, SMOTE,\
SMOTETomek, SMOTEENN, EasyEnsemble, BalanceCascade

from treeinterpreter import treeinterpreter as ti


almost_black = '#262626'

colnames = ['old_index','job_id', 'task_idx','sched_cls', 'priority', 'cpu_requested',
            'mem_requested', 'disk', 'violation'] 

tain_path = r'/home/askrey/Dropbox/Project_step_by_step/3_create_database/csvs/frull_db_2.csv'

X = pd.read_csv(tain_path, header = None, index_col = False ,names = colnames, skiprows = [0],  usecols = [3,4,5,6,7])
y = pd.read_csv(tain_path, header = None, index_col = False ,names = colnames, skiprows = [0],  usecols = [8])
y = y['violation'].values
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.333, random_state=0)
main_x = X.values
main_y = y

verbose = False

# 'Random under-sampling'
US = UnderSampler(verbose=verbose)
x, y = US.fit_transform(main_x, main_y)

ratio = float(np.count_nonzero(y==1)) / float(np.count_nonzero(y==0))
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.333, random_state=0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

clf = RandomForestClassifier(n_estimators=10)
scores = cross_val_score(clf, X_test, y_test)

y_pred = clf.fit(X_train, y_train).predict(X_test)
y_score = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]

prediction, bias, contributions = ti.predict(clf, X_test)

import ipdb; ipdb.set_trace()

for i in range(len(X_test)):
    print "Instance", i
    print "Feature contributions:"
    for c, feature in sorted(zip(contributions[i], 
                                 colnames[3:-1]), 
                             key=lambda x: -abs(x[0].max())):
        print feature, round(c[0], 2)
    print "-"*20 

mean_accuracy = clf.fit(X_train, y_train).score(X_test,y_test,sample_weight=None)
fpr, tpr, thresholds = metrics.roc_curve(y_test, y_score)

roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.savefig('/home/askrey/Dropbox/Project_step_by_step/5_simple_models/new_scripts/random_forest_UnderSampler.pdf')
