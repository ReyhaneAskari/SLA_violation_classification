import csv
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score

# import pylab as pl

tain_path = r'/home/askrey/Dropbox/data.csv'
data = np.loadtxt(open(tain_path,"rb"),delimiter=",")
X = data[:,1:6]
y = data[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.333,
                                                    random_state=0)
clf = RandomForestClassifier(n_estimators=10)
scores = cross_val_score(clf, X_test, y_test)
print("The scores.mean is : %f" %scores.mean())

y_pred = clf.fit(X_train, y_train).predict(X_test)
y_score = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
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
plt.show()

print("Number of mislabeled points out of a total %d points : %d"
    % (X_test.shape[0],(y_test != y_pred).sum()))
print("The mean accuracy is : %f" %mean_accuracy)
