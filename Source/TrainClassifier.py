import numpy as np
import json
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
from sklearn import metrics
from Trainer import *

svc_1 = SVC(kernel='linear')                    # Initializing Classifier

trainer = Trainer()
results = json.load(open('results.xml'))        # Loading the classification result
trainer.results = results

#for i in trainer.results:
 #   print i, trainer.results[i]
#print trainer.results

indices = [int(i) for i in trainer.results]          # Building the dataset now
data = faces.data[indices, :]                        # Image Data

target = [trainer.results[i] for i in trainer.results]          # Target Vector
target = np.array(target).astype(np.int32)


# Train the classifier using 5 fold cross validation
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=0)


# Trained classifier's performance evaluation
def evaluate_cross_validation(clf, X, y, K):
    # create a k-fold cross validation iterator
    cv = KFold(len(y), K, shuffle=True, random_state=0)
    # by default the score used is the one returned by score method of the estimator (accuracy)
    scores = cross_val_score(clf, X, y, cv=cv)
    print "Scores: ", (scores)
    print ("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))

evaluate_cross_validation(svc_1, X_train, y_train, 5)


# Confusion Matrix and Results
def train_and_evaluate(clf, X_train, X_test, y_train, y_test):

    clf.fit(X_train, y_train)

    print ("Accuracy on training set:")
    print (clf.score(X_train, y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)

    print ("Classification Report:")
    print (metrics.classification_report(y_test, y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(y_test, y_pred))

train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)
