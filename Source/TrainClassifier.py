"""
The MIT License (MIT)

Copyright (c) 2016 Izhar Shaikh

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import json
import Tkinter as Tk
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
from sklearn import metrics
from TrainDataset import *

# =====================================================================
# TEST CLASSIFIER FLAG (Disabled by default)
# Enable this flag to test the trained classifier on the training data
# =====================================================================
TEST_CLASSIFIER = True
# =====================================================================

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

# ----------------------------------------------------------------------------------
# Testing the classifier on the training data
# ----------------------------------------------------------------------------------
if TEST_CLASSIFIER is True:

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
    import random

    matplotlib.use('TkAgg')
    top = Tk.Tk()
    top.wm_title("Classifier Testing on the Training Data")

    # Creating the figure to be embedded into the tkinter plot
    f, ax = plt.subplots(1, 1)
    ax.axis('off')       # Initially keeping the Bar graph OFF

    # ax tk.DrawingArea
    # Embedding the Matplotlib figure 'f' into Tkinter canvas
    canvas = FigureCanvasTkAgg(f, master=top)
    canvas.show()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    def checkCallBack():

        index = random.randint(0, 400)
        face = faces.images[index]
        ax.imshow(face, cmap='gray')
        temp = faces.data[index, :]
        temp = np.array(temp).reshape(1, -1)
        status = "SMILING!" if svc_1.predict(temp) == 1 else "NOT SMILING!"
        global checkStatusString
        checkStatusString = "This person is " + status
        var.set(checkStatusString)
        print checkStatusString
        canvas.draw()

    def _quit():
        top.quit()     # stops mainloop
        top.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate


    var = Tk.StringVar()
    checkLabel = Tk.Label(master=top, textvariable=var)
    checkStatusString = ""
    var.set(checkStatusString)
    checkLabel.pack()

    checkButton = Tk.Button(top, text="Check", command=checkCallBack)
    checkButton.pack()

    quitButton = Tk.Button(master=top, text='Quit Application', command=_quit)
    quitButton.pack(side=Tk.BOTTOM)

    top.mainloop()