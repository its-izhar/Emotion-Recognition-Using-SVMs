#!/usr/bin/env python

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

======================================================
Emotion recognition using SVMs (Scikit-learn & OpenCV
======================================================

Author: Izhar Shaikh
License: MIT
Dependencies: Python 2.7, Scikit-Learn, OpenCV 3.0.0,
              Numpy, Scipy, Matplotlib, Tkinter
Instructions: Please checkout Readme.txt & Instructions.txt

The dataset used in this example is Olivetti Faces:
 http://cs.nyu.edu/~roweis/data/olivettifaces.mat

"""

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import json
import subprocess
from sklearn import datasets
import FileDialog                       # Needed for Pyinstaller

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

print(__doc__)



faces = datasets.fetch_olivetti_faces()

# ==========================================================================
# Traverses through the dataset by incrementing index & records the result
# ==========================================================================
class Trainer:
    def __init__(self):
        self.results = {}
        self.imgs = faces.images
        self.index = 0

    def reset(self):
        print "============================================"
        print "Resetting Dataset & Previous Results.. Done!"
        print "============================================"
        self.results = {}
        self.imgs = faces.images
        self.index = 0

    def increment_face(self):
        if self.index + 1 >= len(self.imgs):
            return self.index
        else:
            while str(self.index) in self.results:
                # print self.index
                self.index += 1
            return self.index

    def record_result(self, smile=True):
        print "Image", self.index + 1, ":", "Happy" if smile is True else "Sad"
        self.results[str(self.index)] = smile



# ===================================
# Callback function for the buttons
# ===================================
## smileCallback()              : Gets called when "Happy" Button is pressed
## noSmileCallback()            : Gets called when "Sad" Button is pressed
## updateImageCount()           : Displays the number of images processed
## displayFace()                : Gets called internally by either of the button presses
## displayBarGraph(isBarGraph)  : computes the bar graph after classification is completed 100%
## _begin()                     : Resets the Dataset & Starts from the beginning
## _quit()                      : Quits the Application
## printAndSaveResult()         : Save and print the classification result
## loadResult()                 : Loading the previously stored classification result
## run_once(m)                  : Decorator to allow functions to run only once

def run_once(m):
    def wrapper(*args, **kwargs):
        if not wrapper.has_run:
            wrapper.has_run = True
            return m(*args, **kwargs)
    wrapper.has_run = False
    return wrapper

def smileCallback():
    trainer.record_result(smile=True)
    trainer.increment_face()
    displayFace(trainer.imgs[trainer.index])
    updateImageCount(happyCount=True, sadCount= False)


def noSmileCallback():
    trainer.record_result(smile=False)
    trainer.increment_face()
    displayFace(trainer.imgs[trainer.index])
    updateImageCount(happyCount=False, sadCount=True)


def updateImageCount(happyCount, sadCount):
    global HCount, SCount, imageCountString, countString   # Updating only when called by smileCallback/noSmileCallback
    if happyCount is True and HCount < 400:
        HCount += 1
    if sadCount is True and SCount < 400:
        SCount += 1
    if HCount == 400 or SCount == 400:
        HCount = 0
        SCount = 0
    # --- Updating Labels
    # -- Main Count
    imageCountPercentage = str(float((trainer.index + 1) * 0.25)) \
        if trainer.index+1 < len(faces.images) else "Classification DONE! 100"
    imageCountString = "Image Index: " + str(trainer.index+1) + "/400   " + "[" + imageCountPercentage + " %]"
    labelVar.set(imageCountString)           # Updating the Label (ImageCount)
    # -- Individual Counts
    countString = "(Happy: " + str(HCount) + "   " + "Sad: " + str(SCount) + ")\n"
    countVar.set(countString)


@run_once
def displayBarGraph(isBarGraph):
    ax[1].axis(isBarGraph)
    n_groups = 1                    # Data to plot
    Happy, Sad = (sum([trainer.results[x] == True for x in trainer.results]),
               sum([trainer.results[x] == False for x in trainer.results]))
    index = np.arange(n_groups)     # Create Plot
    bar_width = 0.5
    opacity = 0.75
    ax[1].bar(index, Happy, bar_width, alpha=opacity, color='b', label='Happy')
    ax[1].bar(index + bar_width, Sad, bar_width, alpha=opacity, color='g', label='Sad')
    ax[1].set_ylim(0, max(Happy, Sad)+10)
    ax[1].set_xlabel('Expression')
    ax[1].set_ylabel('Number of Images')
    ax[1].set_title('Training Data Classification')
    ax[1].legend()


@run_once
def printAndSaveResult():
    print trainer.results                       # Prints the results
    with open("../results/results.xml", 'w') as output:
        json.dump(trainer.results, output)        # Saving The Result

@run_once
def loadResult():
    results = json.load(open("../results/results.xml"))
    trainer.results = results


def displayFace(face):
    ax[0].imshow(face, cmap='gray')
    isBarGraph = 'on' if trainer.index+1 == len(faces.images) else 'off'      # Switching Bar Graph ON
    if isBarGraph is 'on':
        displayBarGraph(isBarGraph)
        printAndSaveResult()
    # f.tight_layout()
    canvas.draw()


def _opencv():
    print "\n\n Please Wait. . . ."
    opencvProcess = subprocess.Popen("Train Classifier and Test Video Feed.py", close_fds=True, shell=True)
    # os.system('"Train Classifier.exe"')
    # opencvProcess.communicate()


def _begin():
    trainer.reset()
    global HCount, SCount
    HCount = 0
    SCount = 0
    updateImageCount(happyCount=False, sadCount=False)
    displayFace(trainer.imgs[trainer.index])


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


if __name__ == "__main__":
    # Embedding things in a tkinter plot & Starting tkinter plot
    matplotlib.use('TkAgg')
    root = Tk.Tk()
    root.wm_title("Emotion Recognition Using Scikit-Learn & OpenCV")

    # =======================================
    # Class Instances & Starting the Plot
    # =======================================
    trainer = Trainer()

    # Creating the figure to be embedded into the tkinter plot
    f, ax = plt.subplots(1, 2)
    ax[0].imshow(faces.images[0], cmap='gray')
    ax[1].axis('off')  # Initially keeping the Bar graph OFF

    # ax tk.DrawingArea
    # Embedding the Matplotlib figure 'f' into Tkinter canvas
    canvas = FigureCanvasTkAgg(f, master=root)
    canvas.show()
    canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

    print "Keys in the Dataset: ", faces.keys()
    print "Total Images in Olivetti Dataset:", len(faces.images)

    # Declaring Button & Label Instances
    # =======================================
    smileButton = Tk.Button(master=root, text='Smiling', command=smileCallback)
    smileButton.pack(side=Tk.LEFT)

    noSmileButton = Tk.Button(master=root, text='Not Smiling', command=noSmileCallback)
    noSmileButton.pack(side=Tk.RIGHT)

    labelVar = Tk.StringVar()
    label = Tk.Label(master=root, textvariable=labelVar)
    imageCountString = "Image Index: 0/400   [0 %]"     # Initial print
    labelVar.set(imageCountString)
    label.pack(side=Tk.TOP)

    countVar = Tk.StringVar()
    HCount = 0
    SCount = 0
    countLabel = Tk.Label(master=root, textvariable=countVar)
    countString = "(Happy: 0   Sad: 0)\n"     # Initial print
    countVar.set(countString)
    countLabel.pack(side=Tk.TOP)

    opencvButton = Tk.Button(master=root, text='Load the "Trained Classifier" & Test Output', command=_opencv)
    opencvButton.pack(side=Tk.TOP)

    resetButton = Tk.Button(master=root, text='Reset', command=_begin)
    resetButton.pack(side=Tk.TOP)

    quitButton = Tk.Button(master=root, text='Quit Application', command=_quit)
    quitButton.pack(side=Tk.TOP)

    authorVar = Tk.StringVar()
    authorLabel = Tk.Label(master=root, textvariable=authorVar)
    authorString = "\n\n Developed By: " \
                   "\n Izhar Shaikh " \
                   "\n (izhar.shaikh@ufl.edu) " \
                   "\n [EEL6825 Pattern Recognition - Spring 2016]"     # Initial print
    authorVar.set(authorString)
    authorLabel.pack(side=Tk.BOTTOM)

    root.iconbitmap(r'..\icon\happy-sad.ico')
    Tk.mainloop()                               # Starts mainloop required by Tk
