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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt
import numpy as np
import json

print(__doc__)

from Trainer import *

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk


# Embedding things in a tkinter plot & Starting tkinter plot
matplotlib.use('TkAgg')
root = Tk.Tk()
root.wm_title("Emotion Recognition Using Scikit-Learn & OpenCV")

print "Total Images in Olivetti Dataset:",  len(faces.images)


# =======================================
# Class Instances & Starting the Plot
# =======================================
trainer = Trainer()

# Creating the figure to be embedded into the tkinter plot
f, ax = plt.subplots(1, 2)
ax[0].imshow(faces.images[0], cmap='gray')
ax[1].axis('off')       # Initially keeping the Bar graph OFF

# ax tk.DrawingArea
# Embedding the Matplotlib figure 'f' into Tkinter canvas
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

toolbar = NavigationToolbar2TkAgg(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)


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
    updateImageCount()


def noSmileCallback():
    trainer.record_result(smile=False)
    trainer.increment_face()
    displayFace(trainer.imgs[trainer.index])
    updateImageCount()


def updateImageCount():
    global imageCountString             # Updating only when called by smileCallback/noSmileCallback
    imageCountPercentage = str(float((trainer.index + 1) * 0.25)) \
        if trainer.index+1 < len(faces.images) else "Classification DONE! 100"
    imageCountString = "Image Index: " + str(trainer.index+1) + "/400   " + "[" + imageCountPercentage + " %]"
    var.set(imageCountString)           # Updating the Label (ImageCount)


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
    with open('results.xml', 'w') as output:
        json.dump(trainer.results, output)        # Saving The Result

@run_once
def loadResult():
    results = json.load(open('results.xml'))
    trainer.results = results


def displayFace(face):
    ax[0].imshow(face, cmap='gray')
    isBarGraph = 'on' if trainer.index+1 == len(faces.images) else 'off'      # Switching Bar Graph ON
    if isBarGraph is 'on':
        displayBarGraph(isBarGraph)
        printAndSaveResult()
    # f.tight_layout()
    canvas.draw()


def _begin():
    trainer.reset()
    displayFace(trainer.imgs[trainer.index])


def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate


# =======================================
# Declaring Button Instances (2 Buttons)
# =======================================
smileButton = Tk.Button(master=root, text='Happy', command=smileCallback)
smileButton.pack(side=Tk.LEFT)

noSmileButton = Tk.Button(master=root, text='Sad', command=noSmileCallback)
noSmileButton.pack(side=Tk.RIGHT)

var = Tk.StringVar()
label = Tk.Label(master=root, textvariable=var)
imageCountString = "Image Index: 0/400   [0 %]"     # Initial print
var.set(imageCountString)
label.pack()

button = Tk.Button(master=root, text='Reset', command=_begin)
button.pack(side=Tk.TOP)

button = Tk.Button(master=root, text='Quit Application', command=_quit)
button.pack(side=Tk.BOTTOM)

Tk.mainloop()                               # Starts mainloop required by Tk
