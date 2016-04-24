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
from sklearn import datasets
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
import matplotlib.pyplot as plt

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk


# Embedding things in a tkinter plot & Starting tkinter plot
matplotlib.use('TkAgg')
root = Tk.Tk()
root.wm_title("Faces")

faces = datasets.fetch_olivetti_faces()
print faces.keys()

print "Total Images in Olivetti Dataset:",  len(faces.images)

# ==========================================================================
# Traverses through the dataset by incrementing index & records the result
# ==========================================================================

class Trainer:
    def __init__(self):
        self.results = {}
        self.imgs = faces.images
        self.index = 0

    def increment_face(self):
        if self.index + 1 >= len(self.imgs):
            return self.index
        else:
            while str(self.index) in self.results:
                print self.index
                self.index += 1
            return self.index

    def record_result(self, smile=True):
        self.results[str(self.index)] = smile


# =======================================
# Class Instances & Starting the Plot
# =======================================
trainer = Trainer()

# Creating the figure to be embedded into the tkinter plot
f = plt.figure()
ax = f.add_subplot(111)

# ax tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

toolbar = NavigationToolbar2TkAgg(canvas, root)
toolbar.update()
canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)


# ===================================
# Callback function for the buttons
# ===================================
## smileCallback()  : Gets called when "Happy" Button is pressed
## noSmileCallback(): Gets called when "Sad" Button is pressed
## displayFace()    : Gets called internally by either of the button presses

def smileCallback():
    trainer.record_result(smile=True)
    trainer.increment_face()
    displayFace(trainer.imgs[trainer.index])

def noSmileCallback():
    trainer.record_result(smile=False)
    trainer.increment_face()
    displayFace(trainer.imgs[trainer.index])

def displayFace(face):
    ax.imshow(face, cmap='gray')
    f.canvas.draw()

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

button = Tk.Button(master=root, text='Quit', command=_quit)
button.pack(side=Tk.BOTTOM)

Tk.mainloop()
print trainer.results