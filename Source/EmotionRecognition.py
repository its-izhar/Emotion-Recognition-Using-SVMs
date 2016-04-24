"""
======================================================
Emotion recognition using SVMs (Scikit-learn & OpenCV
======================================================

Author: Izhar Shaikh
License: MIT
Dependencies: Python 2.7, Scikit-Learn, OpenCV 3.0.0,
              Numpy, Scipy, Matplotlib
Instructions: Please checkout Readme.txt & Instructions.txt

The dataset used in this example is Olivetti Faces:
 http://cs.nyu.edu/~roweis/data/olivettifaces.mat

"""

import matplotlib.pyplot as plt
from sklearn import datasets
import Tkinter

faces = datasets.fetch_olivetti_faces()
print faces.keys()

for i in range(10):
    face = faces.images[i]
    plt.subplot(1, 10, i + 1)
    plt.imshow(face.reshape((64, 64)), cmap='gray')
    plt.axis('off')
plt.show()

# ==========================================================================
# Traverses through the dataset by increamenting index & records the result
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
# Class Instances
# =======================================

trainer = Trainer()
top = Tkinter.Tk()


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
    plt.clf()
    plt.imshow(face, cmap='gray')
    plt.axis('off')
    plt.show()



# =======================================
# Declaring Button Instances (2 Buttons)
# =======================================

smileButton = Tkinter.Button(top, text="Happy", command=smileCallback)
noSmileButton = Tkinter.Button(top, text="Sad", command=noSmileCallback)

smileButton.pack()
noSmileButton.pack()
top.mainloop()