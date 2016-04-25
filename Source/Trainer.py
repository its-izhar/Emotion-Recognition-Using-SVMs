from sklearn import datasets


# Fetching the dataset
faces = datasets.fetch_olivetti_faces()
print faces.keys()

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
        if self.index + 1 >= len(self.imgs)/8:
            return self.index
        else:
            while str(self.index) in self.results:
                # print self.index
                self.index += 1
            return self.index

    def record_result(self, smile=True):
        print "Image", self.index + 1, ":", "Happy" if smile is True else "Sad"
        self.results[str(self.index)] = smile
