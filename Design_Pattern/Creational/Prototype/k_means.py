import numpy as np
from centroid_prototype import CentroidPrototype

class K_Means:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=10):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, data):
        self.centroid_prototype = CentroidPrototype(data)
        self.centroids = {}

        for i in range(self.k):
            if i < len(data):
                self.centroids[i] = self.centroid_prototype.clone().data
            else:
                self.centroids[i] = data[np.random.randint(0, len(data))]

        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []

            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)

            previous = dict(self.centroids)

            for classification in self.classes:
                self.centroids[classification] = np.nan_to_num(np.average(self.classes[classification], axis=0))

            isOptimal = True

            for centroid in self.centroids:
                original_centroid = previous[centroid]
