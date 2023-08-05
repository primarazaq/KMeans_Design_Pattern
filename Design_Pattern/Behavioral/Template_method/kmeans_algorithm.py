import numpy as np
import matplotlib.pyplot as plt

class K_Means:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=10):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, data):
        self.centroids = {}
        for i in range(self.k):
            self.centroids[i] = data[i]
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
                self.centroids[classification] = np.average(self.classes[classification], axis=0)
            isOptimal = True
            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]
                if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    isOptimal = False
            if isOptimal:
                break

    def pred(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification

    def plot_clusters(self):
        colors = 10 * ["r", "c", "k", "g", "b"]
        arr = []
        for centroid in self.centroids:
            plt.scatter(self.centroids[centroid][0], self.centroids[centroid][1], s=130, marker="x")
        for classification in self.classes:
            color = colors[classification]
            for features in self.classes[classification]:
                plt.scatter(features[0], features[1], color=color, s=30)
                arr.append([classification])
        print("Total Data:", len(arr), "data")
        for i in range(self.k):
            print("Kluster", i, ":", arr.count([i]), "data")
        plt.show()
