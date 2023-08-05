import numpy as np

class KMeansStrategy:
    def calculate_centroids(self, classes):
        raise NotImplementedError


class AverageStrategy(KMeansStrategy):
    def calculate_centroids(self, classes):
        new_centroids = np.array([np.average(cls, axis=0) for cls in classes])
        return new_centroids


class MedianStrategy(KMeansStrategy):
    def calculate_centroids(self, classes):
        new_centroids = np.array([np.median(cls, axis=0) for cls in classes])
        return new_centroids

