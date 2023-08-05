# k_means.py

import numpy as np

class K_Means:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=10):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def fit(self, data):
        self.centroids = {}

        # Menentukan pusat cluster. k yang pertama akan dijadikan centroid acak pertama
        for i in range(self.k):
            self.centroids[i] = data[i]

        # Mulai Iterasi
        for i in range(self.max_iterations):
            self.classes = {}
            for i in range(self.k):
                self.classes[i] = []

            # Mencari jarak terdekat antara tiap data dengan centroid. Sekaligus mencari data tersebut lebih dekat ke centroid mana
            for features in data:
                distances = [np.linalg.norm(features - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classes[classification].append(features)

            previous = dict(self.centroids)

            # Mencari nilai rata-rata (Means) untuk centroid berikutnya
            for classification in self.classes:
                self.centroids[classification] = np.average(self.classes[classification], axis=0)

            isOptimal = True

            # Membandingkan centroid dengan centroid yang sebelumnya. Apakah sudah optimal/ nilai centroid nya tidak berubah ?
            for centroid in self.centroids:
                original_centroid = previous[centroid]
                curr = self.centroids[centroid]

                if np.sum((curr - original_centroid) / original_centroid * 100.0) > self.tolerance:
                    isOptimal = False

            # agar lebih optimal
            if isOptimal:
                break

    def pred(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification