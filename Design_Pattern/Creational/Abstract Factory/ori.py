import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd 

style.use('ggplot')

# Abstract Factory untuk Algoritma K-Means
class KMeansFactory:
    def create_algorithm(self, k, tolerance, max_iterations):
        return KMeans(k, tolerance, max_iterations)

# Abstract Factory untuk Plotting
class PlottingFactory:
    def create_plotter(self):
        return KMeansPlotter()

class KMeansPlotter:
    def plot(self, km):
        colors = 10 * ["r", "c", "k", "g", "b"]

        arr = []
        for centroid in km.centroids:
            plt.scatter(km.centroids[centroid][0], km.centroids[centroid][1], s=130, marker="x")

        for classification in km.classes:
            color = colors[classification]
            for features in km.classes[classification]:
                plt.scatter(features[0], features[1], color=color, s=30)
                arr.append([classification])

        print("Total Data : ", len(arr), "data")
        for i in range(km.k):
            print("Kluster ", i, ": ", arr.count([i]), "data")
        plt.show()

class KMeans:
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

def main():
    df = pd.read_csv("data/abaloneconverted.csv")
    df = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']]
    dataset = df.astype(float).values.tolist()

    X = df.values
    print(df)

    print("================ K-Means Clustering ================")
    print("           DESIGN PATTERN ABSTRACT FACTORY          ") 
    print("Data : Abalone")
    print("")
    print("Anggota Kelompok : ")
    print("-10119108 Prasetyo Hade MW")
    print("-10119118 Rizky Septiana")
    print("-10119123 Angga Cahya Abadi")
    print("-10119124 Primarazaq Noorshalih Putra Hilmana")
    print("----------------------------------------------------")

    kluster = int(input("Masukkan Jumlah Kluster yang diinginkan : "))

    # Menggunakan Abstract Factory untuk membuat objek KMeans
    kmeans_factory = KMeansFactory()
    kmeans = kmeans_factory.create_algorithm(kluster, tolerance=0.0001, max_iterations=10)
    kmeans.fit(X)

    # Menggunakan Abstract Factory untuk membuat objek plotter dan melakukan plotting
    plotting_factory = PlottingFactory()
    plotter = plotting_factory.create_plotter()
    plotter.plot(kmeans)

if __name__ == "__main__":
    main()