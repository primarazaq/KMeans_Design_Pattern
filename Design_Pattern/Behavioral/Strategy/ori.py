import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

style.use('ggplot')


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


class K_Means:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=10, strategy=AverageStrategy()):
        self.k = k
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.strategy = strategy

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
            new_centroids = self.strategy.calculate_centroids(list(self.classes.values()))
            for idx, centroid in enumerate(self.centroids):
                self.centroids[centroid] = new_centroids[idx]

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


def main():
    df = pd.read_csv("data/abaloneconverted.csv")
    df = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']]
    dataset = df.astype(float).values.tolist()

    X = df.values  # mengembalikkan nilai numpy array
    print(df)

    # TAMPILAN
    print("================ K-Means Clustering ================")
    print("                  DESIGN PATTERN X")
    print("Data : Abalone")
    print("")
    print("Anggota Kelompok : ")
    print("-10119108 Prasetyo Hade MW")
    print("-10119118 Rizky Septiana")
    print("-10119123 Angga Cahya Abadi")
    print("-10119124 Primarazaq Noorshalih Putra Hilmana")
    print("----------------------------------------------------")
    strategy_choice = int(input("Masukkan pilihan Strategi (1 = Average, 2 = Median): "))
    strategy = AverageStrategy() if strategy_choice == 1 else MedianStrategy()
    kluster = int(input("Masukkan Jumlah Kluster yang diinginkan : "))
    km = K_Means(kluster, strategy=strategy)
    km.fit(X)

    # Plotting untuk warna pada grafik
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
    for i in range(kluster):
        print("Kluster ", i, ": ", arr.count([i]), "data")
    plt.show()


if __name__ == "__main__":
    main()