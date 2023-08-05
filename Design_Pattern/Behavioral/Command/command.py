import pandas as pd
import matplotlib.pyplot as plt
from kmeans import K_Means

class KMeansCommand:
    def __init__(self, k=3):
        self.k = k

    def execute(self):
        df = pd.read_csv("data/abaloneconverted.csv")
        df = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']]
        dataset = df.astype(float).values.tolist()

        X = df.values
        print(df)

        print("================ K-Means Clustering ================")
        print("               DESIGN PATTERN COMMAND")
        print("Data : Abalone")
        print("")
        print("Anggota Kelompok : ")
        print("-10119108 Prasetyo Hade MW")
        print("-10119118 Rizky Septiana")
        print("-10119123 Angga Cahya Abadi")
        print("-10119124 Primarazaq Noorshalih Putra Hilmana")
        print("----------------------------------------------------")
        kluster = self.k
        km = K_Means(kluster)
        km.fit(X)

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
