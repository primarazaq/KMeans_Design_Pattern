import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from k_means import K_Means
from k_means_component import K_MeansComponent

style.use('ggplot')

def main():
    df = pd.read_csv("data/abaloneconverted.csv")
    df = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']]
    dataset = df.astype(float).values.tolist()

    X = df.values
    print(df)

    print("================ K-Means Clustering ================")
    print("             DESIGN PATTERN COMPOSITE")
    print("Data : Abalone")
    print("")
    print("Anggota Kelompok : ")
    print("-10119108 Prasetyo Hade MW")
    print("-10119118 Rizky Septiana")
    print("-10119123 Angga Cahya Abadi")
    print("-10119124 Primarazaq Noorshalih Putra Hilmana")
    print("----------------------------------------------------")
    kluster = int(input("Masukkan Jumlah Kluster yang diinginkan : "))

    k_means_component = K_MeansComponent()
    for i in range(kluster):
        k_means = K_Means()
        k_means_component.add_component(k_means)

    k_means_component.fit(X)

    colors = 10 * ["r", "c", "k", "g", "b"]
    arr = []

    for component in k_means_component.components:
        for centroid in component.centroids:
            plt.scatter(component.centroids[centroid][0], component.centroids[centroid][1], s=130, marker="x")

        for classification in component.classes:
            color = colors[classification]
            for features in component.classes[classification]:
                plt.scatter(features[0], features[1], color=color, s=30)
                arr.append([classification])

    print("Total Data : ", len(arr), "data")
    for i in range(kluster):
        print("Kluster ", i, ": ", arr.count([i]), "data")
    plt.show()


if __name__ == "__main__":
    main()
