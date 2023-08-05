import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

from k_means_algorithm import K_Means
from k_means import AverageStrategy, MedianStrategy
from data_processor import read_data

style.use('ggplot')

def main():
    df = pd.read_csv("data/abaloneconverted.csv")
    df = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']]
    dataset = df.astype(float).values.tolist()

    X = df.values  # mengembalikkan nilai numpy array
    print(df)

    # TAMPILAN
    print("================ K-Means Clustering ================")
    print("              DESIGN PATTERN STRATEGY")
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
