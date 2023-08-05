import pandas as pd
from kmeans_factory import KMeansFactory
from plotting_factory import PlottingFactory

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
