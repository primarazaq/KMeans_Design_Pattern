import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd

style.use('ggplot')

from kmeans_algorithm import K_Means

class KMeansTemplate(K_Means):
    def __init__(self, k=3, tolerance=0.0001, max_iterations=10):
        super().__init__(k, tolerance, max_iterations)

    def preprocess_data(self):
        df = pd.read_csv("data/abaloneconverted.csv")
        df = df[['Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight',
                 'Rings']]
        dataset = df.astype(float).values.tolist()
        return dataset

    def display_info(self):
        print("================ K-Means Clustering ================")
        print("          DESIGN PATTERN TEMPLATE METHOD")
        print("Data: Abalone")
        print("")
        print("Anggota Kelompok:")
        print("-10119108 Prasetyo Hade MW")
        print("-10119118 Rizky Septiana")
        print("-10119123 Angga Cahya Abadi")
        print("-10119124 Primarazaq Noorshalih Putra Hilmana")
        print("----------------------------------------------------")

    def get_user_input(self):
        kluster = int(input("Masukkan Jumlah Kluster yang diinginkan: "))
        return kluster

    def main_algorithm(self, X, kluster):
        km = K_Means(kluster)
        km.fit(X)
        km.plot_clusters()

    def run(self):
        self.display_info()
        dataset = self.preprocess_data()
        X = np.array(dataset)
        kluster = self.get_user_input()
        self.main_algorithm(X, kluster)
