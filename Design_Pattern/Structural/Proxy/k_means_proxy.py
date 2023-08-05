# k_means_proxy.py

from k_means import K_Means

class K_Means_Proxy:
    def __init__(self, k=3, tolerance=0.0001, max_iterations=10):
        self.k_means = K_Means(k, tolerance, max_iterations)

    def fit(self, data):
        print("Proxy: Menjalankan metode fit pada K_Means")
        self.k_means.fit(data)

    def pred(self, data):
        print("Proxy: Menjalankan metode pred pada K_Means")
        return self.k_means.pred(data)
