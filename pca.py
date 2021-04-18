import numpy as np
from sklearn.decomposition import PCA


class OurPCA:


    def fit_transform(self, data, n_components):

        pca = PCA(n_components=n_components, random_state=42)
        return pca.fit_transform(data)
