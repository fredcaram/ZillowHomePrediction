from sklearn.decomposition import pca, nmf


class ZillowDataDecomposition2:
    def __init__(self, x, y, n_features):
        self.x = x
        self.y = y
        self.n_features = n_features
        self.pca_model = self.__get_fitted_pca_model__()

    def __get_fitted_pca_model__(self):
        pca_model = pca.PCA(self.n_features)
        pca_model.fit(self.x, self.y)
        return pca_model


class ZillowDataDecomposition(ZillowDataDecomposition2):
    def get_pca_transformed_data(self, x):
        return self.pca_model.transform(x)