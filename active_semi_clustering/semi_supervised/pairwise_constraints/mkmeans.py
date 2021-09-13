import numpy as np

from sklearn.cluster import KMeans
from metric_learn import MMC
from .constraints import preprocess_constraints


class MKMeans:
    def __init__(self, n_clusters=3, max_iter=1000, diagonal=True, rng=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.diagonal = diagonal
        self.rng = rng if rng else np.random.default_rng()

    def fit(self, X, y=None, ml=[], cl=[]):
        X_transformed = X

        assert ml and cl
        ml_graph, cl_graph, _ = preprocess_constraints(ml, cl, X.shape[0])

        pairs = []
        labels = []
        for i, constraints in ml_graph.items():
            for j in constraints:
                pairs.append((i, j))
                labels.append(1)

        for i, constraints in cl_graph.items():
            for j in constraints:
                pairs.append((i, j))
                labels.append(-1)

        mmc = MMC(diagonal=self.diagonal, preprocessor=X, max_iter=self.max_iter, random_state=self.rng)
        mmc.fit(pairs, labels)
        X_transformed = mmc.transform(X)

        kmeans = KMeans(n_clusters=self.n_clusters, init='k-means++', max_iter=self.max_iter, random_state=self.rng)
        kmeans.fit(X_transformed)

        self.labels_ = kmeans.labels_

        return self
