import numpy as np
import sklearn.cluster
import sklearn.metrics
import time

from active_semi_clustering.exceptions import EmptyClustersException, ClusteringNotFoundException
from .constraints import preprocess_constraints


class COPKMeans:
    def __init__(self, n_clusters=3, max_iter=100, rng=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.rng = rng if rng else np.random.RandomState()

    # TODO: Deduplicate from MPCKMeansMF.
    def _graph_to_list(self, graph):
        res = []
        for i, items in graph.items():
            res.extend([(i, j) for j in items if i < j])
        return res

    def fit(self, X, y=None, ml=[], cl=[]):
        preprocess_start = time.perf_counter()
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])
        cl = self._graph_to_list(cl_graph)
        print(f'Preprocess constraints took {time.perf_counter() - preprocess_start:0.2f}s')

        # Initialize cluster centers
        init_start = time.perf_counter()
        cluster_centers = self._init_cluster_centers(X)
        print(f'Initialize centers took {time.perf_counter() - init_start:0.2f}s')

        # Repeat until convergence
        self.initial_centers = cluster_centers
        self.homogeneity_values = []
        self.completeness_values = []
        self.iteration_times = []
        self.labels = []
        for iteration in range(self.max_iter):
            start = time.perf_counter()
            prev_cluster_centers = cluster_centers.copy()

            # Assign clusters
            labels = self._assign_clusters(X, cluster_centers, self._dist, ml_graph, cl_graph)

            # Estimate means
            cluster_centers = self._get_cluster_centers(X, labels)

            # Update stats.
            self.homogeneity_values.append(sklearn.metrics.homogeneity_score(y, labels))
            self.completeness_values.append(sklearn.metrics.completeness_score(y, labels))
            self.labels.append(labels)

            # Check for convergence
            cluster_centers_shift = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(cluster_centers_shift, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)

            self.iteration_times.append(time.perf_counter() - start)

            if converged: break

        print('\t', iteration, converged)

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self

    def _init_cluster_centers(self, X, method='kmpp'):
        if method == 'random':
            return X[self.rng.choice(X.shape[0], self.n_clusters, replace=False), :]

        elif method == 'kmpp':
            centers, _ = sklearn.cluster.kmeans_plusplus(X, self.n_clusters, random_state=self.rng)
            return centers

    def _dist(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def _assign_clusters(self, *args):
        max_retries_cnt = 1000

        for _ in range(max_retries_cnt):
            try:
                return self._try_assign_clusters(*args)

            except ClusteringNotFoundException:
                continue

        raise ClusteringNotFoundException

    def _try_assign_clusters(self, X, cluster_centers, dist, ml_graph, cl_graph):
        labels = np.full(X.shape[0], fill_value=-1)

        data_indices = list(range(X.shape[0]))
        self.rng.shuffle(data_indices)

        for i in data_indices:
            if labels[i] != -1:
                continue
            distances = np.array([dist(X[i], c) for c in cluster_centers])
            # sorted_cluster_indices = np.argsort([dist(x, c) for c in cluster_centers])

            for cluster_index in distances.argsort():
                if not self._violates_constraints(i, cluster_index, labels, ml_graph, cl_graph):
                    labels[i] = cluster_index

                    # Avoid failure case by adding must-link neighbors as in
                    # https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py#L26
                    for j in ml_graph[i]:
                        assert labels[j] == -1
                        labels[j] = cluster_index

                    break

            if labels[i] < 0:
                raise ClusteringNotFoundException

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            raise EmptyClustersException

        return labels

    def _violates_constraints(self, i, cluster_index, labels, ml_graph, cl_graph):
        for j in ml_graph[i]:
            if labels[j] > 0 and cluster_index != labels[j]:
                return True

        for j in cl_graph[i]:
            if cluster_index == labels[j]:
                return True

        return False

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
