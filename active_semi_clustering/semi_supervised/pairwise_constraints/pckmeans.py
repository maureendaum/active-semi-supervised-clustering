import numpy as np
import sklearn.metrics

from active_semi_clustering.exceptions import EmptyClustersException
from .constraints import preprocess_constraints
from .cluster_helpers import find_centroids


class PCKMeans:
    def __init__(self, n_clusters=3, max_iter=100, w=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.w = w

    def fit(self, X, y=None, ml=[], cl=[]):
        # Preprocess constraints
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])

        # Initialize centroids
        cluster_centers = self._initialize_cluster_centers(X, neighborhoods, cl_graph)

        # Repeat until convergence
        self.objective_function_values = []
        self.v_measure_values = []
        for iteration in range(self.max_iter):
            # Assign clusters
            labels = self._assign_clusters(X, cluster_centers, ml_graph, cl_graph, self.w)

            # Estimate means
            prev_cluster_centers = cluster_centers
            cluster_centers = self._get_cluster_centers(X, labels)

            # Compute sum over objective function.
            index = list(range(X.shape[0]))
            error = 0
            for x_i in index:
                error += self._objective_function(X, x_i, cluster_centers, labels[x_i], labels, ml_graph, cl_graph, self.w)
            self.objective_function_values.append(error)
            not_nan = np.where(y != '')
            self.v_measure_values.append(sklearn.metrics.v_measure_score(y[not_nan], labels[not_nan]))

            # Check for convergence
            difference = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(difference, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)

            if converged: break

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self

    def _element_with_cannotlink_to_all(self, X, neighborhoods, cl_graph):
        for i in range(X.shape[0]):
            all_neighbors_have_cannotlink = True
            for neighborhood in neighborhoods:
                has_cannotlink = False
                for j in neighborhood:
                    if j in cl_graph[i]:
                        has_cannotlink = True
                        break
                if not has_cannotlink:
                    all_neighbors_have_cannotlink = False
                    break
            if all_neighbors_have_cannotlink:
                return i
        return None

    def _initialize_cluster_centers(self, X, neighborhoods, cl_graph):
        neighborhood_centers = np.array([X[neighborhood].mean(axis=0) for neighborhood in neighborhoods])
        neighborhood_sizes = np.array([len(neighborhood) for neighborhood in neighborhoods])

        if len(neighborhoods) > self.n_clusters:
            # Select K largest neighborhoods' centroids
            cluster_centers = neighborhood_centers[np.argsort(neighborhood_sizes)[-self.n_clusters:]]
        else:
            if len(neighborhoods) > 0:
                cluster_centers = neighborhood_centers
            else:
                cluster_centers = np.empty((0, X.shape[1]))

            # Look for a point that is connected by cannot-links to every neighborhood set.
            while len(neighborhoods) < self.n_clusters:
                next_el = self._element_with_cannotlink_to_all(X, neighborhoods, cl_graph)
                if next_el is not None:
                    neighborhoods.append([next_el])
                    cluster_centers = np.concatenate([cluster_centers, X[next_el].reshape(1, -1)])
                else:
                    break

            num_remaining_clusters = self.n_clusters - len(neighborhoods)
            if num_remaining_clusters > 0:
                remaining_cluster_centers = find_centroids(num_remaining_clusters, X)
                cluster_centers = np.concatenate([cluster_centers, remaining_cluster_centers])

        return cluster_centers

    def _objective_function(self, X, x_i, centroids, c_i, labels, ml_graph, cl_graph, w):
        distance = 1 / 2 * np.sum((X[x_i] - centroids[c_i]) ** 2)

        ml_penalty = 0
        for y_i in ml_graph[x_i]:
            if labels[y_i] != -1 and labels[y_i] != c_i:
                ml_penalty += w

        cl_penalty = 0
        for y_i in cl_graph[x_i]:
            if labels[y_i] == c_i:
                cl_penalty += w

        return distance + ml_penalty + cl_penalty

    def _assign_clusters(self, X, cluster_centers, ml_graph, cl_graph, w):
        labels = np.full(X.shape[0], fill_value=-1)

        index = list(range(X.shape[0]))
        np.random.shuffle(index)
        for x_i in index:
            labels[x_i] = np.argmin([self._objective_function(X, x_i, cluster_centers, c_i, labels, ml_graph, cl_graph, w) for c_i in range(self.n_clusters)])

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            # print("Empty clusters")
            raise EmptyClustersException

        return labels

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
