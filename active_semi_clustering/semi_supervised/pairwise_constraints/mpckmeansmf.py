import numpy as np
import scipy
import sklearn.metrics
import time

from active_semi_clustering.exceptions import EmptyClustersException
from active_semi_clustering.farthest_first_traversal import weighted_farthest_first_traversal
from .constraints import preprocess_constraints
from .cluster_helpers import find_centroids


# np.seterr('raise')

class MPCKMeansMF:
    """
    MPCK-Means that learns multiple (M) full (F) matrices
    """

    def __init__(self, n_clusters=3, max_iter=100, rng=None, w_m=1, w_c=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.rng = rng if rng else np.random.RandomState()
        self.w_m = w_m
        self.w_c = w_c
        self.w_log = 1
        self.log_det_As = []

    def _graph_to_list(self, graph):
        res = []
        for i, items in graph.items():
            res.extend([(i, j) for j in items if i < j])
        return res

    def fit(self, X, y=None, ml=[], cl=[], random_init=False, stop_early=True):
        # Preprocess constraints
        preprocess_start = time.perf_counter()
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])
        # Get all CL constraints from transitive closure.
        cl = self._graph_to_list(cl_graph)
        print(f'Preprocess constraints took {time.perf_counter() - preprocess_start:0.2f}s')

        # Initialize cluster centers
        init_start = time.perf_counter()
        if random_init:
            cluster_centers = X[self.rng.choice(X.shape[0], self.n_clusters, replace=False), :]
        else:
            cluster_centers = self._initialize_cluster_centers(X, neighborhoods)
        print(f'Initialize clusters took {time.perf_counter() - init_start:0.2f}s')

        # Initialize metrics
        As = [np.identity(X.shape[1]) for i in range(self.n_clusters)]
        self._update_determinants(As)

        # Repeat until convergence
        self.initial_centers = cluster_centers
        self.objective_function_values = []
        self.homogeneity_values = []
        self.completeness_values = []
        self.mutual_info_values = []
        self.iteration_times = []
        self.As = []
        self.labels = []
        self.term_d = []
        self.term_m = []
        self.term_c = []
        self.farthest = []
        self.farthest_points = []
        for iteration in range(self.max_iter):
            start = time.perf_counter()
            prev_cluster_centers = cluster_centers.copy()

            # Find farthest pair of points according to each metric
            farthest = self._find_farthest_pairs_of_points(X, As, cl)

            # Assign clusters
            labels = self._assign_clusters(X, y, cluster_centers, As, farthest, ml_graph, cl_graph, self.labels[-1] if len(self.labels) else np.full(X.shape[0], fill_value=-1))

            # Estimate means
            cluster_centers = self._get_cluster_centers(X, labels)

            # Update metrics
            As = self._update_metrics(X, labels, cluster_centers, farthest, ml_graph, cl_graph)
            self._update_determinants(As)
            # print(As)

            # Compute stats
            error, term_d, term_m, term_c = 0, 0, 0, 0
            for x_i in range(X.shape[0]):
                i_error, i_d, i_m, i_c = self._objective_function(X, x_i, labels, cluster_centers, labels[x_i], As, farthest, ml_graph, cl_graph)
                error += i_error
                term_d += i_d
                term_m += i_m
                term_c += i_c
            self.objective_function_values.append(error)
            self.term_d.append(term_d)
            self.term_m.append(term_m)
            self.term_c.append(term_c)
            if farthest[0]:
                self.farthest.append([f[2] for f in farthest])
                self.farthest_points.append([(f[0], f[1]) for f in farthest])
            self.homogeneity_values.append(sklearn.metrics.homogeneity_score(y, labels))
            self.completeness_values.append(sklearn.metrics.completeness_score(y, labels))
            self.mutual_info_values.append(sklearn.metrics.adjusted_mutual_info_score(y, labels))
            self.As.append(As)
            self.labels.append(labels)

            # Check for convergence
            cluster_centers_shift = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(cluster_centers_shift, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)

            self.iteration_times.append(time.perf_counter() - start)

            if converged and stop_early:
                break

        print('\t', iteration, converged)

        self.cluster_centers_, self.labels_ = cluster_centers, labels
        self.As_ = As

        return self

    def _update_determinants(self, As):
        self.log_det_As = [np.log(max(np.linalg.det(A), 1e-9)) for A in As]

    def _find_farthest_pairs_of_points(self, X, As, cl):
        farthest = [None] * self.n_clusters
        max_distance = [0] * self.n_clusters

        for i, j in cl:
            for cluster_id in range(self.n_clusters):
                distance = self._dist(X[i], X[j], As[cluster_id])
                if distance > max_distance[cluster_id]:
                    max_distance[cluster_id] = distance
                    farthest[cluster_id] = (i, j, distance)

        return farthest

    def _initialize_cluster_centers(self, X, neighborhoods):
        neighborhood_centers = np.array([X[neighborhood].mean(axis=0) for neighborhood in neighborhoods])
        neighborhood_sizes = np.array([len(neighborhood) for neighborhood in neighborhoods])
        neighborhood_weights = neighborhood_sizes / neighborhood_sizes.sum()

        # print('\t', len(neighborhoods), neighborhood_sizes)

        if len(neighborhoods) > self.n_clusters:
            cluster_centers = neighborhood_centers[weighted_farthest_first_traversal(neighborhood_centers, neighborhood_weights, self.n_clusters, rng=self.rng)]
        else:
            if len(neighborhoods) > 0:
                cluster_centers = neighborhood_centers
            else:
                cluster_centers = np.empty((0, X.shape[1]))

            num_remaining = self.n_clusters - len(neighborhoods)
            if num_remaining:
                remaining_cluster_centers = find_centroids(num_remaining, X, self.rng)
                cluster_centers = np.concatenate([cluster_centers, remaining_cluster_centers])

        return cluster_centers

    def _dist(self, x, y, A):
        "(x - y)^T A (x - y)"
        return scipy.spatial.distance.mahalanobis(x, y, A) ** 2

    def _objective_function(self, X, i, labels, cluster_centers, cluster_id, As, farthest, ml_graph, cl_graph):
        # Include max term inside of log because the determinant is sometimes negative.
        term_d = self._dist(X[i], cluster_centers[cluster_id], As[cluster_id]) - self.w_log * self.log_det_As[cluster_id]

        def f_m(i, c_i, j, c_j, As):
            return 1 / 2 * self._dist(X[i], X[j], As[c_i]) + 1 / 2 * self._dist(X[i], X[j], As[c_j])

        def f_c(i, c_i, j, c_j, As, farthest):
            assert c_i == c_j
            return farthest[c_i][2] - self._dist(X[i], X[j], As[c_i])

        term_m = 0
        for j in ml_graph[i]:
            if labels[j] >= 0 and labels[j] != cluster_id:
                term_m += self.w_m * f_m(i, cluster_id, j, labels[j], As)

        term_c = 0
        for j in cl_graph[i]:
            if labels[j] == cluster_id:
                term_c += self.w_c * f_c(i, cluster_id, j, labels[j], As, farthest)

        return term_d + term_m + term_c, term_d, term_m, term_c

    def _assign_clusters(self, X, y, cluster_centers, As, farthest, ml_graph, cl_graph, labels):
        index = list(range(X.shape[0]))
        np.random.shuffle(index)
        for i in index:
            labels[i] = np.argmin(
                [self._objective_function(X, i, labels, cluster_centers, cluster_id, As, farthest, ml_graph, cl_graph)[0] for cluster_id, cluster_center in enumerate(cluster_centers)])

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            # print("Empty clusters")
            raise EmptyClustersException

        return labels

    def _update_metrics(self, X, labels, cluster_centers, farthest, ml_graph, cl_graph):
        As = []

        for cluster_id in range(self.n_clusters):
            X_i = X[labels == cluster_id]
            n = X_i.shape[0]

            if n == 1:
                As.append(np.identity(X_i.shape[1]))
                continue

            A_inv = (X_i - cluster_centers[cluster_id]).T @ (X_i - cluster_centers[cluster_id])

            for i in range(X.shape[0]):
                for j in ml_graph[i]:
                    if labels[i] == cluster_id or labels[j] == cluster_id:
                        if labels[i] != labels[j]:
                            # * 1/4 rather than 1/2 because we'll count both ij and ji.
                            A_inv += 1 / 4 * self.w_m * ((X[i][:, None] - X[j][:, None]) @ (X[i][:, None] - X[j][:, None]).T)

            for i in range(X.shape[0]):
                for j in cl_graph[i]:
                    if labels[i] == cluster_id or labels[j] == cluster_id:
                        if labels[i] == labels[j]:
                            # * 1/2 because we'll count both ij and ji
                            A_inv += 1/2 * self.w_c * (
                                    ((X[farthest[cluster_id][0]][:, None] - X[farthest[cluster_id][1]][:, None]) @ (X[farthest[cluster_id][0]][:, None] - X[farthest[cluster_id][1]][:, None]).T) - (
                                    (X[i][:, None] - X[j][:, None]) @ (X[i][:, None] - X[j][:, None]).T))

            # Handle the case when the matrix is not invertible
            if not self._is_invertible(A_inv):
                # print("Not invertible")
                A_inv += 1e-9 * np.trace(A_inv) * np.identity(A_inv.shape[0])

            A = n * np.linalg.inv(A_inv)

            # Is A positive semidefinite?
            if not np.all(np.linalg.eigvals(A) >= 0):
                # print("Negative definite")
                eigenvalues, eigenvectors = np.linalg.eigh(A)
                A = eigenvectors @ np.diag(np.maximum(0, eigenvalues)) @ np.linalg.inv(eigenvectors)

            As.append(A)

        return As

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def _is_invertible(self, A):
        return A.shape[0] == A.shape[1] and np.linalg.matrix_rank(A) == A.shape[0]
