import numpy as np
import scipy
import sklearn.metrics
import time

from active_semi_clustering.exceptions import EmptyClustersException
from active_semi_clustering.farthest_first_traversal import weighted_farthest_first_traversal
from .constraints import preprocess_constraints
from .cluster_helpers import find_centroids

# np.seterr('raise')

class MPCKMeans:
    "MPCK-Means-S-D that learns only a single (S) diagonal (D) matrix"

    def __init__(self, n_clusters=3, max_iter=10, rng=None, w_m=1, w_c=1):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.rng = rng if rng else np.random.default_rng()
        self.w_m = w_m
        self.w_c = w_c
        self.w_log = 1

    # TODO: Deduplicate from MPCKMeansMF.
    def _graph_to_list(self, graph):
        res = []
        for i, items in graph.items():
            res.extend([(i, j) for j in items if i < j])
        return res

    def fit(self, X, y=None, ml=set(), cl=set(), random_init=False, stop_early=True):
        # Preprocess constraints
        preprocess_start = time.perf_counter()
        ml_graph, cl_graph, neighborhoods = preprocess_constraints(ml, cl, X.shape[0])
        cl = self._graph_to_list(cl_graph)
        print(f'Preprocess constraints took {time.perf_counter() - preprocess_start:0.2f}s')

        # Initialize cluster centers
        init_start = time.perf_counter()
        if random_init:
            # print('Random initialization of cluster centers')
            cluster_centers = X[self.rng.choice(X.shape[0], self.n_clusters, replace=False), :]
        else:
            cluster_centers = self._initialize_cluster_centers(X, neighborhoods)
        print(f'Initialize centers took {time.perf_counter() - init_start:0.2f}s')

        assert cluster_centers.shape[0] == self.n_clusters

        # Initialize metrics
        A = np.identity(X.shape[1])

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
            farthest = self._find_farthest_pairs_of_points(X, A, cl)

            # Assign clusters
            labels = self._assign_clusters(X, y, cluster_centers, A, farthest, ml_graph, cl_graph, self.labels[-1] if len(self.labels) else np.full(X.shape[0], fill_value=-1))

            # Estimate means
            cluster_centers = self._get_cluster_centers(X, labels)

            # Update metrics
            A = self._update_metrics(X, labels, cluster_centers, farthest, ml_graph, cl_graph)

            # Compute objective function value
            error, term_d, term_m, term_c = 0, 0, 0, 0
            for x_i in range(X.shape[0]):
                i_error, i_d, i_m, i_c = self._objective_fn(X, x_i, labels, cluster_centers, labels[x_i], A, farthest, ml_graph, cl_graph)
                error += i_error
                term_d += i_d
                term_m += i_m
                term_c += i_c
            self.objective_function_values.append(error)
            self.term_d.append(term_d)
            self.term_m.append(term_m)
            self.term_c.append(term_c)
            if farthest:
                self.farthest.append(farthest[2])
                self.farthest_points.append((farthest[0], farthest[1]))
            self.homogeneity_values.append(sklearn.metrics.homogeneity_score(y, labels))
            self.completeness_values.append(sklearn.metrics.completeness_score(y, labels))
            self.mutual_info_values.append(sklearn.metrics.adjusted_mutual_info_score(y, labels))
            self.As.append(A)
            self.labels.append(labels)

            # Check for convergence
            cluster_centers_shift = (prev_cluster_centers - cluster_centers)
            converged = np.allclose(cluster_centers_shift, np.zeros(cluster_centers.shape), atol=1e-6, rtol=0)

            if converged and stop_early:
                break

            iteration_time = time.perf_counter() - start
            self.iteration_times.append(iteration_time)

        print('\t', iteration, converged)

        self.cluster_centers_, self.labels_ = cluster_centers, labels

        return self

    def _find_farthest_pairs_of_points(self, X, A, cl):
        # This needs to be restricted to just points with a cannot-link constraint.
        farthest = None
        max_distance = 0

        for i, j in cl:
            distance = self._dist(X[i], X[j], A)
            if distance > max_distance:
                max_distance = distance
                farthest = (i, j, distance)

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

            num_remaining_clusters = self.n_clusters - len(neighborhoods)
            if num_remaining_clusters:
                remaining_cluster_centers = find_centroids(num_remaining_clusters, X, rng=self.rng)
                cluster_centers = np.concatenate([cluster_centers, remaining_cluster_centers])

        return cluster_centers

    def _dist(self, x, y, A):
        "(x - y)^T A (x - y)"
        return scipy.spatial.distance.mahalanobis(x, y, A) ** 2

    def _objective_fn(self, X, i, labels, cluster_centers, cluster_id, A, farthest, ml_graph, cl_graph):
        term_d = self._dist(X[i], cluster_centers[cluster_id], A) - self.w_log * np.log(np.linalg.det(A))  # FIXME is it okay that it might be negative?

        def f_m(i, j, A):
            return self._dist(X[i], X[j], A)

        def f_c(i, j, A, farthest):
            return farthest[2] - self._dist(X[i], X[j], A)

        term_m = 0
        for j in ml_graph[i]:
            if labels[j] >= 0 and labels[j] != cluster_id:
                term_m += self.w_m * f_m(i, j, A)

        term_c = 0
        for j in cl_graph[i]:
            if labels[j] == cluster_id:
                # assert f_c(i, j, A, farthest) >= 0
                term_c += self.w_c * f_c(i, j, A, farthest)

        return term_d + term_m + term_c, term_d, term_m, term_c

    def _assign_clusters(self, X, y, cluster_centers, A, farthest, ml_graph, cl_graph, labels, should_print=False):
        previous_labels = labels.copy()
        index = list(range(X.shape[0]))
        self.rng.shuffle(index)
        for i in index:
            distances = [self._objective_fn(X, i, labels, cluster_centers, cluster_id, A, farthest, ml_graph, cl_graph)[0] for cluster_id, cluster_center in enumerate(cluster_centers)]
            labels[i] = np.argmin(distances)
            if should_print and labels[i] != previous_labels[i]:
                print(f'Moving {i} from cluster with value {distances[labels[i]]} to cluster with value {distances[labels[i]]}')

        # Handle empty clusters
        # See https://github.com/scikit-learn/scikit-learn/blob/0.19.1/sklearn/cluster/_k_means.pyx#L309
        n_samples_in_cluster = np.bincount(labels, minlength=self.n_clusters)
        empty_clusters = np.where(n_samples_in_cluster == 0)[0]

        if len(empty_clusters) > 0:
            # print("Empty clusters")
            raise EmptyClustersException

        return labels

    def _update_metrics(self, X, labels, cluster_centers, farthest, ml_graph, cl_graph):
        N, D = X.shape
        A = np.zeros((D, D))

        for d in range(D):
            term_x = np.sum([(x[d] - cluster_centers[labels[i], d]) ** 2 for i, x in enumerate(X)])

            term_m = 0
            for i in range(N):
                for j in ml_graph[i]:
                    if labels[i] != labels[j]:
                        # 1/4 rather than 1/2 because we'll count this both for i and j.
                        term_m += 1/4 * self.w_m * (X[i, d] - X[j, d]) ** 2

            term_c = 0
            for i in range(N):
                for j in cl_graph[i]:
                    if labels[i] == labels[j]:
                        tmp = ((X[farthest[0], d] - X[farthest[1], d]) ** 2 - (X[i, d] - X[j, d]) ** 2)
                        # 1/2 rather than 1 because we'll count this both for i and j.
                        term_c += 1/2 * self.w_c * max(tmp, 0)

            A[d, d] = N * 1 / max(term_x + term_m + term_c, 1e-9)

        return A

    def _get_cluster_centers(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
