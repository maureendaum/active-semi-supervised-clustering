from active_semi_clustering.exceptions import InconsistentConstraintsException
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Algorithm Refine from https://www.aaai.org/Papers/KDD/1998/KDD98-032.pdf
def find_centroids(n, X):
    j = 10
    s = max(int(0.1 * X.shape[0]), n)
    assert s <= X.shape[0], 'Number of samples > number of data points (%r > %r)' % (s, X.shape[0])
    rng = np.random.default_rng()

    em_centers = []
    for _ in range(j):
        sample = rng.choice(X, s, replace=False)
        while True:
            gm = GaussianMixture(n_components=n).fit(sample)
            num_nonempty = len(np.unique(gm.predict(sample)))
            if num_nonempty == n:
                break
        em_centers.append(gm.means_)
    em_data = np.concatenate(em_centers)
    kmeans = []
    for i in range(j):
        km = KMeans(n_clusters=n, init=em_centers[i]).fit(em_data)
        kmeans.append(km)
    best_fit = np.argmax([km.score(em_data) for km in kmeans])
    return kmeans[best_fit].cluster_centers_

# Taken from https://github.com/Behrouz-Babaki/COP-Kmeans/blob/master/copkmeans/cop_kmeans.py
def preprocess_constraints(ml, cl, n):
    "Create a graph of constraints for both must- and cannot-links"

    # Represent the graphs using adjacency-lists
    ml_graph, cl_graph = {}, {}
    for i in range(n):
        ml_graph[i] = set()
        cl_graph[i] = set()

    def add_both(d, i, j):
        d[i].add(j)
        d[j].add(i)

    for (i, j) in ml:
        ml_graph[i].add(j)
        ml_graph[j].add(i)

    for (i, j) in cl:
        cl_graph[i].add(j)
        cl_graph[j].add(i)

    def dfs(i, graph, visited, component):
        visited[i] = True
        for j in graph[i]:
            if not visited[j]:
                dfs(j, graph, visited, component)
        component.append(i)

    # Run DFS from each node to get all the graph's components
    # and add an edge for each pair of nodes in the component (create a complete graph)
    # See http://www.techiedelight.com/transitive-closure-graph/ for more details
    visited = [False] * n
    neighborhoods = []
    for i in range(n):
        if not visited[i] and ml_graph[i]:
            component = []
            dfs(i, ml_graph, visited, component)
            for x1 in component:
                for x2 in component:
                    if x1 != x2:
                        ml_graph[x1].add(x2)
            neighborhoods.append(component)

    for (i, j) in cl:
        for x in ml_graph[i]:
            add_both(cl_graph, x, j)

        for y in ml_graph[j]:
            add_both(cl_graph, i, y)

        for x in ml_graph[i]:
            for y in ml_graph[j]:
                add_both(cl_graph, x, y)

    for i in ml_graph:
        for j in ml_graph[i]:
            if j != i and j in cl_graph[i]:
                raise InconsistentConstraintsException('Inconsistent constraints between {} and {}'.format(i, j))

    return ml_graph, cl_graph, neighborhoods
