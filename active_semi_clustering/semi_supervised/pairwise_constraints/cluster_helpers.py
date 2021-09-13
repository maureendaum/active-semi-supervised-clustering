import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Algorithm Refine from https://www.aaai.org/Papers/KDD/1998/KDD98-032.pdf
def find_centroids(n, X, rng=None):
    j = 10
    s = max(int(0.1 * X.shape[0]), n)
    assert s <= X.shape[0], 'Number of samples > number of data points (%r > %r)' % (s, X.shape[0])
    rng = rng if rng else np.random.default_rng()

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
        km = KMeans(n_clusters=n, init=em_centers[i], n_init=1).fit(em_data)
        kmeans.append(km)
    best_fit = np.argmax([km.score(em_data) for km in kmeans])
    return kmeans[best_fit].cluster_centers_

def test_gm():
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    gm = GaussianMixture(n_components=2, random_state=0).fit(X)

    # Sanity check that find_centroids is doing something reasonable.
    rng = np.random.default_rng()
    X1 = rng.normal((1, 1), 0.3, (50, 2))
    X2 = rng.normal((2, 2), 0.1, (50, 2))
    X3 = rng.normal((1, 2), 0.2, (50, 2))
    X4 = rng.normal((0.5, 0.5), 0.15, (50, 2))
    Xs = [X1, X2, X3, X4]

    find_centroids(2, np.concatenate(Xs))
    find_centroids(2, np.concatenate([X1, X2]))

    colors = ['navy', 'turquoise', 'darkorange', 'purple', 'green', 'magenta']
    plt.figure()
    for i, X in enumerate(Xs):
        plt.scatter(X[:, 0], X[:, 1], color=colors[i])
    centers = find_centroids(4, np.concatenate(Xs))
    plt.scatter(centers[:, 0], centers[:, 1], marker='x', color='black')
    plt.savefig('figures/x1.png')

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    iris = datasets.load_iris()
    for n, color in enumerate(colors):
        data = iris.data[iris.target == n]
        plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)
    centers = find_centroids(len(colors), iris.data)
    plt.scatter(centers[:, 0], centers[:, 1], marker='o', color='black')
    plt.savefig('figures/iris.png')


if __name__ == '__main__':
    test_gm()
