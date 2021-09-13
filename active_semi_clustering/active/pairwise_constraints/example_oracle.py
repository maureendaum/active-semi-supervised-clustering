import numpy as np

class MaximumQueriesExceeded(Exception):
    pass


class ExampleOracle:
    def __init__(self, labels, max_queries_cnt=20, rng=None):
        self.labels = labels
        self.queries_cnt = 0
        self.max_queries_cnt = max_queries_cnt
        self.ml = []
        self.cl = []
        self.rng = rng if rng else np.random.default_rng()

    def query(self, i, j):
        "Query the oracle to find out whether i and j should be must-linked"
        if self.queries_cnt < self.max_queries_cnt:
            self.queries_cnt += 1
            ml = self.labels[i] == self.labels[j]
            if ml:
                self.ml.append((i, j))
            else:
                self.cl.append((i, j))
            return ml
        else:
            raise MaximumQueriesExceeded

    def query_all(self):
        "Query the oracle max_queries_cnt times on random pairs."
        n = len(self.labels)
        for _ in range(self.queries_cnt, self.max_queries_cnt):
            i, j = self.rng.choice(n, 2, replace=False)
            self.query(i, j)
