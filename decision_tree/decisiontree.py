import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix




class Nodo:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.left = None
        self.right = None
        self.feature_index = None
        self.threshold = None
        self.label = None

    def is_terminal(self):
        return len(self.Y) == 0 or len(set(self.Y)) == 1
    
    @staticmethod
    def gini_from_counts(counts, n):
        if n == 0: 
            return 0.0
        p = counts / n
        return 1.0 - np.sum(p * p)

    @staticmethod
    def entropy_from_counts(counts, n, eps=1e-12):
        if n == 0:
            return 0.0
        p = counts / n
        p = np.clip(p, eps, 1.0)
        return -np.sum(p * np.log2(p))

    def best_split(self, min_samples_leaf=1, impurity_fn=gini_from_counts):
        n, d = self.X.shape
        if n <= 1:
            return None

        classes, y_idx = np.unique(self.Y, return_inverse=True)
        K = len(classes)

        best_imp = np.inf
        best = None

        total_right = np.bincount(y_idx, minlength=K).astype(np.int64)

        for j in range(d):
            order = np.argsort(self.X[:, j], kind="mergesort")
            xj = self.X[order, j]
            yj = y_idx[order]

            left_counts  = np.zeros(K, dtype=np.int64)
            right_counts = total_right.copy()

            for i in range(n - 1):
                c = yj[i]
                left_counts[c]  += 1
                right_counts[c] -= 1

                if xj[i] == xj[i + 1]:
                    continue

                nL = i + 1
                nR = n - nL
                if nL < min_samples_leaf or nR < min_samples_leaf:
                    continue

                gL = impurity_fn(left_counts,  nL)
                gR = impurity_fn(right_counts, nR)
                imp = (nL * gL + nR * gR) / n  # impureza ponderada

                if imp < best_imp:
                    thr = (xj[i] + xj[i + 1]) / 2.0
                    best_imp = imp
                    best = (j, thr)

        return best



class DT:
    def __init__(self, criterion="gini"):
        self.root = None
        if criterion == "gini":
            self.impurity_fn = Nodo.gini_from_counts
        elif criterion == "entropy":
            self.impurity_fn = Nodo.entropy_from_counts

        else:
            raise ValueError("criterion debe ser 'gini' o 'entropy'")

    def fit(self, X, Y):
        self.root = self._grow(X, Y, depth=0)

    def _grow(self, X, Y, depth):
        node = Nodo(X, Y)

        if (node.is_terminal()):
            node.label = Counter(Y).most_common(1)[0][0] if len(Y) else None
            return node

        # pasa la función de impureza elegida
        split = node.best_split(impurity_fn=self.impurity_fn)
        if split is None:
            node.label = Counter(Y).most_common(1)[0][0]
            return node

        j, t = split
        node.feature_index, node.threshold = j, t

        print(f"{'  '*depth}Nodo (depth={depth}) -> Feature {j}, Threshold {t:.4f}, n={len(Y)}")


        left_mask  = X[:, j] <= t
        right_mask = ~left_mask

        node.left  = self._grow(X[left_mask],  Y[left_mask],  depth + 1)
        node.right = self._grow(X[right_mask], Y[right_mask], depth + 1)
        return node


    def _predict_one(self, node, x):
        if node.label is not None or node.is_terminal():
            return node.label
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x)

    def predict(self, X):
        return np.array([self._predict_one(self.root, x) for x in X])