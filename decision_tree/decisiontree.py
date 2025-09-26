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

    def gini_counts(self, counts, n):
        # counts: vector con conteos por clase (long K)
        if n == 0:
            return 0.0
        p = counts / n
        return 1.0 - np.sum(p * p)

    def best_split(self, min_samples_leaf=1):
        n, d = self.X.shape
        if n <= 1:
            return None

        # Mapear etiquetas a 0..K-1 una sola vez por nodo
        classes, y_idx = np.unique(self.Y, return_inverse=True)
        K = len(classes)

        best_gini = np.inf
        best = None

        # Conteos totales a la derecha (antes de partir)
        total_right = np.bincount(y_idx, minlength=K).astype(np.int64)

        for j in range(d):
            # Ordenar por la columna j
            order = np.argsort(self.X[:, j], kind="mergesort")
            xj = self.X[order, j]
            yj = y_idx[order]

            left_counts = np.zeros(K, dtype=np.int64)
            right_counts = total_right.copy()

            # Recorremos posibles cortes entre i e i+1
            for i in range(n - 1):
                c = yj[i]
                left_counts[c]  += 1
                right_counts[c] -= 1

                # Saltar si los valores son iguales (no hay umbral entre ellos)
                if xj[i] == xj[i + 1]:
                    continue

                nL = i + 1
                nR = n - nL
                # hojas mínimas
                if nL < min_samples_leaf or nR < min_samples_leaf:
                    continue

                gL = self.gini_counts(left_counts, nL)
                gR = self.gini_counts(right_counts, nR)
                g  = (nL * gL + nR * gR) / n  # PONDERADO

                if g < best_gini:
                    thr = (xj[i] + xj[i + 1]) / 2.0  # punto medio
                    best_gini = g
                    best = (j, thr)

        return best


class DT:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.root = None
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def fit(self, X, Y):
        self.root = self._grow(X, Y, depth=0)

    def _grow(self, X, Y, depth):
        node = Nodo(X, Y)

        # parada
        if (node.is_terminal() or
            (self.max_depth is not None and depth >= self.max_depth) or
            len(Y) < self.min_samples_split):
            node.label = Counter(Y).most_common(1)[0][0] if len(Y) else None
            return node

        split = node.best_split(min_samples_leaf=self.min_samples_leaf)
        if split is None:
            node.label = Counter(Y).most_common(1)[0][0]
            return node

        j, t = split
        node.feature_index, node.threshold = j, t

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