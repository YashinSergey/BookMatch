import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import normalize

from .base import Recommender


class ItemCF(Recommender):
    def fit(self, neighbors=100, shrinkage=10, batch_size=256):
        matrix = self.data.ratings
        normalized = normalize(matrix.T).tocsr()
        seen = self.data.seen
        blocks = []
        for start in range(0, matrix.shape[1], batch_size):
            stop = min(start + batch_size, matrix.shape[1])
            similarity = (normalized[start:stop] @ normalized.T).toarray()
            support = (seen[:, start:stop].T @ seen).toarray()
            similarity *= support / (support + shrinkage + 1e-12)
            similarity[np.arange(stop - start), np.arange(start, stop)] = 0
            k = min(neighbors, matrix.shape[1] - 1)
            nearest = np.argsort(-similarity, axis=1, kind='stable')[:, :k]
            rows = np.repeat(np.arange(stop - start), k)
            columns = nearest.ravel()
            values = similarity[rows, columns]
            block = csr_matrix((values, (rows, columns)),
                               shape=(stop - start, matrix.shape[1]))
            block.eliminate_zeros()
            blocks.append(block)
        self.similarity = vstack(blocks, format='csr')
        self.centered = matrix.copy()
        self.centered.data -= np.repeat(self.data.user_mean, self.data.user_count)
        return self

    def score(self, user_indices):
        # Для Top-N: сумма похожестей с понравившимися книгами
        return (self.data.positive[user_indices] @ self.similarity.T).toarray()

    def predict(self, user_ids, book_ids):
        users = self.data.users.get_indexer(user_ids)
        items = self.data.items.get_indexer(book_ids)
        result = np.full(len(users), self.data.mean)
        known_items = items >= 0
        result[known_items] = self.data.item_mean[items[known_items]]
        known = (users >= 0) & known_items
        # Для оценки: средняя пользователя + взвешенные отклонения соседей
        neighbors = self.similarity[items[known]]
        numerator = np.asarray(neighbors.multiply(self.centered[users[known]]).sum(axis=1)).ravel()
        denominator = np.asarray(neighbors.multiply(self.data.seen[users[known]]).sum(axis=1)).ravel()
        correction = np.zeros_like(numerator)
        has_neighbors = denominator > 0
        correction[has_neighbors] = numerator[has_neighbors] / denominator[has_neighbors]
        result[known] = self.data.user_mean[users[known]] + correction
        return np.clip(result, 1, 5)
