import numpy as np

from .base import Recommender, top_indices


class Hybrid(Recommender):
    def __init__(self, models, weights, candidates=100):
        super().__init__(next(iter(models.values())).data)
        self.models = models
        self.weights = np.array([weights[name] for name in models], dtype=float)
        if (self.weights < 0).any() or self.weights.sum() <= 0:
            raise ValueError('Веса должны быть неотрицательными с положительной суммой')
        self.weights /= self.weights.sum()
        self.candidates = candidates

    def get_candidates(self, user_indices):
        scores = []
        selected = []
        rows, cols = self.data.seen[user_indices].nonzero()
        for model in self.models.values():
            values = model.score(user_indices).copy()
            values[rows, cols] = -np.inf
            selected.append(top_indices(values, self.candidates))
            finite = np.isfinite(values)
            low = np.where(finite, values, np.inf).min(axis=1, keepdims=True)
            high = np.where(finite, values, -np.inf).max(axis=1, keepdims=True)
            span = high - low
            normalized = np.zeros_like(values)
            np.divide(values - low, span, out=normalized, where=finite & (span > 0))
            scores.append(normalized)
        result = []
        for row, user in enumerate(user_indices):
            items = np.unique(np.concatenate([indices[row] for indices in selected]))
            items = np.setdiff1d(items, self.data.seen[user].indices)
            values = np.column_stack([score[row, items] for score in scores])
            result.append((items, values))
        return result

    def recommend_batch(self, user_indices, n=10):
        result = np.full((len(user_indices), n), -1, dtype=int)
        for row, (items, values) in enumerate(self.get_candidates(user_indices)):
            order = np.argsort(-(values @ self.weights), kind='stable')[:n]
            result[row, :len(order)] = items[order]
        return result
