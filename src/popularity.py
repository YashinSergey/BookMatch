import numpy as np

from .base import Recommender, top_indices


class Popularity(Recommender):
    def __init__(self, data, min_ratings=20):
        super().__init__(data)
        self.values = data.item_mean.copy()
        self.values[data.item_count < min_ratings] = -np.inf

    def score(self, user_indices):
        return np.tile(self.values, (len(user_indices), 1))

    def get_recommendations(self, user_id=None, N=5):
        index = self.data.users.get_indexer([user_id])[0]
        scores = self.values.copy()
        if index >= 0:
            scores[self.data.ratings[index].indices] = -np.inf
        indices = top_indices(scores[None, :], N)[0]
        return [int(self.data.book_ids[i]) for i in indices if np.isfinite(scores[i])]

    def predict(self, user_ids, book_ids):
        indices = self.data.items.get_indexer(book_ids)
        result = np.full(len(indices), self.data.mean)
        known = indices >= 0
        result[known] = self.data.item_mean[indices[known]]
        return result
