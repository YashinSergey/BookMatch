import numpy as np

from .hybrid import Hybrid


class SegmentedHybrid(Hybrid):
    def __init__(self, models, weights, group_weights, candidates=100, short_history=5, diversity=0):
        super().__init__(models, weights, candidates)
        self.group_weights = {
            group: np.array([values[name] for name in models], dtype=float)
            for group, values in group_weights.items()
        }
        self.short_history = short_history
        self.diversity = diversity

    def group(self, user_index):
        if user_index < 0:
            return 'new'
        return 'short' if self.data.user_count[user_index] <= self.short_history else 'active'

    def rank_candidates(self, items, values, user_index, n=10):
        weights = self.group_weights[self.group(user_index)]
        scores = values @ (weights / weights.sum())
        if self.diversity == 0:
            return items[np.argsort(-scores, kind='stable')[:n]]
        content = self.models['Content'].features[items]
        redundancy = np.zeros(len(items))
        available = np.ones(len(items), dtype=bool)
        selected = []
        for _ in range(min(n, len(items))):
            score = (1 - self.diversity) * scores - self.diversity * redundancy
            score[~available] = -np.inf
            index = np.argmax(score)
            selected.append(index)
            available[index] = False
            similarity = (content @ content[index].T).toarray().ravel()
            redundancy = np.maximum(redundancy, similarity)
        return items[selected]

    def recommend_batch(self, user_indices, n=10):
        result = np.full((len(user_indices), n), -1, dtype=int)
        for row, (items, values) in enumerate(self.get_candidates(user_indices)):
            selected = self.rank_candidates(items, values, user_indices[row], n)
            result[row, :len(selected)] = selected
        return result

    def get_recommendations(self, user_id=None, N=5):
        if user_id not in self.data.users:
            return self.models['Popularity'].get_recommendations(N=N)
        return super().get_recommendations(user_id, N)
