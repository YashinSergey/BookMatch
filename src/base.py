import numpy as np


def top_indices(scores, n):
    if n < 1:
        raise ValueError('N должен быть положительным')
    # Стабильная сортировка: при равенстве первым идёт меньший book_id
    return np.argsort(-scores, axis=1, kind='stable')[:, :n]


class Recommender:
    def __init__(self, data):
        self.data = data

    def recommend_batch(self, user_indices, n=10):
        if not 1 <= n <= len(self.data.items):
            raise ValueError('N должен быть от 1 до числа книг')
        scores = self.score(user_indices)
        rows, cols = self.data.seen[user_indices].nonzero()
        scores[rows, cols] = -np.inf
        indices = top_indices(scores, n)
        # Если книг не хватает, оставляем -1 вместо недоступной рекомендации
        for row in range(len(indices)):
            selected_scores = scores[row, indices[row]]
            unavailable = ~np.isfinite(selected_scores)
            indices[row, unavailable] = -1
        return indices

    def get_recommendations(self, user_id, N=5):
        index = self.data.users.get_indexer([user_id])[0]
        if index < 0:
            raise ValueError('Пользователь отсутствует в train')
        indices = self.recommend_batch(np.array([index]), N)[0]
        recommendations = []
        for i in indices:
            if i >= 0:
                recommendations.append(int(self.data.book_ids[i]))
        return recommendations
