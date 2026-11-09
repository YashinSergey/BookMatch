import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .hybrid import Hybrid


def candidate_features(features, users, pool, extra=True):
    values = np.vstack([values for _, values in pool])
    if not extra:
        return values
    pairs = pd.DataFrame({
        'user_id': np.repeat(users, [len(items) for items, _ in pool]),
        'book_id': features.data.book_ids[np.concatenate([items for items, _ in pool])],
    })
    extra_values = features.transform(pairs).drop(columns=['user_id', 'book_id'])
    return np.column_stack([values, extra_values.to_numpy(dtype=np.float32)])


class NeuralHybrid(Hybrid):
    def __init__(self, models, weights, features, extra=True, blend=0.5, candidates=100):
        super().__init__(models, weights, candidates)
        self.features = features
        self.extra = extra
        self.blend = blend

    def fit(self, x, y, hidden=(32, 16), epochs=20, seed=42):
        self.scaler = StandardScaler().fit(x)
        x = self.scaler.transform(x).astype(np.float32)
        self.network = MLPClassifier(hidden_layer_sizes=hidden, batch_size=256,
                                     learning_rate_init=0.001, random_state=seed)
        self.loss = []
        with threadpool_limits(limits=2):
            for _ in range(epochs):
                self.network.partial_fit(x, y, classes=[0, 1])
                self.loss.append(self.network.loss_)
        return self

    def predict_pool(self, users, pool):
        x = candidate_features(self.features, users, pool, self.extra)
        x = self.scaler.transform(x).astype(np.float32)
        scores = self.network.predict_proba(x)[:, 1]
        return np.split(scores, np.cumsum([len(items) for items, _ in pool])[:-1])

    def rank_candidates(self, items, values, scores, n=10, blend=None):
        span = np.ptp(scores)
        normalized = (scores - scores.min()) / span if span > 0 else np.zeros_like(scores)
        blend = self.blend if blend is None else blend
        score = (1 - blend) * (values @ self.weights) + blend * normalized
        return items[np.argsort(-score, kind='stable')[:n]]

    def recommend_batch(self, user_indices, n=10):
        pool = self.get_candidates(user_indices)
        predictions = self.predict_pool(self.data.user_ids[user_indices], pool)
        result = np.full((len(user_indices), n), -1, dtype=int)
        for row, ((items, values), scores) in enumerate(zip(pool, predictions)):
            selected = self.rank_candidates(items, values, scores, n)
            result[row, :len(selected)] = selected
        return result

    def get_recommendations(self, user_id=None, N=5):
        if user_id not in self.data.users:
            return self.models['Popularity'].get_recommendations(N=N)
        return super().get_recommendations(user_id, N)
