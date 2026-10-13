import numpy as np
from surprise import Dataset, Reader, SVD

from .base import Recommender


class SVDRecommender(Recommender):
    def fit(self, train, factors=50, epochs=30, regularization=0.1, learning_rate=0.005, seed=42):
        dataset = Dataset.load_from_df(train[['user_id', 'book_id', 'rating']], Reader(rating_scale=(1, 5)))
        trainset = dataset.build_full_trainset()
        model = SVD(n_factors=factors, n_epochs=epochs, reg_all=regularization,
                    lr_all=learning_rate, random_state=seed)
        model.fit(trainset)
        self.user_factors = np.zeros((len(self.data.users), factors), dtype=np.float32)
        self.item_factors = np.zeros((len(self.data.items), factors), dtype=np.float32)
        self.user_bias = np.zeros(len(self.data.users), dtype=np.float32)
        self.item_bias = np.zeros(len(self.data.items), dtype=np.float32)
        for inner in trainset.all_users():
            index = self.data.users.get_loc(trainset.to_raw_uid(inner))
            self.user_factors[index] = model.pu[inner]
            self.user_bias[index] = model.bu[inner]
        for inner in trainset.all_items():
            index = self.data.items.get_loc(trainset.to_raw_iid(inner))
            self.item_factors[index] = model.qi[inner]
            self.item_bias[index] = model.bi[inner]
        return self

    def score(self, user_indices):
        baseline = self.data.mean + self.user_bias[user_indices, None] + self.item_bias[None, :]
        personal_scores = self.user_factors[user_indices] @ self.item_factors.T
        return baseline + personal_scores

    def predict(self, user_ids, book_ids):
        users = self.data.users.get_indexer(user_ids)
        items = self.data.items.get_indexer(book_ids)
        result = np.full(len(users), self.data.mean)
        known_users = users >= 0
        known_items = items >= 0
        result[known_users] += self.user_bias[users[known_users]]
        result[known_items] += self.item_bias[items[known_items]]
        known = known_users & known_items
        result[known] += np.sum(self.user_factors[users[known]] * self.item_factors[items[known]], axis=1)
        return np.clip(result, 1, 5)
