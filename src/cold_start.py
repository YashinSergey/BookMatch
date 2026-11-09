import numpy as np
import pandas as pd

from .data import InteractionData
from .popularity import Popularity
from .metrics import ranking_metrics


def evaluate_new_users(train, heldout, books, min_ratings=10, seed=42):
    users = np.sort(train.user_id.unique())
    rng = np.random.default_rng(seed)
    cold_users = rng.choice(users, min(5000, len(users) // 2), replace=False)
    warm_train = train[~train.user_id.isin(cold_users)]
    data = InteractionData(warm_train, books)
    model = Popularity(data, min_ratings=min_ratings)
    relevant = heldout[(heldout.rating >= 4) & heldout.user_id.isin(cold_users)]
    assert not relevant.user_id.isin(data.users).any()
    recommended = model.get_recommendations(N=10)
    rows = []
    for targets in relevant.groupby('user_id').book_id.agg(set):
        row = {}
        for k in [5, 10]:
            p, r, n = ranking_metrics(recommended, targets, k)
            row.update({f'precision@{k}': p, f'recall@{k}': r, f'ndcg@{k}': n})
        rows.append(row)
    return pd.DataFrame([{'model': 'Popularity', 'group': 'new',
                          'removed_users': len(cold_users), 'users': len(rows),
                          **pd.DataFrame(rows).mean().to_dict()}])
