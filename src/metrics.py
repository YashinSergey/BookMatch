import numpy as np
import pandas as pd


def ranking_metrics(recommended, relevant, k):
    hits = np.isin(recommended[:k], list(relevant)).astype(float)
    precision = hits.sum() / k
    recall = hits.sum() / len(relevant) if relevant else 0.0
    discounts = 1 / np.log2(np.arange(2, k + 2))
    dcg = np.sum(hits * discounts[:len(hits)])
    ideal = discounts[:min(k, len(relevant))].sum()
    ndcg = dcg / ideal if ideal else 0.0
    return precision, recall, ndcg


def evaluate(model, heldout, users=None, ks=(5, 10), batch_size=256):
    relevant = heldout[heldout.rating >= 4].groupby('user_id').book_id.agg(set)
    if users is None:
        users = relevant.index.to_numpy()
    users = np.asarray([u for u in users if u in relevant.index and u in model.data.users])
    if len(users) == 0:
        raise ValueError('Нет пользователей с релевантными книгами')
    records = []
    for start in range(0, len(users), batch_size):
        batch_users = users[start:start + batch_size]
        user_indices = model.data.users.get_indexer(batch_users)
        recommendations = model.recommend_batch(user_indices, max(ks))
        for user_id, indices in zip(batch_users, recommendations):
            book_ids = model.data.book_ids[indices[indices >= 0]]
            row = {}
            for k in ks:
                p, r, n = ranking_metrics(book_ids, relevant[user_id], k)
                row.update({f'precision@{k}': p, f'recall@{k}': r, f'ndcg@{k}': n})
            records.append(row)
    result = pd.DataFrame(records).mean().to_dict()
    result['users'] = len(users)
    if hasattr(model, 'predict'):
        predictions = model.predict(heldout.user_id.to_numpy(), heldout.book_id.to_numpy())
        result['rmse'] = float(np.sqrt(np.mean((predictions - heldout.rating.to_numpy()) ** 2)))
    return result
