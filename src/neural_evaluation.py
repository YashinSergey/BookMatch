import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .metrics import ranking_metrics, list_diversity


def compare_neural(neural, scores_only, heldout, users, short_history=5):
    data = neural.data
    relevant = heldout.loc[heldout.rating >= 4].set_index('user_id').book_id
    records, shown = [], {}
    with threadpool_limits(limits=2):
        for start in range(0, len(users), 256):
            batch = users[start:start + 256]
            indices = data.users.get_indexer(batch)
            pool = neural.get_candidates(indices)
            full = neural.predict_pool(batch, pool)
            simple = scores_only.predict_pool(batch, pool)
            for user, index, (items, values), full_scores, simple_scores in zip(batch, indices, pool, full, simple):
                group = 'short' if data.user_count[index] <= short_history else 'active'
                lists = {
                    'MLP scores': scores_only.rank_candidates(items, values, simple_scores, blend=1),
                    'MLP features': neural.rank_candidates(items, values, full_scores, blend=1),
                    'Neural hybrid': neural.rank_candidates(items, values, full_scores),
                }
                for name, selected in lists.items():
                    row = {'model': name, 'group': group}
                    for k in [5, 10]:
                        p, r, n = ranking_metrics(data.book_ids[selected], {relevant[user]}, k)
                        row.update({f'precision@{k}': p, f'recall@{k}': r, f'ndcg@{k}': n})
                    row['diversity@10'] = list_diversity(neural.features.content.features, selected)
                    records.append(row)
                    shown.setdefault((name, group), set()).update(selected)
    frame = pd.DataFrame(records)
    result = []
    for name, rows in frame.groupby('model', sort=False):
        for group in ['all', 'short', 'active']:
            part = rows if group == 'all' else rows[rows.group == group]
            if part.empty:
                continue
            items = set().union(*(items for (model, segment), items in shown.items()
                                  if model == name and (group == 'all' or segment == group)))
            result.append({'model': name, 'group': group, 'users': len(part),
                           **part.drop(columns=['model', 'group']).mean().to_dict(),
                           'coverage@10': len(items) / len(data.items)})
    return pd.DataFrame(result)
