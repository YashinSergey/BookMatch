import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .metrics import ranking_metrics, list_diversity


def compare_models(models, hybrid, segmented, heldout, users, batch_size=256):
    data = hybrid.data
    relevant = heldout.loc[heldout.rating >= 4].set_index('user_id').book_id
    content = models['Content'].features
    without_content = hybrid.weights.copy()
    without_content[list(models).index('Content')] = 0
    without_content /= without_content.sum()
    rows = []
    shown = {}
    candidate_hits = []
    with threadpool_limits(limits=2):
        for start in range(0, len(users), batch_size):
            batch = users[start:start + batch_size]
            indices = data.users.get_indexer(batch)
            baseline = {name: model.recommend_batch(indices, 10) for name, model in models.items()}
            pool = hybrid.get_candidates(indices)
            for row, (user, index, candidates) in enumerate(zip(batch, indices, pool)):
                items, values = candidates
                group = segmented.group(index)
                candidate_hits.append(relevant[user] in data.book_ids[items])
                lists = {name: recommendations[row] for name, recommendations in baseline.items()}
                lists['Hybrid'] = items[np.argsort(-(values @ hybrid.weights), kind='stable')[:10]]
                lists['Without Content'] = items[np.argsort(-(values @ without_content), kind='stable')[:10]]
                weights = segmented.group_weights[group]
                lists['Grouped'] = items[np.argsort(-(values @ (weights / weights.sum())), kind='stable')[:10]]
                lists['Grouped + MMR'] = segmented.rank_candidates(items, values, index)
                for name, selected in lists.items():
                    selected = selected[selected >= 0]
                    record = {'model': name, 'group': group}
                    for k in [5, 10]:
                        precision, recall, ndcg = ranking_metrics(data.book_ids[selected], {relevant[user]}, k)
                        record.update({f'precision@{k}': precision, f'recall@{k}': recall, f'ndcg@{k}': ndcg})
                    record['diversity@10'] = list_diversity(content, selected)
                    rows.append(record)
                    shown.setdefault((name, group), set()).update(selected)
    frame = pd.DataFrame(rows)
    result = []
    for name, model_rows in frame.groupby('model', sort=False):
        for group in ['all', 'short', 'active']:
            part = model_rows if group == 'all' else model_rows[model_rows.group == group]
            if part.empty:
                continue
            catalog = set().union(*(items for (model, segment), items in shown.items()
                                    if model == name and (group == 'all' or segment == group)))
            result.append({'model': name, 'group': group, 'users': len(part),
                           **part.drop(columns=['model', 'group']).mean().to_dict(),
                           'coverage@10': len(catalog) / len(data.items)})
    return pd.DataFrame(result), float(np.mean(candidate_hits))
