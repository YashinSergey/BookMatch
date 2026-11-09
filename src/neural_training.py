import numpy as np
from threadpoolctl import threadpool_limits

from .features import HybridFeatures
from .hybrid import Hybrid
from .popularity import Popularity
from .item_cf import ItemCF
from .svd import SVDRecommender
from .neural import candidate_features


def training_pairs(train, books, tags, book_tags, params, weights, config):
    counts = train.groupby('user_id').size()
    eligible = train[(train.rating >= 4) & train.user_id.isin(counts[counts >= 2].index)]
    targets = eligible.groupby('user_id').sample(n=1, random_state=config['seed'])
    inner_train = train.drop(index=targets.index)
    rng = np.random.default_rng(config['seed'])
    users = np.sort(rng.choice(targets.user_id, min(config['train_users'], len(targets)), replace=False))
    target_books = targets.set_index('user_id').book_id
    with threadpool_limits(limits=2):
        features = HybridFeatures().fit(inner_train, books, tags, book_tags)
        data = features.data
        models = {
            'Popularity': Popularity(data, **params['Popularity']),
            'Content': features.content,
            'ItemCF': ItemCF(data).fit(**params['ItemCF']),
            'SVD': SVDRecommender(data).fit(inner_train, seed=config['seed'], **params['SVD']),
        }
        hybrid = Hybrid(models, weights, config['candidates'])
        matrices, labels = [], []
        covered = 0
        for start in range(0, len(users), 256):
            batch = users[start:start + 256]
            pool = hybrid.get_candidates(data.users.get_indexer(batch))
            sampled, kept_users = [], []
            for user, (items, values) in zip(batch, pool):
                positive = np.flatnonzero(data.book_ids[items] == target_books[user])
                if len(positive) == 0:
                    continue
                negatives = np.flatnonzero(data.book_ids[items] != target_books[user])
                negatives = rng.choice(negatives, min(config['negatives'], len(negatives)), replace=False)
                selected = np.concatenate([positive, negatives])
                sampled.append((items[selected], values[selected]))
                kept_users.append(user)
                labels.extend([1] + [0] * len(negatives))
                covered += 1
            if sampled:
                matrices.append(candidate_features(features, np.array(kept_users), sampled))
    summary = {'sampled_users': len(users), 'covered_users': covered,
               'positive_pairs': covered, 'negative_pairs': len(labels) - covered}
    return np.vstack(matrices).astype(np.float32), np.array(labels), summary
