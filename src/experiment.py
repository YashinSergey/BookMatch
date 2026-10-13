import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import load_data, split_ratings, InteractionData
from .popularity import Popularity
from .content import ContentBased
from .item_cf import ItemCF
from .svd import SVDRecommender
from .metrics import evaluate


def run():
    config = json.loads(Path('experiments.json').read_text())
    seed = config['seed']
    books, ratings, tags, book_tags = load_data()
    train, validation, test = split_ratings(ratings, seed)
    data = InteractionData(train, books)
    Path('reports').mkdir(exist_ok=True)

    # Одна выборка validation для всех вариантов
    users = np.sort(validation.loc[validation.rating >= 4, 'user_id'].unique())
    rng = np.random.default_rng(seed)
    users = np.sort(rng.choice(users, min(config['validation_users'], len(users)), replace=False))
    results = []
    models = {}
    selected = {}
    settings = {
        'Popularity': config['popularity'],
        'Content': config['content'],
        'ItemCF': config['item_cf'],
        'SVD': config['svd'],
    }

    with threadpool_limits(limits=2):
        for name, variants in settings.items():
            best_score = -1
            for params in variants:
                if name == 'Popularity':
                    model = Popularity(data, **params)
                elif name == 'Content':
                    model = ContentBased(data).fit(books, tags, book_tags, **params)
                elif name == 'ItemCF':
                    model = ItemCF(data).fit(**params)
                else:
                    model = SVDRecommender(data).fit(train, seed=seed, **params)

                metrics = evaluate(model, validation, users=users)
                row = {'model': name, 'parameters': json.dumps(params)}
                row.update(metrics)
                results.append(row)
                print(name, params, 'nDCG@10:', round(metrics['ndcg@10'], 5), flush=True)
                if metrics['ndcg@10'] > best_score:
                    best_score = metrics['ndcg@10']
                    selected[name] = params
                    models[name] = model

        pd.DataFrame(results).to_csv('reports/validation.csv', index=False)
        test_results = []
        for name, model in models.items():
            metrics = evaluate(model, test)
            row = {'model': name, 'parameters': json.dumps(selected[name])}
            row.update(metrics)
            test_results.append(row)
            print('TEST', name, metrics, flush=True)
        pd.DataFrame(test_results).to_csv('reports/test.csv', index=False)

    return models, books, train, validation, test


if __name__ == '__main__':
    run()
