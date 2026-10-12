from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def load_data(folder='data/archive'):
    folder = Path(folder)
    books = pd.read_csv(folder / 'books.csv', dtype={'isbn': 'string', 'isbn13': 'string'})
    books = books.rename(columns={'id': 'book_id', 'book_id': 'goodreads_book_id'})
    ratings = pd.read_csv(folder / 'ratings.csv').drop_duplicates()
    # Нет дат: пары с разными оценками исключаются, как в EDA
    ratings = ratings[~ratings.duplicated(['user_id', 'book_id'], keep=False)].copy()
    tags = pd.read_csv(folder / 'tags.csv')
    book_tags = pd.read_csv(folder / 'book_tags.csv').drop_duplicates()
    conflict = book_tags.duplicated(['goodreads_book_id', 'tag_id'], keep=False)
    book_tags = book_tags[~conflict & (book_tags['count'] >= 0)].copy()
    assert books.book_id.is_unique
    assert books.goodreads_book_id.is_unique
    assert ratings.rating.between(1, 5).all()
    assert ratings.book_id.isin(books.book_id).all()
    return books, ratings, tags, book_tags


def split_ratings(ratings, seed=42):
    shuffled = ratings.reset_index(drop=True).sample(frac=1, random_state=seed)
    position = shuffled.groupby('user_id').cumcount()
    size = shuffled.groupby('user_id').user_id.transform('size')
    test = shuffled[(size >= 3) & (position == 0)].copy()
    validation = shuffled[(size >= 3) & (position == 1)].copy()
    train = shuffled[(size < 3) | (position >= 2)].copy()
    assert len(train) + len(validation) + len(test) == len(ratings)
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)
    return train, validation, test


class InteractionData:
    def __init__(self, train, books):
        self.user_ids = np.sort(train.user_id.unique())
        self.book_ids = np.sort(books.book_id.unique())
        self.users = pd.Index(self.user_ids)
        self.items = pd.Index(self.book_ids)
        rows = self.users.get_indexer(train.user_id)
        cols = self.items.get_indexer(train.book_id)
        self.ratings = csr_matrix(
            (train.rating.to_numpy(dtype=np.float32), (rows, cols)),
            shape=(len(self.users), len(self.items)),
        )
        self.seen = self.ratings.copy()
        self.seen.data[:] = 1
        self.positive = self.ratings.copy()
        self.positive.data = (self.positive.data >= 4).astype(np.float32)
        self.positive.eliminate_zeros()
        self.item_count = np.asarray(self.seen.sum(axis=0)).ravel()
        self.user_count = np.diff(self.ratings.indptr)
        self.mean = float(train.rating.mean())
        sums = np.asarray(self.ratings.sum(axis=0)).ravel()
        self.item_mean = np.full(len(self.items), self.mean)
        has_ratings = self.item_count > 0
        self.item_mean[has_ratings] = sums[has_ratings] / self.item_count[has_ratings]
        self.user_mean = np.asarray(self.ratings.sum(axis=1)).ravel() / self.user_count
