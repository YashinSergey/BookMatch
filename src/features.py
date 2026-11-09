import numpy as np
import pandas as pd

from .content import ContentBased, SERVICE_TAGS
from .data import InteractionData


class HybridFeatures:
    def fit(self, train, books, tags, book_tags, max_tags=30):
        self.data = InteractionData(train, books)
        self.content = ContentBased(self.data).fit(books, tags, book_tags, max_tags=max_tags)
        self.users = train.groupby('user_id').rating.agg(
            user_mean='mean', user_count='size', user_std='std',
        )
        self.books = train.groupby('book_id').rating.agg(
            book_mean='mean', book_count='size', book_std='std',
        ).reindex(self.data.book_ids)
        self.books['book_mean'] = self.books.book_mean.fillna(self.data.mean)
        self.books = self.books.fillna(0)
        self.users = self.users.fillna(0)

        links = book_tags.merge(tags, on='tag_id')
        links['tag_name'] = links.tag_name.str.lower().str.strip()
        links = links[(links['count'] > 0) & ~links.tag_name.isin(SERVICE_TAGS)]
        links = links.merge(books[['book_id', 'goodreads_book_id']], on='goodreads_book_id')
        links = links.drop_duplicates(['book_id', 'tag_name'])
        topics = links.tag_name.value_counts().head(20).index
        links = links[links.tag_name.isin(topics)]
        themes = pd.crosstab(links.book_id, links.tag_name).clip(upper=1)
        themes = themes.reindex(index=self.data.book_ids, columns=topics, fill_value=0)
        self.books = self.books.join(themes.add_prefix('tag_'))
        return self

    def transform(self, pairs):
        result = pairs[['user_id', 'book_id']].reset_index(drop=True)
        result = result.join(self.users, on='user_id').join(self.books, on='book_id')
        result[['user_mean', 'book_mean']] = result[['user_mean', 'book_mean']].fillna(self.data.mean)
        result = result.fillna(0)
        users = self.data.users.get_indexer(result.user_id)
        items = self.data.items.get_indexer(result.book_id)
        known = (users >= 0) & (items >= 0)
        similarity = np.zeros(len(result), dtype=np.float32)
        # Считаем только заданные пары, без плотной матрицы пользователь-книга
        for start in range(0, len(result), 4096):
            rows = np.flatnonzero(known[start:start + 4096]) + start
            profile = self.content.profiles[users[rows]]
            content = self.content.features[items[rows]]
            similarity[rows] = np.asarray(profile.multiply(content).sum(axis=1)).ravel()
        result['history_similarity'] = similarity
        return result
