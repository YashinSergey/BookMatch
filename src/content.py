import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .base import Recommender, top_indices


SERVICE_TAGS = {
    'to-read', 'currently-reading', 'read', 'favorites', 'favourites',
    'owned', 'books-i-own', 'my-books', 'library', 'default', 'wishlist',
    'wish-list', 'to-buy', 'owned-books', 'my-library', 'kindle', 'ebook',
    'ebooks', 'e-book', 'e-books', 'audiobook', 'audiobooks', 'audio',
}


class ContentBased(Recommender):
    def fit(self, books, tags, book_tags, max_tags=20, min_df=2):
        links = book_tags.merge(tags, on='tag_id', validate='many_to_one')
        links['tag_name'] = links.tag_name.str.lower().str.strip()
        links = links[~links.tag_name.isin(SERVICE_TAGS) & (links['count'] > 0)]
        links = links.sort_values(['goodreads_book_id', 'count'], ascending=[True, False])
        links = links.drop_duplicates(['goodreads_book_id', 'tag_name'])
        links = links.groupby('goodreads_book_id').head(max_tags)
        # Тег с дефисами сохраняется как единый токен
        links['token'] = links.tag_name.str.replace(r'\W+', '_', regex=True)
        tag_text = links.groupby('goodreads_book_id').token.agg(' '.join)
        catalog = books.set_index('book_id').loc[self.data.book_ids]
        titles = catalog.original_title.fillna('').str.strip()
        titles = titles.where(titles != '', catalog.title)
        profiles = titles + ' ' + catalog.goodreads_book_id.map(tag_text).fillna('')
        self.vectorizer = TfidfVectorizer(min_df=min_df, max_df=0.95, dtype=np.float32)
        self.features = self.vectorizer.fit_transform(profiles)
        self.profiles = normalize(self.data.positive @ self.features).tocsr()
        return self

    def score(self, user_indices):
        return (self.profiles[user_indices] @ self.features.T).toarray()

    def get_similar_books(self, book_id, N=5):
        index = self.data.items.get_indexer([book_id])[0]
        if index < 0:
            raise ValueError(f'Книга {book_id} отсутствует в каталоге')
        scores = (self.features[index] @ self.features.T).toarray()
        scores[0, index] = -np.inf
        indices = top_indices(scores, min(N, len(self.data.items) - 1))[0]
        return [int(self.data.book_ids[i]) for i in indices]
