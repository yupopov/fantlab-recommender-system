import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymorphy2 import MorphAnalyzer
from string import punctuation
from tqdm.auto import tqdm
import pandas as pd
from collections import defaultdict

punctuation_rus = list(punctuation)
punctuation_rus.extend(['«', '»', "''", '``', '„', '“', '..', '...', '—', '–', '…', '’', '№'])
stopwords_rus = stopwords.words('russian')
stopwords_rus = set(stopwords_rus) | set(punctuation_rus)  # transforming into a set for faster inclusion checking
morph_analyzer = MorphAnalyzer()


class TitleParser:
    def __init__(self, tokenize_fn=word_tokenize, morph_analyzer=morph_analyzer):
        self.tokenize_fn = word_tokenize
        self.morph = morph_analyzer

    def preprocess_text(self, text: str, text_type: str, normalize_words=True) -> str:
        """
        Preprocess the text: tokenize, remove stopwords,
        and (possibly) lemmatize by a pymorhpy2's MorphAnalyzer
        (probably will only be useful for linear models)
        """
        assert text_type in ('work_info', 'author', 'award', 'genre_label'), 'Unknown text type!'
        # simple initial preprocessing
        text = text.lower().replace('ё', 'е')
        # join_symbol = ' ' if text_type == 'work_info' or text_type == 'genre_label' else '_'
        join_symbol = ' ' if text_type == 'work_info' else '_'
        if text_type in ('author', 'genre_label'):
            tokens = [word for word in text.lower().split()]
            # perhaps leaving just the last name works better?
            # check later
            # author = text.lower().split()[-1]
        else:
            tokens = []
            for word in self.tokenize_fn(text):
                # filter stopwords
                if word not in stopwords_rus:
                    word_normal_form = self.morph.parse(word)[0].normal_form
                    # some inflected forms of stopwords aren't included in the stopwords list,
                    # so filter once more
                    if word_normal_form not in stopwords_rus:
                        tokens.append(word_normal_form if normalize_words else word)
        # elif self.morph is not None:
        #     tokens = [self.morph.parse(word)[0].normal_form for word in self.tokenize_fn(text) if word not in stopwords_rus]
        # perhaps the stopwords list should be modified,
        # because we don't want inflected forms of stopwords for neural embeddings too
        #     tokens = [word for word in tokens if word not in stopwords_rus]
        # else:
        #     tokens = [word for word in self.tokenize_fn(text) if word not in stopwords_rus]
        return join_symbol.join(tokens)

    def parse_classificatory(self, classificatory: dict):
        # do something with total_count?
        # a numeric feature?
        genre_infos = []
        genre_groups = classificatory.get('genre_group', [])
        for genre_group in genre_groups:
            genre_group_id = genre_group['genre_group_id']
            genres = genre_group.get('genre', [])
            genre_labels = []
            for genre in genres:
                genre_label = genre.get('label') or ''
                genre_labels.append(self.preprocess_text(genre_label, text_type='genre_label'))
                genre_id = genre.get('genre_id', [])
                genre_vote = genre.get('votes', [])
                genre_weight = genre.get('percent', [])
                genre_infos.append((genre_id, genre_vote, genre_weight))
            self.title_features['genre_infos'] = genre_infos
            self.title_features[f'genre_group_{genre_group_id}_labels'] = ' '.join(genre_labels)

    def parse_awards(self, awards: dict):
        # ignoring nominated awards for now
        awards_won = awards.get('win', [])
        title_awards = set()  # order??
        for award in awards_won:
            award_name = award.get('award_rusname', '')
            title_awards.add(self.preprocess_text(award_name, text_type='award'))
        self.title_features['awards'] = ' '.join(title_awards)

    def parse_title_info(self, title_info: dict) -> dict:
        self.title_features = defaultdict(str)
        # record work id and work name
        self.title_features['work_id'] = title_info['work_id']
        self.title_features['work_name'] = title_info['work_name']

        # preprocess and record work info
        for work_property in (
                'work_name_bonus',  # work name bonus, perhaps useless
                'work_description',  # work description
                'work_notes'  # work notes, probably unavailable for most of the works and useless, but still
        ):
            work_property_ = title_info.get(work_property) or ''
            self.title_features[work_property] = self.preprocess_text(work_property_, text_type='work_info')

        # preprocess and record authors
        authors = title_info.get('authors', [])
        self.title_features['authors'] = ' '.join(
            (self.preprocess_text(author['name'], text_type='author') for author in authors if
             author['type'] == 'autor')
        )

        # record language (useless?)
        self.title_features['lang'] = title_info.get('lang') or ''

        # record genre labels
        classificatory = title_info.get('classificatory', {})
        self.parse_classificatory(classificatory)

        # record awards
        awards = title_info.get('awards') or {}  # some fields can explicitly contain None
        self.parse_awards(awards)

        return self.title_features

    def __call__(self, title_infos: list) -> pd.DataFrame:
        self.features = [self.parse_title_info(title_info) for title_info in tqdm(title_infos)]
        self.features = pd.DataFrame(self.features).set_index('work_id').fillna('')
        return self.features