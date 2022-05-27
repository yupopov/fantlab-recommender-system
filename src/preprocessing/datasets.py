import json
import os
from dataclasses import dataclass
import gzip

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pack_sequence
from scipy.sparse import coo_matrix, csr_matrix, save_npz
from lightfm.data import Dataset

from .mark_weights import mark_transforms_dict
from .time_weights import transform_dates
from .item_features_buildup import get_tag_ids_, get_item_feature_weights, \
    filter_by_work_id


def print_stats(marks_df):
        n_marks = len(marks_df)
        n_work_ids = marks_df.work_id.nunique()
        n_user_ids = marks_df.user_id.nunique()
        print(f'Marks: {n_marks}\tUnique titles: {n_work_ids}\tUnique users: {n_user_ids}\n')


def filter_by_marks_count_work(marks_df, min_marks_work: int = 50):
    """
    Filter the dataframe, leaving only works with more than `min_marks_work` marks.
    """
    # print(f'Deleting works with less than {min_marks_work} marks...')
    marks_count_by_work = marks_df.groupby('work_id').mark.count()
    works_with_enough_marks = marks_count_by_work.loc[lambda n_marks: n_marks >= min_marks_work].index.tolist()
    marks_df = marks_df.query('work_id in @works_with_enough_marks')

    print(f'Stats after filtering:')
    print_stats(marks_df)
    return marks_df


def filter_by_marks_count_user(marks_df, min_marks_user: int = 20):
    """
    Filter the dataframe, leaving only users with more than `min_marks` marks.
    """
    # print(f'Deleting users with less than {min_marks_user} marks...')
    marks_count_by_user = marks_df.groupby('user_id').mark.count()
    users_with_enough_marks = marks_count_by_user.loc[lambda n_marks: n_marks >= min_marks_user].index.tolist()
    marks_df = marks_df.query('user_id in @users_with_enough_marks')

    print(f'Stats after filtering:')
    print_stats(marks_df)
    return marks_df


def make_seqs(user_items: list, max_seq_len: int = 20):
    seqs = []
    for i in range(0, len(user_items), max_seq_len):
        seq_end = len(user_items) - i
        seq_start = max(seq_end - max_seq_len, 0)
        seqs.append(user_items[seq_start: seq_end])
    return seqs


class LeftPaddedDataset(torch.utils.data.Dataset):
    def __init__(self, seqs: list, vocab: dict,
        max_seq_len: int = 20, mode: str ='train'):
        """
        A Dataset for the language model, padded on the left
        seqs: sequences from a train/val/test split
        max_seq_len: maximal sequence len to pad/cut
        mode: indicates whether to provide targets ('mode' = train)
        or just items (`mode` = pred). Changes the type of the output :(
        """
        self.seqs = seqs
        self.vocab = vocab
        self.padding_ix = self.vocab['<PAD>']
        self.max_seq_len = max_seq_len
        if mode not in ('train', 'pred'):
            raise ValueError('Unknown mode (must be "train" or "pred")')
        self.mode = mode

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, index):
        item = [self.vocab[item] for item in self.seqs[index]]
        # cut the text
        item = item[:self.max_seq_len]
        # pad FROM THE LEFT
        item = [self.padding_ix] * max(self.max_seq_len - len(item), 0) + item 
        if self.mode == 'train':
            # We try to predict the next item in the sequence,
            # So the labels are items shifted by 1 to the right
            item, label = item[:-1], item[1:]
            return item, label
        return item
            
    def collate_fn(self, batch):
        """
        Technical method to form a batch to feed into a language model
        """
        # items = pack_sequence([torch.tensor(pair[0]) for pair in batch], enforce_sorted=False)
        # labels = pack_sequence([torch.tensor(pair[1]) for pair in batch], enforce_sorted=False)
        if self.mode == 'train':
            items = torch.LongTensor([pair[0] for pair in batch])
            labels = torch.LongTensor([pair[1] for pair in batch])
            return items, labels
        return torch.LongTensor(batch)


@dataclass
class FMDataset:
    train_data: coo_matrix
    train_weights: coo_matrix
    test_data: coo_matrix
    work_features: csr_matrix
    dataset: Dataset


@dataclass
class RNNDataset:
    train_data: list
    train_dataset: LeftPaddedDataset
    val_data: list
    val_dataset: LeftPaddedDataset
    pred_data: list
    pred_dataset: LeftPaddedDataset
    train_interactions: coo_matrix
    test_interactions: coo_matrix
    item_vocab: dict
    user_vocab: dict
    embs: torch.Tensor
    

class FMDatasetMaker:
    """
    A class that filters the marks table by
    - Deleting old marks
    - Deleting users with few marks,
    then encondes the user and work ids, constructs the sparse matrix
    from the filtered dataframe and saves encodings and the matrix 
    to the disk.
    """
    def __init__(self,
                n_last_years: int = 5,
                min_marks_user: int = 20,
                marks_df_path='data/raw/work_marks.csv.gz',
                item_features_path='data/raw/item_features.json.gz'):
        self.n_last_years = n_last_years
        # self.min_marks_work = min_marks_work
        # self.min_marks_user = min_marks_user
        print('Loading marks...')
        self.marks_df = pd.read_csv(marks_df_path, parse_dates=['date'])
        print('Loading item features...')
        with gzip.open(item_features_path, 'rt') as f:
            self.item_features = json.load(f)
        print('Done.')

    def filter_by_date(self):
        """
        Filter the dataframe, leaving only marks from n last years.
        """
        print('Stats before filtering by date:')
        print_stats(self.marks_df)

        date_thresh = pd.Timestamp.today().date() - \
            pd.tseries.offsets.YearBegin() - \
            pd.DateOffset(years=self.n_last_years)
        print(f'Deleting marks dated before {date_thresh.year}...')
        self.marks_df = self.marks_df.query('date >= @date_thresh')

        print(f'Stats after filtering by date:')
        print_stats(self.marks_df)

    def make_train_test_data(self,
                            time_q: float = 0.8,
                            min_marks_user_train: int = 20,
                            min_marks_work: int = 10,
                            marks_transform: str = 'decoupling',
                            time_weights_params: dict={},
                            # time weights stuff...
                            ):
        self.filter_by_date()
        split_date = self.marks_df.date.quantile(time_q)
        print(f'Splitting the marks by {split_date}...')
        marks_df_train, marks_df_test = \
            self.marks_df.query('date <= @split_date'), self.marks_df.query('date > @split_date')
        print('Train set stats:')
        print_stats(marks_df_train)
        print('Test set stats:')
        print_stats(marks_df_test)

        # If we drop the marks which are present only in the train
        # or the test set, the model loses the ability to recommend new works
        # (ones that appeared during test period)
        # We hope that adding item features (tags and so on) will help
        # print('Dropping works with interactions only in the train or the test set...')
        # works_train = marks_df_train.work_id.unique()
        # works_test = marks_df_test.work_id.unique()
        # works_joint = np.setinterset1d(works_train, works_test)

        # Users with few marks in the train set aren't likely to get good predictions
        # User features can change it though
        print(f'Dropping users with less than {min_marks_user_train} marks in the train set...')
        marks_df_train = filter_by_marks_count_user(marks_df_train,
                                                    min_marks_user_train)
                                                  
        # Drop users with marks only in the test period
        # This is the only mandatory filtering
        print('Dropping users from the test set with no marks in the train set...')
        users_train = marks_df_train.user_id.unique()
        marks_df_test = marks_df_test.query('user_id in @users_train')
        print('Stats after filtering:')
        print_stats(marks_df_test)

        # Works with few marks in the test set aren't likely to get recommended
        # Work features can change that, perhaps
        print(f'Dropping works with less than {min_marks_work} marks in the test set...')
        marks_df_test = filter_by_marks_count_work(marks_df_test,
                                                  min_marks_work)

        # Add interaction weights
        print(f'Computing train mark weights...')
        mark_transform = mark_transforms_dict[marks_transform]
        marks_df_train['mark_weight'] = \
            marks_df_train.groupby('user_id').mark.transform(mark_transform)
        print('Computing date weights...')
        date_transform = transform_dates # add hyperparameters here!
        marks_df_train = transform_dates(marks_df_train, **time_weights_params)

        # The weight of an interaction is simply a product
        # of mark weight and date weight
        marks_df_train['weight'] = \
            marks_df_train[['mark_weight', 'time_weight']].prod(axis=1)
        marks_df_train.drop(columns=['mark_weight', 'time_weight'], inplace=True)

        # Сonstruct train and test interaction matrices
        print('Constructing train dataset...')
        user_ids = marks_df_train.user_id.unique().tolist()
        work_ids = np.union1d(marks_df_train.work_id.unique(), marks_df_test.work_id.unique())

        # get tag ids
        self.item_features = filter_by_work_id(self.item_features, work_ids)
        tag_ids = get_tag_ids_(self.item_features)

        dataset = Dataset()
        dataset.fit(user_ids, work_ids, item_features=tag_ids)
        train_data, train_weights = dataset.build_interactions(
            marks_df_train[['user_id', 'work_id', 'weight']].to_numpy()
            )
        # Constructing item features from tags
        # add hyperparameters in config
        print('Adding item features...')
        item_feature_weights = get_item_feature_weights(self.item_features)
        work_features = dataset.build_item_features(
          item_feature_weights, normalize=False
          )
        print('Constructing test dataset...')
        test_data, _ = dataset.build_interactions(
            marks_df_test[['user_id', 'work_id']].to_numpy()
            )

        fm_dataset = FMDataset(
          train_data,
          train_weights,
          test_data,
          work_features,
          dataset
          )
        print('Done.')
        return fm_dataset

    def make_sparse_matrix(self):
        self.filter_by_date()
        self.filter_by_marks_count()
        # enumerate user and work ids
        # to create a sparse marks matrix by its indices
        user_dict = dict(enumerate(self.marks_df.user_id.unique().tolist()))
        user_dict = {value: key for key, value in user_dict.items()}

        work_dict = dict(enumerate(self.marks_df.work_id.unique().tolist()))
        work_dict = {value: key for key, value in work_dict.items()}

        marks_matrix = csr_matrix((self.marks_df.mark.values, 
                           (self.marks_df.user_id.map(user_dict).values,
                            self.marks_df.work_id.map(work_dict).values)),
                          dtype=np.int8)
        
        print('Saving sparse marks matrix, user and work ids enumerations to files...')
        os.makedirs('data/interim', exist_ok=True)
        with open('data/interim/user_dict.json', 'w') as f: # hardcoding the paths for now
            # user_dict = {int(user_id): user_num for user_id, user}
            json.dump(user_dict, f)
        with open('data/interim/work_dict.json', 'w') as f:
            # work_dict = json.dumps(work_dict, indent=4)
            json.dump(work_dict, f)

        save_npz('data/interim/marks.npz', marks_matrix)
        print('Done.')


class RNNDatasetMaker:
    def __init__(
      self,
      n_last_years=10,
      time_q=0.9,
      marks_df_path='data/raw/work_marks.csv.gz',
      embs_path='data/interim/bert_embeddings.pt',
      item2emb_ix_path='data/interim/key2index.json.gz',
      min_marks_work_train=50,
      min_marks_work_test=10,
      min_marks_user_train=10,
      max_seq_len=20,
      valid_size=0.1,
      random_state=17
    ):
        self.n_last_years = n_last_years
        self.time_q = time_q
        self.min_marks_work_train = min_marks_work_train
        self.min_marks_work_test = min_marks_work_test
        self.min_marks_user_train = min_marks_user_train
        self.max_seq_len = max_seq_len
        self.valid_size=valid_size
        self.random_state=random_state
        print('Loading marks...')
        self.marks_df = pd.read_csv(marks_df_path, parse_dates=['date'])
        print('Loading embeddings...')
        self.item_embs = torch.load(embs_path)
        with gzip.open(item2emb_ix_path, 'rt') as f:
            item2emb_ix = json.load(f)
        self.item2emb_ix = {int(key): value for key, value in item2emb_ix.items()}
        print('Done.')

    def filter_by_date(self):
        """
        Filter the dataframe, leaving only marks from n last years.
        """
        print('Stats before filtering by date:')
        print_stats(self.marks_df)

        date_thresh = pd.Timestamp.today().date() - \
            pd.tseries.offsets.YearBegin() - \
            pd.DateOffset(years=self.n_last_years)
        print(f'Deleting marks dated before {date_thresh.year}...')
        self.marks_df = self.marks_df.query('date >= @date_thresh')

        print(f'Stats after filtering by date:')
        print_stats(self.marks_df)

    def get_rnn_dataset(self):
        self.filter_by_date()
        split_date = self.marks_df.date.quantile(self.time_q)
        print(f'Splitting the marks by {split_date}...')
        marks_df_train, marks_df_test = \
            self.marks_df.query('date <= @split_date'), self.marks_df.query('date > @split_date')
        print('Train set stats:')
        print_stats(marks_df_train)
        print('Test set stats:')
        print_stats(marks_df_test)

        # Drop works with few marks from train dataset
        print(f'Dropping works with less than {self.min_marks_work_train} marks in the train set...')
        works_train = marks_df_train.work_id.value_counts().\
            loc[lambda x: x >= self.min_marks_work_train].index.tolist()
        marks_df_train = marks_df_train.query('work_id in @works_train')
        print('Train set stats:')
        print_stats(marks_df_train)
        print(f'Dropping works with less than {self.min_marks_work_test} marks in the test set...')
        works_test = marks_df_test.work_id.value_counts().\
            loc[lambda x: x >= self.min_marks_work_test].index.tolist()
        marks_df_test = marks_df_test.query('work_id in @works_test')
        print('Test set stats:')
        print_stats(marks_df_test)
        
        # Dropping works present only in the test or in the train set
        works_joint = np.intersect1d(works_train, works_test)
        print('Dropping works with interactions only in the train or the test set...')
        marks_df_train = marks_df_train.query('work_id in @works_joint')
        marks_df_test = marks_df_test.query('work_id in @works_joint')
        print('Train set stats:')
        print_stats(marks_df_train)
        print('Test set stats:')
        print_stats(marks_df_test)
        print(f'Dropping users with less than {self.min_marks_user_train} marks in the train set...')
        marks_df_train = filter_by_marks_count_user(marks_df_train,
                                                    self.min_marks_user_train)
                                                  
        # Drop users with marks only in the test period
        print('Dropping users from the test set with no marks in the train set...')
        users_train = marks_df_train.user_id.unique()
        marks_df_test = marks_df_test.query('user_id in @users_train')
        print('Stats after filtering:')
        print_stats(marks_df_test)

        print('Constructing test dataset...')
        user_ids = marks_df_train.user_id.unique().tolist()
        work_ids = np.union1d(marks_df_train.work_id.unique(), marks_df_test.work_id.unique())

        # Construct test interactions as a sparse matrix
        dataset = Dataset()
        dataset.fit(user_ids, work_ids)
        test_data, _ = dataset.build_interactions(
            marks_df_test[['user_id', 'work_id']].to_numpy()
            )
        print('Constructing train and val RNN datasets...')
        # Construct train interactions for prediction
        train_data, _ = dataset.build_interactions(
            marks_df_train[['user_id', 'work_id']].to_numpy()
            )
        # Save user and item enumerations
        user2fm_ix = dataset.mapping()[0]
        item2fm_ix = dataset.mapping()[2]
        # Leave only the embeddings corresponding to the 
        # items in our dataset, and get them in the same order as in test data
        # we need: emb row num: fm_dataset col num
        fm_ix2item = {value: key for key, value in item2fm_ix.items()} # fm_dataset col num: item_id
        fm_ix2emb_ix = {key: self.item2emb_ix[value] for key, value in fm_ix2item.items()} # fm_dataset col num: emb row num
        emb_ix2fm_ix = {value: key for key, value in fm_ix2emb_ix.items()} # emb row num: fm_dataset col num
        # sort just in case
        emb_ix2fm_ix = {key: value for key, value in sorted(emb_ix2fm_ix.items(), key=lambda x: x[1])}
        emb_ix_perm = list(emb_ix2fm_ix.keys())
        self.item_embs = self.item_embs[emb_ix_perm] # item_embs: (n_train_items, n_item_features)
        # Add padding as the last embedding element
        item2fm_ix['<PAD>'] = len(item2fm_ix)
        self.item_embs = torch.cat((self.item_embs,
          torch.zeros(1, self.item_embs.shape[1])))

        # Get sequences of item ids for each user, in chronological order
        # and break them down into subsequences of length <= `max_seq_len`
        marks_df_train.sort_values(by=['user_id', 'date'], inplace=True)
        user_seqs = marks_df_train.\
            groupby('user_id').work_id.\
            apply(lambda x: make_seqs(x.tolist(), max_seq_len=self.max_seq_len))
        # Get the last subsequence to predict future interactions
        user_seqs_for_pred = user_seqs.apply(lambda x: x[0])
        # Permute the users so that they are in the same
        # order as in the test sparse matrix
        user_test_data_order = list(user2fm_ix.keys())
        user_seqs_for_pred = user_seqs_for_pred.loc[user_test_data_order]
        pred_seqs = user_seqs_for_pred.tolist()
        # Split users into train and validation parts
        train_user_ids, val_user_ids = train_test_split(
          user_ids, test_size=self.valid_size, random_state=self.random_state
          )
        assert len(np.intersect1d(train_user_ids, val_user_ids)) == 0, \
            'Train and val user sets intersect!'
        # Obtain lists with sequences of work ids
        train_seqs = user_seqs.loc[train_user_ids].sum()
        val_seqs = user_seqs.loc[val_user_ids].sum()
        print(f'Total {len(train_user_ids)} users, {len(train_seqs)} sequences of length <= {self.max_seq_len} in train set.')
        print(f'Total {len(val_user_ids)} users, {len(val_seqs)} sequences of length <= {self.max_seq_len} in valid set.')
        
        # Define the datasets
        train_dataset = LeftPaddedDataset(train_seqs, item2fm_ix, self.max_seq_len)
        val_dataset = LeftPaddedDataset(val_seqs, item2fm_ix, self.max_seq_len)
        pred_dataset = LeftPaddedDataset(pred_seqs, item2fm_ix,
            self.max_seq_len, mode='pred')

        rnn_dataset = RNNDataset(
          train_data=train_seqs,
          train_dataset=train_dataset,
          val_data=val_seqs,
          val_dataset=val_dataset,
          pred_data=pred_seqs,
          pred_dataset=pred_dataset,
          train_interactions=train_data,
          test_interactions=test_data,
          item_vocab=item2fm_ix,
          user_vocab=user2fm_ix,
          embs=self.item_embs,
        )
        print('Done.')
        return rnn_dataset