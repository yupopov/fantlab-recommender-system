import json
import os
from dataclasses import dataclass
import gzip

import numpy as np
import pandas as pd
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

@dataclass
class FMDataset:
    train_data: coo_matrix
    train_weights: coo_matrix
    test_data: coo_matrix
    work_features: csr_matrix
    dataset: Dataset


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
                            min_marks_work_test: int = 10,
                            marks_transform: str = 'decoupling',
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
        print(f'Dropping users with less that {min_marks_user_train} marks in the train set...')
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
        print(f'Dropping works with less than {min_marks_work_test} marks in the test set...')
        marks_df_test = filter_by_marks_count_work(marks_df_test,
                                                  min_marks_work_test)

        # Add interaction weights
        print(f'Computing train mark weights...')
        mark_transform = mark_transforms_dict[marks_transform]
        marks_df_train['mark_weight'] = \
            marks_df_train.groupby('user_id').mark.transform(mark_transform)
        print('Computing date weights...')
        date_transform = transform_dates # add hyperparameters here!
        marks_df_train = transform_dates(marks_df_train)

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




