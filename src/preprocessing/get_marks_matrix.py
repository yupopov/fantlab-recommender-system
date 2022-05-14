import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


def print_stats(marks_df):
        n_marks = len(marks_df)
        n_work_ids = marks_df.work_id.nunique()
        n_user_ids = marks_df.user_id.nunique()
        print(f'Marks: {n_marks}\tUnique titles: {n_work_ids}\tUnique users: {n_user_ids}\n')


def filter_by_marks_count_work(marks_df, min_marks_work: int = 50):
    """
    Filter the dataframe, leaving only works with more than `min_marks_work` marks.
    """
    print(f'Deleting works with less than {min_marks_work} marks...')
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
    print(f'Deleting users with less than {min_marks_user} marks...')
    marks_count_by_user = marks_df.groupby('user_id').mark.count()
    users_with_enough_marks = marks_count_by_user.loc[lambda n_marks: n_marks >= min_marks_user].index.tolist()
    marks_df = marks_df.query('user_id in @users_with_enough_marks')

    print(f'Stats after filtering:')
    print_stats(marks_df)
    return marks_df

class SparseMatrixMaker:
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
                marks_df_path='data/raw/work_marks.csv.gz'):
        self.n_last_years = n_last_years
        # self.min_marks_work = min_marks_work
        # self.min_marks_user = min_marks_user
        self.marks_df = pd.read_csv(marks_df_path, parse_dates=['date'])

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
                            min_marks_work_test: int = 10
                            ):
        self.filter_by_date()
        split_date = self.marks_df.date.quantile(time_q)
        print(f'Splitting the marks by {split_date}...')
        marks_df_train, marks_df_test = \
            self.marks_df.query('date <= @split_date'), self.marks_df.query('date <= @split_date')
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
        # Works features can change that, perhaps
        print(f'Dropping works with less than {min_marks_work_test} in the test set...')
        marks_df_test = filter_by_marks_count_work(marks_df_test,
                                                  min_marks_work_test)

        # Now it's time to construct train and test datasets
        # Have to see if lightfm can do that for me with Dataset class
        users = marks_df_train.user_id.unique().tolist()
        
        
        

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




