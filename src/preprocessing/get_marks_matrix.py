import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, save_npz


class SparseMatrixMaker:
    """
    A class that filters the marks table by
    - Deleting old marks
    - Deleting users with few marks,
    then encondes the user and work ids, constructs the sparse matrix
    from the filtered dataframe and saves encodings and the matrix 
    to the disk.
    """
    def __init__(self, n_last_years: int = 5, min_marks: int = 20, marks_df_path='data/raw/work_marks.csv.gz'):
        self.n_last_years = n_last_years
        self.min_marks = min_marks
        self.marks_df = pd.read_csv(marks_df_path, parse_dates=['date'])

    def print_stats(self):
        n_marks = len(self.marks_df)
        n_work_ids = self.marks_df.work_id.nunique()
        n_user_ids = self.marks_df.user_id.nunique()
        print(f'Marks: {n_marks}\tUnique titles: {n_work_ids}\tUnique users: {n_user_ids}\n')

    def filter_by_date(self):
        """
        Filter the dataframe, leaving only marks from n last years.
        """
        print('Stats before filtering by date:')
        self.print_stats()

        date_thresh = pd.Timestamp.today().date() - \
        pd.tseries.offsets.YearBegin() - \
        pd.DateOffset(years=self.n_last_years)
        print(f'Deleting marks dated before {date_thresh.year}...')
        self.marks_df = self.marks_df.query('date >= @date_thresh')

        print(f'Stats after filtering by date:')
        self.print_stats()

    def filter_by_marks_count(self):
        """
        Filter the dataframe, leaving only users with more than `min_marks` marks.
        """
        print(f'Deleting users with less than {self.min_marks} marks...')
        marks_count_by_user = self.marks_df.groupby('user_id').mark.count()
        users_with_enough_marks = marks_count_by_user.loc[lambda n_marks: n_marks >= self.min_marks].index.tolist()
        marks_df = self.marks_df.query('user_id in @users_with_enough_marks')

        print(f'Stats after filtering:')
        print(f'Marks: {len(marks_df)}\tUnique titles: {marks_df.work_id.nunique()}\tUnique users: {marks_df.user_id.nunique()}')

        # marks_df.to_csv(result_file_name, index=False, compression='zip')

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




