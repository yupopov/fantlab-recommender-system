from functools import partial

import numpy as np
import pandas as pd


"""
See mark_weights.py for general discussion of interaction weights.
It is clear that newer interactions (marks, in our case) should get
higher weights than older ones. Here is the function that does that.
The general algorithm is as follows:
1) min-max normalize mark dates (`user` corresponds to normalizing user-by-user,
`global` corresponds to normalizing by max and min dates in the whole set);
2) map normalized timestamps to [eps_time, 1] (the oldest interactions 
thus get weight eps_time);
3) optionally apply some nonlinear transformation to the weights
(we use power transforms)
"""

def transform_dates(marks_df: pd.DataFrame,
                            min_max_normalize='user',
                            eps_time: float = 0.2,
                            power: float = 1
                            ):
    # going with personal min and max timestamps for now
    # using absolute timestamps is also a valid option?
    # min_ts = marks_df.date.min()
    # max_ts = marks_df.date.max()
    
    # get min and max dates
    if min_max_normalize == 'user':
        user_min_max_date = marks_df.groupby('user_id').date.agg(['min', 'max']).\
            rename(columns={'min': 'min_date', 'max': 'max_date'})
        marks_df = marks_df.merge(user_min_max_date, on='user_id')
    elif min_max_normalize == 'global':
        marks_df['min_date'] = marks_df.date.min()
        marks_df['max_date'] = marks_df.date.max()
    else:
        raise ValueError('Unknown normalization mode')

    # min-max normalize and map to [eps_time, 1]
    marks_df['time_weight'] = (marks_df['date'] - marks_df['min_date']) / \
        (marks_df['max_date'] - marks_df['min_date'] + pd.Timedelta(1, 'D'))
    marks_df['time_weight'] = (marks_df['time_weight'] + eps_time) / \
        (1 + eps_time)
    
    marks_df['time_weight'] = marks_df['time_weight'] ** power

    marks_df.drop(columns=['min_date', 'max_date'], inplace=True)

    return marks_df