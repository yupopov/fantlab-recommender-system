import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    

"""
Matrix factorization methods (such as LightFM that we're using here)
only allow implicit feedback (that is, a training matrix is essentially
binary, where 1 means 'there was an interaction between user u and item i'.
However, in this project we mostly deal with explicit feedback (ratings).
But it is intuitively clear that different marks mean essentially
different interactions (consider the case of marks 1 and 10).
One of the ways to incorporate the information about the marks 
into a factorization model is providing an interactions weight matrix
(of the same form as the training matrix) that will scale gradient updates.

Intuition tells us that it may be reasonable to assign 
negative weights to low marks. However by now,
we are not sure whether matrix factorization methods
(such as LightFM) can accept negative interaction weights, and to what
behavior during training it may lead.
So the mark weights have to: be positive (1).

In an ideal world, user ratings on a scale of 1 to 10
are normally distributed with the mean of 5, and 
a linear increase in rating means a linear increase in the 
user internal preferences. However, it is obviously not the case
in the real world: although we can be more or less sure that
the relationship between rankings and user preferences is monotonic,
the most popular mark is around 7 or 8, 
and marks distributions are heavily skewed to the right
(you can visualize that using `show_marks_transforms`)
So the mark weights have to:
(2) increase monotonously as marks increase;
(3) increase non-linearly, taking into account the marks
distribution for each user.

Below are some mark -> weight transforms that satisfy conditions (1)-(3).
"""

def transform_marks_gaussian(marks: pd.Series, eps=0.001, add_min=True):
    """
    A gaussian linear marks transform
    (doesn't satisfy condition (3), so not used in dataset creation as is)
    """
    marks = (marks - marks.mean()) / (marks.std() + eps)
    # negative ratings?
    if add_min:
        marks -= marks.min()
    # marks_df.marks = marks
    # return marks_df
    return marks


def transform_marks_minmax(marks: pd.Series, eps=0.01):
    """
    A simple min-max linear transform
    (doesn't satisfy condition (3), so not used in dataset creation as is)
    """
    marks -= marks.min()
    marks = marks / marks.max()
    # normalize so that lowest marks get weight `eps`
    return (marks + eps) / (1 + eps)


def transform_marks_decoupling(marks: pd.Series):
    """
    A decoupling rating transform
    (see article "A study of methods for normalizing user ratings in collaborative filtering":
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.3197&rep=rep1&type=pdf)
    """

    # mark_probs = marks.value_counts(normalize=True).sort_index()
    # mark_weights = (np.cumsum(mark_probs) - mark_probs / 2).to_dict()
    # the same stuff in numpy (works faster)
    unique_marks, counts = np.unique(marks, return_counts=True)
    mark_probs = counts / counts.sum()
    mark_weights = mark_probs.cumsum() - mark_probs / 2
    mark_weights = dict(zip(unique_marks, mark_weights))

    return marks.replace(mark_weights)


def transform_marks_sigmoid(marks: pd.Series):
    """
    Take a sigmoid of the gaussian transform of the ratings
    (particularly, mean value mark gets weight 0.5)
    """
    marks = transform_marks_gaussian(marks, add_min=False)
    return sigmoid(marks)


def show_marks_transforms(marks: pd.Series):
    marks_minmax = transform_marks_minmax(marks)
    marks_decoupled = transform_marks_decoupling(marks)
    marks_sigmoid = transform_marks_sigmoid(marks)

    plt.figure(figsize=(10, 8))
    plt.plot(np.sort(marks.unique()), np.sort(marks_minmax.unique()), label='minmax')
    plt.plot(np.sort(marks.unique()), np.sort(marks_sigmoid.unique()), label='sigmoid')
    plt.plot(np.sort(marks.unique()), np.sort(marks_decoupled.unique()), label='decoupled')
    plt.hist(marks, density=True)
    plt.legend()
    plt.grid(True);

mark_transforms_dict={
  'gaussian': transform_marks_gaussian,
  'minmax': transform_marks_minmax,
  'decoupling': transform_marks_decoupling,
  'sigmoid': transform_marks_sigmoid,
}
