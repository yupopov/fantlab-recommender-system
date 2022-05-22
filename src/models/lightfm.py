import numpy as np

from tqdm.auto import tqdm
# from lightfm.evaluation import precision_at_k
from src.models import get_top_k_predictions_with_label

def precision(true: np.array, predicted: np.array):
    return len(np.intersect1d(true, predicted)) / len(predicted)


def recall(true: np.array, predicted: np.array):
    return len(np.intersect1d(true, predicted)) / len(true)

'''
This is test handmade precision_at_k function.
It is not used in the model fitting

def precision_at_k(model, test_interactions, train_interactions=None,
    k=10, user_features=None, item_features=None, num_threads=2):
    test_interactions_csr = test_interactions.tocsr()
    if train_interactions is not None:
        train_interactions_csr = train_interactions.tocsr()

    num_items = test_interactions.shape[1]
    item_ids = np.arange(num_items)
    user_ids, _ = test_interactions_csr.nonzero()
    user_ids = np.unique(user_ids).tolist()

    precisions = []

    for user in tqdm(user_ids):
        user_predicts = model.predict(
          user, item_ids,
          user_features=user_features, item_features=item_features,
          num_threads=num_threads
        )
        if train_interactions is not None:
            user_train_interactions = train_interactions_csr[user].toarray().\
                flattten().astype(bool)
            user_predicts = np.where(
              ~user_train_interactions, user_predicts, -np.Inf
              )
        top_k_predictions = np.argsort(user_predicts)[-1:-k+1:-1]

        user_test_interactions = test_interactions_csr[user].toarray().\
            flatten().nonzero()[0]
        user_precision = precision(user_test_interactions, top_k_predictions)
        precisions.append(user_precision)
    
    precisions = np.array(precisions)
    return precisions
'''

'''
Moved to the separate module. Saved here as a working copy just in case.

def get_top_k_predictions_with_labels(model, test_interactions, train_interactions=None,
    k=10, user_features=None, item_features=None, batch_size=50, num_threads=2):
    """
    Get indices of top k predictions (ordered) for test interactions
    and their labels (whether users interacted with them or not)
    """
    test_interactions_csr = test_interactions.tocsr()
    if train_interactions is not None:
        train_interactions_csr = train_interactions.tocsr()

    num_items = test_interactions.shape[1]
    item_ids = np.arange(num_items)
    user_ids, _ = test_interactions_csr.nonzero()
    user_ids = np.unique(user_ids)

    top_k_predictions = []
    labels = []

    for batch_start in tqdm(range(0, len(user_ids), batch_size)):
        user_batch = user_ids[batch_start: batch_start + batch_size]
        batch_len = len(user_batch)

        if model.__class__.__name__ == 'LightFM':
            user_ids_batch = np.repeat(user_batch, repeats=num_items)
            item_ids_batch = np.repeat(
                item_ids.reshape(-1, 1),repeats=batch_len, axis=1).T.flatten()
            batch_predicts = model.predict(
                user_ids_batch, item_ids_batch,
                user_features=user_features, item_features=item_features,
                num_threads=num_threads
                ).reshape(batch_len, num_items)
        elif model.__class__.__name__ == 'LinearRecommender':
            batch_predicts = model.predict(user_batch).toarray()
            # print(batch_predicts.shape)
        if train_interactions is not None:
            batch_train_interactions = train_interactions_csr[user_batch].\
                toarray().astype(bool)
            batch_predicts = np.where(
              ~batch_train_interactions, batch_predicts, -np.Inf
              )
        # batch_top_k_predictions = np.argsort(-batch_predicts, axis=1)[:, :k]
        # batch_top_k_pred_indices = np.argsort(-batch_predicts, axis=1)[:, :k]
        # batch_top_k_pred_indices = np.argpartition(-batch_predicts, range(k), axis=1)[:, :k]
        # get the indices of top k predicted items for each user in batch
        # (yeah, i know, looks like mumbo-jumbo, but it's the fastest way,
        # see https://stackoverflow.com/questions/42184499/cannot-understand-numpy-argpartition-output/42186357#42186357)
        # permute the indices so that k top predictions for each user
        # are in the first k positions (but unsorted themselves)
        # and all other indices
        # print(type(batch_predicts))
        batch_top_k_pred_indices_unsorted = np.argpartition(-batch_predicts, k, axis=1)[:, :k]
        # apply the permutation to predictions row-by-row
        # so that we have top k predicted values for each user
        # (still unsorted)
        # actually, this alone is enough if you want to compute 
        # precision or recall, because they don't care about
        # the order of top k predicted items, but we hope 
        # to calculate average precision@k in that manner as well sometimes
        batch_predicts_top_k_unsorted = np.take_along_axis(
            batch_predicts, batch_top_k_pred_indices_unsorted, axis=1
            )
        # now sort the remaining k predicted values
        # and sort top k indices again
        batch_top_k_pred_indices = np.take_along_axis(
            batch_top_k_pred_indices_unsorted,
            np.argsort(batch_predicts_top_k_unsorted), axis=1
        )
        batch_test_interactions = test_interactions_csr[user_batch].toarray()
        # sort the test interactions so that the items with indices
        # of top k predicted elements for each user appear first
        batch_test_interactions_sorted = np.take_along_axis(
            batch_test_interactions, batch_top_k_pred_indices, axis=1
        )

        top_k_predictions.extend(batch_top_k_pred_indices.tolist())
        labels.extend(batch_test_interactions_sorted.tolist())

        # batch_precisions = batch_test_interactions_sorted.sum(axis=1) / k
        # precisions.extend(batch_precisions.tolist())

        # return batch_pred_ranks, batch_test_interactions

        # for test_interactions, top_k_predictions in \
        #     zip(batch_test_interactions, batch_top_k_predictions):
        #     user_test_interactions = test_interactions.nonzero()[0]
        #     user_precision = precision(user_test_interactions, top_k_predictions)
        #     precisions.append(user_precision)
    
    # precisions = np.array(precisions)
    top_k_predictions = np.array(top_k_predictions)
    labels = np.array(labels)
    return top_k_predictions, labels
'''


def my_precision_at_k(model, test_interactions, train_interactions=None,
    k=10, user_features=None, item_features=None, batch_size=50, num_threads=2):

    '''
    Calculates and returns precision@k metric faster than the LightFM's builtin
    '''

    test_interactions_csr = test_interactions.tocsr()
    if train_interactions is not None:
        train_interactions_csr = train_interactions.tocsr()

    num_items = test_interactions.shape[1]
    item_ids = np.arange(num_items)
    user_ids, _ = test_interactions_csr.nonzero()
    user_ids = np.unique(user_ids)

    precisions = []

    for batch_start in tqdm(range(0, len(user_ids), batch_size)):
        user_batch = user_ids[batch_start: batch_start + batch_size]
        batch_len = len(user_batch)

        user_ids_batch = np.repeat(user_batch, repeats=num_items)
        item_ids_batch = np.repeat(
            item_ids.reshape(-1, 1),repeats=batch_len, axis=1).T.flatten()
        batch_predicts = model.predict(
            user_ids_batch, item_ids_batch,
            user_features=user_features, item_features=item_features,
            num_threads=num_threads
            )
        batch_predicts = batch_predicts.reshape(batch_len, num_items)
        if train_interactions is not None:
            batch_train_interactions = train_interactions_csr[user_batch].\
                toarray().astype(bool)
            batch_predicts = np.where(
              ~batch_train_interactions, batch_predicts, -np.Inf
              )
        # batch_top_k_predictions = np.argsort(-batch_predicts, axis=1)[:, :k]
        # batch_top_k_pred_indices = np.argsort(-batch_predicts, axis=1)[:, :k]
        # batch_top_k_pred_indices = np.argpartition(-batch_predicts, range(k), axis=1)[:, :k]
        # get the indices of top k predicted items for each user in batch
        # (yeah, i know, looks like mumbo-jumbo, but it's the fastest way,
        # see https://stackoverflow.com/questions/42184499/cannot-understand-numpy-argpartition-output/42186357#42186357)
        # permute the indices so that k top predictions for each user
        # are in the first k positions (but unsorted themselves)
        # and all other indices
        batch_top_k_pred_indices_unsorted = np.argpartition(-batch_predicts, k, axis=1)[:, :k]
        # apply the permutation to predictions row-by-row
        # so that we have top k predicted values for each user
        # (still unsorted)
        # actually, this alone is enough if you want to compute 
        # precision or recall, because they don't care about
        # the order of top k predicted items, but we hope 
        # to calculate average precision@k in that manner as well sometimes
        batch_predicts_top_k_unsorted = np.take_along_axis(
            batch_predicts, batch_top_k_pred_indices_unsorted, axis=1
            )
        # now sort the remaining k predicted values
        # and sort top k indices again
        batch_top_k_pred_indices = np.take_along_axis(
            batch_top_k_pred_indices_unsorted,
            np.argsort(batch_predicts_top_k_unsorted), axis=1
        )
        batch_test_interactions = test_interactions_csr[user_batch].toarray()
        # sort the test interactions so that the items with indices
        # of top k predicted elements for each user appear first
        batch_test_interactions_sorted = np.take_along_axis(
            batch_test_interactions, batch_top_k_pred_indices, axis=1
        )

        batch_precisions = batch_test_interactions_sorted.sum(axis=1) / k
        precisions.extend(batch_precisions.tolist())

        # return batch_pred_ranks, batch_test_interactions

        # for test_interactions, top_k_predictions in \
        #     zip(batch_test_interactions, batch_top_k_predictions):
        #     user_test_interactions = test_interactions.nonzero()[0]
        #     user_precision = precision(user_test_interactions, top_k_predictions)
        #     precisions.append(user_precision)
    
    precisions = np.array(precisions)
    return precisions


# fit function
def fit_lightfm(model, fm_dataset, use_item_features=True,
    fit_params: dict = {}, precision_params: dict={}):

    item_features = fm_dataset.work_features if use_item_features else None

    model.fit(interactions=fm_dataset.train_data,
              item_features=item_features,
              sample_weight=fm_dataset.train_weights,
              **fit_params)
    
    # train_precision = my_precision_at_k(model, fm_dataset.train_data,
    #                                  item_features=item_features,
    #                                  **precision_params).mean()
    # test_precision = my_precision_at_k(model, fm_dataset.test_data,
    #                                 train_interactions=fm_dataset.train_data,
    #                                 item_features=item_features, 
    #                                 **precision_params).mean()

    # print(f'Train precision: {train_precision:.4f}')
    # print(f'Test precision: {test_precision:.4f}')

    print('Computing top k recommendations for each user...')
    print('Train set:')
    train_top_k_preds, train_labels = get_top_k_predictions_with_labels(
        model, fm_dataset.train_data,
        item_features=item_features,
        **precision_params)
    print('Test set:')
    test_top_k_preds, test_labels = get_top_k_predictions_with_labels(
        model, fm_dataset.test_data,
        train_interactions=fm_dataset.train_data,
        item_features=item_features,
        **precision_params)

    train_precision = train_labels.sum(axis=1).mean() / precision_params['k']
    test_precision = test_labels.sum(axis=1).mean() / precision_params['k']
    
    print(f'Train precision: {train_precision:.4f}')
    print(f'Test precision: {test_precision:.4f}')

    return model, train_top_k_preds, test_top_k_preds