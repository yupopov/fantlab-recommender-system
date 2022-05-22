from sklearn.preprocessing import normalize

from scipy.sparse import coo_matrix, csr_matrix, csc_matrix

class LinearRecommender:
    """
    A recommender system that assumes linearity of user-item interactions
    (that is, computes user preferences during the train period
    as the sum of embeddings of the items with which user interacted
    during test period with weights given in `weights` matrix,
    and computes item similariy as cosine similarity of items' embeddings)
    """
    def __init__(self, fm_dataset,
                 item_embs,
                 item2emb_ix: dict,
                 normalize_embs=True):
        self.weights = fm_dataset.train_weights.tocsr() # (n_users, n_train_items)
        self.item_embs = item_embs # (n_items, n_item_features)
        self.item2fm_ix = fm_dataset.dataset.mapping()[2] # item id: fm_dataset row num
        self.item2emb_ix = {int(key): value for key, value in item2emb_ix.items()} # item_id: emb row num
        # `weights` columns and `item_embs` rows
        # correspond to different item ids (may be even of different lengths),
        # so leave only columns corresponding to item ids in `weights`
        # in the same order as in `weights`
        self.permute_embs()
        self.normalize_embs = normalize_embs
        # normalize embeddings
        if self.normalize_embs:
            self.item_embs = normalize(self.item_embs, norm='l2', axis=1)
        self.item_embs = csc_matrix(self.item_embs)

    def permute_embs(self):
        # we need: emb row num: fm_dataset row num
        fm_ix2item = {value: key for key, value in self.item2fm_ix.items()} # fm_dataset row num: item_id
        fm_ix2emb_ix = {key: self.item2emb_ix[value] for key, value in fm_ix2item.items()} # fm_dataset row num: emb row num
        emb_ix2fm_ix = {value: key for key, value in fm_ix2emb_ix.items()} # emb row num: fm_dataset row num
        emb_ix2fm_ix = {key: value for key, value in sorted(emb_ix2fm_ix.items(), key=lambda x: x[1])}
        emb_ix_perm = list(emb_ix2fm_ix.keys())
        self.item_embs = self.item_embs[emb_ix_perm] # item_embs: (n_train_items, n_item_features)

    def predict(self, user_ids):
        # sum item embeddings with corresponing interaction weights
        # for each user to obtain vectors that aggregate user interactions
        # and normalize these vectors
        user_agg_vectors = self.weights[user_ids] @ self.item_embs # (n_users, n_item_features)
        if self.normalize_embs:
            user_agg_vectors = normalize(user_agg_vectors,
                                        norm='l2',
                                        axis=1) # (n_users, n_item_features)
        # compute cosine similarity of aggregated vectors
        # with item vectors (we interpret cosine similarity of
        # two feature vectors as degree of their items similarity)
        predictions = user_agg_vectors @ self.item_embs.T
        return predictions