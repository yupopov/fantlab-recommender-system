from lightfm.evaluation import precision_at_k

def fit_lightfm(model, fm_dataset, fit_params: dict = {}):
    model.fit(interactions=fm_dataset.train_data,
              item_features=fm_dataset.work_features,
              sample_weight=fm_dataset.train_weights,
              **fit_params)
    
    train_precision = precision_at_k(model, fm_dataset.train_data,
                                     item_features=fm_dataset.work_features,
                                     k=10).mean()
    test_precision = precision_at_k(model, fm_dataset.test_data,
                                    train_interactions=fm_dataset.train_data,
                                    item_features=fm_dataset.work_features,
                                    k=10).mean()

    print(f'Train precision: {train_precision:.4f}')
    print(f'Test precision: {test_precision:.4f}')
    