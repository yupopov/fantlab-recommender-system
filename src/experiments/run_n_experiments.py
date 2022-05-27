import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from src.models.rnn_recommender import RecurrentLanguageModel, RecurrentRecommender
from src.models.trainer import Trainer

def run_experiment(dataset,
                   net_config: dict,
                   trainer_config: dict,
                   net_class=RecurrentLanguageModel,
                   trainer_class=Trainer,
                   random_seed=17
                   ):

    # initialize random seed
    np.random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True

    # initialize the datasets
    # and the dataloaders


    train_dataset = dataset.train_dataset
    train_dataloader = DataLoader(train_dataset, batch_size=trainer_config['batch_size'], collate_fn=train_dataset.collate_fn, shuffle=True)


    val_dataset = dataset.val_dataset
    val_dataloader = DataLoader(val_dataset, batch_size=trainer_config['batch_size'], collate_fn=train_dataset.collate_fn, shuffle=True)
  
    # initialize the model and the trainer
    model = net_class(net_config, dataset.item_vocab, dataset.embs) 
    trainer = trainer_class(trainer_config)

    trainer.fit(model, train_dataloader, val_dataloader)

    return trainer

def plot_experiments(keys: list, experiments_results: dict, bottom=4, top=8.5, n_experiments=5):
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    for n in range(0, n_experiments):
        avg_val_acc = experiments_results['train_loss_hists'][n]
        ax[0].plot(avg_val_acc)        
        ax[0].set_ylim(bottom=bottom, top=top)
        ax[0].grid(True)
        ax[0].set_xlabel('n_epochs')
        ax[0].set_ylabel('train loss')
        ax[0].set_title(f'Avg train loss vs num of epochs')

        avg_val_loss = experiments_results['val_loss_hists'][n]
        ax[1].plot(avg_val_loss)
        ax[1].set_ylim(bottom=bottom, top=top)
        ax[1].grid(True)
        ax[1].set_xlabel('n_epochs')
        ax[1].set_ylabel('val loss')
        ax[1].set_title(f'Avg val loss vs num of epochs')

    plt.tight_layout();

def run_n_experiments(dataset,
                      net_config: dict,
                      trainer_config: dict,
                      bottom=4,
                      top=8.5,
                      n_experiments=5):
    """
    Run n experiments with the same params
    but different random seeds,
    record the accuracy and loss histories 
    for each run.

    Parameters:
    train_params: dict
    Parameters for the run_experimnet function
    """

    exp_results = defaultdict(list)
    exp_results['train_loss_hists'] = []
    exp_results['val_loss_hists'] = []

    # Run n experiments with different random seeds and save the results
    for i in range(n_experiments):
        print(f'Starting experiment {i}...')
        running_trainer = run_experiment(dataset, net_config, trainer_config, random_seed=i)
     
        exp_results['train_loss_hists'].append(running_trainer.history['train_loss'])
        # exp_results['train_acc_hists'].append(running_trainer.history['train_acc'])
        exp_results['val_loss_hists'].append(running_trainer.history['val_loss'])
        # exp_results['val_acc_hists'].append(running_trainer.history['val_acc'])

        # if save_model:
        #     model = running_trainer.model
        #     optimizer = running_trainer.opt
        #     scheduler = running_trainer.scheduler

        #     exp_results['models'].append(running_trainer.model.state_dict())
        #     exp_results['optimizers'].append(running_trainer.opt.state_dict())
        #     exp_results['schedulers'].append(running_trainer.scheduler.state_dict())

    exp_results['train_loss_hists'] = np.array(exp_results['train_loss_hists'])
    # exp_results['train_acc_hists'] = np.array(exp_results['train_acc_hists'])
    exp_results['val_loss_hists'] = np.array(exp_results['val_loss_hists'])
    # exp_results['val_acc_hists'] = np.array(exp_results['val_acc_hists'])
    
    # Calculate some statistics of the experiments
    exp_results['avg_train_loss_hist'] = exp_results['train_loss_hists'].mean(axis=0)
    # exp_results['avg_train_acc_hist'] = exp_results['train_acc_hists'].mean(axis=0)
    exp_results['avg_val_loss_hist'] = exp_results['val_loss_hists'].mean(axis=0)
    # exp_results['avg_val_acc_hist'] = exp_results['val_acc_hists'].mean(axis=0)
    
    # exavg_accuracy_history =  exp_results['val_acc_hists'].mean(axis=0)

    min_val_losses = exp_results['val_loss_hists'].max(axis=1)
    min_val_loss = min_val_losses.max()
    best_run = min_val_losses.argmax()
    best_epoch =  exp_results['val_loss_hists'][best_run].argmax()

    exp_results['max_accuracy'] = min_val_loss
    exp_results['best_run'] = best_run
    
    # if save_model:
    #     exp_results['best_model'] = exp_results['models'][best_run]
    #     exp_results['best_model_optimizer'] = exp_results['optimizers'][best_run]
    #     exp_results['best_model_scheduler'] = exp_results['schedulers'][best_run]
    #     del exp_results['models'], exp_results['optimizers'], exp_results['scheduler']
    

    # experiment_name = make_experiment_name(experiment_params['net_config'], experiment_params['trainer_config'])
    
    print(f'Ran {n_experiments} experiments.')
    print(f'Min val loss {min_val_loss:.4f} achieved by run {best_run} on epoch {best_epoch}.')
    print(f'Avg min loss: {min_val_losses.mean():.4f}')
    
    print(exp_results['train_loss_hists'])
    plot_experiments(exp_results.keys(), exp_results, bottom=nottom, top=top, n_experiments=n_experiments)
    

    return exp_results