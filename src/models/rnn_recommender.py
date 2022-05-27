import numpy as np
import torch
from torch import nn
from torch.nn.functional import sigmoid, relu, elu, tanh
from torch.nn import Module, Embedding, LSTM, RNN, GRU, Linear, Sequential, Dropout, \
    CrossEntropyLoss, Sigmoid, Tanh, ReLU, ELU, BatchNorm1d
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.cuda

from src.preprocessing.datasets import LeftPaddedDataset
from src.models.get_top_k_predictions_with_label import get_top_k_predictions_with_labels
from src.models.trainer import Trainer


cell_types = {
            "RNN": RNN,
            "GRU": GRU,
            "LSTM": LSTM
            }

activation_types = {
            "sigmoid": Sigmoid(),
            "tanh": Tanh(),
            "relu": ReLU(),
            "elu": ELU(),
        }


class RecurrentLanguageModel(Module):
    """
    A recurrent network that tries to predict next 
    item in a sequence.
    """
    def __init__(self, config: dict, vocab: dict, embs: torch.Tensor):
        super().__init__()
        self.vocab = vocab
        self.config = config
        self.embs = Embedding.from_pretrained(
          embs, freeze=config.get('freeze_embs', True), 
          padding_idx=len(vocab)-1)
        # By RNNDataset construction, padding is the last
        # row in the embedding and consists of zeros
        cell_class = cell_types[config["cell_type"]]
        self.cell = cell_class(input_size=embs.size(1),
                               batch_first=True,
                               hidden_size=config["hidden_size"],
                               num_layers=config["num_layers"],
                               dropout=config["cell_dropout"],
                               bidirectional=False,
                               )
        self.out_activation = activation_types[config["out_activation"]]
        self.out_dropout = Dropout(config["out_dropout"])
        cur_out_size = config["hidden_size"] 
        # cur_out_size = config["hidden_size"] * config["num_layers"]
        # if config["bidirectional"]:
        #     cur_out_size *= 2
        out_layers = []
        for cur_hidden_size in config["out_sizes"]:
            layer = Sequential(
                Linear(cur_out_size, cur_hidden_size),
                self.out_activation,
                self.out_dropout
            )
            # out_layers.append(Linear(cur_out_size, cur_hidden_size))
            out_layers.append(layer)
            cur_out_size = cur_hidden_size
        out_layers.append(Linear(cur_out_size, len(self.vocab)))
        self.out_proj = Sequential(*out_layers)

    def forward(self, input):
        # input: (batch_size, max_seq_len)
        embedded = self.embs(input)
        # embedded: (batch_size, max_seq_len, self.embs.shape[1])
        hidden_states, _ = self.cell(embedded)
        # hidden_states: (batch_size, max_seq_len, self.cell.hidden_size)
        # Now pass each hidden state through a sequence
        # of dense layers independently
        return self.out_proj(hidden_states) # (batch_size, max_seq_len, len(self.vocab))
        # return predicts


class RecurrentRecommender(Module):
    """
    A wrapper over RecurrentLanguageModel
    """
    def __init__(self, lm: RecurrentLanguageModel,
        pred_dataset: LeftPaddedDataset, #user_vocab: dict
        ):
        super().__init__()
        self.lm = lm
        self.device = 'cuda' if next(self.lm.parameters()).is_cuda else 'cpu'
        self.pred_dataset = pred_dataset
        # self.user_vocab = user_vocab

    def forward(self, user_ids: list):
        seq_batch = self.pred_dataset.collate_fn(
          [self.pred_dataset[user_id] for user_id in user_ids]
        ).to(self.device) # seq_batch: (len(user_ids), self.pred_dataset.max_seq_len)
        batch_preds = self.lm(seq_batch)
        # batch_preds contains all hidden states of the last RNN layer
        # batch_preds: (len(user_ids), self.pred_dataset.max_seq_len, len(self.model.vocab))
        # Leave the last hidden state of the last layer
        # We use it to predict the next item in the sequence
        # The last index in the last dimension corresponds to padding,
        # so drop it as well
        batch_preds = batch_preds[:, -1, :-1].detach().cpu().numpy()
        # batch_preds: (len(user_ids), len(self.model.vocab))
        return batch_preds







        