import torch
from torch import nn
from torch.nn.functional import sigmoid, relu, elu, tanh
from torch.nn import Module, Embedding, LSTM, RNN, GRU, Linear, Sequential, Dropout, \
    CrossEntropyLoss
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

cell_types = {
            "RNN": RNN,
            "GRU": GRU,
            "LSTM": LSTM
            }

class RecurrentRecommender(Module):
    def __init__(self, config: dict, vocab: dict, embs: torch.Tensor):
        super().__init__()
        self.vocab = vocab
        self.config = config
        self.embs = Embedding.from_pretrained(
          embs, freeze=config.get('freeze_embs', True), 
          padding_idx=len(vocab)-1)
        cell_class = cell_types[config["cell_type"]]
        self.cell = cell_class(input_size=embs.size(1),
                               batch_first=True,
                               hidden_size=config["hidden_size"],
                               num_layers=config["num_layers"],
                               dropout=config["cell_dropout"],
                               bidirectional=False,
                               )
        activation_types = {
            "sigmoid": sigmoid,
            "tanh": tanh,
            "relu": relu,
            "elu": elu,
        }
        self.out_activation = activation_types[config["out_activation"]]
        self.out_dropout = Dropout(config["out_dropout"])
        # cur_out_size = config["hidden_size"] * config["num_layers"]
        cur_out_size = config["hidden_size"] 
        # if config["bidirectional"]:
        #     cur_out_size *= 2
        out_layers = []
        for cur_hidden_size in config["out_sizes"]:
            out_layers.append(Linear(cur_out_size, cur_hidden_size))
            cur_out_size = cur_hidden_size
        out_layers.append(Linear(cur_out_size, len(self.vocab)))
        self.out_proj = Sequential(*out_layers)

    def forward(self, input):
        embedded = self.embs(input)
        hidden_states, _ = self.cell(embedded)
        # if isinstance(last_state, tuple):
        #     last_state = last_state[0]
        # last_state = last_state.transpose(0, 1)
        # last_state = last_state.reshape(last_state.size(0), -1)
        self.out_activation
        self.out_dropout
        return self.out_proj(hidden_states)
        # return predicts


        