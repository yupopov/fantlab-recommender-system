from typing import Dict
import dill

import torch
from numpy import asarray
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from tqdm.notebook import tqdm

from .rnn_recommender import RecurrentRecommender


class Trainer:
    def __init__(self, config: Dict):
        """
        Fits end evaluates given model with Adam optimizer.
         Hyperparameters are specified in `config`
        Possible keys are:
            - n_epochs: number of epochs to train
            - lr: optimizer learning rate
            - weight_decay: l2 regularization weight
            - device: on which device to perform training ("cpu" or "cuda")
            - verbose: whether to print anything during training
        :param config: configuration for `Trainer`
        """
        self.config = config
        self.n_epochs = config["n_epochs"]
        opt_state_dict = config.get('opt', None) # get the optimizer state dict, if present
        # self.setup_opt_fn = config.get('optimizer_fn',
        #     lambda model: Adam(model.parameters(),
        #                         config["lr"],
        #                         weight_decay=config["weight_decay"])
        # )
        self.opt_cls = config.get('optimizer_cls', torch.optim.Adam)
        self.opt_params = config.get('optimizer_params', {})
        # self.setup_scheduler_fn = config.get('scheduler_fn', 
        #     lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=1, verbose=False)
        # )
        self.scheduler_cls = config.get('scheduler_cls', torch.optim.lr_scheduler.ReduceLROnPlateau)
        self.scheduler_params = config.get('scheduler_params', {})
        self.model = None
        self.opt = None
        self.scheduler = None
        self.history = None
        self.loss_fn = CrossEntropyLoss()
        self.device = config["device"]
        self.verbose = config.get("verbose", True)

    def fit(self, model, train_loader, val_loader):
        """
        Fits model on training data, each epoch evaluates on validation data
        :param model: PyTorch model
        :param train_loader: DataLoader for training data
        :param val_loader: DataLoader for validation data
        :return:
        """
        self.model = model.to(self.device)
        if self.opt is None:
            # self.opt = self.setup_opt_fn(self.model) #.to(self.device)
            self.opt = self.opt_cls(self.model.parameters(), **self.opt_params)
        if self.scheduler is None:
            # self.scheduler = self.setup_scheduler_fn(self.opt) #.to(self.device)
            self.scheduler = self.scheduler_cls(self.opt, **self.scheduler_params)

        # self.history = {"train_loss": [], 'train_acc': [], "val_loss": [], "val_acc": []}
        self.history = {"train_loss": [], "val_loss": []}
        for epoch in range(self.n_epochs):
            print(f'Training epocn {epoch}')
            train_info = self._train_epoch(train_loader)
            val_info = self._val_epoch(val_loader)
            self.history["train_loss"].append(train_info["epoch_train_loss"])
            # self.history["train_acc"].append(train_info["epoch_train_acc"])
            self.history["val_loss"].append(val_info["epoch_val_loss"])
            # self.history["val_acc"].append(val_info["epoch_val_acc"])
        return self.model.eval()

    def _train_epoch(self, train_loader):
        self.model.train()
        if self.verbose:
            train_loader = tqdm(train_loader)
        
        running_train_loss = 0.
        running_train_acc = 0.

        for batch in train_loader:
            self.model.zero_grad()
            texts, labels = batch

            texts = texts.to(self.device)
            labels = labels.to(self.device)

            preds = self.model.forward(texts)
            # a dirty hack so that one can compute cross entropy,
            # perhaps better to rewrite the model
            # output itself so it's of the right size
            preds = preds.permute((0, 2, 1))
            loss = self.loss_fn(preds, labels)

            loss.backward()
            self.opt.step()

            preds_class = preds.argmax(dim=1)
            # acc = (preds_class == labels.data).float().mean().data.item()             

            running_train_loss += loss.item()
            # running_train_acc += acc

        epoch_train_loss = running_train_loss / len(train_loader)
        # epoch_train_acc = running_train_acc / len(train_loader)
        if self.verbose:
            # print(f'Train loss = {epoch_train_loss:.3}, Train acc = {epoch_train_acc:.3}')
            print(f'Train loss = {epoch_train_loss:.3}')
            # train_loader.set_description(f"Epoch train loss={epoch_train_loss:.3}; Epoch train acc:{epoch_train_acc:.3}")

        # return {"epoch_train_acc": epoch_train_acc, "epoch_train_loss": epoch_train_loss}
        return {"epoch_train_loss": epoch_train_loss}


    def _val_epoch(self, val_loader):
        self.model.eval()
        # all_logits = []
        # all_labels = []
        if self.verbose:
            val_loader = tqdm(val_loader)

        running_val_loss = 0.
        # running_val_acc = 0.

        with torch.no_grad():
            for batch in val_loader:
                texts, labels = batch

                texts = texts.to(self.device)
                labels = labels.to(self.device)

                preds = self.model.forward(texts)
                preds = preds.permute((0, 2, 1))
                loss = self.loss_fn(preds, labels)

                # preds_class = preds.argmax(dim=1)
                # acc = (preds_class == labels.data).float().mean().data.item()             

                running_val_loss += loss.item()
                # running_val_acc += acc
                # all_logits.append(logits)
                # all_labels.append(labels)
                
        # all_labels = torch.cat(all_labels).to(self.device)
        # all_logits = torch.cat(all_logits)
        # loss = CrossEntropyLoss()(all_logits, all_labels).item()
        # acc = (all_logits.argmax(1) == all_labels).float().mean().item()

        epoch_val_loss = running_val_loss / len(val_loader)
        # epoch_val_acc = running_val_acc / len(val_loader)

        self.scheduler.step(metrics=epoch_val_loss)
        if self.verbose:
            # print(f'Val loss = {epoch_val_loss:.3}, Val acc = {epoch_val_acc:.3}')
            print(f'Val loss = {epoch_val_loss:.3}')
            # val_loader.set_description(f"Epoch val loss={epoch_val_loss:.3}; Epoch val acc:{epoch_val_acc:.3}")
        # return {"epoch_val_acc": epoch_val_acc, "epoch_val_loss": epoch_val_loss}
        return {"epoch_val_loss": epoch_val_loss}

    def predict(self, test_loader):
        if self.model is None:
            raise RuntimeError("You should train the model first")
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in test_loader:
                texts, labels = batch
                logits = self.model.forward(texts.to(self.device))
                predictions.extend(logits.argmax(1).tolist())
        return asarray(predictions)

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("You should train the model first")
        # delete the unserializable elements from config
        # (optimizer and scheduler constructors)
        # we won't need them later, because the optimizer
        # and the scheduler are recreated from checkpoint 
        # for key in 'optimizer_fn', 'scheduler_fn':
        #     del self.config[key]
        checkpoint = {"config": self.model.config,
                      "trainer_config": self.config,
                      "vocab": self.model.vocab.state_dict(),
                      "emb_matrix": self.model.emb_matrix,
                      'optimizer_cls': self.opt.__class__,
                      'scheduler_cls': self.scheduler.__class__,
                      "opt": self.opt.state_dict(),
                      "scheduler": self.scheduler.state_dict(),
                      "model": self.model.state_dict()}
        torch.save(checkpoint, path, pickle_module=dill)

    @classmethod
    def load(cls, path: str):
        ckpt = torch.load(path)
        keys = ["config", "trainer_config", "vocab", "emb_matrix", "opt", 'scheduler', 'model']
        for key in keys:
            if key not in ckpt:
                raise RuntimeError(f"Missing key {key} in checkpoint")

        new_model = RecurrentRecommender(ckpt["config"], ckpt["vocab"], ckpt["emb_matrix"])
        new_model.load_state_dict(ckpt["model"])

        new_trainer = cls(ckpt["trainer_config"])
        new_trainer.model = new_model
        new_trainer.model.to(new_trainer.device)

        # new_opt = new_trainer.setup_opt_fn(new_trainer.model)
        new_opt = new_trainer.opt_cls(new_trainer.model.parameters(),
            **ckpt['trainer_config'].get('opt_params', {}))
        new_opt.load_state_dict(ckpt['opt'])
        new_trainer.opt = new_opt

        # new_scheduler = new_trainer.setup_scheduler_fn(new_trainer.opt)
        new_scheduler = new_trainer.scheduler_cls(new_trainer.opt,
            **ckpt['trainer_config'].get('scheduler_params', {}))
        new_scheduler.load_state_dict(ckpt['scheduler'])
        new_trainer.scheduler = new_scheduler

        return new_trainer