#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import re
import torch

from pathlib import Path

def extract_epoch_step(p):
    match = re.search(r'epoch=(\d+)-step=(\d+)', str(p))
    epoch = int(match.group(1)) if match else -1
    step = int(match.group(2)) if match else -1
    return epoch, step

def get_latest_checkpoint(ckpt_root):
    # Get all checkpoints in ckpt_root
    checkpoint_files = list(Path(ckpt_root).glob("epoch=*.ckpt"))
    
    if not checkpoint_files: return Path('NOT_FOUND')
    
    # Sort by epoch and step
    checkpoint_files.sort(key=extract_epoch_step, reverse=True)
    
    return checkpoint_files[0]


# In[ ]:


import torch

class ModelCheckpoint(object):
    def __init__(self, save_path, mode='max'):
        # Input args
        self.save_path = save_path
        self.mode = mode

        # Other args
        self.best_epoch = 0
        self.best_result = -1 if mode == 'max' else float('inf')

    def can_record(self, result):
        record_max = self.mode == 'max' and result > self.best_result
        record_min = self.mode == 'min' and result < self.best_result
        return record_max or record_min

    def record(self, model, result, epoch):
        if self.save_path is not None and self.can_record(result):
            ## Update records
            self.best_result = result
            self.best_epoch = epoch

            ## Save to disk
            print(f'Saving {epoch=} {result=:0.4f}')
            torch.save(model.state_dict(), self.save_path)


# In[ ]:


import numpy as np

def compute_class_weights(ys):
    _, freqs = np.unique(ys, return_counts=True)
    weights = freqs / len(ys)

    return weights


# In[ ]:


import numpy as np

def compute_partial_weights(ys, num_classes=18):
    counts = np.zeros(num_classes, dtype=int)
    labels, freqs = np.unique(ys, return_counts=True)
    
    # Update counts based on the label frequencies
    for idx, l in enumerate(labels): counts[l] = freqs[idx]

    weights = counts / len(ys)

    return weights

# sample_ys = train_y
# partial_weights = compute_partial_weights(sample_ys)
    
# print(make_header(f'Partial Weights)'))
# print(f'{len(sample_ys)=} {partial_weights}')


# In[ ]:


import torch
import torchmetrics
import statistics
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from collections import defaultdict
from core_utilities import mem_usage

class Trainer(object):
    def __init__(
        self,
        model,
        config,
        class_weights=None,
        batch_weights=False,
    ):
        # Input args
        self.model = model
        self.config = config
        self.class_weights = None if class_weights is None else torch.tensor(
            class_weights,
            dtype=torch.float,
            device=config.device,
            requires_grad=False
        )
        self.batch_weights = batch_weights

        # Metrics
        self.metrics = defaultdict(list)

        # Train and validation metrics
        self.train_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=model.out_channels)
        self.valid_acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=model.out_channels)

    def log(self, name, value):
        self.metrics[name].append(value.item())

    def get_weights(self, ys):
        # If batch weights are enabled, compute weights for the batch
        if self.batch_weights:
            ## Convert tensor to numpy
            ys = ys.cpu().numpy()

            ## Compute weights
            weights = compute_partial_weights(ys, num_classes=self.config.num_classes)

            ## Convert weights to tensor
            weights = torch.tensor(
                weights,
                dtype=torch.float,
                device=self.config.device,
                requires_grad=False
            )
        else:
            weights = self.class_weights

        return weights
        
    def training_step(self, batch, batch_idx):
        # Get X and y
        X, y = batch
        X, y = X.to(self.config.device), y.to(self.config.device)
        
        # Compute logits
        y_pred = self.model(X)
        
        # Compute and log loss
        loss = F.cross_entropy(y_pred, y, weight=self.get_weights(y))
        self.log("loss", loss)

        # Compute and log metrics
        acc = self.train_acc(y_pred.cpu(), y.cpu())
        self.log('train_acc', acc)
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Get X and y
        X, y = batch
        X, y = X.to(self.config.device), y.to(self.config.device)
        
        # Compute logits
        y_pred = self.model(X)
        
        # Compute loss
        loss = F.cross_entropy(y_pred, y, weight=self.get_weights(y))
        self.log("valid_loss", loss)

        # Compute and log metrics
        acc = self.valid_acc(y_pred.cpu(), y.cpu())
        self.log('valid_acc', acc)
        
        return loss

    def update_progress(self, pbar, optimizer, p_info):
        pbar.set_postfix(
            lr=f'{optimizer.param_groups[0]["lr"]:0.6f}',
            mem=f'{mem_usage():0.2f} GB',
            **p_info,
        )

    def progress_info(self, epoch, step, name='train'):
        return {
            'epoch': epoch,
            'step': step,
            **dict(map(lambda v: (v[0], statistics.mean(v[1])), self.metrics.items()))
        }

    def fit(
        self,
        train_loader,
        val_loader,
        monitor='valid_acc',
        mode='max',
        train_steps=None,
        valid_steps=None,
        train_batch_fn=lambda step, epoch, batch: batch,
    ):
        # Training status
        metrics = defaultdict(list)
        
        # Get optimizer configuration
        optimizer_config = self.configure_optimizers()
        optimizer = optimizer_config['optimizer']
        scheduler = optimizer_config['lr_scheduler']

        # Checkpoint Configuration
        chpt = ModelCheckpoint(self.config.checkpoint, mode=mode)

        # Train loop
        for epoch in (pbar := tqdm(range(self.config.epochs))):
            ## Enable training
            self.model.train()
            
            ## Train one epoch
            for idx, batch in enumerate(train_loader):
                ### Apply batch_fn
                batch = train_batch_fn(idx, epoch, batch)
                
                ### Reset gradients
                optimizer.zero_grad()
                
                ### Process one batch
                loss = self.training_step(batch, idx)

                ### Compute gradients
                loss.backward()

                ### Apply gradient
                optimizer.step()

                ### Update progress info
                p_info = self.progress_info(epoch, idx, loss)
                self.update_progress(pbar, optimizer, p_info)

                ### Apply step limitations
                if train_steps and train_steps == idx: break

            with torch.no_grad():
                ## Disable training
                self.model.eval()
            
                ## Validate one epoch
                for idx, batch in enumerate(val_loader):
                    ### Process one batch
                    loss = self.validation_step(batch, idx)
    
                    ### Update progress info
                    p_info = self.progress_info(epoch, idx, loss)
                    self.update_progress(pbar, optimizer, p_info)
    
                    ### Apply step limitations
                    if valid_steps and valid_steps == idx: break

            ## Execute scheduler step
            scheduler.step(statistics.mean(self.metrics['valid_loss']))

            ## Record training metrics
            for k, v in self.metrics.items():
                metrics[k].append(
                    statistics.mean(self.metrics[k])
                )

            ## Clear epoch metrics
            self.metrics = defaultdict(list)
        
            ## Update checkpoint
            chpt.record(self.model, metrics[monitor][-1], epoch)

            ## Stop if we are not making progress
            if (chpt.best_epoch + self.config.optimizer.early_stopping) <= epoch:
                print(f'Early stopping at {epoch=}')
                break

        return metrics
                
    def configure_optimizers(self, verbose=True):
        # Define Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.optimizer.lr,
            weight_decay=self.config.optimizer.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.config.optimizer.patience,
            factor=self.config.optimizer.lr_decay,
            min_lr=self.config.optimizer.min_lr,
            verbose=verbose,
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }


# In[ ]:


def train_batch_turn_one_class_off_per_epoch(trainer, step, epoch, batch, verbose=True):
    # Get X and y
    X, y = batch
    
    # Initialize parameters
    num_classes = trainer.config.num_classes
    start_epoch = trainer.config.turn_off_start_epoch
    end_epoch = trainer.config.turn_off_end_epoch

    # Turn off one class per epoch after num_classes epochs have completed
    # Turn off for num_classes subsequent epochs
    if epoch >= start_epoch and epoch < end_epoch:
        ## Get the class to turn off
        turned_off_class = (epoch - start_epoch) % num_classes

        ## Log the turned_off_class
        if step == 0 and verbose: print(f'{epoch=} {turned_off_class=}')

        ## Find indices of samples which are not turned off
        on_indices = (y != turned_off_class)

        ## Construct batch with ON samples
        batch = X[on_indices], y[on_indices]
    
    return batch

# class TestTrainer():
#     def __init__(self, config):
#         self.config = config

# @dataclass
# class TrainTestConfig:
#     num_classes: int = 3

#     turn_off_start_epoch: int = 2
#     turn_off_end_epoch: int = 6

# batch = torch.randn(4, 16), torch.tensor([0, 1, 1, 2])
# t = TestTrainer(TrainTestConfig())
# step, epoch = 0, 4

# X, y = train_batch_turn_one_class_off_per_epoch(t, step, epoch, batch)
# X, y


# In[ ]:


import torch
import random
import torch.nn.functional as F

def get_train_batch_mixup_fn(config):
    # MixUp configuration
    num_classes = config.num_classes
    items_to_mix = config.items_to_mix
    mixup_prob = config.mixup_prob
    mixup_num_samples = config.mixup_num_samples

    # MixUp start and end epochs
    mixup_start_epoch = config.mixup_start_epoch
    mixup_end_epoch = config.mixup_end_epoch

    def apply_mixup(batch):
        Xs, ys = batch

        # MixUp conditions
        # 1. The number of elements in the batch may be less than the number of mixup samples.
        # 2. The number of mixup samples may not be aligned to items_to_mix bounday

        # Handle Mixup Condition#1
        num_elements = len(Xs)
        mixup_samples = min(num_elements, mixup_num_samples)

        # Handle Mixup Condition#2
        mixup_samples = mixup_samples - mixup_samples%items_to_mix
        
        # print(f'{ys=}')
        # print(f'{mixup_samples=} {items_to_mix=}')
        # print(f'{len(Xs[:mixup_samples].split(items_to_mix))=}')

        # X mixup
        Xs_mixed = torch.mean(
            torch.stack(
                Xs[:mixup_samples].split(mixup_samples // items_to_mix)
            ),
            dim=0
        )
        # print(f'{Xs_mixed.shape=}')

        # y mixup
        ys_mixed = torch.mean(
            torch.stack(
                F.one_hot(ys, num_classes=num_classes).split(mixup_samples // items_to_mix)
            ),
            dim=0,
            dtype=torch.float,
        )
        # print(f'{ys_mixed.shape=}')

        # X mixup + X and y mixup + y
        if ys.shape[0] > mixup_samples:
            # print(f'Remaining: {len(ys[mixup_samples:])=}')
            Xs_combined = torch.concat([Xs_mixed, Xs[mixup_samples:]], dim=0)
            ys_combined = torch.concat(
                [
                    ys_mixed,
                    F.one_hot(ys[mixup_samples:], num_classes=num_classes, dtype=torch.float)
                ],
                dim=0,
                dtype=torch.float,
            )
        else:
            Xs_combined = Xs_mixed
            ys_combined = ys_mixed
        
        return Xs_combined, ys_combined
        
    def fn(step, epoch, batch):
        # epoch should be in [mixup_start_epoch, mixup_end_epoch)
        if (epoch >= mixup_start_epoch and epoch < mixup_end_epoch) and random.random() <= mixup_prob:
            # print(f'Mixing')
            return apply_mixup(batch)
        else:
            # print(f'Unmixed')
            return batch

    return fn

# from dataclasses import dataclass, field

# @dataclass
# class MixUpConfig():
#     num_classes: int = 18
#     items_to_mix: int = 2
#     mixup_prob: float = .4
#     mixup_num_samples: int = 64

#     # MixUp start and end epochs
#     mixup_start_epoch: int = 18
#     mixup_end_epoch: int = 30

# mixup_config = MixUpConfig()
# print(f'{mixup_config=}')
# num_items = 10

# Xs = torch.randn(num_items, 4, 16)
# ys = torch.randint(mixup_config.num_classes, (num_items,))
# # print(f'{Xs=}\n{ys=}')
# mixup_fn = get_train_batch_mixup_fn(mixup_config)

# step, epoch, batch = 0, 18, (Xs, ys)
# Xs, ys = mixup_fn(step, epoch, batch)
# print(f'{Xs.shape=}\n{ys.shape=}')

