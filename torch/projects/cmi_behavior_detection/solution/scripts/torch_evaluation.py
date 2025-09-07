#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from transforms import apply_transforms
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

def get_predict_fn(model, device):
    model.to(device)
    model.eval()
    
    def fn(X):
        ## Compute predictions
        logits = model(X.to(device))

        # Add a batch dimension if it's squeezed by the model
        logits = logits.unsqueeze(0) if logits.ndim == 1 else logits
        
        return F.softmax(logits, dim=1)
    
    return fn

def get_multi_predict_fn(models, device):
    for m in models:    
        m.to(device)
        m.eval()

    def prob_fn(m, X):
        # Compute predictions
        logits = m(X.to(device))

        # Add a batch dimension if it's squeezed by the model
        logits = logits.unsqueeze(0) if logits.ndim == 1 else logits

        return F.softmax(logits, dim=1)
    
    def fn(X):
        return torch.mean(
            torch.stack(list(map(lambda m: prob_fn(m, X), models))),
            dim=0
        ) 
    
    return fn

def get_slices_fn(slice_dim, slice_size, stride, transforms=[]):
    def fn(item):
        item_size = item.shape[slice_dim]
        # print(f'{item.shape=}')
        starts = np.concatenate([
            np.arange(0, item_size-slice_size, step=stride),
            [max(item_size-slice_size, 0)],
        ])

        slices = list(map(
            lambda start: apply_transforms(item[start:start+slice_size], transforms),
            starts
        ))
        # print(f'{starts=}')
        # print(f'{len(slices)=} {slices[0].shape=}')

        return slices

    return fn

def get_slices_predict_fn(predict_fn, slices_fn, transforms):
    def fn(X):
        slices = apply_transforms(np.stack(slices_fn(X)), transforms)
        preds = predict_fn(slices).mean(0, keepdim=True)
        
        return preds

    return fn

@torch.inference_mode()
def evaluate_dataset(dataset, predict_fn):
    # Result store
    y_preds, ys = [], []

    # Compute predictions and add them to the store
    # count = 0
    for X, y in tqdm(dataset, desc='Prediction'):
        preds = torch.argmax(predict_fn(X), dim=1)
        
        y_preds.append(preds.item())
        ys.append(y.item())

        # count += 1
        # if count == 10:
        #     break

    # Compute confusion matrix
    cm = confusion_matrix(ys, y_preds)
    display = ConfusionMatrixDisplay(cm)

    # Plot confusion matrix
    display.plot()
    plt.show()

    return np.array(y_preds), np.array(ys)


# In[ ]:


import torch

def load_models(model_weights, device):
    def load_fn(args):
        name, (m, p) = args
        
        print(f'Loading [({name})]:: {p=}')
        m.load_state_dict(
            torch.load(
                p,
                weights_only=True,
                map_location=torch.device(device)
            )
        )

        return m

    models = list(map(load_fn, model_weights.items()))
    return models

