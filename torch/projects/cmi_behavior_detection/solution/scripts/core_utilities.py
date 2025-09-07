#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch

def is_interactive():
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE', '') == 'Interactive'
#     return True

def mem_usage():
    # CUDA Memory Usage in GB
    return torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0


# In[ ]:


def make_header(s = ''):
    return f'\n{s}\n{"="*len(s)}\n'


# In[ ]:


import torch
import random
import os

import numpy as np

def set_seed(seed = 42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    
    # CUDA
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)


# In[ ]:


from matplotlib import pyplot as plt

def make_grid_plot(num_items, cols=3, size=3, **kwargs):
    # Compute rows and columns
    num_items = num_items or 1
    rows = (num_items + cols - 1)//cols
    cols = min(cols, num_items)
    
    # Create axes
    fig, axes = plt.subplots(
        rows, cols,
        figsize=np.array([cols, rows])*size,
        **kwargs,
    )

    # Flatten axes for easy iteration
    axes = axes.flatten() if num_items > 1 else [axes]
    
    return fig, axes[:num_items]


# In[ ]:


import pickle

def save_pkl(p, obj):
    with open(p, 'wb') as f:
        pickle.dump(obj, f)
        f.close()

def load_pkl(p):
    with open(p, 'rb') as f:
        obj = pickle.load(f)
        f.close()

    return obj

