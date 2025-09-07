#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##############################
# These transforms use numpy #
# Batch dimensions are not expected #
#####################################

import random
import torch
import numpy as np

def Slice(slice_len):
    def fn(X, y=None):
        ## Compute slice limits
        current_len = X.shape[0]
        wiggle_len = current_len - slice_len

        if wiggle_len > 0:
            start = random.randrange(wiggle_len)
            end = start + slice_len

            ### Apply slice limits
            orig_shape = X.shape
            X = X[start:end]
            # print(f'{orig_shape=} {X.shape}=')

        return X if y is None else (X, y)

    return fn

def FixLength(sequence_len):
    def fn(X, y=None):
        ## Compute padding length
        current_len = X.shape[0]
        pad_len = sequence_len - current_len
        # print(f'{pad_len=} {current_len=}')
    
        ## Pad the sequence if necessary
        if pad_len > 0:
            pad_data = np.zeros((pad_len, X.shape[1]), dtype=X.dtype)
            X = np.concatenate([X, pad_data], axis=0)
        elif pad_len < 0:
            X = X[:sequence_len]

        return X if y is None else (X, y)

    return fn

def ToType():
    def fn(X, y=None):
        X = X.astype('float32')
        y = None if y is None else y.astype('int64')
        
        return X if y is None else (X, y)

    return fn

def ToTensor():
    def fn(X, y=None):
        X = torch.tensor(X)
        
        return X if y is None else (X, y)

    return fn

def Transpose(dims=(1, 0)):
    def fn(X, y=None):
        # Convert X to channel-first configuration
        X = np.transpose(X, dims)

        return X if y is None else (X, y)

    return fn

def apply_transforms(x, transforms):
    for t in transforms:
        x = t(*x) if x is tuple else t(x)

    return x


# In[ ]:


import cv2
import numpy as np

def Resize(height=None, width=None):
    def fn(X, y=None):
        if height or width:
            ## Compute new shape
            current_h, current_w = X.shape
            new_h = height or current_h
            new_w = width or current_w
            # print(f'({current_h=}, {current_w=}) {new_h=} {new_w=}')
    
            ## Apply resize operation
            X = cv2.resize(X, (new_w, new_h), interpolation = cv2.INTER_AREA)
            # print(f'{X.shape=}')

        return X if y is None else (X, y)

    return fn

# x = np.random.rand(30, 9)
# x_resized = Resize(42)(x)
# print(f'{make_header("Resize Operation")}')
# print(f'{x.shape=} {x_resized.shape}')


# In[ ]:


import random
import numpy as np

def Flip(dim=0, prob=.5):
    def fn(X, y=None):
        X = flip_fn(X) if random.random() <= prob else X
        
        return X if y is None else (X, y)
            
    def flip_fn(X):
        return np.flip(X, axis=dim)

    return fn

# x = np.random.rand(30, 9)
# x_flipped = Flip(prob=1.0)(x)
# print(f'{make_header("Flip Operation")}')
# print(f'{np.all(np.equal(x[0], x_flipped[-1]))}')


# In[ ]:


import random
import numpy as np

def Dropout(prob=.5, drop_prob=.2):
    def fn(X, y=None):
        X = drop_fn(X) if random.random() <= prob else X
        
        return X if y is None else (X, y)
            
    def drop_fn(X):
        keep_mask = np.random.rand(*X.shape) > drop_prob
        return X*keep_mask

    return fn

# x = np.random.rand(30, 9)
# x_dropped = Dropout(prob=1.0, drop_prob=.5)(x)
# print(f'{make_header("Drop Operation")}')
# print(f'{np.equal(x, x_dropped).sum() / x.size}')


# In[ ]:


import random
import numpy as np

def Holes(dim=0, size=8, count=4, prob=.5):
    def fn(X, y=None):
        X = holes_fn(X) if random.random() <= prob else X
        
        return X if y is None else (X, y)

    def holes_fn(X):
        # Duplicate X for holes
        X_holes = X.copy()

        # Hole references
        hole_ref = np.random.randint(X.shape[dim]-size, size=count)
        # print(f'{X.shape=} {hole_ref=}')

        # Apply hole references
        for hole in hole_ref:
            channel = np.random.randint(X.shape[dim + 1])
            # print(f'{hole=} {channel=}')
            X_holes[hole:hole+size, channel] = 0
        
        return X_holes

    return fn

# x = np.random.rand(30, 9)
# x_holes = Holes(prob=1.0)(x)
# print(f'{make_header("Holes Operation")}')
# print(f'{np.equal(x, x_holes).sum() / x.size}')


# In[ ]:


import random
import numpy as np

def Clip(limits=(-1, 1)):
    def fn(X, y=None):
        X = clip_fn(X)
        
        return X if y is None else (X, y)

    def clip_fn(X):
        return np.clip(X, *limits)

    return fn

# x = np.random.rand(30, 9)
# x_clipped = Clip(limits=(.2, .5))(x)
# print(f'{make_header("Clip Operation")}')
# print(f'{np.logical_or(x < .2, x > .5).sum() / x.size} {np.logical_or(x_clipped < .2, x_clipped > .5).sum()}')


# In[ ]:


import random

def OffChannel(start, size, prob=.5):
    def fn(X, y=None):
        X = off_fn(X) if random.random() <= prob else X
        
        return X if y is None else (X, y)

    def off_fn(X):
        # Duplicate X for holes
        X_tof = X.copy()

        # Update input data
        X_tof[:, start:start+size] = 0

        return X_tof

    return fn

# from core_utilities import make_header

# x = np.random.rand(30, 270)
# start, size = 7, len(data_config.tof_columns)
# x_off = OffChannel(start, size, prob=.5)(x)
# print(f'{make_header("OffChannel Operation")}')
# print(f'{np.equal(0, x_off[..., start:start+size]).sum()  / x[..., start:start+size].size}')


# In[ ]:


import random

def LambdaX(lambda_fn, prob=.5):
    def fn(X, y=None):
        X = lambda_fn(X) if random.random() <= prob else X
        
        return X if y is None else (X, y)

    return fn

# from core_utilities import make_header

# x = np.random.rand(30, 5)
# start, size = 7, len(data_config.tof_columns)
# x_lambdaX = LambdaX(lambda x: x*2, prob=1.)(x)
# print(f'{make_header("LambdaX Operation")}')
# print(f'{np.equal(x_lambdaX, x*2).sum()} {x.size}')

