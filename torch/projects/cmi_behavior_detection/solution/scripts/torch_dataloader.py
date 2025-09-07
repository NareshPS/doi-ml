#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from core_utilities import make_header

def dataloader_params(config, pin_memory=True):
    return dict(
        batch_size=config.batch_size,
        shuffle=True if config.name == 'train' else False,
        pin_memory=pin_memory,
        num_workers=config.dataloader_workers,
        persistent_workers=True,
        # drop_last=True,
        # num_workers=1,
    )

# train_loader_params = dataloader_params(train_config)
# valid_loader_params = dataloader_params(valid_config)

# if 'info.dataloader.params' in config.features:
#     print(
#         make_header('Train'),
#         f'\n{train_loader_params=}'
#     )
#     print(
#         make_header('Validation'),
#         f'\n{valid_loader_params=}'
#     )


# In[ ]:


from torch.utils.data import TensorDataset, DataLoader
from core_utilities import make_header

def create_dataloader(dataset, train_config, collate_fn=None):
    # Define dataloader
    loader = DataLoader(dataset, **dataloader_params(train_config), collate_fn=collate_fn)
    
    return loader

def summarize_dataloader(name, loader):
    item = next(iter(loader))
    Xs, ys = item if type(item) is list else (item, None)
    
    if type(Xs) is list:
        Xs_info = '\n'.join(map(lambda x: f'{x.shape=} {x.dtype}', Xs))
    else:
        print(type(Xs))
        Xs_info = f'{Xs.shape=} {Xs.dtype}'

    ys_info = f'{ys.shape=} {ys.dtype=} {ys=}' if ys is not None else ''
    
    print(
        make_header(f"Dataloader: {name}"),
        f'\n{Xs_info} {ys_info}'
    )


# # MixUp

# In[ ]:


import torch
import random
import numpy as np

from typing import List, Tuple

def get_mixup_collate_fn(items_to_mix=2, mixup_num_samples=128, num_classes=18, prob=.4):
    def mixup_label(labels):
        l_mixed = np.zeros(num_classes)
        for l in labels: l_mixed[l] += 1
        
        l_mixed /= len(labels)

        # print(f'{labels=} {l_mixed=}')
        return l_mixed

    def onehot(label):
        oh_label = np.zeros(num_classes)
        oh_label[label] = 1

        return oh_label

    def apply_mixup(data):
        # print(f'{len(data)=} {type(data[0])} {len(data[0])=} {data[0][0].shape=} {data[0][1]=}')
        Xs, ys = zip(*data)

        # MixUp conditions
        # 1. The number of elements in the batch may be less than the number of mixup samples.
        # 2. The number of mixup samples may not be aligned to items_to_mix bounday

        # Handle Mixup Condition#1
        num_elements = len(Xs)
        mixup_samples = min(num_elements, mixup_num_samples)

        # Handle Mixup Condition#2
        mixup_samples = mixup_samples - mixup_samples%items_to_mix
        
        # print(f'{ys=}')

        # X mixup
        Xs_mixed = np.array(list(
            # Sum Xs to mix
            map(sum, zip(*np.array_split(Xs[:mixup_samples], items_to_mix)))
        )) / items_to_mix
        # print(f'{Xs_mixed.shape}')

        # y mixup
        ys_mixed = np.array(list(
            # Mix labels
            map(mixup_label, zip(*np.array_split(ys[:mixup_samples], items_to_mix)))
        ))
        # print(f'{ys_mixed.shape}')

        # X mixup + X and y mixup + y
        if ys[mixup_samples:]:
            # print(f'Remaining: {len(ys[mixup_samples:])=}')
            Xs_combined = np.concatenate([Xs_mixed, Xs[mixup_samples:]], axis=0)
            ys_combined = np.concatenate(
                [
                    ys_mixed,
                    np.array(list(map(onehot, ys[mixup_samples:])))
                ],
                axis=0
            )
        else:
            Xs_combined = Xs_mixed
            ys_combined = ys_mixed
        
        return [torch.tensor(Xs_combined), torch.tensor(ys_combined)]

    def no_mixup(data):
        Xs, ys = zip(*data)
        Xs = torch.tensor(np.array(Xs))
        ys = torch.tensor(np.array(ys), dtype=torch.long)

        return [Xs, ys]
        
    def fn(data: List[Tuple[torch.Tensor, torch.Tensor]]):
        if random.random() <= prob:
            # print(f'Mixing')
            return apply_mixup(data)
        else:
            # print(f'Unmixed')
            return no_mixup(data)

    return fn


# # Mixing Samples

# In[ ]:


import torch
import random
import numpy as np

def onehot(labels, num_classes):
    num_elements = len(labels)
    
    oh_label = np.zeros((num_elements, num_classes))
    
    oh_label[np.arange(num_elements, dtype=int), labels] = 1

    return oh_label

def make_collater(collate_fn, prob=.4):
    def skip_fn(Xs, ys):
        Xs = torch.tensor(np.array(Xs))
        ys = torch.tensor(np.array(ys), dtype=torch.long)

        return [Xs, ys]
        
    def fn(data: list[tuple[torch.Tensor, torch.Tensor]]):
        Xs, ys = zip(*data)
        
        if random.random() <= prob:
            # print(f'Apply collater')
            return collate_fn(Xs, ys)
        else:
            # print(f'Skip collater')
            return skip_fn(Xs, ys)

    return fn


# In[ ]:


import torch
import numpy as np
    
def get_mixing_fn(mix_Xys, items_to_mix=2, max_mix_samples=128, num_classes=18):
    def mix_labels(labels):
        l_mixed = np.zeros(num_classes)
        for l in labels: l_mixed[l] += 1
        
        l_mixed /= len(labels)
    
        # print(f'{labels=} {l_mixed=}')
        return l_mixed

    def mix_fn(Xs, ys):
        # Mix conditions
        # 1. The number of elements in the batch may be less than the number of mixing samples.
        # 2. The number of mix samples may not be aligned to items_to_mix bounday.

        # Handle Mixing Condition#1
        num_elements = len(Xs)
        mix_samples = min(num_elements, max_mix_samples)

        # Handle Mixing Condition#2
        mix_samples = mix_samples - mix_samples%items_to_mix
        
        # print(f'{ys=} {mix_samples=}')

        # Apply mix_Xys to get Xs_mixed and ys_mixed
        Xs_mixed, ys_mixed = zip(*map(
            lambda args: mix_Xys(args[0], args[1]),
            zip(
                map(np.stack, np.array_split(Xs[:mix_samples], num_elements // items_to_mix)),
                np.array_split(
                    onehot(ys[:mix_samples], num_classes),
                    num_elements // items_to_mix
                ),
            )
        ))
        Xs_mixed, ys_mixed = np.concatenate(Xs_mixed), np.concatenate(ys_mixed)
    
        # print(f'mix_fn():: {Xs_mixed.shape=} {ys_mixed.shape=}')

        # X_mixed + X and y_mixed + y
        if ys[mix_samples:]:
            print(f'Remaining: {len(ys[mix_samples:])=}')
            Xs_combined = np.concatenate([Xs_mixed, Xs[mix_samples:]], axis=0)
            ys_combined = np.concatenate(
                [
                    ys_mixed,
                    onehot(ys[mix_samples:], num_classes)
                ],
                axis=0
            )
        else:
            Xs_combined = Xs_mixed
            ys_combined = ys_mixed
        
        return [torch.tensor(Xs_combined), torch.tensor(ys_combined)]

    return mix_fn


# # CutMix

# In[ ]:


import numpy as np

def get_cutmix_fn(min_size=.4, max_size=.6, dim=2, num_classes=18):
    def fn(Xs, ys):
        # print(f'{Xs.shape=} {ys.shape=} {dim=}')
        # print(f'{Xs=}')
        # Sample a size for the cut
        size = np.random.rand()*(max_size - min_size) + min_size

        num_elements = np.ma.size(Xs, 0)
        dim_size = np.ma.size(Xs, dim)
        cut_size = int(dim_size*size)
        # print(f'{dim_size=} {cut_size=}')

        # Find cut points in the source samples (all except the first sample)
        cut_points = np.random.randint(dim_size - cut_size, size=(num_elements - 1))
        # print(f'{cut_points=}')

        # Create a sample placeholder of the first sample to paste the cuts.
        X = Xs[None, 0].copy()

        # Find pivot points on the placeholder to paste the cuts.
        paste_points = np.random.choice(list(range(0, dim_size - cut_size, cut_size)), cut_points.shape)
        # print(f'{paste_points=}')

        # Iterate over the pivots and update the placeholder
        for idx, (cut_point, paste_point) in enumerate(zip(cut_points, paste_points)):
            # print(f'{cut_point)=}')
            X[..., paste_point: paste_point+cut_size] = Xs[idx+1, :, cut_point: cut_point+cut_size]

        # print(f'{X.shape=} {X=}')

        # Create a copy of ys to modify
        ys = ys.copy()
        # print(f'{ys=}')
        
        # Update the share of cut sources
        ys[1:] = ys[1:]*size
        # print(f'{ys=}')

        # Update the share of the placeholder
        ys[0] = ys[0]*(1 - size*(num_elements - 1))

        # Update cutmix label
        y = ys.sum(0)[None, ...]
        # print(f'{y.shape=} {y=}')

        # print(f'{X.shape=} {y.shape=}')

        return X, y

    return fn

# num_items = 8

# Xs = list(map(lambda v: np.full((3, 5), v), range(num_items)))
# ys = np.random.randint(18, size=(num_items,)).tolist()
# # print(f'{Xs=}\n{ys=}')

# collater = make_collater(
#     collate_fn=get_mixing_fn(
#         mix_Xys=get_cutmix_fn(
#             min_size=train_config.min_cut_size,
#             max_size=train_config.max_cut_size,
#             dim=2,
#             num_classes=model_config.out_channels,
#         ),
#         items_to_mix=train_config.items_to_cutmix,
#         max_mix_samples=train_config.max_cutmix_samples,
#         num_classes=model_config.out_channels,
#     ),
#     # prob=train_config.cutmix_prob,
#     prob=1.,
# )

# Xs, ys = collater(list(zip(Xs, ys)))
# print(f'{Xs.shape=}\n{ys.shape=}')
# print(f'{Xs=}\n{ys=}')

