#!/usr/bin/env python
# coding: utf-8

# # Channel Shuffle

# In[ ]:


import torch
import torch.nn as nn

class SliceAndShuffle(nn.Module):
    def __init__(self, num_slices, dim):
        super().__init__()
        # Input args
        self.num_slices = num_slices
        self.dim = dim

    def forward_train(self, x):
        # Get target shape
        shape = x.shape
        new_shape = (*x.shape[:self.dim], self.num_slices, -1)
        # print(f'{shape=} {new_shape=}')

        # Reshape the tensor to the target shape
        x = x.reshape(new_shape)
        # print(f'{x.shape=} {xr.shape=}')

        # Shuffle
        perm = torch.randperm(self.num_slices)
        # print(f'{perm=}')
        
        x = x[:, :, perm, :]

        # Remove shuffle dimension
        x = x.reshape((*shape[:self.dim], -1))

        return x

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return x

# x = torch.randn(2, 2, 8)
# l =  SliceAndShuffle(num_slices=4, dim=2)
# x, l(x)


# # Composite Convolution Layers

# In[ ]:


import torch.nn as nn

from collections import OrderedDict

class ConvNormActivation(nn.Module):
    def __init__(
        self,
        name,
        in_channels, out_channels, kernel_size,
        conv_layer=nn.Conv1d,
        norm_layer=nn.BatchNorm1d,
        activation_layer=nn.ReLU,
        bias=True
    ):
        super().__init__()

        # Input args
        self.name = name
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.activation_layer = activation_layer

        self.bias = bias

        # Layers
        self.block = self.make_block()

    def make_block(self):
        items = [
            (
                self.name + '_conv',
                self.conv_layer(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    kernel_size=self.kernel_size,
                    bias=self.bias,
                )
            )
        ]

        if self.norm_layer is not None:
            items.append(
                (self.name + '_bn', self.norm_layer(self.out_channels))
            )

        if self.activation_layer is not None:
            items.append(
                (self.name + '_relu', self.activation_layer(inplace=True))
            )

        return nn.Sequential(OrderedDict(items))

    def forward(self, x):
        x = self.block(x)

        return x


# In[ ]:


import torch.nn as nn

from collections import OrderedDict

class ConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels, out_channels, kernel_size,
        norm=nn.BatchNorm1d,
        activation=nn.ReLU,
        dropout=None,
        **conv_kwargs,
    ):
        super().__init__()

        # Input args
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Layers
        self.block = nn.Sequential(
            OrderedDict(filter(
                lambda kv: kv[1] is not None,
                [
                    (
                        'conv',
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            **conv_kwargs,
                        )
                    ),
                    ('bn', norm(out_channels) if norm is not None else None),
                    ('relu', activation(inplace=True) if activation is not None else None),
                    ('dropout', nn.Dropout(dropout) if dropout else None),
                ]
            ))
        )

    def forward(self, x):
        x = self.block(x)

        return x

# from torchinfo import summary

# in_channels, out_channels, kernel_size = 64, 128, 3
# x = torch.randn(1, in_channels, 10)
# l = ConvBlock1d(in_channels, out_channels, kernel_size, padding='same', bias=False)

# print(summary(
#     model=l, 
#     input_data=x,
#     col_names=["input_size", "output_size", "num_params", "trainable"],
#     col_width=20,
#     row_settings=["var_names"]
# ))
# l(x).shape


# # Squeeze and Excitation Layers

# In[ ]:


import torch.nn as nn

from torchinfo import summary

class ChannelSeBlock(nn.Module):
    def __init__(
        self,
        in_channels, squeeze_channels,
        activation=nn.ReLU,
        conv=ConvBlock1d,
        pool=nn.AdaptiveAvgPool1d,
        scaler=nn.Sigmoid,
    ):
        """ChannelSeBlock is a Squeeze-And-Excitation block.

        Args:
            in_channels (int): The number of input channels.
            squeeze_channels (int): The number of channels to squeeze to.
        """
        super().__init__()

        self.pool_m = pool(1)
        self.conv_1_m = conv(
            in_channels, squeeze_channels, 1,
            activation=activation,
            norm=None, bias=False,
        )
        self.conv_2_m = conv(
            squeeze_channels, in_channels, 1,
            norm=None, activation=None, bias=False,
        )
        self.scaler_m = scaler()

    def forward(self, x):
        inp = x

        # 1. (...) -> (..., in_channels, 1, 1)
        x = self.pool_m(x)

        # 2. -> (..., squeeze_channels, 1, 1)
        x = self.conv_1_m(x)

        # 3. -> (..., in_channels, 1, 1)
        x = self.conv_2_m(x)

        # 4. Scale
        x = self.scaler_m(x)
        # print(x)
        x = inp * x

        return x

# import torch

# in_channels = 32
# x = torch.randn(1, in_channels, 10)
# l = ChannelSeBlock(
#     in_channels,
#     squeeze_channels=8,
#     # scaler=SigmoidBottleneckScaler
# )

# print(summary(
#     model=l, 
#     input_data=x,
#     col_names=["input_size", "output_size", "num_params", "trainable"],
#     col_width=20,
#     row_settings=["var_names"]
# ))
# l(x)


# In[ ]:


import torch.nn as nn

from collections import OrderedDict

class LinearSeBlock(nn.Module):
    def __init__(self, in_channels, squeeze_channels, scaler=nn.Sigmoid):
        """LinearSeBlock is a Squeeze-And-Excitation block.

        Args:
            in_channels (int): The number of input channels.
            squeeze_channels (int): The number of channels to squeeze to.
            scaler (nn.Module): Module to scale the input.
        """
        super().__init__()

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.block = nn.Sequential(OrderedDict([
            # ('pool', ),
            ('linear_1', nn.Linear(in_channels, squeeze_channels)),
            ('relu', nn.ReLU(inplace=True)),
            ('linear_2', nn.Linear(squeeze_channels, in_channels)),
            ('scaler', scaler())
        ]))

    def forward(self, x):
        # 1. Save the input for scaling (..., time_steps, in_channels)
        inp = x

        # 2. Reduce time_steps (..., time_steps, in_channels) -> (..., in_channels, 1)
        x = self.pool(x.permute(0, 2, 1))
        # print(f'pool: {x.shape=}')

        # Apply block
        x = self.block(x.squeeze(-1)).unsqueeze(1)

        # Scale the input
        x = inp * x

        return x

# from torchinfo import summary

# in_channels = 64
# x = torch.randn(1, 10, in_channels)
# l = LinearSeBlock(in_channels, squeeze_channels=8)

# print(summary(
#     model=l, 
#     input_data=x,
#     col_names=["input_size", "output_size", "num_params", "trainable"],
#     col_width=20,
#     row_settings=["var_names"]
# ))
# l(x).shape


# # Deep Convolution Blocks

# In[ ]:


import torch.nn as nn

from collections import OrderedDict

class ResidualConvBlock1d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        squeeze_channels=None,
        se_scaler=nn.Sigmoid,
        dropout=None,
        activation=nn.ReLU,
        pool=nn.MaxPool1d,
        num_blocks=2,
        **conv_kwargs,
    ):
        super().__init__()

        # Input args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.squeeze_channels=squeeze_channels
        self.dropout = dropout
        self.activation = activation
        self.num_blocks = num_blocks
        self.conv_kwargs = conv_kwargs

        # Layers
        self.conv_m = self.make_cnn_block()
        if squeeze_channels: self.se_block = LinearSeBlock(out_channels, squeeze_channels, scaler=se_scaler)
        self.projection_m = self.create_projection()
        self.activation_m = activation(inplace=True)
        self.pool_m = pool(2) if pool is not None else None
        if dropout: self.dropout_m = nn.Dropout(dropout)

    def make_cnn_block(self):
        # Configure containers
        blocks = []

        # Create blocks
        for idx in range(self.num_blocks):
            ## Create one block
            block = ConvBlock1d(
                in_channels=self.in_channels if idx == 0 else self.out_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                activation=self.activation,
                **self.conv_kwargs
            )

            # Update containers
            blocks.append((f'conv_{idx}', block))

        # Combine all the blocks into a CNN module
        return nn.Sequential(OrderedDict(blocks))

    def create_projection(self):
        if self.in_channels == self.out_channels:
            return None

        return ConvBlock1d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            activation=None,
            **self.conv_kwargs
        )

    def forward(self, x):
        # 1. Create a shortcut (..., in_channels, time_steps)
        shortcut_x = x

        # 2. Apply conv module -> (..., in_channels, time_steps)
        x = self.conv_m(x)

        # 3. Apply Se block (requires channel_last input) -> (..., in_channels, time_steps)
        if self.squeeze_channels:
            # print(f'se_block::{x.shape=}')
            x = self.se_block(x.permute(0, 2, 1)).permute(0, 2, 1)

        # 4. Apply projection module if necessary
        if self.projection_m is not None:
            shortcut_x = self.projection_m(shortcut_x)

        # 5. Add shortcut to x
        x = x + shortcut_x

        # 6. Apply activation, pool, and dropout
        x = self.activation_m(x)
        if self.pool_m is not None: x = self.pool_m(x)
        x = self.dropout_m(x) if self.dropout else x

        return x

# from torchinfo import summary

# in_channels, out_channels, kernel_size = 32, 64, 3
# x = torch.randn(2, in_channels, 10)

# m = ResidualConvBlock1d(
#     in_channels, out_channels, kernel_size,
#     squeeze_channels=out_channels//8,
#     dropout=.1,
#     pool=None,
#     num_blocks=2,
#     padding='same',
#     bias=False,
# )

# print(summary(
#     model=m, 
#     input_data=x,
#     col_names=["input_size", "output_size", "num_params", "trainable"],
#     col_width=20,
#     row_settings=["var_names"]
# ))
# m(x).shape


# In[ ]:


import torch.nn as nn

from collections import OrderedDict

class BDConvBlock(nn.Module):
    def __init__(
        self,
        name,
        in_channels,
        out_channels,
        kernel_size,
        dropout,
        squeeze_channels=None,
        se_scaler=nn.Sigmoid,
        **conv_kwargs
    ):
        super().__init__()

        # Input args
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Layers
        self.block = nn.Sequential(
            OrderedDict(filter(
                lambda kv: kv[1] is not None,
                [
                    (
                        name + '_conv',
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size,
                            **conv_kwargs,
                        )
                    ),
                    (name + '_bn', nn.BatchNorm1d(out_channels)),
                    (name + '_relu', nn.ReLU(inplace=True)),
                    (name + '_dropout', nn.Dropout(dropout) if dropout else None),
                ]
            ))
        )
        self.se_block = None if squeeze_channels is None else ChannelSeBlock(
            in_channels=out_channels,
            squeeze_channels=squeeze_channels,
            scaler=se_scaler,
        )

    def forward(self, x):
        x = self.block(x)

        x = x if self.se_block is None else self.se_block(x)

        return x

class BDLinearBlock(nn.Module):
    def __init__(self, name, in_channels, out_channels, dropout, **linear_kwargs):
        super().__init__()

        # Input args
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Layers
        self.block = nn.Sequential(
            OrderedDict(filter(
                lambda kv: kv[1] is not None,
                [
                    (name + '_linear', nn.Linear(in_channels, out_channels, **linear_kwargs)),
                    (name + '_relu', nn.ReLU(inplace=True)),
                    (name + '_dropout', nn.Dropout(dropout) if dropout else None),
                ]
            ))
        )

    def forward(self, x):
        x = self.block(x)

        return x

# block = BDConvBlock('block', 9, 512, 3, None, padding='same')
# block = BDLinearBlock('block', 512, 128, None, bias=False)
# block


# # Attention Blocks

# In[ ]:


import torch.nn as nn

class SimpleAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        # Input args
        self.channels = channels

        # Layers
        self.score_m = nn.Sequential(
            nn.Linear(channels, 1),
            nn.Tanh(),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        # Save input (..., time_steps, channels)
        inp = x

        # Get attention scores (..., time_steps, channels) -> (..., time_steps, channels)
        scores = self.score_m(x)
        # print(f'{x.shape=} {scores.shape=}\n{x=}\n{scores=}')

        # Scale the input with the scores
        x = inp * scores

        # Sum time-steps
        x = x.sum(dim=1)

        return x

# from torchinfo import summary

# channels = 16
# x = torch.randn(2, 8, channels)
# m = SimpleAttention(channels)

# print(summary(
#     model=m, 
#     input_data=x,
#     col_names=["input_size", "output_size", "num_params", "trainable"],
#     col_width=20,
#     row_settings=["var_names"]
# ))
# x_attn = m(x)
# print(f'{x_attn.shape=}')
# print(f'{x_attn.shape=} {x[0, :, 0]=}\n{x_attn[0]=}')


# # Other Layers

# In[ ]:


import torch.nn as nn

class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()

        # Input args
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


# # SE Block Scalers

# In[ ]:


import torch.nn as nn

class SigmoidBottleneckScaler(nn.Module):
    """SigmoidBottleneckScaler scales the inputs in Squeeze-Excitation layers.
    The simplest scaler is nn.Sigmoid. The bottleneck scaler zeroes out non-preferential
    channels.
    """
    def __init__(self, cutoff_prob=.5, **kwargs):
        super().__init__()

        # Input args
        self.cutoff_prob = cutoff_prob
        
        # Layers
        self.sigmoid_m = nn.Sigmoid(**kwargs)

    def forward(self, x):
        # Get channel scaling factors using sigmoid
        sigmoid_x = self.sigmoid_m(x)

        # Clip sigmoid values lower than .5
        x = (sigmoid_x - self.cutoff_prob).clamp(min=0)
        # print(f'SigmoidBottleneckScaler:: {x=}')

        # Set sigmoid values > .5 to 1.
        x = (x*100).clamp(max=1.)

        if (x == 1.).all() or (x == 0.).all():
            # print(f'SigmoidBottleneckScaler:: shortcut')
            x = sigmoid_x
        # else:
        #     print(f'SigmoidBottleneckScaler:: no-shortcut')

        return x
        
# import torch

# x = torch.randn(1, 16)
# l = SigmoidBottleneckScaler(cutoff_prob=.6)

# l(x)


# In[ ]:


import torch.nn as nn

class SoftmaxBottleneckScaler(nn.Module):
    """SigmoidBottleneckScaler scales the inputs in Squeeze-Excitation layers.
    The bottleneck scaler zeroes out non-preferential
    channels.
    """
    def __init__(self, k, **kwargs):
        super().__init__()

        # Input args
        self.k = k
        
        # Layers
        self.softmax_m = nn.Softmax(**kwargs)

    def forward(self, x):        
        # Get channel scaling factors using softmax
        softmax_x = self.softmax_m(x)
        # print(f'SoftmaxBottleneckScaler:: {softmax_x=}')

        # Compute cutoff_prob based on k
        cutoff_prob, _ = softmax_x.kthvalue(self.k, dim=1, keepdims=True)
        # print(f'SoftmaxBottleneckScaler:: {cutoff_prob.shape=}\n{cutoff_prob=}')

        # Clip softmax values lower than cutoff_prob
        x = (softmax_x - cutoff_prob).clamp(min=0)

        # Set softmax values > cutoff_prob to 1.
        x = (x*10).clamp(max=1.)
        # print(f'SoftmaxBottleneckScaler:: {x=}')

        if (x == 1.).all() or (x == 0.).all():
            # print(f'SoftmaxBottleneckScaler:: shortcut')
            x = softmax_x

        return x
        
# import torch

# x = torch.randn(1, 16, 1, 1)
# l = SoftmaxBottleneckScaler(k=8, dim=1)

# l(x)

