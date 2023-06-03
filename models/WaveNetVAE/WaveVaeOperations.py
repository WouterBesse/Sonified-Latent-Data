import torch
from torch import nn
import numpy as np
import math
from torch import Tensor
from torch.nn import functional as F

"""
Weight initialisation functions
"""
def xavieru_init(mod):
    if hasattr(mod, 'weight') and mod.weight is not None:
        nn.init.xavier_uniform_(mod.weight, gain=nn.init.calculate_gain(mod.activation) if hasattr(mod, 'activation') else 1.0)
    if hasattr(mod, 'bias') and mod.bias is not None:
        mod.bias.data.zero_()

def xaviern_init(mod):
    if hasattr(mod, 'weight') and mod.weight is not None:
        nn.init.xavier_normal_(mod.weight, gain=nn.init.calculate_gain(mod.activation) if hasattr(mod, 'activation') else 1.0)
    if hasattr(mod, 'bias') and mod.bias is not None:
        mod.bias.data.zero_()

def kaimingu_init(mod):
    if hasattr(mod, 'weight') and mod.weight is not None:
        nn.init.kaiming_normal_(mod.weight, a=0.1, nonlinearity=mod.activation if hasattr(mod, 'activation') else 'leaky_relu')
    if hasattr(mod, 'bias') and mod.bias is not None:
        mod.bias.data.zero_()

def kaimingn_init(mod):
    if hasattr(mod, 'weight') and mod.weight is not None:
        nn.init.kaiming_normal_(mod.weight, a=0.1, nonlinearity=mod.activation if hasattr(mod, 'activation') else 'leaky_relu')
    if hasattr(mod, 'bias') and mod.bias is not None:
        mod.bias.data.zero_()

    
"""
Torch Modules
"""
class CausalConvolution1D(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 1, dilation = 0, bias = False, init_type = 'kaiming_n', activation='leaky_relu') -> None:
        super(CausalConvolution1D, self).__init__()

        self.padding = (kernel_size - 1) * dilation
        self.conv = Conv1dWrap(in_channels=in_channels, 
                               out_channels=out_channels, 
                               kernel_size=kernel_size, 
                               padding = 0, 
                               dilation = dilation, 
                               bias = bias,
                               init_type = init_type,
                               activation=activation)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)
    
class Conv1dWrap(nn.Conv1d):
    """
    Simple wrapper that ensures initialization
    Source: https://github.com/hrbigelow/ae-wavenet/blob/master/wavenet.py#L167
    """
    def __init__(self, init_type = 'xavier_u', activation='leaky_relu', **kwargs):
        super(Conv1dWrap, self).__init__(**kwargs)
        init_type = init_type
        self.activation = activation
        if init_type == 'xavier_u':
            self.apply(xavieru_init)
        elif init_type == 'xavier_n':
            self.apply(xaviern_init)
        elif init_type == 'kaiming_u':
            self.apply(kaimingu_init)
        elif init_type == 'kaiming_n':
            self.apply(kaimingn_init)


class Jitter(nn.Module):
    """
    Jitter implementation from [Chorowski et al., 2019].
    During training, each latent vector can replace either one or both of
    its neighbors. As in dropout, this prevents the model from
    relying on consistency across groups of tokens. Additionally,
    this regularization also promotes latent representation stability
    over time: a latent vector extracted at time step t must strive
    to also be useful at time steps t − 1 or t + 1.
    """

    def __init__(self, probability=0.12):
        super(Jitter, self).__init__()

        self.probability = probability

    def forward(self, quantized):
        original_quantized = quantized.clone()
        new_quantized = quantized.clone()
        length = original_quantized.size(2)

        for i in range(length):
            """
            Each latent vector is replace with either of its neighbors with a certain probability
            (0.12 from the paper).
            """
            replace = [True, False][np.random.choice([1, 0], p=[self.probability, 1 - self.probability])]
            if replace:
                if i == 0:
                    neighbor_index = i + 1
                elif i == length - 1:
                    neighbor_index = i - 1
                else:
                    """
                    "We independently sample whether it is to
                    be replaced with the token right after
                    or before it."
                    """
                    neighbor_index = i + np.random.choice([-1, 1], p=[0.5, 0.5])
                new_quantized[:, :, i] = original_quantized[:, :, neighbor_index]

        return new_quantized
    
class Upsampling(nn.Module):
    """
    Module for upsampling conditional features
    Sourced from: https://github.com/hrbigelow/ae-wavenet/blob/master/wavenet.py#L142
    """
    def __init__(self, n_chan, filter_sz, stride, bias=True, name=None, init_type='kaiming_n', activation='leaky_relu'):
        super(Upsampling, self).__init__()
        # See upsampling_notes.txt: padding = filter_sz - stride
        # and: left_offset = left_wing_sz - end_padding
        end_padding = stride - 1
        self.activation = activation

        self.tconv = nn.ConvTranspose1d(n_chan, n_chan, filter_sz, stride,
                padding=filter_sz - stride, bias=bias)
        
        if init_type == 'xavier_u':
            self.apply(xavieru_init)
        elif init_type == 'xavier_n':
            self.apply(xaviern_init)
        elif init_type == 'kaiming_u':
            self.apply(kaimingu_init)
        elif init_type == 'kaiming_n':
            self.apply(kaimingn_init)

    def forward(self, lc):
        """
        B, T, S, C: batch_sz, timestep, less-frequent timesteps, input channels
        lc: (B, C, S)
        returns: (B, C, T)
        """
        lc_up = self.tconv(lc)
        return lc_up