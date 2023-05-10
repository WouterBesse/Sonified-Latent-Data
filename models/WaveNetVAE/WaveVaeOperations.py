import torch
from torch import nn
import numpy as np
import math
from torch import Tensor

"""
Util functions
"""    
def normalisedConvTranspose2d(in_channels, out_channels, kernel_size,
                    weight_normalization=True, **kwargs):
    freq_axis_kernel_size = kernel_size[0]
    m = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, **kwargs)
    m.weight.data.fill_(1.0 / freq_axis_kernel_size)
    m.bias.data.zero_()
    if weight_normalization:
        return nn.utils.weight_norm(m)
    else:
        return m

"""
Torch Modules
"""
class CausalConvolution1D(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size = 1, dilation = 0, bias = False) -> None:
        super(CausalConvolution1D, self).__init__()

        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, padding = 0, dilation = dilation, bias = bias)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (self.padding, 0))
        return self.conv(x)

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

class ResidualConv1dGLU(nn.Module):

    def __init__(self, residual_channels, gate_channels, kernel_size, skip_out_channels = None, cin_channels = -1, dropout= 1 - 0.95, dilation = 1, bias = False):
        super(ResidualConv1dGLU, self).__init__()

        self.dropout = nn.Dropout(p = dropout)
#       dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        self.dil_conv = CausalConvolution1D(residual_channels, 
                                  gate_channels, 
                                  kernel_size,
                                  dilation = dilation,
                                  bias = bias)

        self.conv1cond = nn.Conv1d(cin_channels, 
                                   gate_channels, 
                                   kernel_size = 1, 
                                   padding = 0, 
                                   dilation = 1, 
                                   bias = bias)

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1_out = nn.Conv1d(gate_out_channels, 
                                   residual_channels, 
                                   kernel_size = kernel_size,
                                   bias=bias, 
                                   padding = 'same')
        
        self.conv1_skip = nn.Conv1d(gate_out_channels, 
                                    skip_out_channels, 
                                    kernel_size = kernel_size, 
                                    bias=bias, 
                                    padding = 'same')
        self.splitdim = 1

    def forward(self, x, c):
        """Forward
        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
        Returns:
            Tensor: output
        """
        residual = x
        # x = self.dropout(x)
        condition = self.conv1cond(c)

        x = self.dil_conv(x) # Dilated convolution

        # x = x[:, :, :residual.size(-1)] # Remove future time steps

        a, b = x.split(x.size(self.splitdim) // 2, dim=self.splitdim) # Get filter and gate

        # local conditioning
        ca, cb = c.split(condition.size(self.splitdim) // 2, dim=self.splitdim) # Get filter and gate from condition
        filt, gate = a + ca, b + cb # Combine filters and gates

        x = torch.tanh(filt) * torch.sigmoid(gate)

        # For skip connection
        s = self.conv1_skip(x)

        # For residual connection
        x = self.conv1_out(x)
        x += residual
        
        return x, s

