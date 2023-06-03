import math
import numpy as np
import torch
from torch import nn
import models.WaveNetVAE.WaveVaeOperations as WOP
# import WaveVaeOperations as WOP

class Wavenet(nn.Module):
    """WaveNet class"""
    def __init__(self, 
                 layers = 10, 
                 stacks = 2, 
                 out_channels = 256, 
                 res_channels = 384, 
                 skip_channels = 256, 
                 gate_channels = 768, 
                 cond_channels = -1, 
                 kernel_size = 3, 
                 upsample_scales = None,
                 bias = True,
                 init_type = 'kaiming_u',
                 activation = 'leaky_relu'):
        
        super().__init__()

        # Upsample audio to size of residual channels
        self.first_conv = WOP.Conv1dWrap(in_channels = 1,
                                    out_channels = res_channels, 
                                    kernel_size=1,
                                    dilation=1, 
                                    bias=bias,
                                    init_type=init_type,
                                    activation=activation)
        
        self.activation = nn.LeakyReLU(negative_slope=0.1, inplace=True) if activation == 'leaky_relu' else nn.ReLU(inplace=True)

        # Make WaveNet layers
        receptive_field = 1
        self.dilations = []
        self.dilated_queues = []
        self.conv_layers = nn.ModuleList()

        for stack in range(stacks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for layer in range(layers):
                
                final_layer = (stack + 1 == stacks and layer + 1 == layers)
                
                resdilconv = WaveNetLayer(
                    residual_channels = res_channels,
                    gate_channels = gate_channels,
                    kernel_size = kernel_size,
                    skip_out_channels = skip_channels,
                    cin_channels = cond_channels,
                    dilation = new_dilation,
                    bias = bias,
                    init_type = init_type,
                    activation = activation,
                )
                self.conv_layers.append(resdilconv)

                receptive_field += additional_scope
                additional_scope *= 2
                new_dilation *= 2

        self.receptive_field = receptive_field
        print("WaveNet Receptive Field: ", self.receptive_field)
        
        self.final_convs = nn.Sequential(
            self.activation,
            WOP.Conv1dWrap(in_channels=skip_channels,
                      out_channels=skip_channels,
                      kernel_size = 1,
                      bias = True,
                      init_type=init_type,
                      activation=activation),
            self.activation,
            WOP.Conv1dWrap(in_channels = skip_channels, 
                      out_channels = out_channels, 
                      kernel_size = 1,
                      bias = True,
                      init_type=init_type,
                      activation=activation),
            self.activation
        )

        # Convolutions for upsampling latent space condition
        self.lc_upsample = nn.Sequential()
        iterator = enumerate(zip([8, 6, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2]))
        for i, (filt_sz, stride) in iterator: 
            name = f'Upsampling_{i}(filter_sz={filt_sz}, stride={stride})'
            mod = WOP.Upsampling(128, filt_sz, stride, name=name)
            self.lc_upsample.add_module(str(i), mod)

    def forward(self, x, c = None, verbose = False):
        """Forward step
        Args:
            x (Tensor): Mulaw encoded audio signal, shape (B x 1 x T)
            c (Tensor): Local conditioning features, shape (B x cin_channels x timesteps)
              Also type of input tensor must be FloatTensor, not LongTensor
        Returns:
            Tensor: output, shape B x out_channels x T
        """

        # Upsample local conditioning features
        if verbose:
            print("Condition before upsampling: ", c.size())   
        c = self.lc_upsample(c)
        if verbose:
            print("Condition and x after c upsampling: ", c.size(), x.size())
        
        assert c.size(-1) == x.size(-1) # Make sure audio and local conditioning have same timesteps
        
        # Feed data to network
        x = self.first_conv(x)

        skips = 0
        for layer in self.conv_layers:
            x, s = layer(x, c)
            skips += s
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        x = skips
        x = self.final_convs(x)
        return x
    

class WaveNetLayer(nn.Module):

    def __init__(self, 
                 residual_channels, 
                 gate_channels, 
                 kernel_size, 
                 skip_out_channels = None, 
                 cin_channels = -1, 
                 dilation = 1, 
                 bias = False, 
                 init_type='kaiming_n', 
                 activation='leaky_relu'):
        super(WaveNetLayer, self).__init__()

#       dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

        self.dil_conv = WOP.CausalConvolution1D(residual_channels, 
                                  gate_channels, 
                                  kernel_size,
                                  dilation = dilation,
                                  bias = bias,
                                  init_type = init_type,
                                  activation = activation)

        self.conv1cond = WOP.Conv1dWrap(in_channels = cin_channels, 
                                   out_channels = gate_channels, 
                                   kernel_size = 1, 
                                   padding = 0, 
                                   dilation = 1, 
                                   bias = bias,
                                   init_type = init_type,
                                   activation=activation)

        # conv output is split into two groups
        gate_out_channels = gate_channels // 2
        self.conv1_out = WOP.Conv1dWrap(in_channels=gate_out_channels, 
                                   out_channels=residual_channels, 
                                   kernel_size = kernel_size,
                                   bias=bias, 
                                   padding = 'same',
                                   init_type = init_type,
                                   activation=activation)
        
        self.conv1_skip = WOP.Conv1dWrap(in_channels = gate_out_channels, 
                                    out_channels = skip_out_channels, 
                                    kernel_size = kernel_size, 
                                    bias=bias, 
                                    padding = 'same',
                                    init_type = init_type,
                                    activation=activation)
        self.splitdim = 1
        # self.apply(xavier_init)

    def forward(self, x, c):
        """Forward
        Args:
            x (Tensor): B x C x T
            c (Tensor): B x C x T, Local conditioning features
        Returns:
            x (Tensor): B x residual_channels x T
            s (Tensor): B x skip_channels x T
        """
        residual = x
        condition = self.conv1cond(c)

        x = self.dil_conv(x) # Dilated convolution

        a, b = x.split(x.size(self.splitdim) // 2, dim=self.splitdim) # Get filter and gate
        ca, cb = condition.split(condition.size(self.splitdim) // 2, dim=self.splitdim) # Get filter and gate from local condition
        filt, gate = a + ca, b + cb # Combine filters and gates

        x = torch.tanh(filt) * torch.sigmoid(gate)

        # For skip connection
        s = self.conv1_skip(x)

        # For residual connection
        x = self.conv1_out(x)
        x += residual
        
        return x, s