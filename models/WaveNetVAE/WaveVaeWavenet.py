import math
import numpy as np
import torch
from torch import nn
import models.WaveNetVAE.WaveVaeOperations as WOP
# import WaveVaeOperations as WOP

class Wavenet(nn.Module):

    def __init__(self, 
                 layers = 10, 
                 stacks = 2, 
                 out_channels = 256, 
                 res_channels = 512, 
                 skip_channels = 512, 
                 gate_channels = 1024, 
                 cond_channels = -1, 
                 kernel_size = 3, 
                 freq_axis_kernel_size = 3, 
                 dropout = 1 - 0.95, 
                 upsample_conditional_features = True, 
                 upsample_scales = None,
                 bias = True):
        
        super().__init__()

        #assert layers % stacks == 0
        # self.upsample = upsample_conditional_features

        self.first_conv = WOP.Conv1dWrap(in_channels = 1,
                                    out_channels = res_channels, 
                                    kernel_size=1,
                                    dilation=1, 
                                    bias=bias)
        print("changed!")
        
        # self.emb = nn.Sequential(nn.Embedding(out_channels, res_channels, padding_idx=out_channels // 2 - 1),
        #                          nn.Tanh())
        
        # self.skip_conv = WOP.Conv1dWrap(in_channels = res_channels, 
        #                            out_channels = res_channels, 
        #                            kernel_size = 1, 
        #                            bias = bias)

        # Wavenet layers
        receptive_field = 1
        self.dilations = []
        self.dilated_queues = []
        self.conv_layers = nn.ModuleList()

        for stack in range(stacks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for layer in range(layers):
                
                final_layer = (stack + 1 == stacks and layer + 1 == layers)
                
                resdilconv = WOP.ResidualConv1dGLU(
                    residual_channels = res_channels,
                    gate_channels = gate_channels,
                    kernel_size = kernel_size,
                    skip_out_channels = skip_channels,
                    cin_channels = cond_channels,
                    dilation = new_dilation,
                    dropout = dropout,
                    bias = bias,
                    final_layer = final_layer
                )
                self.conv_layers.append(resdilconv)

                receptive_field += additional_scope
                additional_scope *= 2
                new_dilation *= 2

        self.receptive_field = receptive_field
        print("WaveNet Receptive Field: ", self.receptive_field)
        
        self.final_convs = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            WOP.Conv1dWrap(in_channels=skip_channels,
                      out_channels=skip_channels,
                      kernel_size = 1,
                      bias = bias),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            WOP.Conv1dWrap(in_channels = skip_channels, 
                      out_channels = out_channels, 
                      kernel_size = 1,
                      bias = bias),
            # nn.ReLU(inplace=True),
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
            x (Tensor): One-hot encoded audio signal, shape (B x C x T)
            c (Tensor): Local conditioning features,
              shape (B x cin_channels x T)
              Also type of input tensor must be FloatTensor, not LongTensor
            softmax (bool): Whether applies softmax or not.
        Returns:
            Tensor: output, shape B x out_channels x T
        """

        # B x 1 x C x T
        if verbose:
            print("Condition before upsampling: ", c.size())
            
        c = self.lc_upsample(c)
        # B x C x T

        if verbose:
            print("Condition and x after c upsampling: ", c.size(), x.size())
        
        assert c.size(-1) == x.size(-1)
        
        # Feed data to network
        x = self.first_conv(x)
        # x = self.emb(x).transpose(1, 2)
        # skip = self.skip_conv(x)
        skips = 0
        for layer in self.conv_layers:
            x, s = layer(x, c)
            skips += s
        # skips *= math.sqrt(1.0 / len(self.conv_layers))

        x = skips
        x = self.final_convs(x)
        return x