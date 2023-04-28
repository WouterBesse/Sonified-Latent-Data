import math
import numpy as np
import torch
from torch import nn
import models.WaveNetVAE.WaveVaeOperations as WOP

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
        
        super(Wavenet, self).__init__()

        #assert layers % stacks == 0
        self.upsample = upsample_conditional_features

        self.first_conv = nn.Conv1d(in_channels = out_channels,
                                    out_channels = res_channels, 
                                    kernel_size=1,
                                    dilation=1, 
                                    bias=bias)
        
        self.skip_conv = nn.Conv1d(in_channels = res_channels, 
                                   out_channels = res_channels, 
                                   kernel_size = 1, 
                                   bias = bias)

        # Wavenet layers
        receptive_field = 1
        self.dilations = []
        self.dilated_queues = []
        self.conv_layers = nn.ModuleList()

        for stack in range(stacks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for layer in range(layers):
                resdilconv = WOP.ResidualConv1dGLU(
                    residual_channels = res_channels,
                    gate_channels = gate_channels,
                    kernel_size = kernel_size,
                    skip_out_channels = skip_channels,
                    cin_channels = cond_channels,
                    dilation = new_dilation,
                    dropout = dropout,
                )
                self.conv_layers.append(resdilconv)

                receptive_field += additional_scope
                additional_scope *= 2
                new_dilation *= 2

        self.receptive_field = receptive_field
        print("WaveNet Receptive Field: ", self.receptive_field)
        
        self.final_convs = nn.Sequential(
            nn.Conv1d(skip_channels,
                      out_channels,
                      kernel_size = 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.BatchNorm1d(out_channels),
            nn.Conv1d(in_channels = out_channels, 
                      out_channels = out_channels, 
                      kernel_size = 1),
            nn.ReLU(inplace=True),
        )

        # Convolutions for upsampling latent space condition
        self.upsample_conv_seq = nn.Sequential()
        if upsample_conditional_features:
            self.upsample_conv = []
            for s in upsample_scales:
                freq_axis_padding = (freq_axis_kernel_size - 1) // 2
                convt = WOP.normalisedConvTranspose2d(1, 1, (freq_axis_kernel_size, s),
                                        padding=(freq_axis_padding, 0),
                                        dilation=1, stride=(1, s),
                                        weight_normalization= True)
                self.upsample_conv.append(convt)
                # assuming we use [0, 1] scaled features
                # this should avoid non-negative upsampling output
                self.upsample_conv.append(nn.ReLU(inplace = True))
                
            self.upsample_conv_seq = nn.Sequential(*self.upsample_conv) 
        else:
            self.upsample_conv = None

    def forward(self, x, c = None, softmax = True):
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

        B, _, T = x.size()
        # print(c.size())

        # B x 1 x C x T
        print(c.size())
        c = c.unsqueeze(1)

        c = self.upsample_conv_seq(c)
        # B x C x T
        c = c.squeeze(1)

        print(c.size(), x.size())
        assert c.size(-1) == x.size(-1)
        
        # Feed data to network
        x = self.first_conv(x)
        skip = self.skip_conv(x)
        for layer in self.conv_layers:
            x, skip = layer(x, c, skip)

        x = skip
        x = self.final_convs(x)
        return x
        