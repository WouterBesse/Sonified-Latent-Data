import math
import torch
from torch import nn, from_numpy
from torch.utils.data import Dataset
from torchaudio.transforms import MuLawEncoding, MFCC, Resample, MuLawDecoding
import torchaudio
import models.WaveNetVAE.WaveVaeOperations as WOP
from models.WaveNetVAE.WaveVaeWavenet import Wavenet
# import WaveVaeOperations as WOP
# from WaveVaeWavenet import Wavenet
from tqdm.auto import tqdm
import random
import numpy as np
import soundfile as sf
import os


class Decoder(nn.Module):
    """
    VAE Decoder
    """

    def __init__(self, out_channels, upsamples, zsize=128, use_jitter=True,
                 jitter_probability=0.12, use_kaiming_normal=True):
        super().__init__()

        self.use_jitter = use_jitter
        if use_jitter:
            self.jitter = WOP.Jitter(jitter_probability)

        # self.linear = nn.Linear(int(zsize), int(input_size[1] // hidden_dim))

        """
        The jittered latent sequence is passed through a single
        convolutional layer with filter length 3 and 128 hidden
        units to mix information across neighboring timesteps.
        (https://github.com/swasun/VQ-VAE-Speech/blob/master/src/models/wavenet_decoder.py#L50)
        """
        self.conv_1 = nn.Conv1d(in_channels=zsize,
                                out_channels=128,
                                kernel_size=2,
                                padding='same')

        if use_kaiming_normal:
            self.conv_1 = nn.utils.weight_norm(self.conv_1)
            nn.init.kaiming_normal_(self.conv_1.weight)

        self.wavenet = Wavenet(
            layers=10,
            stacks=2,
            out_channels=out_channels,
            res_channels=386,
            skip_channels=256,
            gate_channels=768,
            cond_channels=128,
            kernel_size=3,
            upsample_conditional_features=True,
            upsample_scales=upsamples,
        )

        self.receptive_field = self.wavenet.receptive_field

    def forward(self, x, cond, jitter, verbose):
        """Forward step
        Args:
            x (Tensor): Mono audio signal, shape (B x 1 x T)
            c (Tensor): Local conditioning features,
              shape (B x cin_channels x T)
            xsize (Tuple): Size of condition before flattening
            jitter (Bool): Argument deciding if we should jitter our condition or not
        Returns:
            X (Tensor): Reconstructed result, shape (B x 1 x T)
        """
        condition = cond
        if self.use_jitter and jitter:
            condition = self.jitter(condition)
        if verbose:
            print("X size before wavenet: ", x.size())

        condition = self.conv_1(condition)

        x = self.wavenet(x, condition, verbose)

        return x


class Encoder(nn.Module):
    """
    VAE Encoder
    """

    def __init__(self, input_size, hidden_dim=768, zsize=128, resblocks=2, relublocks=4):
        super().__init__()

        features, timesteps = input_size
        self.zsize = zsize
        self.ReL = nn.LeakyReLU(negative_slope=0.1)

        """
        Preprocessing convolutions with residual connections
        """
        self.conv_1 = nn.Conv1d(features,
                                hidden_dim,
                                kernel_size=3,
                                padding='same')

        self.conv_2 = nn.Conv1d(features,
                                hidden_dim,
                                kernel_size=3,
                                padding='same')

        """
        Downsample in the time axis by a factor of 2
        """
        self.downsample = nn.Conv1d(hidden_dim,
                                    hidden_dim,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)

        """
        Residual convs
        """
        self.resblocks = nn.ModuleList()
        for _ in range(resblocks):
            self.resblocks.append(
                nn.Conv1d(hidden_dim,
                          hidden_dim,
                          kernel_size=3,
                          padding='same'))

        """
        Relu blocks
        """
        self.relublocks = nn.ModuleList()
        for _ in range(relublocks):
            self.relublocks.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim,
                              hidden_dim,
                              kernel_size=3,
                              padding='same'),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    nn.Conv1d(hidden_dim,
                              hidden_dim,
                              kernel_size=3,
                              padding='same'),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True)))

        """
        The linear block from the WaveNet VQVAE paper.
        This is deceptively not an actual linear layer, but just a convolution layer.
        """
        self.linear = nn.Conv1d(hidden_dim,
                                zsize * 2,
                                kernel_size=1,
                                bias=False,
                                padding='same')

    def forward(self, x, verbose):
        """Forward step
        Args:
            x (Tensor): MFCC, shape (B x features x timesteps)
        Returns:
            zcomb[:, :self.zsize] (Tensor): Latent space mean, shape (B x zsize)
            zcomb[:, self.zsize:] (Tensor): Latent space variance, shape (B x zsize)
            x_size (Tuple): Size of condition before flattening, shape (B x hidden_dim x timesteps)
        """

        net = self.conv_1(x)
        conv = self.conv_2(x)
        x = self.ReL(net) + self.ReL(conv)

        # Downsample
        if verbose:
            print("Before downsample: ", x.size())
        x = self.ReL(self.downsample(x))
        if verbose:
            print("After downsample: ", x.size())

        # Residual convs
        for resblock in self.resblocks:
            xres = self.ReL(resblock(x))
            x = xres + x

        # Relu blocks
        for relblock in self.relublocks:
            xrelu = relblock(x)
            x = x + xrelu
        x = self.ReL(x)

        z_double = self.linear(x)

        mu, log_var = torch.split(z_double, self.zsize, dim=1)

        return mu, log_var


class WaveNetVAE(nn.Module):

    def __init__(self, input_size, num_hiddens, upsamples, zsize=32, resblocks=2, out_channels=256, devices=['cuda:0', 'cuda:1']):
        super(WaveNetVAE, self).__init__()
        
        self.out_channels = out_channels
        self.softmax = nn.Softmax(dim=1)
        self.devices = devices

        self.encoder = Encoder(
            input_size=input_size,
            hidden_dim=num_hiddens,
            zsize=zsize,
            resblocks=resblocks,
        )
        
        self.encoder = self.encoder.to(devices[0])

        self.decoder = Decoder(
            out_channels=out_channels,
            upsamples=upsamples,
            zsize=zsize
        )
        
        self.decoder = self.decoder.to(devices[1])

        self.receptive_field = self.decoder.receptive_field
        self.mulaw = MuLawEncoding()
        self.mudec = MuLawDecoding()
        self.N = torch.distributions.Normal(0, 1)
        # self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        # self.N.scale = self.N.scale.cuda()

    def sample(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        # return torch.normal(mu, std)
        # eps = torch.randn(*mu.size()).to(mu.get_device())
        eps = self.N.sample(mu.shape).to(mu.get_device())
        z = mu + std * eps
        return z

    def samplenew(self, mu, logvar):
        if self.training:
            z = torch.randn_like(mu).mul(torch.exp(0.5 * logvar)).add_(mu)
        else:
            z = mu

        return z

    def forward(self, xau, xspec, jitter, verbose = False):
        """Forward step
        Args:
            xau (Tensor): Audio, shape (B x 1 x T)
            xspec (Tensor): MFCC, shape (B x features x timestpes)
            jitter (Bool): To jitter the latent space condition or not
        Returns:
            x_hat (Tensor): Reconstructed Audio, shape (B x 1 x T)
            mean (Tensor): Mean of latent space, shape (B x zsize)
            var (Tensor): Variance of latent space, shape (B x zsize)
        """
        mean, log_var = self.encoder(xspec.to(self.devices[0]), verbose)

        z = None
        if self.training:
            z = self.samplenew(mean, log_var)
        else:
            z = mean

        x_hat = self.decoder(xau.to(self.devices[1]), z.to(self.devices[1]), jitter, verbose)

        return x_hat.to(self.devices[0]), mean.to(self.devices[0]), log_var.to(self.devices[0])
    
    def sample_value(self, x, device, quantization_channels=256):
        # print(x[:, :, -1].size())
        pdf = self.softmax(x[:, :, -1]).to(device)
        
        cdf = torch.cumsum(pdf, dim=1).to(device)
        batch_size = cdf.size()[0]
        sample_prob = torch.rand(batch_size).to(device)
        pred = torch.zeros(batch_size, dtype=torch.float32).to(device)
        
        for i, prob in enumerate(sample_prob):
            # pred[i] = cdf[i].searchsorted(prob)
            pred[i] = torch.searchsorted(cdf[i], prob)
        # print(probs.size())
        # max_prob = torch.argmax(probs,dim=1).to(device)
        
        # max_prob = max_prob + ((1**0.5)*torch.randn(1)).type(torch.LongTensor).to(device)
        # print(max_prob.size())
        return pred
    
    def inference(self, dataloader, size = 4096, device='cuda'):

        audio_gen = torch.zeros(1, 1, size).to(device[0])
        print(audio_gen.size())
        audio2 = []
        first_loop = True
        for batch_idx, (onehot_input, mfcc_input, target, _) in enumerate(tqdm(dataloader)):
            
            if first_loop:
                audio_gen = onehot_input.to(device[0])
                snippet_gen, _, _ = self.forward(onehot_input.to(device[0]), mfcc_input.to(device[0]), False)
                if self.out_channels == 256:
                    snippet_gen = self.sample_value(snippet_gen, device[0])

                # print(audio_gen.size(), snippet_gen.unsqueeze(0).unsqueeze(0).size())
                audio_gen = torch.cat((audio_gen, snippet_gen.unsqueeze(0)), 1)
                # audio_gen = snippet_gen
                first_loop = False
            else:
                # print("Gen: ", audio_gen[:, -4096:][:, -5:-1].detach().cpu())
                # print("One: ", onehot_input[:, -5:-1].detach().cpu())
                # print("Gen: ", audio_gen[:, -4096:].detach().cpu().size())
                # print("One: ", onehot_input.detach().cpu().size())
                # print("========")
                # print(audio_gen[:, -4096:].size())
                snippet_gen, _, _ = self.forward(audio_gen.to(device[0]).type(torch.LongTensor)[:, -4096:], mfcc_input.to(device[0]), False)
                # snippet_gen, _, _ = self.forward(audio_gen[:, -4096:], mfcc_input.to(device), False)
                if self.out_channels == 256:
                    snippet_gen = self.sample_value(snippet_gen, device[0])
                # print(snippet_gen.item())
                audio_gen = torch.cat((audio_gen, snippet_gen.unsqueeze(0)), 1)

#         if self.out_channels == 256:
#             audio_gen = self.mudec(audio_gen)
            
        return audio_gen





