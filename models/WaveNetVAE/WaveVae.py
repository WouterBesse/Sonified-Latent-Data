import torch
from torch import nn
from torchaudio.transforms import MuLawEncoding, MuLawDecoding
import models.WaveNetVAE.WaveVaeOperations as WOP
from models.WaveNetVAE.WaveVaeWavenet import Wavenet
from tqdm.auto import tqdm
import torch.distributions as dist


class Decoder(nn.Module):
    """
    WaveNetVAE Decoder
    """
    def __init__(self, out_channels, upsamples, zsize=128, use_jitter=True,
                 jitter_probability=0.12, init_type='kaiming_n'):
        super().__init__()

        self.use_jitter = use_jitter
        if use_jitter:
            self.jitter = WOP.Jitter(jitter_probability)

        """
        The jittered latent sequence is passed through a single
        convolutional layer with filter length 3 and 128 hidden
        units to mix information across neighboring timesteps.
        (https://github.com/swasun/VQ-VAE-Speech/blob/master/src/models/wavenet_decoder.py#L50)
        """
        self.mix_conv = WOP.Conv1dWrap(in_channels=zsize,
                                out_channels=128,
                                kernel_size=3,
                                padding='same',
                                init_type = init_type)

        self.wavenet = Wavenet(
            layers=10,
            stacks=2,
            out_channels=out_channels,
            res_channels=384,
            skip_channels=256,
            gate_channels=768,
            cond_channels=128,
            kernel_size=3,
            upsample_conditional_features=True,
            upsample_scales=upsamples,
            init_type = init_type
        )

        self.receptive_field = self.wavenet.receptive_field

    def forward(self, x, cond, jitter, verbose):
        """Forward step
        Args:
            x (Tensor): Mono audio signal, shape (B x 1 x T)
            c (Tensor): Local conditioning features, shape (B x cin_channels x T_mfcc)
            jitter (Bool): Argument deciding if we should jitter our condition or not
        Returns:
            X (Tensor): Reconstructed result, shape (B x 256 x T)
        """
        condition = cond
        if self.use_jitter and jitter:
            condition = self.jitter(condition)
        if verbose:
            print("X size before wavenet: ", x.size())

        condition = self.mix_conv(condition)

        x = self.wavenet(x, condition, verbose)

        return x


class Encoder(nn.Module):
    """
    WaveNETVAE Encoder
    """

    def __init__(self, input_size, hidden_dim=768, zsize=128, resblocks=2, relublocks=4, init_type = 'kaiming_n'):
        super().__init__()

        features, timesteps = input_size
        self.zsize = zsize
        self.ReL = nn.LeakyReLU(negative_slope=0.1)

        """
        Preprocessing convolutions with residual connections
        """
        self.conv_1 = WOP.Conv1dWrap(in_channels=features,
                                out_channels=hidden_dim,
                                kernel_size=3,
                                padding='same',
                                init_type = init_type)

        self.conv_2 = WOP.Conv1dWrap(in_channels=features,
                                out_channels=hidden_dim,
                                kernel_size=3,
                                padding='same',
                                init_type = init_type)

        """
        Downsample in the time axis by a factor of 2
        """
        self.downsample = WOP.Conv1dWrap(in_channels=hidden_dim,
                                    out_channels=hidden_dim,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1,
                                    init_type = init_type)

        """
        Residual convs
        """
        self.resblocks = nn.ModuleList()
        for _ in range(resblocks):
            self.resblocks.append(
                WOP.Conv1dWrap(in_channels=hidden_dim,
                          out_channels=hidden_dim,
                          kernel_size=3,
                          padding='same'),
                          init_type = init_type)

        """
        Relu blocks
        """
        self.relublocks = nn.ModuleList()
        for _ in range(relublocks):
            self.relublocks.append(
                nn.Sequential(
                    WOP.Conv1dWrap(in_channels=hidden_dim,
                              out_channels=hidden_dim,
                              kernel_size=3,
                              padding='same',
                              init_type = init_type),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    WOP.Conv1dWrap(in_channels=hidden_dim,
                              out_channels=hidden_dim,
                              kernel_size=3,
                              padding='same',
                              init_type = init_type),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True)))

        """
        The linear block from the WaveNet VQVAE paper.
        This is deceptively not an actual linear layer, but just a convolution layer.
        """
        self.linear = nn.Conv1d(in_channels=hidden_dim,
                                out_channels=zsize * 2,
                                kernel_size=1,
                                bias=False,
                                padding='same',
                                init_type = init_type)

    def forward(self, x, verbose):
        """Forward step
        Args:
            x (Tensor): MFCC, shape (B x features x timesteps)
        Returns:
            mu (Tensor): Latent space mean, shape (B x zsize x timesteps//2)
            log_var (Tensor): Latent space variance, shape (B x zsize x timesteps//2)
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
    """
    Full WaveNetVAE model
    """

    def __init__(self, input_size, num_hiddens, upsamples, zsize=32, resblocks=2, out_channels=256, init_type='kaiming_n'):
        super(WaveNetVAE, self).__init__()
        
        self.out_channels = out_channels

        self.encoder = Encoder(
            input_size=input_size,
            hidden_dim=num_hiddens,
            zsize=zsize,
            resblocks=resblocks,
            init_type = init_type
        )

        self.decoder = Decoder(
            out_channels=out_channels,
            upsamples=upsamples,
            zsize=zsize,
            init_type = init_type
        )

        self.receptive_field = self.decoder.receptive_field
        self.mulaw = MuLawEncoding()
        self.mudec = MuLawDecoding()
        self.N = torch.distributions.Normal(0, 1)

    def sample(self, mu, logvar):
        """Sample from latent space
        Args:
            mu (Tensor): Mean of latent space, shape (B x zsize x timesteps//2)
            logvar (Tensor): Variance of latent space, shape (B x zsize x timesteps//2)
        Returns:
            z (Tensor): Sampled latent space, shape (B x zsize x timesteps//2)
        """
        if self.training:
            z = torch.randn_like(mu).mul(torch.exp(0.5 * logvar)).add_(mu)
        else:
            z = mu
        return z

    def forward(self, xau, xspec, jitter, verbose = False):
        """Forward step
        Args:
            xau (Tensor): Audio, shape (B x 1 x T)
            xspec (Tensor): MFCC, shape (B x features x timesteps)
            jitter (Bool): To jitter the latent space condition or not
        Returns:
            x_hat (Tensor): Reconstructed Audio, shape (B x 1 x T)
            mean (Tensor): Mean of latent space, shape (B x zsize x timesteps//2)
            var (Tensor): Variance of latent space, shape (B x zsize x timesteps//2)
        """
        mean, log_var = self.encoder(xspec, verbose)

        z = None
        if self.training:
            z = self.sample(mean, log_var)
        else:
            z = mean

        x_hat = self.decoder(xau, z, jitter, verbose)

        return x_hat, mean, log_var
    
    def sample_value(self, x, temperature = 1.0, device = 'cuda'):
        """
        Sample from softmax distribution or convert to sample value
        """
        x = x[..., -1]
        if temperature > 0:
            # sample from softmax distribution
            x /= temperature
            x = x.squeeze()
            prob = torch.nn.functional.softmax(x, dim=0)

            cd = dist.Categorical(prob)
            x = cd.sample()
        else:
            # convert to sample value
            x = torch.argmax(x, 0)
        return x.to(device)
    
    def inference(self, dataloader, size = 4096, device='cuda'):

        audio_gen = torch.zeros(1, 1, size).to(device[0])
        print(audio_gen.size())
        first_loop = True
        for batch_idx, (waveform, mfcc_input) in enumerate(tqdm(dataloader)):
            # Loop through all snippets, generate new sample and append to audio_gen
            # First snippet is generated from original audio
            if first_loop:
                audio_gen = waveform.to(device[0])[...,:4096]
                snippet_gen, _, _ = self.forward(audio_gen[...,-4096:].to(device[0]).unsqueeze(1), mfcc_input.to(device[0]), False)
                if self.out_channels == 256:
                    snippet_gen = self.sample_value(snippet_gen, 1.0, device[0])
                
                audio_gen = torch.cat((audio_gen, snippet_gen.unsqueeze(0).unsqueeze(0)), 1)
                first_loop = False
            else:
                snippet_gen, _, _ = self.forward(audio_gen[..., -4096:].to(device[0]).unsqueeze(1), mfcc_input.to(device[0]), False)
                if self.out_channels == 256:
                    snippet_gen = self.sample_value(snippet_gen, 1.0, device[0])

                audio_gen = torch.cat((audio_gen, snippet_gen.unsqueeze(0).unsqueeze(0)), 1)
            
        return audio_gen





