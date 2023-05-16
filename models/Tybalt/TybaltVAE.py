import torch
from torch import nn

def linear_block(input_size, output_size):
    block = nn.Sequential(
        nn.Linear(input_size, output_size),
        nn.BatchNorm1d(output_size),
        nn.LeakyReLU(0.1))
    return block

class Decoder(nn.Module):

    def __init__(self, output_size, zsize):
        super(Decoder, self).__init__()

        self.decode = nn.Sequential(
            nn.Linear(zsize, 1000),
            nn.Sigmoid(),
            nn.Linear(1000, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decode(x)
class Encoder(nn.Module):

    def __init__(self, input_size, zsize):
        super(Encoder, self).__init__()

        self.linear_1 = linear_block(input_size, 1000)
        self.linear_mu = linear_block(1000, 32)
        self.linear_var = linear_block(1000, 32)

    def forward(self, x):
        x = self.linear_1(x)
        mu = self.linear_mu(x)
        log_var = self.linear_var(x)

        return mu, log_var


class TybaltVAE(nn.Module):

    def __init__(self, input_size, output_size, zsize=32):
        super(TybaltVAE, self).__init__()

        self.encoder = Encoder(
            input_size=input_size,
            zsize=zsize,
        )

        self.decoder = Decoder(
            output_size=output_size,
            zsize=zsize
        )

    def sample(self, mu, log_var):
        if self.training:
            z = torch.randn_like(mu).mul(torch.exp(0.5 * log_var)).add_(mu)
        else:
            z = mu

        return z

    def forward(self, x, verbose = False):
        """Forward step
        Args:
            x (Tensor): Input data, shape (B x T)
            verbose (Bool): To print extra verbose info or not
        Returns:
            x_hat (Tensor): Reconstructed input, shape (B x T)
            mean (Tensor): Mean of latent space, shape (B x zsize)
            var (Tensor): Variance of latent space, shape (B x zsize)
        """
        mean, log_var = self.encoder(x)
        z = self.sample(mean, log_var)
        x_hat = self.decoder(z)

        return x_hat, mean, log_var