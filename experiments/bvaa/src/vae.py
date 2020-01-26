
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from encoder import Encoder
from decoder import Decoder

from utils import weights_init


class VAE(nn.Module):
	def __init__(self, img_size, latent_dim, encoder=None, decoder=None):
		super(VAE, self).__init__()
		if list(img_size[1:]) not in [[32,32], [64,64]]:
			raise RuntimeError("{} sized images not supported. Only (None, 32, 32) and (None, 64, 64) supported. Build your own architecture or reshape images!".format(img_size))
		self.latent_dim = latent_dim
		self.img_size = img_size
		self.num_pixels = self.img_size[1] * self.img_size[2]
		if encoder is None:
			encoder = Encoder(self.img_size, self.latent_dim)
		if decoder is None:
			decoder = Decoder(self.img_size, self.latent_dim)
		
		self.encoder = encoder
		self.decoder = decoder
		self.reset_parameters()

	def reparameterize(self, mean, logvar):
		if self.training:
			std = torch.exp(0.5 * logvar)
			eps = torch.randn_like(std)
			return mean + std * eps

		else:
			return mean


	def forward(self, x):
		latent_distribution = self.encoder(x)
		latent_sample = self.reparameterize(*latent_distribution)
		reconstruct = self.decoder(latent_sample)
		return reconstruct, latent_distribution, latent_sample

	def reset_parameters(self):
		self.apply(weights_init)

	def sample_latent(self, x):
		latent_distribution = self.encoder(x)
		latent_sample = self.reparameterize(*latent_distribution)
		return latent_sample
