import numpy as np

import torch
import torch.nn as nn

class Decoder(nn.Module):
	r"""
	References:
            [1] Burgess, Christopher P., et al. "Understanding disentangling in
            $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
	"""
	def __init__(self, img_size, latent_dim):

		super(Decoder, self).__init__()

		hidden_channels = 32
		kernel_size = 4
		hidden_dim = 256

		self.img_size = img_size
		self.latent_dim = latent_dim
		
		self.reshape = (hidden_channels, kernel_size, kernel_size)

		n_channels = self.img_size[0]

		self.lin1 = nn.Linear(latent_dim, hidden_dim)
		self.lin2 = nn.Linear(hidden_dim, hidden_dim)
		self.lin3 = nn.Linear(hidden_dim, np.product(self.reshape))

		cnn_kwargs = dict(stride=2, padding=1)

		if self.img_size[1] == self.img_size[2] == 64:
			self.convT_64 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size, **cnn_kwargs)

		self.convT1 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size, **cnn_kwargs)
		self.convT2 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size, **cnn_kwargs)
		self.convT3 = nn.ConvTranspose2d(hidden_channels, n_channels, kernel_size, **cnn_kwargs)

	def forward(self, z):
		batch_size = z.size(0)

		x = torch.relu(self.lin1(z))
		x = torch.relu(self.lin2(x))
		x = torch.relu(self.lin3(x))
		
		x = x.view(batch_size, *self.reshape)

		if self.img_size[1] == self.img_size[2] == 64:
			x = torch.relu(self.convT_64(x))

		x = torch.relu(self.convT1(x))
		x = torch.relu(self.convT2(x))

		x = torch.sigmoid(self.convT3(x))

		return x