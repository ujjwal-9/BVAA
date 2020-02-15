
import abc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


LOSSES = ["betaH", "betaB"]
RECON_DIST = ["bernoulli", "laplace", "gaussian"]


def get_loss_fn(loss_name, **kwargs_parse):

	kwargs_all = dict(rec_dist=kwargs_parse['rec_dist'], steps_anneal=kwargs_parse['reg_anneal'])

	if loss_name == 'betaH':
		return BetaHLoss(beta=kwargs_parse["betaH_B"], **kwargs_all)

	elif loss_name == 'betaB':
		return BetaBLoss(C_init=kwargs_parse['betaB_initC'], 
						 C_fin=kwargs_parse['betaB_finC'], 
						 gamma=kwargs_parse['betaB_G'],
						 **kwargs_all)

	else:
		assert loss_name not in LOSSES
		raise ValueError("Unknown loss: {}".format(loss_name))


class BaseLoss(abc.ABC):

	def __init__(self, record_loss_every=50, rec_dist="bernoulli", steps_anneal=0):
		self.n_train_steps = 0
		self.record_loss_every = record_loss_every
		self.rec_dist = rec_dist
		self.steps_anneal = steps_anneal

	@abc.abstractmethod
	def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
		"""
		Calculates loss for a batch of data.

		Parameters
		----------
		data : torch.Tensor
			Input data (e.g. batch of images). Shape : (batch_size, n_chan,
			height, width).

		recon_data : torch.Tensor
			Reconstructed data. Shape : (batch_size, n_chan, height, width).

		latent_dist : tuple of torch.tensor
			sufficient statistics of the latent dimension. E.g. for gaussian
			(mean, log_var) each of shape : (batch_size, latent_dim).

		is_train : bool
			Whether currently in train mode.

		storer : dict
			Dictionary in which to store important variables for vizualisation.

		kwargs:
			Loss specific arguments
		"""

	def _pre_call(self, is_train, storer):
		if is_train:
			self.n_train_steps += 1

		if not is_train or self.n_train_steps % self.record_loss_every == 1:
			storer = storer
		else:
			storer = None

		return storer


class BetaHLoss(BaseLoss):
	
	def __init__(self, beta=4, **kwargs):
		super().__init__(**kwargs)
		self.beta = beta

	def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
		storer = _pre_call(is_train, storer)

		reconst_loss = _reconstruction_loss(data, recon_data, storer=storer, distribution=self.rec_dist)

		kl_loss = _kl_normal_loss(*latent_dist, storer)
		anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal) if is_train else 1)

		loss = reconst_loss + anneal_reg * (self.beta * kl_loss)

		if storer is not None:
			storer['loss'].append(loss.item())

		return loss

class BetaULoss(BaseLoss):
	
	def __init__(self, beta=4, **kwargs):
		super().__init__(**kwargs)
		self.beta = beta

	def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
		storer = _pre_call(is_train, storer)

		reconst_loss = _reconstruction_loss(data, recon_data, storer=storer, distribution=self.rec_dist)

		kl_loss = _kl_normal_loss(*latent_dist, storer)
		anneal_reg = (linear_annealing(0, 1, self.n_train_steps, self.steps_anneal) if is_train else 1)

		loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
		
		loss = reconst_loss + anneal_reg * (self.beta * kl_loss)

		if storer is not None:
			storer['loss'].append(loss.item())

		return loss


class BetaBLoss(BaseLoss):

	def __init__(self, C_init=0., C_fin=20., gamma=100., **kwargs):
		super().__init__(**kwargs)
		self.gamma = gamma
		self.C_fin = C_fin
		self.C_init = C_init

	def __call__(self, data, recon_data, latent_dist, is_train, storer, **kwargs):
		storer = self._pre_call(is_train, storer)

		reconst_loss = _reconstruction_loss(data, recon_data, storer=storer, distribution=self.rec_dist)

		kl_loss = _kl_normal_loss(*latent_dist, storer)

		C = (linear_annealing(self.C_init, self.C_fin, self.n_train_steps, self.steps_anneal)
			 if is_train else self.C_fin)

		loss = reconst_loss + self.gamma * (kl_loss - C).abs()

		if storer is not None:
			storer['loss'].append(loss.item())

		return loss


def _reconstruction_loss(data, recon_data, distribution="bernoulli", storer=None):
	batch_size, n_channels, height, width = recon_data.size()
	is_colored = n_channels == 3

	if distribution == "bernoulli":
		loss = F.binary_cross_entropy(recon_data, data, reduction="sum")

	elif distribution == "gaussian":
		loss = F.mse_loss(recon_data * 255, data * 255, reduction="sum") / 255

	elif distribution == "laplace":
		loss = F.l1_loss(recon_data, data, reduction="sum")
		loss = loss * 3
		loss = loss * (loss != 0)

	else:
		assert distribution not in RECON_DIST
		raise ValueError("Unknown distribution: {}".format(distribution))

	loss = loss / batch_size

	if storer is not None:
		storer['reconst_loss'].append(loss.item())

	return loss


def _kl_normal_loss(mean, logvar, storer=None):
	latent_dim = mean.size(1)
	latent_kl = 0.5 * (-1 - logvar + mean.pow(2) + logvar.exp()).mean(dim=0)
	total_kl = latent_kl.sum()

	if storer is not None:
		storer['kl_loss'].append(total_kl.item())
		for i in range(latent_dim):
			storer['kl_loss_' + str(i)].append(latent_kl[i].item())

	return total_kl

def linear_annealing(init, fin, step, annealing_steps):
	if annealing_steps == 0:
		return fin
	assert fin > init
	delta = fin - init
	annealed = min(init + delta * step / annealing_steps, fin)
	return annealed


class SiameseLoss(nn.Module):
	def __init__(self, margin):
		super(Loss, self).__init__()
		self.margin = margin
	
	def forward(self, output1, output2, label):
		"""Define the computation performed at every call."""
		euclidean_distance = F.pairwise_distance(output1, output2)
		distance_from_margin = torch.clamp(torch.pow(euclidean_distance, 2) - self.margin, max=50.0)
		exp_distance_from_margin = torch.exp(distance_from_margin)
		distance_based_loss = (1.0 + math.exp(-self.margin)) / (1.0 + exp_distance_from_margin)
		similar_loss = -0.5 * (1 - label) * torch.log(distance_based_loss)
		dissimilar_loss = -0.5 * label * torch.log(1.0 - distance_based_loss)
		return torch.mean(similar_loss + dissimilar_loss)
	
	def predict(self, output1, output2, threshold_factor=0.5):
		"""Predict a dissimilarity label given two embeddings.
		Return `1` if dissimilar.
		"""
		return F.pairwise_distance(output1, output2) > self.margin * threshold_factor


class DistanceBasedLoss(nn.Module):
	def __init__(self, margin):
		"""Set parameters of distance-based loss function."""
		super(DistanceBasedLoss, self).__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		"""Define the computation performed at every call."""
		euclidean_distance = F.pairwise_distance(output1, output2)
		distance_from_margin = torch.clamp(torch.pow(euclidean_distance, 2) - self.margin, max=50.0)
		exp_distance_from_margin = torch.exp(distance_from_margin)
		distance_based_loss = (1.0 + math.exp(-self.margin)) / (1.0 + exp_distance_from_margin)
		similar_loss = -0.5 * (1 - label) * torch.log(distance_based_loss)
		dissimilar_loss = -0.5 * label * torch.log(1.0 - distance_based_loss)
		return torch.mean(similar_loss + dissimilar_loss)
	
	def predict(self, output1, output2, threshold_factor=0.5):
		"""Predict a dissimilarity label given two embeddings.
		Return `1` if dissimilar.
		"""
		return F.pairwise_distance(output1, output2) > self.margin * threshold_factor


class ContrastiveLoss(nn.Module):
	def __init__(self, margin):
		"""Set parameters of contrastive loss function."""
		super(ContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		"""Define the computation performed at every call."""
		euclidean_distance = F.pairwise_distance(output1, output2)
		clamped = torch.clamp(self.margin - euclidean_distance, min=0.0)
		similar_loss = (1 - label) * 0.5 * torch.pow(euclidean_distance, 2)
		dissimilar_loss = label * 0.5 * torch.pow(clamped, 2)
		contrastive_loss = similar_loss + dissimilar_loss

		return torch.mean(contrastive_loss)

	def predict(self, output1, output2, threshold_factor=0.5):
		"""Predict a dissimilarity label given two embeddings.
		Return `1` if dissimilar.
		"""
		return F.pairwise_distance(output1, output2) > self.margin * threshold_factor


class SSIM(nn.Module):
	"""Wrapper class used to compute the structural similarity index."""

	def __init__(self, window_size=11, size_average=True):
		super(SSIM, self).__init__()
		self.window_size = window_size
		self.size_average = size_average
		self.channel = 1
		self.window = create_window(window_size, self.channel)

	def forward(self, img1, img2):
		"""Execute the computation of the structural similarity index."""
		(_, channel, _, _) = img1.size()

		if channel == self.channel and self.window.data.type() == img1.data.type():
			window = self.window
		else:
			window = create_window(self.window_size, channel)

			if img1.is_cuda:
				window = window.cuda(img1.get_device())
			window = window.type_as(img1)

			self.window = window
			self.channel = channel

		return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def gaussian(window_size, sigma):
	"""Compute gaussian window, that is a tensor with values of the bell curve."""
	gauss = torch.Tensor(
		[math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
	return gauss/gauss.sum()


def create_window(window_size, channel):
	"""Generate a two dimensional window with desired number of channels."""
	_1D_window = gaussian(window_size, 1.5).unsqueeze(1)
	_2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
	window = torch.Tensor(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
	return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
	"""Compute the structural similarity index between two images."""
	mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
	mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

	mu1_sq = mu1.pow(2)
	mu2_sq = mu2.pow(2)
	mu1_mu2 = mu1*mu2

	sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
	sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
	sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

	C1 = 0.01**2
	C2 = 0.03**2

	ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
													(sigma1_sq + sigma2_sq + C2))

	if size_average:
		return ssim_map.mean()
	else:
		return ssim_map.mean(1).mean(1).mean(1)