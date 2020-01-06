
import abc
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

LOSSES = ["VAE", "betaB"]
RECON_DIST = ["bernoulli", "laplace", "gaussian"]


def get_loss_fn(loss_name, **kwargs_parse):

	kwargs_all = dict(rec_dist=kwargs_parse['rec_dist'], steps_anneal=kwargs_parse['reg_anneal'])

	if loss_name = 'VAE':
		return BetaHLoss(beta=1, **kwargs_all)

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

	def _pre_call(self, is_train, storer):
		if is_train:
			self.n_train_steps += 1

		if not is_train or self.n_train_steps % self.record_loss_every == 1:
			storer = storer
		
		else:
			storer = None

		return storer


class BetaHLoss(BaseLoss):
	r"""
    Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    beta : float, optional
        Weight of the kl divergence.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.

    References
    ----------
        [1] Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with
        a constrained variational framework." (2016).
    """
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


class BetaBLoss(BaseLoss):
	r"""
	Compute the Beta-VAE loss as in [1]

    Parameters
    ----------
    C_init : float, optional
        Starting annealed capacity C.

    C_fin : float, optional
        Final annealed capacity C.

    gamma : float, optional
        Weight of the KL divergence term.

    kwargs:
        Additional arguments for `BaseLoss`, e.g. rec_dist`.


	References
    ----------
        [1] Burgess, Christopher P., et al. "Understanding disentangling in
        $\beta$-VAE." arXiv preprint arXiv:1804.03599 (2018).
	"""

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
    """Linear annealing of a parameter."""
    if annealing_steps == 0:
        return fin
    assert fin > init
    delta = fin - init
    annealed = min(init + delta * step / annealing_steps, fin)
    return annealed