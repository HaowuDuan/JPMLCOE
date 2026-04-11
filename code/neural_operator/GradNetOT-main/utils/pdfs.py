import torch
import torch.nn.functional as F
import torch.distributions as D

class MultivariateNormal:
    def __init__(self, mean, cov):
        """
        mean: Tensor of shape (d,)
        cov: Tensor of shape (d, d) (covariance matrix, must be symmetric PSD)
        """
        self.mean = mean
        self.cov = cov
        self.d = mean.shape[0]
        
        # Precompute Cholesky for sampling and log det
        self.L = torch.linalg.cholesky(cov)
        self.log_det_cov = 2 * torch.sum(torch.log(torch.diag(self.L)))
        self.cov_inv = torch.cholesky_inverse(self.L)
        self.norm_const = -0.5 * self.d * torch.log(torch.tensor(2 * torch.pi))  # normalizing constant without det

    def sample(self, n):
        """
        Sample n points from the MVN.
        Returns: Tensor of shape (n, d)
        """
        z = torch.randn(n, self.d)
        return self.mean + z @ self.L.T

    def log_pdf(self, x):
        """
        Compute log PDF at points x.
        x: Tensor of shape (..., d), can be on CPU or GPU.
        Returns: Tensor of shape (...)
        """
        device = x.device
        mean = self.mean.to(device)
        cov_inv = self.cov_inv.to(device)

        diff = x - mean
        mahal = torch.einsum('...i,ij,...j->...', diff, cov_inv, diff)
        return self.norm_const - 0.5 * self.log_det_cov - 0.5 * mahal


class MixtureOfGaussians(torch.distributions.Distribution):
    def __init__(self, weights, means, covariances, device=torch.device('cpu')):
        """
        weights: Tensor of shape [num_components]
        means: Tensor of shape [num_components, 2]
        covariances: Tensor of shape [num_components, 2, 2]
        """
        super().__init__()
        self.num_components = weights.shape[0]
        self.cat = D.Categorical(weights.to(device))
        self.components = D.MultivariateNormal(loc=means.to(device), covariance_matrix=covariances.to(device))

    def sample(self, num_samples):
        idx = self.cat.sample((num_samples,))
        return self.components.sample((num_samples,))[torch.arange(num_samples), idx]

    def log_pdf(self, x):
        # x: [..., 2]
        comp_log_probs = self.components.log_prob(x.unsqueeze(-2))  # [..., num_components]
        log_mix_weights = torch.log(self.cat.probs.to(x.device))
        return torch.logsumexp(log_mix_weights + comp_log_probs, dim=-1)