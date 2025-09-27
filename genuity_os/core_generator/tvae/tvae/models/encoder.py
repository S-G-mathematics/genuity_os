import torch
import torch.nn as nn
import torch.nn.functional as F


class TVAEEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, config):
        super().__init__()
        self.config = config
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.fc_mu = nn.Linear(config.hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, latent_dim)
        # VampPrior pseudo-inputs
        if config.use_vampprior:
            self.num_pseudos = 10
            self.pseudo_inputs = nn.Parameter(torch.randn(self.num_pseudos, input_dim))

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def get_vampprior(self):
        if self.config.use_vampprior:
            h = F.relu(self.fc1(self.pseudo_inputs))
            h = F.relu(self.fc2(h))
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
        return None, None
