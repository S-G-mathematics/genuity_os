import torch
import torch.nn as nn
import torch.nn.functional as F


class TVAEDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, config, n_cont, n_cat):
        super().__init__()
        self.config = config
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.n_cont = n_cont
        self.n_cat = n_cat
        self.fc1 = nn.Linear(latent_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.hidden_dim)
        # Multi-head: separate heads for continuous and categorical
        if config.use_multi_head_decoder:
            self.cont_head = nn.Linear(config.hidden_dim, n_cont)
            self.cat_head = nn.Linear(config.hidden_dim, n_cat)
        else:
            self.out = nn.Linear(config.hidden_dim, output_dim)
        # Transformer attention (optional)
        if config.use_transformer_attention:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim, nhead=4
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        else:
            self.transformer = None

    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        if self.transformer is not None:
            h = self.transformer(h.unsqueeze(1)).squeeze(1)
        if self.config.use_multi_head_decoder:
            cont_out = self.cont_head(h)
            cat_out = self.cat_head(h)
            return cont_out, cat_out
        else:
            return self.out(h)
