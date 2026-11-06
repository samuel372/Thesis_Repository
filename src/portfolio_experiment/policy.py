"""A self-contained shared-stock policy for transfer between unrelated baskets.

The same encoder and scorer process every stock. Averaging stock embeddings gives
basket context without ticker IDs or a privileged input position. SB3 still owns
PPO and supplies the default optimiser class and optimiser settings.
"""
import torch
from torch import nn
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy


class SharedNetwork(nn.Module):
    def __init__(self, n, features, width):
        super().__init__()
        self.n, self.features = n, features
        self.encoder = nn.Sequential(nn.Linear(features, width), nn.Tanh(),
                                     nn.Linear(width, width), nn.Tanh())
        self.scorer = nn.Sequential(nn.Linear(2*width, width), nn.Tanh(), nn.Linear(width, 1))
        self.latent_dim_pi, self.latent_dim_vf = n, width

    def forward(self, features):
        rows = self.encoder(features.reshape(-1, self.n, self.features))
        context = rows.mean(dim=1)
        scores = self.scorer(torch.cat([rows, context[:, None, :].expand_as(rows)], dim=-1)).squeeze(-1)
        # Pass the allocation scores to the environment's clipping and normalisation.
        return scores, context

    def forward_actor(self, features):
        return self.forward(features)[0]

    def forward_critic(self, features):
        return self.forward(features)[1]


class SharedStockPolicy(ActorCriticPolicy):
    def __init__(self, *args, hidden_size=64, **kwargs):
        self.hidden_size = hidden_size
        super().__init__(*args, **kwargs)

    def _build(self, lr_schedule):
        if not isinstance(self.action_dist, DiagGaussianDistribution):
            raise ValueError("This shared policy uses ordinary PPO Gaussian actions, not gSDE.")
        n, features = self.observation_space.shape
        self.mlp_extractor = SharedNetwork(n, features, self.hidden_size)
        self.action_net = nn.Identity()
        self.value_net = nn.Linear(self.hidden_size, 1)
        # One shared exploration scale preserves consistency when stocks swap
        # positions. Independent per-position scales would encode slot identity.
        self.log_std = nn.Parameter(torch.full((1,), self.log_std_init))
        # Construct the optimiser using the class and options supplied by SB3.
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        return {**super()._get_constructor_parameters(), "hidden_size": self.hidden_size}
