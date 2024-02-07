import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.distributions import Normal, Categorical

import numpy as np
import torch

from hw1.roble.infrastructure import pytorch_util as ptu
from hw1.roble.policies.base_policy import BasePolicy

class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, *args, **kwargs):
        super().__init__()

        # Initialize neural networks for discrete and continuous action spaces
        if self._discrete:
            self._logits_na = ptu.build_mlp(input_size=self._ob_dim,
                                            output_size=self._ac_dim,
                                            params=self._network)
            self._logits_na.to(ptu.device)
            self._optimizer = optim.Adam(self._logits_na.parameters(), self._learning_rate)
        else:
            self._mean_net = ptu.build_mlp(input_size=self._ob_dim,
                                           output_size=self._ac_dim,
                                           params=self._network)
            self._mean_net.to(ptu.device)
            if self._deterministic:
                self._optimizer = optim.Adam(self._mean_net.parameters(), self._learning_rate)
            else:
                self._logstd = nn.Parameter(torch.zeros(self._ac_dim, dtype=torch.float32, device=ptu.device))
                self._logstd.to(ptu.device)
                self._optimizer = optim.Adam(itertools.chain([self._logstd], self._mean_net.parameters()), self._learning_rate)

        # Initialize baseline network if required
        if self._nn_baseline:
            self._baseline = ptu.build_mlp(input_size=self._ob_dim, output_size=1, params=self._network)
            self._baseline.to(ptu.device)
            self._baseline_optimizer = optim.Adam(self._baseline.parameters(), self._learning_rate)
        else:
            self._baseline = None

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    """def get_action(self, obs: np.ndarray) -> np.ndarray:
        # Handle shape of observation and ensure no gradient computation
        #obs = ptu.from_numpy(obs)
        with torch.no_grad():
            action_distribution = self.forward(obs)
            if self._discrete:
                return ptu.to_numpy(action_distribution.sample())
            else:
                return ptu.to_numpy(action_distribution.mean)"""
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # Convert numpy array to PyTorch tensor
        obs = ptu.from_numpy(obs) if isinstance(obs,np.ndarray) else obs
        with torch.no_grad():
            action_distribution = self.forward(obs)
            if self._discrete:
                action = action_distribution.sample()
            else:
                action = action_distribution.mean if isinstance(action_distribution, torch.distributions.Distribution) else action_distribution
            return ptu.to_numpy(action)

    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    """def forward(self, observation: torch.FloatTensor):
        if self._discrete:
            logits = self._logits_na(observation)
            return distributions.Categorical(logits=logits)
        else:
            mean = self._mean_net(observation)
            if self._deterministic:
                return distributions.Delta(mean)
            else:
                std = torch.exp(self._logstd)
                return distributions.Normal(mean, std)"""""

    def forward(self, observation: torch.FloatTensor):
        if self._discrete:
            logits = self._logits_na(observation)
            return Categorical(logits=logits)
        else:
            mean = self._mean_net(observation)
            if self._deterministic:
                return mean
            else:
                std = torch.exp(self._logstd)
                return Normal(mean, std)
class MLPPolicySL(MLPPolicy):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loss = nn.MSELoss()

    def update(self, observations, actions, adv_n=None, acs_labels_na=None, qvals=None):
        observations = ptu.from_numpy(observations)
        actions = ptu.from_numpy(actions)
        self._optimizer.zero_grad()
        pred_actions_dist = self(observations)
        if hasattr(pred_actions_dist, 'mean'):
            pred_actions = pred_actions_dist.mean
        else:
            pred_actions = pred_actions_dist
        loss = self._loss(pred_actions, actions)
        loss.backward()
        self._optimizer.step()
        return {'Training Loss': ptu.to_numpy(loss)}

    def update_idm(self, observations, actions, next_observations, adv_n=None, acs_labels_na=None, qvals=None):
        combined_obs = np.concatenate((observations, next_observations), axis=1)
        combined_obs = ptu.from_numpy(combined_obs)
        actions = ptu.from_numpy(actions)
        self._optimizer.zero_grad()
        pred_actions = self(combined_obs)
        loss = self._loss(pred_actions, actions)
        loss.backward()
        self._optimizer.step()
        return {'Training Loss IDM': ptu.to_numpy(loss)}
