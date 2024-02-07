from hw1.roble.infrastructure.replay_buffer import ReplayBuffer
from hw1.roble.policies.MLP_policy import MLPPolicySL
from .base_agent import BaseAgent
import torch
import numpy as np
import pickle
from hw1.roble.infrastructure import pytorch_util as ptu

class BCAgent(BaseAgent):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, env, agent_params, **kwargs):
        super(BCAgent, self).__init__()

        self.env_params = kwargs
        # actor/policy
        self._actor = MLPPolicySL(
            **self._agent_params,
            deterministic=False,
            nn_baseline=False,
        )

        self.idm_params = self._agent_params
        # Explicitly set the input dimension for the IDM
        self.idm_params['ob_dim'] = env.observation_space.shape[0] * 2

        self._idm = MLPPolicySL(
            **self.idm_params,
            deterministic=True,
            nn_baseline=False,
        )

        # replay buffer
        self.reset_replay_buffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = self._actor.update(ob_no, ac_na)
        return log

    def train_idm(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        #combined_obs = np.concatenate((ob_no, next_ob_no), axis=1)
        log = self._idm.update_idm(ob_no, ac_na,next_ob_no)
        return log

    def use_idm(self, paths):
        self._idm.eval()
        all_labelled_data = []

        for episode_idx in range(len(paths)):
            observations = torch.tensor(paths[episode_idx]["observation"], dtype=torch.float32)
            next_observations = torch.tensor(paths[episode_idx]["next_observation"], dtype=torch.float32)

            full_input = torch.cat((observations, next_observations), dim=1)
            action = self._idm.get_action(full_input)

            ep_labelled_data = {
                "observation": observations.squeeze().numpy(),
                "next_observation": next_observations.squeeze().numpy(),
                "action": action.squeeze(),
                "terminal": np.array(paths[episode_idx]["terminal"]),
                "reward": np.array(paths[episode_idx]["reward"]),
                "image_obs": np.array(paths[episode_idx]["image_obs"]),
            }
            all_labelled_data.append(ep_labelled_data)
            print("Index: ", episode_idx, "was labelled")

        save_path = self.env_params["expert_data"].replace("expert_data_", "labelled_data_")
        with open(save_path, "wb") as f:
            pickle.dump(all_labelled_data, f)
            print("Saved labelled data to labelled_data.pkl")

    def add_to_replay_buffer(self, paths):
        self._replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self._replay_buffer.sample_random_data(batch_size)

    def save(self, path):
        return self._actor.save(path)

    def reset_replay_buffer(self):
        self._replay_buffer = ReplayBuffer(self._agent_params['max_replay_buffer_size'])
