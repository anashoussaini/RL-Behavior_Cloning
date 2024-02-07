from hw1.roble.infrastructure.utils import *
import numpy as np

class ReplayBuffer(object):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, max_size=1000000):
        # Initialize the buffer to store rollouts
        self._paths = []
        self._obs = None
        self._acs = None
        self._rews = None
        self._next_obs = None
        self._terminals = None
        self._max_size = max_size  # Ensuring max size is set correctly

    def __len__(self):
        return self._obs.shape[0] if self._obs is not None else 0

    def add_rollouts(self, paths, concat_rew=True):
        # Convert paths to component arrays
        new_data = convert_listofrollouts(paths, concat_rew)

        # Iterate over each attribute and handle concatenation
        for i, attr in enumerate(['_obs', '_acs', '_rews', '_next_obs', '_terminals']):
            attr_data = getattr(self, attr)
            new_data_i = new_data[i]

            # Debugging information
            print(f"Processing {attr}: existing data shape: {attr_data.shape if attr_data is not None else 'None'}, new data shape: {new_data_i.shape}")

            # Reshape new_data_i to 2D if it's not already
            if len(new_data_i.shape) != 2:
                # Assuming the first dimension is correct, reshape to make the second dimension 1
                new_data_i = new_data_i.reshape(new_data_i.shape[0], -1)
                print(f"Reshaped new data for {attr} to: {new_data_i.shape}")

            if attr_data is None:
                setattr(self, attr, new_data_i[-self._max_size:])
            else:
                combined_data = np.concatenate([attr_data, new_data_i])[-self._max_size:]
                setattr(self, attr, combined_data)

    def sample_random_data(self, batch_size):
        # Ensure there is enough data to sample
        if self.__len__() < batch_size:
            raise ValueError("Not enough data in replay buffer to sample the requested batch size.")

        rand_indices = np.random.choice(self._obs.shape[0], batch_size, replace=False)
        return (self._obs[rand_indices], self._acs[rand_indices], self._rews[rand_indices], self._next_obs[rand_indices], self._terminals[rand_indices])

    def sample_recent_data(self, batch_size=1):
        if self.__len__() < batch_size:
            raise ValueError("Not enough data in replay buffer to sample the requested recent data.")
        return (self._obs[-batch_size:], self._acs[-batch_size:], self._rews[-batch_size:], self._next_obs[-batch_size:], self._terminals[-batch_size:])
