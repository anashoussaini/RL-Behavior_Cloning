import numpy as np
import time
import torch
import collections
from hw1.roble.infrastructure import pytorch_util as ptu

############################################
############################################

def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array')):
    ob = env.reset()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0

    while True:
        if render:
            render_environment(env, render_mode, image_obs)

        obs.append(ob)
        ac, _ = process_action(policy.get_action(ob))
        acs.append(ac)
        ob, rew, done, _ = env.step(ac)
        record_step_data(next_obs, rewards, ob, rew)

        if check_rollout_end(done, steps, max_path_length):
            terminals.append(True)
            break
        else:
            terminals.append(False)
            steps += 1

    return create_path_dictionary(obs, image_obs, acs, rewards, next_obs, terminals)

def sample_trajectories(env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array')):
    timesteps_this_batch = 0
    paths = []
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)
    return paths, timesteps_this_batch

def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array')):
    paths = []
    for _ in range(ntraj):
        path = sample_trajectory(env, policy, max_path_length, render, render_mode)
        paths.append(path)
    return paths

############################################
############################################

def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation": np.array(obs, dtype=np.float32),
            "image_obs": np.array(image_obs, dtype=np.uint8),
            "reward": np.array(rewards, dtype=np.float32),
            "action": np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}

def convert_listofrollouts(paths, concat_rew=True):
    observations, actions, rewards, next_observations, terminals = zip(*[(path["observation"], path["action"], path["reward"], path["next_observation"], path["terminal"]) for path in paths])
    rewards = np.concatenate(rewards) if concat_rew else list(itertools.chain.from_iterable(rewards))
    return np.concatenate(observations), np.concatenate(actions), rewards, np.concatenate(next_observations), np.concatenate(terminals)

############################################
############################################

def get_pathlength(path):
    return len(path["reward"])

def flatten(matrix):
    if isinstance(matrix, collections.abc.Sequence) and isinstance(matrix[0], (collections.abc.Sequence, np.ndarray)):
        return [item for sublist in matrix for item in sublist]
    else:
        return matrix

# Additional helper functions for clarity and modularity
def render_environment(env, render_mode, image_obs):
    if 'rgb_array' in render_mode:
        img = env.render(mode='rgb_array')
        image_obs.append(img)
    if 'human' in render_mode:
        env.render(mode='human')
        time.sleep(env.model.opt.timestep)

def process_action(action):
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    return action, action[0] if len(action.shape) > 0 else action

def record_step_data(next_obs, rewards, ob, rew):
    next_obs.append(ob)
    rewards.append(rew)

def check_rollout_end(done, steps, max_path_length):
    return done or steps >= max_path_length

def create_path_dictionary(obs, image_obs, acs, rewards, next_obs, terminals):
    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.stack(image_obs, axis=0) if image_obs else np.array([]),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32)
    }

