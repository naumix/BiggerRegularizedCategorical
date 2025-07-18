import numpy as np
from jaxrl.utils import Batch

class RewardNormalizer(object):
    def __init__(self, num_seeds: int, target_entropy: float, discount: float = 0.99, v_max: float = 10.0, max_steps: int | None = None):
        self.returns_min_norm = np.zeros(num_seeds) + np.inf
        self.returns_max_norm = np.zeros(num_seeds) - np.inf           
        self.effective_horizon = 1 / (1 - discount)
        self.discount = discount
        self.v_max = v_max
        self.target_entropy = target_entropy        
        self.max_steps = max_steps
        self.step = 0
        self.rewards = np.zeros((num_seeds, max_steps)) if max_steps is not None else [[] for _ in range(num_seeds)]
        
    def _calculate_returns_variable_length_trajectory(self, rewards_traj: list, truncate: bool):
        values = np.zeros_like(rewards_traj)
        bootstrap = rewards_traj.mean() * self.effective_horizon if truncate else 0.0
        for i in reversed(range(rewards_traj.shape[0])):
            values[i] = rewards_traj[i] + self.discount * bootstrap
            bootstrap = values[i]
        return values.min(axis=-1), values.max(axis=-1)
    
    def _calculate_returns_fixed_length_trajectory(self):
        values = np.zeros_like(self.rewards)
        bootstrap = self.rewards.mean(-1) * self.effective_horizon
        for i in reversed(range(values.shape[-1])):
            values[:, i] = self.rewards[:, i] + self.discount * bootstrap
            bootstrap = values[:, i]
        return values.min(axis=-1), values.max(axis=-1)
        
    def _update_variable_length_trajectory(self, rewards: np.ndarray, terminal: np.ndarray, truncate: np.ndarray):
        for i, reward in enumerate(rewards):
            self.rewards[i].append(reward)
        done = np.logical_or(terminal, truncate)
        if done.any():
            indx = done.nonzero()[0]
            for j in indx:
                rewards_traj = np.asarray(self.rewards[j])
                value_min, value_max = self._calculate_returns_variable_length_trajectory(rewards_traj, truncate[j])
                self.returns_min_norm[j] = min(self.returns_min_norm[j], value_min) 
                self.returns_max_norm[j] = max(self.returns_max_norm[j], value_max) 
                self.rewards[j] = []
                
    def _update_fixed_length_trajectory(self, rewards: np.ndarray, terminal: np.ndarray, truncate: np.ndarray):
        self.rewards[:, self.step] = rewards
        dones = np.logical_or(terminal, truncate)
        if self.step == self.max_steps - 1:
            assert dones.all()
            v_min, v_max = self._calculate_returns_fixed_length_trajectory()
            self.returns_min_norm = np.where(v_min < self.returns_min_norm, v_min, self.returns_min_norm)
            self.returns_max_norm = np.where(v_max > self.returns_max_norm, v_max, self.returns_max_norm)            
            self.step = 0
        else:
            self.step += 1
        
    def update(self, rewards: np.ndarray, terminal: np.ndarray, truncate: np.ndarray):
        if self.max_steps is not None:
            self._update_fixed_length_trajectory(rewards, terminal, truncate)
        else:
            self._update_variable_length_trajectory(rewards, terminal, truncate)
            
    def normalize(self, batches: Batch, temperature: np.ndarray):
        denominator = np.where(self.returns_max_norm > np.abs(self.returns_min_norm), self.returns_max_norm, np.abs(self.returns_min_norm))
        denominator = (denominator - temperature * self.effective_horizon * self.target_entropy / 2) / self.v_max
        denominator = denominator[batches.task_ids]
        rewards = batches.rewards / denominator
        return Batch(observations=batches.observations, actions=batches.actions, rewards=rewards, masks=batches.masks, next_observations=batches.next_observations, task_ids=batches.task_ids)
   