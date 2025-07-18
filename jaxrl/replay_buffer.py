import gymnasium as gym
import numpy as np

import os
import pickle

from jaxrl.utils import Batch

class ParallelReplayBuffer:
    def __init__(self, observation_space: gym.spaces.Box, action_dim: int, capacity: int, num_tasks: int):
        self.observations = np.empty((num_tasks, capacity, observation_space.shape[-1]), dtype=observation_space.dtype)
        self.actions = np.empty((num_tasks, capacity, action_dim), dtype=np.float32)
        self.rewards = np.empty((num_tasks, capacity, ), dtype=np.float32)
        self.masks = np.empty((num_tasks, capacity, ), dtype=np.float32)
        self.next_observations = np.empty((num_tasks, capacity, observation_space.shape[-1]), dtype=observation_space.dtype)
        self.size = 0
        self.insert_index = 0
        self.capacity = capacity
        self.n_parts = 4
        self.num_tasks = num_tasks

    def insert(self, observation: np.ndarray, action: np.ndarray, reward: float, mask: float, next_observation: np.ndarray):
        self.observations[:, self.insert_index] = observation
        self.actions[:, self.insert_index] = action
        self.rewards[:, self.insert_index] = reward
        self.masks[:, self.insert_index] = mask
        self.next_observations[:, self.insert_index] = next_observation
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, num_batches: int):
        indx = np.random.randint(self.size * self.num_tasks, size=(num_batches, batch_size))
        task_indx, sample_indx = np.divmod(indx, self.size)
        observations = self.observations[task_indx, sample_indx, :]
        actions = self.actions[task_indx, sample_indx, :]
        rewards = self.rewards[task_indx, sample_indx]
        masks = self.masks[task_indx, sample_indx]
        next_observations = self.next_observations[task_indx, sample_indx, :]
        return Batch(observations=observations,
                     actions=actions,
                     rewards=rewards,
                     masks=masks,
                     next_observations=next_observations,
                     task_ids=task_indx)    

    def sample_task_batches(self, batch_size_per_task: int | None = None):
        batch_size = 32 if batch_size_per_task is None else batch_size_per_task
        indxs = np.random.randint(self.size, size=batch_size)        
        task_ids = np.zeros((self.num_tasks, batch_size), dtype=np.int32) + np.arange(self.num_tasks, dtype=np.int32)[:, None]
        return Batch(observations=self.observations[:, indxs],
                     actions=self.actions[:, indxs],
                     rewards=self.rewards[:, indxs],
                     masks=self.masks[:, indxs],
                     next_observations=self.next_observations[:, indxs],
                     task_ids=task_ids)  
        
    def save(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        # because of memory limits, we will dump the buffer into multiple files
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_chunk = [
                self.observations[:, i*chunk_size : (i+1)*chunk_size],
                self.actions[:, i*chunk_size : (i+1)*chunk_size],
                self.rewards[:, i*chunk_size : (i+1)*chunk_size],
                self.masks[:, i*chunk_size : (i+1)*chunk_size],
                self.next_observations[:, i*chunk_size : (i+1)*chunk_size]
            ]

            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            pickle.dump(data_chunk, open(data_path_chunk, 'wb'))
        # Save also size and insert_index
        pickle.dump((self.size, self.insert_index), open(os.path.join(save_dir, 'buffer_info'), 'wb'))

    def load(self, save_dir: str):
        data_path = os.path.join(save_dir, 'buffer')
        chunk_size = self.capacity // self.n_parts

        for i in range(self.n_parts):
            data_path_splitted = data_path.split('buffer')
            data_path_splitted[-1] = f'_chunk_{i}{data_path_splitted[-1]}'
            data_path_chunk = 'buffer'.join(data_path_splitted)
            data_chunk = pickle.load(open(data_path_chunk, "rb"))

            self.observations[:, i*chunk_size : (i+1)*chunk_size], \
            self.actions[:, i*chunk_size : (i+1)*chunk_size], \
            self.rewards[:, i*chunk_size : (i+1)*chunk_size], \
            self.masks[:, i*chunk_size : (i+1)*chunk_size], \
            self.next_observations[:, i*chunk_size : (i+1)*chunk_size] = data_chunk
        self.size, self.insert_index = pickle.load(open(os.path.join(save_dir, 'buffer_info'), 'rb'))

