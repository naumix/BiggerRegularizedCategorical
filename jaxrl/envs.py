import numpy as np
import gymnasium as gym
import random
from gymnasium.wrappers import FlattenObservation, RescaleAction, TimeLimit
    
class FlattenObservationShadowhandWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        subspaces = env.observation_space
        obs_box  = subspaces['observation']
        goal_box = subspaces['desired_goal']
        low  = np.concatenate((obs_box.low,  goal_box.low),  axis=-1)
        high = np.concatenate((obs_box.high, goal_box.high), axis=-1)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=obs_box.dtype)

    def observation(self, obs: dict) -> np.ndarray:
        return np.concatenate((obs['observation'], obs['desired_goal']), axis=-1)

def _make_env_dmc(env_name: str, seed: int = 0) -> gym.Env:
    from jaxrl.additional_tasks.dmc_envs import cheetah, walker, hopper, reacher, ball_in_cup, pendulum, fish
    from dm_control import suite
    suite.ALL_TASKS = suite.ALL_TASKS + suite._get_tasks('custom')
    suite.TASKS_BY_DOMAIN = suite._get_tasks_by_domain(suite.ALL_TASKS)
    from shimmy import DmControlCompatibilityV0
    domain_name, task_name = env_name.split("-")
    env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={'random': seed},
    )
    env = DmControlCompatibilityV0(env, render_mode='rgb_array', render_kwargs={'camera_id': 1})
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = FlattenObservation(env)
    env = RescaleAction(env, -1.0, 1.0)
    return env

def _make_env_metaworld(env_name: str, seed: int = 0) -> gym.Env:
    try:
        constructor = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
    except:
        from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)
        constructor = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE[env_name]
    env = constructor(seed=int(seed))
    env = TimeLimit(env, 200)
    return env
    
def _make_env_humanoidbench(env_name: str, seed: int = 0) -> gym.Env:
    import humanoid_bench
    from humanoid_bench.env import ROBOTS, TASKS
    env = gym.make(env_name, autoreset=False)
    return env

def _make_env_shadowhand(env_name: str, seed: int = 0) -> gym.Env:
    from jaxrl.additional_tasks import dex_envs
    env_version = 'v1'
    reward_type = 'sparse'
    name = f'{env_name}-rotate-{env_version}'
    env = gym.make(name, reward_type=reward_type)
    env = FlattenObservationShadowhandWrapper(env)
    return env

def make_env(env_name: str, seed: int = 0) -> gym.Env:
    env = None
    if '-goal-observable' in env_name:
        env = _make_env_metaworld(env_name, seed)
    elif '-v0' in env_name:
        env = _make_env_humanoidbench(env_name, seed)
    elif '-' in env_name:
        env = _make_env_dmc(env_name, seed)
    else:
        env = _make_env_shadowhand(env_name, seed)
    return env

class ParallelEnv():
    
    def __init__(self, env_names: list, seed: int = 0):
        
        np.random.seed(seed)
        random.seed(seed)
        
        envs = []
        obs_dims = np.zeros(len(env_names), dtype=np.int32)
        act_dims = np.zeros(len(env_names), dtype=np.int32)
        for i, env_name in enumerate(env_names):
            envs.append(make_env(env_name, seed))
            obs_dims[i] = envs[-1].observation_space.shape[0]
            act_dims[i] = envs[-1].action_space.shape[0]

        max_state_dim = int(np.max(obs_dims))
        max_action_dim = int(np.max(act_dims))
        state_dim_differences = max_state_dim - obs_dims

        
        observation_space = gym.spaces.Box(low=(np.ones(max_state_dim, dtype=np.float64)[None, :] - np.inf).repeat(len(envs), axis=0),
                                           high=(np.ones(max_state_dim, dtype=np.float64)[None, :] + np.inf).repeat(len(envs), axis=0),
                                           shape=(len(envs), max_state_dim),
                                           dtype=envs[-1].observation_space.dtype)        
                
        action_space = gym.spaces.Box(low=(np.ones(max_action_dim, dtype=np.float64)[None, :] * -1).repeat(len(envs), axis=0),
                                      high=(np.ones(max_action_dim, dtype=np.float64)[None, :]).repeat(len(envs), axis=0),
                                      shape=(len(envs), max_action_dim),
                                      dtype=envs[-1].action_space.dtype)
    
        
        self.envs = envs
        self.obs_dims = obs_dims
        self.act_dims = act_dims
        self.state_dim_differences = state_dim_differences
        self.observation_space = observation_space
        self.action_space = action_space
        self.num_tasks = len(envs)
        
    def _reset_idx(self, idx: int):
        seed = np.random.randint(0, 1e8)
        state, _ = self.envs[idx].reset(seed=seed)
        state = np.concatenate((state, np.zeros(self.state_dim_differences[idx], dtype=np.float32)), axis=0)
        return state
    
    def generate_masks(self, terminals: np.ndarray, truncates: np.ndarray):
        masks = 1 - (terminals * (1 - truncates))
        return masks
    
    def reset_where_done(self, states: np.ndarray, terminals: np.ndarray, truncates: np.ndarray):
        for j, (terminal, truncate) in enumerate(zip(terminals, truncates)):
            if (terminal == True) or (truncate == True):
                states[j], terminals[j], truncates[j] = self._reset_idx(j), False, False
        return states, terminals, truncates
    
    def reset(self):
        states = []
        for i, env in enumerate(self.envs):
            states.append(self._reset_idx(i))
        return np.stack(states)

    def _get_goal(self, info: dict):
        if 'success' in info:
            goal = info['success']
        elif 'is_success' in info:
            goal = info['is_success']
        elif 'solved' in info:
            goal = info['solved']
        else:
            goal = 0
        return goal
        
    def step(self, actions: np.ndarray):
        states, rewards, terminals, truncates, goals = [], [], [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            state, reward, terminal, truncate, info = env.step(action[:self.act_dims[i]])
            state = np.concatenate((state, np.zeros(self.state_dim_differences[i], dtype=np.float64)), axis=0)
            states.append(state)
            rewards.append(reward)
            terminals.append(terminal)
            truncates.append(truncate)
            goals.append(self._get_goal(info))
        return np.stack(states), np.stack(rewards), np.stack(terminals), np.stack(truncates), np.stack(goals)   

    def evaluate(self, agent, num_episodes, temperature=0.0, render=False, max_render_steps=5000, render_frameskip=4):
        n_rollouts = np.zeros(self.num_tasks)
        returns = np.zeros(self.num_tasks)
        goals = np.zeros(self.num_tasks)
        mask = np.ones(self.num_tasks)
        mask_goals = np.ones(self.num_tasks)
        observations = self.reset()
        if render:
            renders = []
        i = 0
        while True:
            if render:
                if i % render_frameskip == 0:
                    if i < max_render_steps:
                        env_renders = self.render()
                        renders.append(env_renders)
            actions = agent.sample_actions(observations, temperature=temperature)
            #actions = envs.action_space.sample()
            next_observations, rewards, terms, truns, success = self.step(actions)
            returns += rewards * mask
            goals += success * mask_goals
            mask_goals = np.where(success, 0, mask_goals)
            mask_goals = np.where(np.logical_or(terms, truns), 1, mask_goals)
            observations = next_observations
            n_rollouts += np.logical_or(terms, truns)
            observations, terms, truns = self.reset_where_done(observations, terms, truns)
            mask = np.where(n_rollouts >= num_episodes, 0, 1)
            i += 1
            if n_rollouts.min() == num_episodes:
                break
        if render:
            renders = np.stack(renders)
            renders = np.transpose(renders, (1, 0, 4, 2, 3))
            return {'goal': goals/num_episodes, 'return': returns/num_episodes, 'renders': renders}
        else:
            return {'goal': goals/num_episodes, 'return': returns/num_episodes}

    def render(self):
        renders = []
        for i, env in enumerate(self.envs):
            render = env.render()
            renders.append(render)
        renders = np.stack(renders)
        return renders
