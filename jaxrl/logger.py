import numpy as np
import wandb


def log_to_wandb(step: int, infos: dict, suffix: str = ''):
    dict_to_log = {'timestep': step}
    for info_key in infos:
        for seed, value in enumerate(infos[info_key]):
            dict_to_log[f'seed{seed}/{info_key}{suffix}'] = value
    wandb.log(dict_to_log, step=step)
    
def get_wandb_video(renders: np.ndarray, fps: int = 15):
    videos = []
    for i in range(renders.shape[0]):
        video = wandb.Video(renders[i], fps=fps, format='mp4')
        videos.append(video)
    return videos
    
class EpisodeRecorder:
    
    def __init__(self, num_seeds: int):
        self.returns_online = np.zeros(num_seeds)
        self.goals_online = np.zeros(num_seeds)
        self.counts = np.zeros(num_seeds)
        self.returns_online_episode = np.zeros(num_seeds)
        self.goals_online_episode = np.zeros(num_seeds)
        self.num_seeds = num_seeds
        
    def update(self, rewards: np.ndarray, goals: np.ndarray, terminals: np.ndarray, truncates: np.ndarray):
        self.returns_online_episode += rewards
        self.goals_online_episode += goals
        if terminals.any() or truncates.any():
            done = np.logical_or(terminals, truncates)
            self.counts = np.where(done, self.counts+1, self.counts)
            self.goals_online_episode[self.goals_online_episode > 0.0] = 1.0
            self.goals_online = np.where(done, self.goals_online+self.goals_online_episode, self.goals_online)
            self.returns_online = np.where(done, self.returns_online+self.returns_online_episode, self.returns_online)
            self.returns_online_episode = np.where(done, 0, self.returns_online_episode)
            self.goals_online_episode = np.where(done, 0, self.goals_online_episode)
        
    def _get_scores(self):
        self.counts = np.where(self.counts==0.0, 1e-8, self.counts)
        infos_online_eval = {'goal_online': self.goals_online/self.counts, 'return_online': self.returns_online/self.counts}
        self.returns_online = np.zeros(self.num_seeds)
        self.goals_online = np.zeros(self.num_seeds)
        self.counts = np.zeros(self.num_seeds)
        print(infos_online_eval)
        return infos_online_eval
    
    def log(self, FLAGS, agent, replay_buffer, reward_normalizer, step, eval_env=None, render=False):
        batches_info = replay_buffer.sample_task_batches()
        batches_info = reward_normalizer.normalize(batches_info, agent.get_temperature())
        infos = agent.get_infos(batches_info)
        infos_online_eval = self._get_scores()
        infos = {**infos, **infos_online_eval}
        if FLAGS.offline_evaluation:
            eval_stats = eval_env.evaluate(agent, num_episodes=FLAGS.eval_episodes, temperature=0.0, render=render)
            if render:
                eval_stats['renders'] = get_wandb_video(eval_stats['renders'])
            infos = {**infos, **eval_stats}
        if FLAGS.log_to_wandb:
            log_to_wandb(step, infos)
        return infos
    