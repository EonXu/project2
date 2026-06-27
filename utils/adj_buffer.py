import numpy as np
from .util import get_dim_from_space
from .segment_tree import SumSegmentTree, MinSegmentTree
import torch


def _cast(x):
    return x.transpose(2, 0, 1, 3)


class AdjBuffer(object):
    def __init__(self, policy_info, policy_agents, num_factor, buffer_size, episode_length, use_same_share_obs,
                 use_avail_acts, use_reward_normalization=False, gamma=0.97, gae_lambda=0.95, hidden_size=64,
                 adj_return_adv_coef=1.0, adj_factor_adv_coef=0.0, seed=0):
        """
        Replay buffer class for training RNN policies. Stores entire episodes rather than single transitions.

        :param policy_info: (dict) maps policy id to a dict containing information about corresponding policy.
        :param policy_agents: (dict) maps policy id to list of agents controled by corresponding policy.
        :param buffer_size: (int) max number of transitions to store in the buffer.
        :param use_same_share_obs: (bool) whether all agents share the same centralized observation.
        :param use_avail_acts: (bool) whether to store what actions are available.
        :param use_reward_normalization: (bool) whether to use reward normalization.
        """
        self.policy_info = policy_info
        self.rng = np.random.RandomState(int(seed))

        self.policy_buffers = {
            p_id: AdjPolicyBuffer(
                buffer_size,
                episode_length,
                len(policy_agents[p_id]),
                num_factor,
                self.policy_info[p_id]['obs_space'],
                self.policy_info[p_id]['share_obs_space'],
                self.policy_info[p_id]['act_space'],
                use_same_share_obs,
                use_avail_acts,
                use_reward_normalization,
                gamma,
                gae_lambda,
                hidden_size,
                adj_return_adv_coef,
                adj_factor_adv_coef,
                seed=int(seed) + 1000 + i,
            )
            for i, p_id in enumerate(self.policy_info.keys())
        }

    def __len__(self):
        return self.policy_buffers['policy_0'].filled_i

    def insert(self, num_insert_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts, adj=None,
               prob_adj=None, q_tot=None, f_v=None, f_q=None, rnn_states=None):
        """
        Insert a set of episodes into buffer. If the buffer size overflows, old episodes are dropped.

        :param num_insert_episodes: (int) number of episodes to be added to buffer
        :param obs: (dict) maps policy id to numpy array of observations of agents corresponding to that policy
        :param share_obs: (dict) maps policy id to numpy array of centralized observation corresponding to that policy
        :param acts: (dict) maps policy id to numpy array of actions of agents corresponding to that policy
        :param rewards: (dict) maps policy id to numpy array of rewards of agents corresponding to that policy
        :param dones: (dict) maps policy id to numpy array of terminal status of agents corresponding to that policy
        :param dones_env: (dict) maps policy id to numpy array of terminal status of env
        :param valid_transition: (dict) maps policy id to numpy array of whether the corresponding transition is valid of agents corresponding to that policy
        :param avail_acts: (dict) maps policy id to numpy array of available actions of agents corresponding to that policy

        :return: (np.ndarray) indexes in which the new transitions were placed.
        """
        for p_id in self.policy_info.keys():
            idx_range = self.policy_buffers[p_id].insert(num_insert_episodes, np.array(obs[p_id]),
                                                         np.array(share_obs[p_id]), np.array(acts[p_id]),
                                                         np.array(rewards[p_id]), np.array(dones[p_id]),
                                                         np.array(dones_env[p_id]), np.array(avail_acts[p_id]),
                                                         np.array(adj[p_id]), np.array(prob_adj[p_id]),
                                                         np.array(q_tot[p_id]),
                                                         np.array(f_v[p_id]), np.array(f_q[p_id]),
                                                         np.array(rnn_states[p_id]))
        return idx_range

    '''def sample(self, batch_size,data_chunk_length,num_mini_batch):
        """
        Sample a set of episodes from buffer, uniformly at random.
        :param batch_size: (int) number of episodes to sample from buffer.

        :return: obs: (dict) maps policy id to sampled observations corresponding to that policy
        :return: share_obs: (dict) maps policy id to sampled observations corresponding to that policy
        :return: acts: (dict) maps policy id to sampled actions corresponding to that policy
        :return: rewards: (dict) maps policy id to sampled rewards corresponding to that policy
        :return: dones: (dict) maps policy id to sampled terminal status of agents corresponding to that policy
        :return: dones_env: (dict) maps policy id to sampled environment terminal status corresponding to that policy
        :return: valid_transition: (dict) maps policy_id to whether each sampled transition is valid or not (invalid if corresponding agent is dead)
        :return: avail_acts: (dict) maps policy_id to available actions corresponding to that policy
        """
        replace = self.__len__() < batch_size
        inds = self.rng.choice(self.__len__(), batch_size, replace=replace)
        obs_batch, share_obs_batch, dones_batch, dones_env_batch, adj_batch, prob_adj_batch, advantages_batch, f_advts_batch, rnn_obs_batch = {}, {}, {}, {}, {}, {}, {}, {}, {}
        for p_id in self.policy_info.keys():
            obs_batch[p_id], share_obs_batch[p_id], dones_batch[p_id], dones_env_batch[p_id], adj_batch[p_id], prob_adj_batch[p_id], advantages_batch[p_id], f_advts_batch[p_id], rnn_obs_batch[p_id] = self.policy_buffers[p_id].sample_inds(inds,data_chunk_length,num_mini_batch)

        return obs_batch, share_obs_batch, dones_batch, dones_env_batch, adj_batch, prob_adj_batch, advantages_batch, f_advts_batch, rnn_obs_batch'''

    def compute_advantage(self, idx, value_normalizer=None):

        for p_id in self.policy_info.keys():
            self.policy_buffers[p_id].compute_advantage(idx)
        return idx


class AdjPolicyBuffer(object):
    def __init__(self, buffer_size, episode_length, num_agents, num_factor, obs_space, share_obs_space, act_space,
                 use_same_share_obs, use_avail_acts, use_reward_normalization=False, gamma=0.97, gae_lambda=0.95,
                 hidden_size=64, adj_return_adv_coef=1.0, adj_factor_adv_coef=0.0, seed=0):
        """
        Buffer class containing buffer data corresponding to a single policy.

        :param buffer_size: (int) max number of episodes to store in buffer.
        :param episode_length: (int) max length of an episode.
        :param num_agents: (int) number of agents controlled by the policy.
        :param obs_space: (gym.Space) observation space of the environment.
        :param share_obs_space: (gym.Space) centralized observation space of the environment.
        :param act_space: (gym.Space) action space of the environment.
        :use_same_share_obs: (bool) whether all agents share the same centralized observation.
        :use_avail_acts: (bool) whether to store what actions are available.
        :param use_reward_normalization: (bool) whether to use reward normalization.
        """
        self.buffer_size = buffer_size
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.num_factor = num_factor
        self.use_same_share_obs = use_same_share_obs
        self.use_avail_acts = use_avail_acts
        self.use_reward_normalization = use_reward_normalization
        self.filled_i = 0
        self.current_i = 0
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.hidden_size = hidden_size
        self.adj_return_adv_coef = float(adj_return_adv_coef)
        self.adj_factor_adv_coef = float(adj_factor_adv_coef)

        self.rng = np.random.RandomState(int(seed))
        # obs
        if obs_space.__class__.__name__ == 'Box':
            obs_shape = obs_space.shape
            share_obs_shape = share_obs_space.shape
        elif obs_space.__class__.__name__ == 'list':
            obs_shape = obs_space
            share_obs_shape = share_obs_space
        else:
            raise NotImplementedError

        self.obs = np.zeros((self.episode_length + 1, self.buffer_size,
                             self.num_agents, obs_shape[0]), dtype=np.float32)

        if self.use_same_share_obs:
            self.share_obs = np.zeros((self.episode_length + 1, self.buffer_size, share_obs_shape[0]), dtype=np.float32)
        else:
            self.share_obs = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, share_obs_shape[0]),
                                      dtype=np.float32)

        # action
        act_dim = np.sum(get_dim_from_space(act_space))
        self.acts = np.zeros((self.episode_length, self.buffer_size, self.num_agents, act_dim), dtype=np.float32)
        if self.use_avail_acts:
            self.avail_acts = np.ones((self.episode_length + 1, self.buffer_size, self.num_agents, act_dim),
                                      dtype=np.float32)

        # rewards
        self.rewards = np.zeros((self.episode_length, self.buffer_size, self.num_agents, 1), dtype=np.float32)

        # default to done being True
        self.dones = np.ones_like(self.rewards, dtype=np.float32)
        self.dones_env = np.ones((self.episode_length, self.buffer_size, 1), dtype=np.float32)
        self.adj = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, self.num_factor),
                            dtype=np.int64)
        self.prob_adj = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, self.num_factor),
                                 dtype=np.float32)
        self.qtot = np.zeros((self.episode_length, self.buffer_size, 1), dtype=np.float32)
        self.margin_q = np.zeros((self.episode_length, self.buffer_size, self.num_agents, 1), dtype=np.float32)
        self.f_q = np.zeros((self.episode_length, self.buffer_size, self.num_factor + self.num_agents, 1),
                            dtype=np.float32)
        self.advantage = np.zeros((self.episode_length, self.buffer_size, 1), dtype=np.float32)
        self.f_v = np.zeros((self.episode_length, self.buffer_size, self.num_factor + self.num_agents, 1),
                            dtype=np.float32)
        self.f_advt = np.zeros((self.episode_length, self.buffer_size, self.num_factor, 1), dtype=np.float32)
        self.rnn_obs = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, self.hidden_size),
                                dtype=np.float32)

    def __len__(self):
        return self.filled_i

    def compute_advantage(self, idx, value_normalizer=None):
        """
        Build graph-policy credit from the trajectory that the sampled graph
        actually produced.

        The old implementation recursively accumulated ``factor_q-factor_v``
        only when exactly the same node set appeared at the next timestep.
        That is not a temporal-difference residual, and it drops the credit
        chain exactly at join/leave/recover events where the topology changes.

        The primary signal below is a discounted team return with a
        timestep-and-roster baseline over the recent adjacency buffer.  It is
        valid across topology changes and credits the sampled graph using real
        rewards.  A centered factor Q-V term is retained only as a lower-weight
        auxiliary signal to distinguish factors selected at the same step.
        """
        requested_idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        if requested_idx.size == 0 or self.filled_i <= 0:
            return idx

        # Every occupied slot from 0..filled_i-1 is valid, including after the
        # circular buffer has become full. Recompute all occupied episodes so
        # their return baselines stay consistent as the recent buffer changes.
        reference_idx = np.arange(self.filled_i, dtype=np.int64)
        episode_idx = reference_idx
        self.f_advt[:, episode_idx] = 0.0
        obs_ref = np.take(
            self.obs[:-1],
            reference_idx,
            axis=1,
        )
        active_ref = ~np.all(
            obs_ref <= -0.999,
            axis=-1,
        )
        active_count_ref = active_ref.sum(axis=-1)

        rewards_ref = np.take(
            self.rewards,
            reference_idx,
            axis=1,
        )[..., 0]
        dones_env_ref = np.take(
            self.dones_env,
            reference_idx,
            axis=1,
        )[..., 0]
        active_den = np.maximum(active_count_ref, 1).astype(np.float32)
        team_rewards_ref = (
            rewards_ref * active_ref.astype(np.float32)
        ).sum(axis=-1) / active_den
        expected_transition_shape = (
            self.episode_length,
            reference_idx.size,
        )
        if team_rewards_ref.shape != expected_transition_shape:
            raise RuntimeError(
                "AdjBuffer transition axes are not [time, episode]: "
                "got {}, expected {}".format(
                    team_rewards_ref.shape,
                    expected_transition_shape,
                )
            )

        returns_ref = np.zeros_like(team_rewards_ref, dtype=np.float32)
        running_return = np.zeros(
            (reference_idx.size,),
            dtype=np.float32,
        )
        for step in reversed(range(self.episode_length)):
            continuation = (
                1.0 - dones_env_ref[step]
            ).astype(np.float32)
            running_return = (
                team_rewards_ref[step]
                + self.gamma * continuation * running_return
            )
            returns_ref[step] = running_return

        # Use recent episodes at the same timestep and roster size as a
        # control variate.  The shock schedule is deterministic in the current
        # experiment, while graph choices and outcomes vary between episodes.
        return_adv_ref = np.zeros_like(returns_ref, dtype=np.float32)
        for step in range(self.episode_length):
            step_rosters = active_count_ref[step]
            for roster_size in np.unique(step_rosters):
                roster_mask = step_rosters == roster_size
                roster_values = returns_ref[step, roster_mask]
                if roster_values.size > 1:
                    baseline = float(roster_values.mean())
                else:
                    # Before the adjacency buffer is populated, do not invent
                    # a return advantage from a single sample.
                    baseline = float(roster_values[0])
                return_adv_ref[step, roster_mask] = (
                    roster_values - baseline
                )

        graph_adv = np.take(
            return_adv_ref,
            episode_idx,
            axis=1,
        )

        current_adj = np.take(
            self.adj[:-1, :, :, :self.num_factor],
            episode_idx,
            axis=1,
        ).astype(np.int64, copy=False)
        current_obs = np.take(
            self.obs[:-1],
            episode_idx,
            axis=1,
        )
        current_active = ~np.all(
            current_obs <= -0.999,
            axis=-1,
        )
        factor_size = current_adj.sum(axis=2)
        factor_alive_count = (
            current_adj * current_active[:, :, :, None]
        ).sum(axis=2)
        valid_factor = (
            (factor_size > 0)
            & (factor_alive_count == factor_size)
            & (np.take(
                self.dones_env,
                episode_idx,
                axis=1,
            )[..., :1] < 0.5)
        )

        if self.adj_factor_adv_coef != 0.0:
            local_adv = (
                np.take(
                    self.f_q[:, :, :self.num_factor],
                    episode_idx,
                    axis=1,
                )[..., 0]
                - np.take(
                    self.f_v[:, :, :self.num_factor],
                    episode_idx,
                    axis=1,
                )[..., 0]
            )
            valid_count = np.maximum(
                valid_factor.sum(axis=-1, keepdims=True),
                1,
            ).astype(np.float32)
            local_mean = (
                local_adv * valid_factor.astype(np.float32)
            ).sum(axis=-1, keepdims=True) / valid_count
            local_adv = (
                local_adv - local_mean
            ) * valid_factor.astype(np.float32)
        else:
            local_adv = np.zeros_like(valid_factor, dtype=np.float32)

        def _standardize_valid(values, mask):
            valid_values = values[mask]
            if valid_values.size <= 1:
                return np.zeros_like(values, dtype=np.float32)
            mean = float(valid_values.mean())
            std = float(valid_values.std())
            if not np.isfinite(std) or std < 1e-5:
                std = 1.0
            standardized = (values - mean) / (std + 1e-5)
            standardized[~mask] = 0.0
            return standardized.astype(np.float32)

        # A graph is one structured action per transition. Normalize its
        # return advantage once per transition rather than once per selected
        # factor; otherwise large rosters receive more statistical weight only
        # because they use more factor slots.
        valid_graph_transition = valid_factor.any(axis=-1)
        graph_adv = _standardize_valid(
            graph_adv,
            valid_graph_transition,
        )
        graph_factor_adv = np.repeat(
            graph_adv[:, :, None],
            self.num_factor,
            axis=2,
        ) * valid_factor.astype(np.float32)
        if self.adj_factor_adv_coef != 0.0:
            local_adv = _standardize_valid(local_adv, valid_factor)
        else:
            local_adv = np.zeros_like(local_adv, dtype=np.float32)

        combined_adv = (
            self.adj_return_adv_coef * graph_factor_adv
            + self.adj_factor_adv_coef * local_adv
        )
        combined_adv[~valid_factor] = 0.0
        f_advt_values = self.f_advt[..., 0]
        f_advt_values[:, episode_idx, :] = combined_adv

        return idx

    def insert(self, num_insert_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts, adj=None,
               prob_adj=None, qtot=None, f_v=None, f_q=None, rnn_obs=None):
        """
        Insert a set of episodes corresponding to this policy into buffer. If the buffer size overflows, old transitions are dropped.

        :param num_insert_steps: (int) number of transitions to be added to buffer
        :param obs: (np.ndarray) observations of agents corresponding to this policy.
        :param share_obs: (np.ndarray) centralized observations of agents corresponding to this policy.
        :param acts: (np.ndarray) actions of agents corresponding to this policy.
        :param rewards: (np.ndarray) rewards of agents corresponding to this policy.
        :param dones: (np.ndarray) terminal status of agents corresponding to this policy.
        :param dones_env: (np.ndarray) environment terminal status.
        :param valid_transition: (np.ndarray) whether each transition is valid or not (invalid if agent was dead during transition)
        :param avail_acts: (np.ndarray) available actions of agents corresponding to this policy.

        :return: (np.ndarray) indexes of the buffer the new transitions were placed in.
        """

        # obs: [step, episode, agent, dim]

        episode_length = acts.shape[0]
        assert episode_length == self.episode_length, ("different dimension!")

        if self.current_i + num_insert_episodes <= self.buffer_size:
            idx_range = np.arange(self.current_i, self.current_i + num_insert_episodes)
        else:
            num_left_episodes = self.current_i + num_insert_episodes - self.buffer_size
            idx_range = np.concatenate((np.arange(self.current_i, self.buffer_size), np.arange(num_left_episodes)))

        if self.use_same_share_obs:
            # remove agent dimension since all agents share centralized observation
            share_obs = share_obs[:, :, 0]

        self.obs[:, idx_range] = obs.copy()
        self.share_obs[:, idx_range] = share_obs.copy()
        self.acts[:, idx_range] = acts.copy()
        self.rewards[:, idx_range] = rewards.copy()
        self.dones[:, idx_range] = dones.copy()
        self.dones_env[:, idx_range] = dones_env.copy()
        self.adj[:, idx_range] = adj.copy()
        self.prob_adj[:, idx_range] = prob_adj.copy()
        self.qtot[:, idx_range] = qtot.copy()
        self.f_v[:, idx_range] = f_v.copy()
        self.f_q[:, idx_range] = f_q.copy()
        self.rnn_obs[:, idx_range] = rnn_obs.copy()

        if self.use_avail_acts:
            self.avail_acts[:, idx_range] = avail_acts.copy()

        self.current_i = idx_range[-1] + 1
        self.filled_i = min(self.filled_i + len(idx_range), self.buffer_size)

        return idx_range

    # def sample_inds(self, data_chunk_length, num_mini_batch):
    #     """
    #     Sample a set of transitions from buffer from the specified indices.
    #     :param sample_inds: (np.ndarray) indices of samples to return from buffer.
    #
    #     :return: obs: (np.ndarray) sampled observations corresponding to that policy
    #     :return: share_obs: (np.ndarray) sampled observations corresponding to that policy
    #     :return: acts: (np.ndarray) sampled actions corresponding to that policy
    #     :return: rewards: (np.ndarray) sampled rewards corresponding to that policy
    #     :return: dones: (np.ndarray) sampled terminal status of agents corresponding to that policy
    #     :return: dones_env: (np.ndarray) sampled environment terminal status corresponding to that policy
    #     :return: valid_transition: (np.ndarray) whether each sampled transition in episodes are valid or not (invalid if corresponding agent is dead)
    #     :return: avail_acts: (np.ndarray) sampled available actions corresponding to that policy
    #     """
    #     batch_size = self.episode_length * self.buffer_size
    #     data_chunks = batch_size // data_chunk_length
    #     mini_batch_size = data_chunks // num_mini_batch
    #     rand = torch.randperm(data_chunks).numpy()
    #     sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
    #
    #     obs = self.obs[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
    #     #np.stack(self.dones_env[:-1].transpose(1,0,2,3),axis=0)
    #     #acts = self.acts.reshape(batch_size,self.num_agents,-1)
    #     dones = np.concatenate((np.zeros((1, self.buffer_size, self.num_agents,1), dtype=np.float32),self.dones[:-1]))
    #     dones_env = np.concatenate((np.zeros((1, self.buffer_size,1), dtype=np.float32),self.dones_env[:-1]))
    #     adj = self.adj[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
    #     prob_adj = self.prob_adj[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
    #     advantage = self.advantage
    #
    #     rnn_obs = self.rnn_obs[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
    #
    #     '''advantage_copy = advantage.copy()
    #     advantage_copy[dones_env == 1.0] = np.nan
    #     mean_advantage = np.nanmean(advantage_copy[1:])
    #     std_advantage = np.nanstd(advantage_copy[1:])
    #     advantage[1:] = (advantage[1:] - mean_advantage) / (std_advantage + 1e-10)'''
    #     advantages = advantage.transpose(1,0,2).reshape(batch_size,-1)
    #
    #     '''margin_advt = self.margin_advt
    #     margin_advt_copy = margin_advt.copy()
    #     margin_advt_copy[dones == 1.0] = np.nan
    #     mean_advt = np.nanmean(margin_advt_copy[1:])
    #     std_advt = np.nanstd(margin_advt_copy[1:])
    #     margin_advt[1:] = (margin_advt[1:] - mean_advt) / (std_advt + 1e-10)
    #     margin_advts = margin_advt.transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)'''
    #
    #     f_advt = self.f_advt
    #     f_advt_copy = f_advt.copy()
    #     f_advt_copy[np.tile(dones_env,self.num_factor) == 1.0] = np.nan
    #     mean_advt_f = np.nanmean(f_advt_copy)
    #     std_advt_f = np.nanstd(f_advt_copy)
    #     '''f_advt_copy = f_advt.copy()
    #     f_advt_copy[np.tile(dones_env,self.num_factor) == 1.0] = np.nan
    #     mean_advt_f = np.nanmean(f_advt_copy.reshape(batch_size,self.num_factor,-1),axis=0)
    #     std_advt_f = np.nanstd(f_advt_copy.reshape(batch_size,self.num_factor,-1),axis=0)'''
    #     f_advt = (f_advt - mean_advt_f) / (std_advt_f + 1e-5)
    #     f_advts = f_advt.transpose(1,0,2,3).reshape(batch_size,self.num_factor,-1)
    #     #rewards = self.rewards.reshape(batch_size,self.num_agents,-1)
    #     dones = dones.transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
    #     dones_env = dones_env.transpose(1,0,2).reshape(batch_size,-1)
    #     if self.use_same_share_obs:
    #         share_obs = self.share_obs[:-1].transpose(1,0,2).reshape(batch_size,-1)
    #     else:
    #         share_obs = self.share_obs[:-1].transpose(1,0,2,3).reshape(batch_size,self.num_agents,-1)
    #
    #     '''if self.use_avail_acts:
    #         avail_acts = self.avail_acts[:-1].reshape(batch_size,self.num_agents,-1)
    #     else:
    #         avail_acts = None'''
    #     for indices in sampler:
    #         obs_batch = []
    #         share_obs_batch = []
    #         dones_batch = []
    #         dones_env_batch = []
    #         adj_batch = []
    #         prob_adj_batch = []
    #         advantages_batch = []
    #         #margin_advts_batch = []
    #         f_advts_batch = []
    #         rnn_obs_batch = []
    #         for i in indices:
    #             ind = i * data_chunk_length
    #             obs_batch.append(obs[ind:ind+data_chunk_length])
    #             share_obs_batch.append(share_obs[ind:ind+data_chunk_length])
    #             dones_batch.append(dones[ind:ind+data_chunk_length])
    #             dones_env_batch.append(dones_env[ind:ind+data_chunk_length])
    #             adj_batch.append(adj[ind:ind+data_chunk_length])
    #             prob_adj_batch.append(prob_adj[ind:ind+data_chunk_length])
    #             advantages_batch.append(advantages[ind:ind+data_chunk_length])
    #             #margin_advts_batch.append(margin_advts[ind:ind+data_chunk_length])
    #             f_advts_batch.append(f_advts[ind:ind+data_chunk_length])
    #             rnn_obs_batch.append(rnn_obs[ind:ind+data_chunk_length])
    #         obs_batch = np.stack(obs_batch,axis=0)
    #         share_obs_batch = np.stack(share_obs_batch,axis=0)
    #         dones_batch = np.stack(dones_batch,axis=0)
    #         dones_env_batch = np.stack(dones_env_batch,axis=0)
    #         adj_batch = np.stack(adj_batch,axis=0)
    #         prob_adj_batch = np.stack(prob_adj_batch,axis=0)
    #         advantages_batch = np.stack(advantages_batch,axis=0)
    #         #margin_advts_batch = np.stack(margin_advts_batch,axis=0)
    #         f_advts_batch = np.stack(f_advts_batch,axis=0)
    #         rnn_obs_batch = np.stack(rnn_obs_batch,axis=0)
    #
    #         yield obs_batch, share_obs_batch, dones_batch, dones_env_batch, adj_batch, prob_adj_batch, advantages_batch, f_advts_batch, rnn_obs_batch
    def sample_inds(self, data_chunk_length, num_mini_batch):
        """
        只从 filled_i 范围内采样，避免未填充 episode 污染邻接训练。
        """
        valid_episodes = int(self.filled_i)
        if valid_episodes <= 0:
            raise RuntimeError("AdjPolicyBuffer.sample_inds called before any episode is inserted.")

        batch_size = self.episode_length * valid_episodes
        data_chunk_length = min(int(data_chunk_length), batch_size)

        data_chunks = max(1, batch_size // data_chunk_length)
        num_mini_batch = max(1, min(int(num_mini_batch), data_chunks))
        mini_batch_size = max(1, data_chunks // num_mini_batch)

        rand = self.rng.permutation(data_chunks)
        sampler = [
            rand[i * mini_batch_size:(i + 1) * mini_batch_size]
            for i in range(num_mini_batch)
        ]

        obs_seq = self.obs[:-1, :valid_episodes]
        obs = obs_seq.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)

        # Intra-episode dynamic fix:
        # Adj/GAT training needs the alive mask of the *current* graph state.
        # The replay buffer's self.dones stores next-state masks, and a zero
        # prefix makes initially empty capacity slots look alive at s0.  Use
        # Wolfpack's padded all--1 observations to reconstruct current inactive
        # slots for every sampled transition, including s0.
        dones = np.all(obs_seq <= -0.999, axis=-1, keepdims=True).astype(np.float32)

        # Keep environment termination semantics from the original shifted
        # dones_env path; only per-slot activity is reconstructed from obs.
        dones_env = np.concatenate((
            np.zeros((1, valid_episodes, 1), dtype=np.float32),
            self.dones_env[:-1, :valid_episodes]
        ))

        adj = self.adj[:-1, :valid_episodes].transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        prob_adj = self.prob_adj[:-1, :valid_episodes].transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        rnn_obs = self.rnn_obs[:-1, :valid_episodes].transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)

        advantage = self.advantage[:, :valid_episodes]
        advantages = advantage.transpose(1, 0, 2).reshape(batch_size, -1)

        f_advt = self.f_advt[:, :valid_episodes]

        # Identify valid selected factors. Advantages were already normalized
        # once per structured graph action in compute_advantage().
        active_seq = ~np.all(obs_seq <= -0.999, axis=-1)
        adj_seq = self.adj[:-1, :valid_episodes, :, :self.num_factor]
        factor_size = adj_seq.sum(axis=2)
        factor_alive_count = (
            adj_seq * active_seq[:, :, :, None]
        ).sum(axis=2)
        valid_factor = (
            (factor_size > 0)
            & (factor_alive_count == factor_size)
            & (dones_env[..., 0, None] < 0.5)
        )
        # Do not normalize again over factor slots: that would overweight
        # larger rosters solely because they select more factors.
        f_advt = np.where(
            np.isfinite(f_advt),
            f_advt,
            0.0,
        ).astype(np.float32, copy=False)
        f_advt[~valid_factor[..., None]] = 0.0
        f_advt = np.clip(f_advt, -5.0, 5.0)
        f_advts = f_advt.transpose(1, 0, 2, 3).reshape(batch_size, self.num_factor, -1)

        dones = dones.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        dones_env = dones_env.transpose(1, 0, 2).reshape(batch_size, -1)

        if self.use_same_share_obs:
            share_obs = self.share_obs[:-1, :valid_episodes].transpose(1, 0, 2).reshape(batch_size, -1)
        else:
            share_obs = self.share_obs[:-1, :valid_episodes].transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents,
                                                                                           -1)

        for indices in sampler:
            obs_batch = []
            share_obs_batch = []
            dones_batch = []
            dones_env_batch = []
            adj_batch = []
            prob_adj_batch = []
            advantages_batch = []
            f_advts_batch = []
            rnn_obs_batch = []

            for i in indices:
                ind = i * data_chunk_length
                obs_batch.append(obs[ind:ind + data_chunk_length])
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                dones_batch.append(dones[ind:ind + data_chunk_length])
                dones_env_batch.append(dones_env[ind:ind + data_chunk_length])
                adj_batch.append(adj[ind:ind + data_chunk_length])
                prob_adj_batch.append(prob_adj[ind:ind + data_chunk_length])
                advantages_batch.append(advantages[ind:ind + data_chunk_length])
                f_advts_batch.append(f_advts[ind:ind + data_chunk_length])
                rnn_obs_batch.append(rnn_obs[ind:ind + data_chunk_length])

            yield (
                np.stack(obs_batch, axis=0),
                np.stack(share_obs_batch, axis=0),
                np.stack(dones_batch, axis=0),
                np.stack(dones_env_batch, axis=0),
                np.stack(adj_batch, axis=0),
                np.stack(prob_adj_batch, axis=0),
                np.stack(advantages_batch, axis=0),
                np.stack(f_advts_batch, axis=0),
                np.stack(rnn_obs_batch, axis=0),
            )
