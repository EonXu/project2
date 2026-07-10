import numpy as np
from .util import get_dim_from_space
from .segment_tree import SumSegmentTree, MinSegmentTree
import torch


def _cast(x):
    return x.transpose(2, 0, 1, 3)


class AdjBuffer(object):
    def __init__(self, policy_info, policy_agents, num_factor, buffer_size, episode_length, use_same_share_obs,
                 use_avail_acts, use_reward_normalization=False, gamma=0.97, gae_lambda=0.95, hidden_size=64,
                 adj_return_adv_coef=1.0, adj_factor_adv_coef=0.0, seed=0,
                 use_adj_delayed_triplet_credit=False,
                 adj_delayed_triplet_credit_coef=0.0,
                 adj_delayed_triplet_credit_window=0,
                 adj_delayed_triplet_credit_cap=0.0,
                 adj_delayed_triplet_credit_min_reward=0.0,
                 adj_delayed_triplet_credit_positive_only=False,
                 adj_delayed_triplet_credit_min_adv=0.0,
                 adj_delayed_triplet_credit_require_future_match=False,
                 use_adj_delayed_triplet_success_gate=False,
                 adj_delayed_triplet_success_gate_min_adv=0.0,
                 adj_delayed_triplet_success_gate_scale=1.0,
                 adj_delayed_triplet_success_gate_floor=0.0,
                 adj_delayed_triplet_future_overlap_min_nodes=3,
                 adj_delayed_triplet_partial_match_weight=0.5,
                 use_adj_capture_to_win_credit=False,
                 adj_capture_to_win_credit_coef=0.0,
                 adj_capture_to_win_credit_min_outcome_adv=0.5,
                 adj_capture_to_win_credit_scale=0.75,
                 adj_capture_to_win_credit_cap=0.35,
                 adj_capture_to_win_credit_require_future_match=False):
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
                use_adj_delayed_triplet_credit=use_adj_delayed_triplet_credit,
                adj_delayed_triplet_credit_coef=adj_delayed_triplet_credit_coef,
                adj_delayed_triplet_credit_window=adj_delayed_triplet_credit_window,
                adj_delayed_triplet_credit_cap=adj_delayed_triplet_credit_cap,
                adj_delayed_triplet_credit_min_reward=adj_delayed_triplet_credit_min_reward,
                adj_delayed_triplet_credit_positive_only=adj_delayed_triplet_credit_positive_only,
                adj_delayed_triplet_credit_min_adv=adj_delayed_triplet_credit_min_adv,
                adj_delayed_triplet_credit_require_future_match=adj_delayed_triplet_credit_require_future_match,
                use_adj_delayed_triplet_success_gate=use_adj_delayed_triplet_success_gate,
                adj_delayed_triplet_success_gate_min_adv=adj_delayed_triplet_success_gate_min_adv,
                adj_delayed_triplet_success_gate_scale=adj_delayed_triplet_success_gate_scale,
                adj_delayed_triplet_success_gate_floor=adj_delayed_triplet_success_gate_floor,
                adj_delayed_triplet_future_overlap_min_nodes=adj_delayed_triplet_future_overlap_min_nodes,
                adj_delayed_triplet_partial_match_weight=adj_delayed_triplet_partial_match_weight,
                use_adj_capture_to_win_credit=use_adj_capture_to_win_credit,
                adj_capture_to_win_credit_coef=adj_capture_to_win_credit_coef,
                adj_capture_to_win_credit_min_outcome_adv=adj_capture_to_win_credit_min_outcome_adv,
                adj_capture_to_win_credit_scale=adj_capture_to_win_credit_scale,
                adj_capture_to_win_credit_cap=adj_capture_to_win_credit_cap,
                adj_capture_to_win_credit_require_future_match=adj_capture_to_win_credit_require_future_match,
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
                 hidden_size=64, adj_return_adv_coef=1.0, adj_factor_adv_coef=0.0, seed=0,
                 use_adj_delayed_triplet_credit=False,
                 adj_delayed_triplet_credit_coef=0.0,
                 adj_delayed_triplet_credit_window=0,
                 adj_delayed_triplet_credit_cap=0.0,
                 adj_delayed_triplet_credit_min_reward=0.0,
                 adj_delayed_triplet_credit_positive_only=False,
                 adj_delayed_triplet_credit_min_adv=0.0,
                 adj_delayed_triplet_credit_require_future_match=False,
                 use_adj_delayed_triplet_success_gate=False,
                 adj_delayed_triplet_success_gate_min_adv=0.0,
                 adj_delayed_triplet_success_gate_scale=1.0,
                 adj_delayed_triplet_success_gate_floor=0.0,
                 adj_delayed_triplet_future_overlap_min_nodes=3,
                 adj_delayed_triplet_partial_match_weight=0.5,
                 use_adj_capture_to_win_credit=False,
                 adj_capture_to_win_credit_coef=0.0,
                 adj_capture_to_win_credit_min_outcome_adv=0.5,
                 adj_capture_to_win_credit_scale=0.75,
                 adj_capture_to_win_credit_cap=0.35,
                 adj_capture_to_win_credit_require_future_match=False):
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
        self.use_adj_delayed_triplet_credit = bool(
            use_adj_delayed_triplet_credit
        )
        self.adj_delayed_triplet_credit_coef = max(
            0.0,
            float(adj_delayed_triplet_credit_coef),
        )
        self.adj_delayed_triplet_credit_window = max(
            0,
            int(adj_delayed_triplet_credit_window),
        )
        self.adj_delayed_triplet_credit_cap = max(
            0.0,
            float(adj_delayed_triplet_credit_cap),
        )
        self.adj_delayed_triplet_credit_min_reward = float(
            adj_delayed_triplet_credit_min_reward
        )
        self.adj_delayed_triplet_credit_positive_only = bool(
            adj_delayed_triplet_credit_positive_only
        )
        self.adj_delayed_triplet_credit_min_adv = max(
            0.0,
            float(adj_delayed_triplet_credit_min_adv),
        )
        self.adj_delayed_triplet_credit_require_future_match = bool(
            adj_delayed_triplet_credit_require_future_match
        )
        self.use_adj_delayed_triplet_success_gate = bool(
            use_adj_delayed_triplet_success_gate
        )
        self.adj_delayed_triplet_success_gate_min_adv = float(
            adj_delayed_triplet_success_gate_min_adv
        )
        self.adj_delayed_triplet_success_gate_scale = max(
            1e-6,
            float(adj_delayed_triplet_success_gate_scale),
        )
        self.adj_delayed_triplet_success_gate_floor = min(
            1.0,
            max(0.0, float(adj_delayed_triplet_success_gate_floor)),
        )
        self.adj_delayed_triplet_future_overlap_min_nodes = min(
            3,
            max(1, int(adj_delayed_triplet_future_overlap_min_nodes)),
        )
        self.adj_delayed_triplet_partial_match_weight = min(
            1.0,
            max(0.0, float(adj_delayed_triplet_partial_match_weight)),
        )
        self.use_adj_capture_to_win_credit = bool(
            use_adj_capture_to_win_credit
        )
        self.adj_capture_to_win_credit_coef = max(
            0.0,
            float(adj_capture_to_win_credit_coef),
        )
        self.adj_capture_to_win_credit_min_outcome_adv = float(
            adj_capture_to_win_credit_min_outcome_adv
        )
        self.adj_capture_to_win_credit_scale = max(
            1e-6,
            float(adj_capture_to_win_credit_scale),
        )
        self.adj_capture_to_win_credit_cap = max(
            0.0,
            float(adj_capture_to_win_credit_cap),
        )
        self.adj_capture_to_win_credit_require_future_match = bool(
            adj_capture_to_win_credit_require_future_match
        )

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
        self.delayed_triplet_credit = np.zeros_like(self.f_advt)
        self.delayed_triplet_success_gate = np.zeros_like(self.f_advt)
        self.delayed_triplet_future_match = np.zeros_like(self.f_advt)
        self.delayed_triplet_future_exact = np.zeros_like(self.f_advt)
        self.delayed_triplet_future_partial = np.zeros_like(self.f_advt)
        self.capture_to_win_triplet_credit = np.zeros_like(self.f_advt)
        self.capture_to_win_quality_gate = np.zeros_like(self.f_advt)
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
        self.delayed_triplet_credit[:, episode_idx] = 0.0
        self.delayed_triplet_success_gate[:, episode_idx] = 0.0
        self.delayed_triplet_future_match[:, episode_idx] = 0.0
        self.delayed_triplet_future_exact[:, episode_idx] = 0.0
        self.delayed_triplet_future_partial[:, episode_idx] = 0.0
        self.capture_to_win_triplet_credit[:, episode_idx] = 0.0
        self.capture_to_win_quality_gate[:, episode_idx] = 0.0
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
            # Factor critics are order-specific networks.  Comparing a pair
            # factor directly with a triplet factor mixes critic scales and in
            # run22 biased the learned graph toward low-order pairs.  Keep the
            # auxiliary credit local to alternatives of the same factor order.
            centered_local_adv = np.zeros_like(local_adv, dtype=np.float32)
            for order in np.unique(factor_size[valid_factor]):
                order_mask = valid_factor & (factor_size == order)
                valid_count = np.maximum(
                    order_mask.sum(axis=-1, keepdims=True),
                    1,
                ).astype(np.float32)
                order_mean = (
                    local_adv * order_mask.astype(np.float32)
                ).sum(axis=-1, keepdims=True) / valid_count
                centered_local_adv += (
                    (local_adv - order_mean)
                    * order_mask.astype(np.float32)
                )
            local_adv = centered_local_adv
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

        def _standardize_valid_by_order(values, mask, orders):
            standardized = np.zeros_like(values, dtype=np.float32)
            if not mask.any():
                return standardized
            for order in np.unique(orders[mask]):
                order_mask = mask & (orders == order)
                order_standardized = _standardize_valid(values, order_mask)
                standardized[order_mask] = order_standardized[order_mask]
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
            local_adv = _standardize_valid_by_order(
                local_adv,
                valid_factor,
                factor_size,
            )
        else:
            local_adv = np.zeros_like(local_adv, dtype=np.float32)

        delayed_triplet_credit = np.zeros_like(local_adv, dtype=np.float32)
        delayed_triplet_success_gate = np.zeros_like(local_adv, dtype=np.float32)
        delayed_triplet_future_match = np.zeros_like(local_adv, dtype=np.float32)
        delayed_triplet_future_exact = np.zeros_like(local_adv, dtype=np.float32)
        delayed_triplet_future_partial = np.zeros_like(local_adv, dtype=np.float32)
        if (
            self.use_adj_delayed_triplet_credit
            and self.adj_delayed_triplet_credit_coef > 0.0
            and self.adj_delayed_triplet_credit_window > 0
        ):
            reward_signal = (
                team_rewards_ref
                - float(self.adj_delayed_triplet_credit_min_reward)
            ).astype(np.float32, copy=False)
            reward_signal = np.maximum(reward_signal, 0.0)
            window = int(self.adj_delayed_triplet_credit_window)
            triplet_factor_mask = valid_factor & (factor_size == 3)
            success_gate = np.ones_like(reward_signal, dtype=np.float32)
            if self.use_adj_delayed_triplet_success_gate:
                graph_delayed_signal = np.zeros_like(
                    reward_signal,
                    dtype=np.float32,
                )
                for step in range(self.episode_length):
                    discount = 1.0
                    alive = np.ones(reference_idx.size, dtype=np.float32)
                    for offset in range(window):
                        future_step = step + offset
                        if future_step >= self.episode_length:
                            break
                        graph_delayed_signal[step] += (
                            discount
                            * alive
                            * reward_signal[future_step]
                        )
                        alive *= (
                            1.0
                            - dones_env_ref[future_step].astype(np.float32)
                        )
                        discount *= float(self.gamma)

                graph_delayed_adv = np.zeros_like(
                    graph_delayed_signal,
                    dtype=np.float32,
                )
                for step in range(self.episode_length):
                    step_rosters = active_count_ref[step]
                    for roster_size in np.unique(step_rosters):
                        roster_mask = step_rosters == roster_size
                        roster_values = graph_delayed_signal[
                            step,
                            roster_mask,
                        ]
                        if roster_values.size > 1:
                            baseline = float(roster_values.mean())
                        else:
                            baseline = float(roster_values[0])
                        graph_delayed_adv[step, roster_mask] = (
                            roster_values - baseline
                        )
                graph_delayed_adv = _standardize_valid(
                    graph_delayed_adv,
                    valid_graph_transition,
                )
                success_gate = (
                    (
                        graph_delayed_adv
                        - float(self.adj_delayed_triplet_success_gate_min_adv)
                    )
                    / float(self.adj_delayed_triplet_success_gate_scale)
                )
                success_gate = np.clip(success_gate, 0.0, 1.0).astype(
                    np.float32,
                    copy=False,
                )
                if self.adj_delayed_triplet_success_gate_floor > 0.0:
                    gate_floor = float(
                        self.adj_delayed_triplet_success_gate_floor
                    )
                    success_gate = (
                        gate_floor
                        + (1.0 - gate_floor) * success_gate
                    ).astype(np.float32, copy=False)
                success_gate[~valid_graph_transition] = 0.0
            if self.adj_delayed_triplet_credit_require_future_match:
                delayed_signal = np.zeros_like(local_adv, dtype=np.float32)
                triplet_sets = []
                for step in range(self.episode_length):
                    step_sets = []
                    for ep_pos in range(reference_idx.size):
                        factor_sets = []
                        for factor_idx in range(self.num_factor):
                            if not triplet_factor_mask[
                                step,
                                ep_pos,
                                factor_idx,
                            ]:
                                continue
                            nodes = tuple(
                                np.flatnonzero(
                                    current_adj[
                                        step,
                                        ep_pos,
                                        :,
                                        factor_idx,
                                    ] > 0
                                ).tolist()
                            )
                            if len(nodes) == 3:
                                factor_sets.append(nodes)
                        step_sets.append(factor_sets)
                    triplet_sets.append(step_sets)

                min_overlap = int(
                    self.adj_delayed_triplet_future_overlap_min_nodes
                )
                partial_match_weight = float(
                    self.adj_delayed_triplet_partial_match_weight
                )
                for step in range(self.episode_length):
                    for ep_pos in range(reference_idx.size):
                        current_triplets = {}
                        for factor_idx in range(self.num_factor):
                            if not triplet_factor_mask[
                                step,
                                ep_pos,
                                factor_idx,
                            ]:
                                continue
                            nodes = tuple(
                                np.flatnonzero(
                                    current_adj[
                                        step,
                                        ep_pos,
                                        :,
                                        factor_idx,
                                    ] > 0
                                ).tolist()
                            )
                            if len(nodes) == 3:
                                current_triplets[factor_idx] = nodes
                        if not current_triplets:
                            continue
                        discount = 1.0
                        for offset in range(window):
                            future_step = step + offset
                            if future_step >= self.episode_length:
                                break
                            future_reward = float(
                                reward_signal[future_step, ep_pos]
                            )
                            if future_reward > 0.0:
                                future_sets = triplet_sets[
                                    future_step
                                ][ep_pos]
                                for factor_idx, nodes in (
                                    current_triplets.items()
                                ):
                                    match_weight = 0.0
                                    node_set = set(nodes)
                                    for future_nodes in future_sets:
                                        overlap = len(
                                            node_set.intersection(future_nodes)
                                        )
                                        if overlap < min_overlap:
                                            continue
                                        if overlap >= 3:
                                            match_weight = 1.0
                                            break
                                        match_weight = max(
                                            match_weight,
                                            partial_match_weight,
                                        )
                                    if match_weight > 0.0:
                                        delayed_triplet_future_match[
                                            step,
                                            ep_pos,
                                            factor_idx,
                                        ] = max(
                                            delayed_triplet_future_match[
                                                step,
                                                ep_pos,
                                                factor_idx,
                                            ],
                                            match_weight,
                                        )
                                        if match_weight >= 1.0:
                                            delayed_triplet_future_exact[
                                                step,
                                                ep_pos,
                                                factor_idx,
                                            ] = 1.0
                                        else:
                                            delayed_triplet_future_partial[
                                                step,
                                                ep_pos,
                                                factor_idx,
                                            ] = 1.0
                                        delayed_signal[
                                            step,
                                            ep_pos,
                                            factor_idx,
                                        ] += (
                                            discount
                                            * future_reward
                                            * match_weight
                                        )
                            if dones_env_ref[future_step, ep_pos] >= 0.5:
                                break
                            discount *= float(self.gamma)

                delayed_adv = np.zeros_like(delayed_signal, dtype=np.float32)
                for step in range(self.episode_length):
                    step_rosters = active_count_ref[step]
                    for roster_size in np.unique(step_rosters):
                        roster_mask = step_rosters == roster_size
                        factor_mask = (
                            triplet_factor_mask[step]
                            & roster_mask[:, None]
                        )
                        factor_values = delayed_signal[step][factor_mask]
                        if factor_values.size == 0:
                            continue
                        if factor_values.size > 1:
                            baseline = float(factor_values.mean())
                        else:
                            baseline = float(factor_values[0])
                        delayed_adv[step][factor_mask] = (
                            delayed_signal[step][factor_mask]
                            - baseline
                        )
                delayed_adv = _standardize_valid(
                    delayed_adv,
                    triplet_factor_mask,
                )
            else:
                delayed_signal = np.zeros_like(reward_signal, dtype=np.float32)
                for step in range(self.episode_length):
                    discount = 1.0
                    alive = np.ones(reference_idx.size, dtype=np.float32)
                    for offset in range(window):
                        future_step = step + offset
                        if future_step >= self.episode_length:
                            break
                        delayed_signal[step] += (
                            discount
                            * alive
                            * reward_signal[future_step]
                        )
                        alive *= (
                            1.0
                            - dones_env_ref[future_step].astype(np.float32)
                        )
                        discount *= float(self.gamma)
                delayed_adv = np.zeros_like(delayed_signal, dtype=np.float32)
                for step in range(self.episode_length):
                    step_rosters = active_count_ref[step]
                    for roster_size in np.unique(step_rosters):
                        roster_mask = step_rosters == roster_size
                        roster_values = delayed_signal[step, roster_mask]
                        if roster_values.size > 1:
                            baseline = float(roster_values.mean())
                        else:
                            baseline = float(roster_values[0])
                        delayed_adv[step, roster_mask] = (
                            roster_values - baseline
                        )
                delayed_adv = _standardize_valid(
                    delayed_adv,
                    valid_graph_transition,
                )
            min_delayed_adv = float(self.adj_delayed_triplet_credit_min_adv)
            if self.adj_delayed_triplet_credit_positive_only:
                delayed_adv = np.maximum(
                    delayed_adv - min_delayed_adv,
                    0.0,
                )
            elif min_delayed_adv > 0.0:
                delayed_adv = np.where(
                    np.abs(delayed_adv) >= min_delayed_adv,
                    delayed_adv,
                    0.0,
                )
            delayed_triplet_credit = (
                delayed_adv
                if self.adj_delayed_triplet_credit_require_future_match
                else delayed_adv[:, :, None]
            )
            delayed_triplet_success_gate = (
                success_gate[:, :, None]
                * triplet_factor_mask.astype(np.float32)
            )
            delayed_triplet_credit = (
                delayed_triplet_credit
                * delayed_triplet_success_gate
            )
            delayed_triplet_credit = (
                delayed_triplet_credit
                * triplet_factor_mask.astype(np.float32)
                * float(self.adj_delayed_triplet_credit_coef)
            )
            if self.adj_delayed_triplet_credit_cap > 0.0:
                delayed_triplet_credit = np.clip(
                    delayed_triplet_credit,
                    -float(self.adj_delayed_triplet_credit_cap),
                    float(self.adj_delayed_triplet_credit_cap),
                )

        capture_to_win_triplet_credit = np.zeros_like(
            local_adv,
            dtype=np.float32,
        )
        capture_to_win_quality_gate = np.zeros_like(
            local_adv,
            dtype=np.float32,
        )
        if (
            self.use_adj_capture_to_win_credit
            and self.adj_capture_to_win_credit_coef > 0.0
        ):
            valid_episode = valid_graph_transition.any(axis=0)
            episode_return = team_rewards_ref.sum(axis=0).astype(
                np.float32,
                copy=False,
            )
            episode_outcome_adv = np.zeros_like(
                episode_return,
                dtype=np.float32,
            )
            valid_returns = episode_return[valid_episode]
            if valid_returns.size > 1:
                outcome_mean = float(valid_returns.mean())
                outcome_std = float(valid_returns.std())
                if not np.isfinite(outcome_std) or outcome_std < 1e-5:
                    outcome_std = 1.0
                episode_outcome_adv[valid_episode] = (
                    (valid_returns - outcome_mean)
                    / (outcome_std + 1e-5)
                )
            outcome_gate = (
                (
                    episode_outcome_adv
                    - float(self.adj_capture_to_win_credit_min_outcome_adv)
                )
                / float(self.adj_capture_to_win_credit_scale)
            )
            outcome_gate = np.clip(outcome_gate, 0.0, 1.0).astype(
                np.float32,
                copy=False,
            )
            outcome_gate[~valid_episode] = 0.0

            selective_success_gate = delayed_triplet_success_gate.copy()
            if self.use_adj_delayed_triplet_success_gate:
                gate_floor = float(self.adj_delayed_triplet_success_gate_floor)
                if 0.0 < gate_floor < 1.0:
                    selective_success_gate = np.clip(
                        (selective_success_gate - gate_floor)
                        / (1.0 - gate_floor),
                        0.0,
                        1.0,
                    )
                elif gate_floor >= 1.0:
                    selective_success_gate = np.zeros_like(
                        selective_success_gate,
                        dtype=np.float32,
                    )

            if self.adj_capture_to_win_credit_require_future_match:
                future_match_gate = delayed_triplet_future_match
            else:
                future_match_gate = triplet_factor_mask.astype(np.float32)
            capture_to_win_quality_gate = (
                outcome_gate[None, :, None]
                * selective_success_gate
                * future_match_gate
                * triplet_factor_mask.astype(np.float32)
            ).astype(np.float32, copy=False)

            graph_positive_credit = np.maximum(
                graph_adv[:, :, None],
                0.0,
            ).astype(np.float32, copy=False)
            capture_to_win_triplet_credit = (
                capture_to_win_quality_gate
                * graph_positive_credit
                * float(self.adj_capture_to_win_credit_coef)
            )
            if self.adj_capture_to_win_credit_cap > 0.0:
                valid_abs_graph = np.abs(graph_adv[valid_graph_transition])
                graph_abs_scale = (
                    float(valid_abs_graph.mean())
                    if valid_abs_graph.size > 0
                    else 1.0
                )
                if not np.isfinite(graph_abs_scale) or graph_abs_scale < 1e-6:
                    graph_abs_scale = 1.0
                credit_cap = (
                    graph_abs_scale
                    * float(self.adj_capture_to_win_credit_cap)
                )
                capture_to_win_triplet_credit = np.clip(
                    capture_to_win_triplet_credit,
                    0.0,
                    credit_cap,
                )

        combined_adv = (
            self.adj_return_adv_coef * graph_factor_adv
            + self.adj_factor_adv_coef * local_adv
            + delayed_triplet_credit
            + capture_to_win_triplet_credit
        )
        combined_adv[~valid_factor] = 0.0
        f_advt_values = self.f_advt[..., 0]
        f_advt_values[:, episode_idx, :] = combined_adv
        delayed_triplet_values = self.delayed_triplet_credit[..., 0]
        delayed_triplet_values[:, episode_idx, :] = delayed_triplet_credit
        delayed_triplet_success_gate_values = (
            self.delayed_triplet_success_gate[..., 0]
        )
        delayed_triplet_success_gate_values[:, episode_idx, :] = (
            delayed_triplet_success_gate
        )
        delayed_triplet_future_match_values = (
            self.delayed_triplet_future_match[..., 0]
        )
        delayed_triplet_future_match_values[:, episode_idx, :] = (
            delayed_triplet_future_match
        )
        delayed_triplet_future_exact_values = (
            self.delayed_triplet_future_exact[..., 0]
        )
        delayed_triplet_future_exact_values[:, episode_idx, :] = (
            delayed_triplet_future_exact
        )
        delayed_triplet_future_partial_values = (
            self.delayed_triplet_future_partial[..., 0]
        )
        delayed_triplet_future_partial_values[:, episode_idx, :] = (
            delayed_triplet_future_partial
        )
        capture_to_win_credit_values = (
            self.capture_to_win_triplet_credit[..., 0]
        )
        capture_to_win_credit_values[:, episode_idx, :] = (
            capture_to_win_triplet_credit
        )
        capture_to_win_quality_gate_values = (
            self.capture_to_win_quality_gate[..., 0]
        )
        capture_to_win_quality_gate_values[:, episode_idx, :] = (
            capture_to_win_quality_gate
        )

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
    def _recent_episode_indices(self, recent_episode_window=0):
        valid_episodes = int(self.filled_i)
        if valid_episodes <= 0:
            raise RuntimeError(
                "AdjPolicyBuffer.sample_inds called before any episode is inserted."
            )
        recent_episode_window = int(recent_episode_window or 0)
        if recent_episode_window <= 0 or recent_episode_window >= valid_episodes:
            return np.arange(valid_episodes, dtype=np.int64)

        end = int(self.current_i) % int(self.buffer_size)
        start = (end - recent_episode_window) % int(self.buffer_size)
        return (
            np.arange(
                start,
                start + recent_episode_window,
                dtype=np.int64,
            )
            % int(self.buffer_size)
        )

    def sample_inds(
            self,
            data_chunk_length,
            num_mini_batch,
            recent_episode_window=0):
        """
        只从 filled_i 范围内采样，避免未填充 episode 污染邻接训练。
        """
        episode_indices = self._recent_episode_indices(
            recent_episode_window
        )
        valid_episodes = int(episode_indices.size)
        self.last_sample_episode_count = valid_episodes
        self.last_sample_recent_window = int(recent_episode_window or 0)
        self.last_sample_episode_indices = episode_indices.copy()

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

        obs_seq = np.take(self.obs[:-1], episode_indices, axis=1)
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
        dones_env_seq = np.take(
            self.dones_env[:-1],
            episode_indices,
            axis=1,
        )
        dones_env = np.concatenate((
            np.zeros((1, valid_episodes, 1), dtype=np.float32),
            dones_env_seq,
        ))

        adj_seq_full = np.take(self.adj[:-1], episode_indices, axis=1)
        prob_adj_seq = np.take(
            self.prob_adj[:-1],
            episode_indices,
            axis=1,
        )
        rnn_obs_seq = np.take(self.rnn_obs[:-1], episode_indices, axis=1)
        adj = adj_seq_full.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        prob_adj = prob_adj_seq.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        rnn_obs = rnn_obs_seq.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)

        advantage = np.take(self.advantage, episode_indices, axis=1)
        advantages = advantage.transpose(1, 0, 2).reshape(batch_size, -1)

        f_advt = np.take(self.f_advt, episode_indices, axis=1)
        delayed_triplet_credit = np.take(
            self.delayed_triplet_credit,
            episode_indices,
            axis=1,
        )
        delayed_triplet_success_gate = np.take(
            self.delayed_triplet_success_gate,
            episode_indices,
            axis=1,
        )
        delayed_triplet_future_match = np.take(
            self.delayed_triplet_future_match,
            episode_indices,
            axis=1,
        )
        delayed_triplet_future_exact = np.take(
            self.delayed_triplet_future_exact,
            episode_indices,
            axis=1,
        )
        delayed_triplet_future_partial = np.take(
            self.delayed_triplet_future_partial,
            episode_indices,
            axis=1,
        )
        capture_to_win_triplet_credit = np.take(
            self.capture_to_win_triplet_credit,
            episode_indices,
            axis=1,
        )
        capture_to_win_quality_gate = np.take(
            self.capture_to_win_quality_gate,
            episode_indices,
            axis=1,
        )

        # Identify valid selected factors. Advantages were already normalized
        # once per structured graph action in compute_advantage().
        active_seq = ~np.all(obs_seq <= -0.999, axis=-1)
        adj_seq = adj_seq_full[:, :, :, :self.num_factor]
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
        delayed_triplet_credit = np.where(
            np.isfinite(delayed_triplet_credit),
            delayed_triplet_credit,
            0.0,
        ).astype(np.float32, copy=False)
        delayed_triplet_credit[~valid_factor[..., None]] = 0.0
        delayed_triplet_credit = np.clip(delayed_triplet_credit, -5.0, 5.0)
        delayed_triplet_credits = (
            delayed_triplet_credit
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        delayed_triplet_success_gate = np.where(
            np.isfinite(delayed_triplet_success_gate),
            delayed_triplet_success_gate,
            0.0,
        ).astype(np.float32, copy=False)
        delayed_triplet_success_gate[~valid_factor[..., None]] = 0.0
        delayed_triplet_success_gate = np.clip(
            delayed_triplet_success_gate,
            0.0,
            1.0,
        )
        delayed_triplet_success_gates = (
            delayed_triplet_success_gate
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        delayed_triplet_future_match = np.where(
            np.isfinite(delayed_triplet_future_match),
            delayed_triplet_future_match,
            0.0,
        ).astype(np.float32, copy=False)
        delayed_triplet_future_match[~valid_factor[..., None]] = 0.0
        delayed_triplet_future_match = np.clip(
            delayed_triplet_future_match,
            0.0,
            1.0,
        )
        delayed_triplet_future_matches = (
            delayed_triplet_future_match
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        delayed_triplet_future_exact = np.where(
            np.isfinite(delayed_triplet_future_exact),
            delayed_triplet_future_exact,
            0.0,
        ).astype(np.float32, copy=False)
        delayed_triplet_future_exact[~valid_factor[..., None]] = 0.0
        delayed_triplet_future_exact = np.clip(
            delayed_triplet_future_exact,
            0.0,
            1.0,
        )
        delayed_triplet_future_exacts = (
            delayed_triplet_future_exact
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        delayed_triplet_future_partial = np.where(
            np.isfinite(delayed_triplet_future_partial),
            delayed_triplet_future_partial,
            0.0,
        ).astype(np.float32, copy=False)
        delayed_triplet_future_partial[~valid_factor[..., None]] = 0.0
        delayed_triplet_future_partial = np.clip(
            delayed_triplet_future_partial,
            0.0,
            1.0,
        )
        delayed_triplet_future_partials = (
            delayed_triplet_future_partial
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        capture_to_win_triplet_credit = np.where(
            np.isfinite(capture_to_win_triplet_credit),
            capture_to_win_triplet_credit,
            0.0,
        ).astype(np.float32, copy=False)
        capture_to_win_triplet_credit[~valid_factor[..., None]] = 0.0
        capture_to_win_triplet_credit = np.clip(
            capture_to_win_triplet_credit,
            -5.0,
            5.0,
        )
        capture_to_win_triplet_credits = (
            capture_to_win_triplet_credit
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        capture_to_win_quality_gate = np.where(
            np.isfinite(capture_to_win_quality_gate),
            capture_to_win_quality_gate,
            0.0,
        ).astype(np.float32, copy=False)
        capture_to_win_quality_gate[~valid_factor[..., None]] = 0.0
        capture_to_win_quality_gate = np.clip(
            capture_to_win_quality_gate,
            0.0,
            1.0,
        )
        capture_to_win_quality_gates = (
            capture_to_win_quality_gate
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )

        dones = dones.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        dones_env = dones_env.transpose(1, 0, 2).reshape(batch_size, -1)

        if self.use_same_share_obs:
            share_obs_seq = np.take(
                self.share_obs[:-1],
                episode_indices,
                axis=1,
            )
            share_obs = share_obs_seq.transpose(1, 0, 2).reshape(batch_size, -1)
        else:
            share_obs_seq = np.take(
                self.share_obs[:-1],
                episode_indices,
                axis=1,
            )
            share_obs = share_obs_seq.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents,
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
            delayed_triplet_credit_batch = []
            delayed_triplet_success_gate_batch = []
            delayed_triplet_future_match_batch = []
            delayed_triplet_future_exact_batch = []
            delayed_triplet_future_partial_batch = []
            capture_to_win_triplet_credit_batch = []
            capture_to_win_quality_gate_batch = []
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
                delayed_triplet_credit_batch.append(
                    delayed_triplet_credits[ind:ind + data_chunk_length]
                )
                delayed_triplet_success_gate_batch.append(
                    delayed_triplet_success_gates[ind:ind + data_chunk_length]
                )
                delayed_triplet_future_match_batch.append(
                    delayed_triplet_future_matches[ind:ind + data_chunk_length]
                )
                delayed_triplet_future_exact_batch.append(
                    delayed_triplet_future_exacts[ind:ind + data_chunk_length]
                )
                delayed_triplet_future_partial_batch.append(
                    delayed_triplet_future_partials[ind:ind + data_chunk_length]
                )
                capture_to_win_triplet_credit_batch.append(
                    capture_to_win_triplet_credits[ind:ind + data_chunk_length]
                )
                capture_to_win_quality_gate_batch.append(
                    capture_to_win_quality_gates[ind:ind + data_chunk_length]
                )
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
                np.stack(delayed_triplet_credit_batch, axis=0),
                np.stack(delayed_triplet_success_gate_batch, axis=0),
                np.stack(delayed_triplet_future_match_batch, axis=0),
                np.stack(delayed_triplet_future_exact_batch, axis=0),
                np.stack(delayed_triplet_future_partial_batch, axis=0),
                np.stack(capture_to_win_triplet_credit_batch, axis=0),
                np.stack(capture_to_win_quality_gate_batch, axis=0),
                np.stack(rnn_obs_batch, axis=0),
            )
