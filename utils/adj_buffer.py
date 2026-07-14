import numpy as np
from .util import get_dim_from_space
from .segment_tree import SumSegmentTree, MinSegmentTree
import torch

from .pair_credit import (
    CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH,
    canonical_capture_factor_catalog,
    compute_capture_anchored_pair_credit,
    compute_capture_to_win_outcome_gate,
    compute_capture_to_win_triplet_outcome_advantage,
    scale_capture_to_win_outcome_credit,
)
from .graph_sampling import (
    build_previous_adjacency_sequence,
    build_previous_done_sequence,
    require_ready_graph_advantage,
    select_outcome_contrast_complete_episodes,
    write_graph_advantage_sequence,
)


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
                 adj_capture_to_win_credit_require_future_match=False,
                 use_adj_pair_triplet_complementary_credit=False,
                 adj_pair_pursuit_credit_coef=0.0,
                 adj_pair_pursuit_credit_window=20,
                 adj_pair_pursuit_credit_cap=0.20,
                 adj_pair_pursuit_credit_min_reward=0.0):
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
                use_adj_pair_triplet_complementary_credit=use_adj_pair_triplet_complementary_credit,
                adj_pair_pursuit_credit_coef=adj_pair_pursuit_credit_coef,
                adj_pair_pursuit_credit_window=adj_pair_pursuit_credit_window,
                adj_pair_pursuit_credit_cap=adj_pair_pursuit_credit_cap,
                adj_pair_pursuit_credit_min_reward=adj_pair_pursuit_credit_min_reward,
            )
            for i, p_id in enumerate(self.policy_info.keys())
        }

    def __len__(self):
        return self.policy_buffers['policy_0'].filled_i

    def insert(self, num_insert_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts, adj=None,
               prob_adj=None, q_tot=None, f_v=None, f_q=None, rnn_states=None,
               capture_counts=None, success_now=None,
               capture_factor_matches=None,
               capture_candidate_only_matches=None,
               capture_candidate_behavior=None,
               capture_identity_candidates=None):
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
            policy_dones_env = np.array(dones_env[p_id])
            policy_capture_counts = (
                capture_counts.get(p_id)
                if isinstance(capture_counts, dict)
                else capture_counts
            )
            policy_success_now = (
                success_now.get(p_id)
                if isinstance(success_now, dict)
                else success_now
            )
            policy_capture_factor_matches = (
                capture_factor_matches.get(p_id)
                if isinstance(capture_factor_matches, dict)
                else capture_factor_matches
            )
            policy_capture_identity_candidates = (
                capture_identity_candidates.get(p_id)
                if isinstance(capture_identity_candidates, dict)
                else capture_identity_candidates
            )
            policy_capture_candidate_only_matches = (
                capture_candidate_only_matches.get(p_id)
                if isinstance(capture_candidate_only_matches, dict)
                else capture_candidate_only_matches
            )
            policy_capture_candidate_behavior = (
                capture_candidate_behavior.get(p_id)
                if isinstance(capture_candidate_behavior, dict)
                else capture_candidate_behavior
            )
            if policy_capture_counts is None:
                policy_capture_counts = np.zeros_like(
                    policy_dones_env,
                    dtype=np.float32,
                )
            if policy_success_now is None:
                policy_success_now = np.zeros_like(
                    policy_dones_env,
                    dtype=np.float32,
                )
            idx_range = self.policy_buffers[p_id].insert(num_insert_episodes, np.array(obs[p_id]),
                                                         np.array(share_obs[p_id]), np.array(acts[p_id]),
                                                         np.array(rewards[p_id]), np.array(dones[p_id]),
                                                         policy_dones_env, np.array(avail_acts[p_id]),
                                                         np.array(adj[p_id]), np.array(prob_adj[p_id]),
                                                         np.array(q_tot[p_id]),
                                                         np.array(f_v[p_id]), np.array(f_q[p_id]),
                                                         np.array(rnn_states[p_id]),
                                                         np.asarray(policy_capture_counts),
                                                         np.asarray(policy_success_now),
                                                          policy_capture_factor_matches,
                                                          policy_capture_candidate_only_matches,
                                                          policy_capture_candidate_behavior,
                                                          policy_capture_identity_candidates)
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
                 adj_capture_to_win_credit_require_future_match=False,
                 use_adj_pair_triplet_complementary_credit=False,
                 adj_pair_pursuit_credit_coef=0.0,
                 adj_pair_pursuit_credit_window=20,
                 adj_pair_pursuit_credit_cap=0.20,
                 adj_pair_pursuit_credit_min_reward=0.0):
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
        # Stable slot generations distinguish a resident episode from a later
        # episode that overwrites the same circular-buffer slot. Outcome
        # contrast support is consumable across adjacency updates: a sparse
        # historical episode may supplement one update, but must not be
        # repeatedly replayed as the missing class in every later update.
        self.episode_generation = np.full(
            self.buffer_size,
            -1,
            dtype=np.int64,
        )
        self.outcome_support_used = np.zeros(
            self.buffer_size,
            dtype=bool,
        )
        self._next_episode_generation = 0
        self._next_outcome_support_round = 0
        self._cached_outcome_support_round = None
        self._cached_outcome_support_signature = None
        self._cached_outcome_support_selection = None
        self.outcome_generation_update_count = 0
        self.outcome_slot_overwrite_count = 0
        self.outcome_generation_conflict_count = 0
        self.outcome_invalid_used_state_count = 0
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
        self.use_adj_pair_triplet_complementary_credit = bool(
            use_adj_pair_triplet_complementary_credit
        )
        self.adj_pair_pursuit_credit_coef = max(
            0.0,
            float(adj_pair_pursuit_credit_coef),
        )
        self.adj_pair_pursuit_credit_window = max(
            1,
            int(adj_pair_pursuit_credit_window),
        )
        self.adj_pair_pursuit_credit_cap = max(
            0.0,
            float(adj_pair_pursuit_credit_cap),
        )
        self.adj_pair_pursuit_credit_min_reward = float(
            adj_pair_pursuit_credit_min_reward
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
        self.capture_counts = np.zeros_like(self.dones_env, dtype=np.float32)
        self.success_now = np.zeros_like(self.dones_env, dtype=np.float32)
        self.capture_factor_matches = np.zeros(
            (
                self.episode_length,
                self.buffer_size,
                self.num_factor,
            ),
            dtype=np.float32,
        )
        self.num_candidate_factor = len(
            canonical_capture_factor_catalog(self.num_agents, 3)
        )
        self.capture_candidate_only_matches = np.zeros(
            (
                self.episode_length,
                self.buffer_size,
                self.num_candidate_factor,
            ),
            dtype=np.float32,
        )
        self.capture_candidate_behavior = np.zeros(
            (
                self.episode_length,
                self.buffer_size,
                self.num_candidate_factor,
                4,
            ),
            dtype=np.float32,
        )
        self.capture_identity_candidates = np.zeros_like(
            self.dones_env,
            dtype=np.float32,
        )
        self.adj = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, self.num_factor),
                            dtype=np.int64)
        self.prob_adj = np.zeros((self.episode_length + 1, self.buffer_size, self.num_agents, self.num_factor),
                                 dtype=np.float32)
        self.qtot = np.zeros((self.episode_length, self.buffer_size, 1), dtype=np.float32)
        self.margin_q = np.zeros((self.episode_length, self.buffer_size, self.num_agents, 1), dtype=np.float32)
        self.f_q = np.zeros((self.episode_length, self.buffer_size, self.num_factor + self.num_agents, 1),
                            dtype=np.float32)
        self.advantage = np.zeros((self.episode_length, self.buffer_size, 1), dtype=np.float32)
        # A computed graph advantage may legitimately be zero. Track whether
        # each value was actually written so an unwritten replay default cannot
        # silently erase a class-complete outcome cohort.
        self.graph_advantage_ready = np.zeros_like(
            self.advantage,
            dtype=bool,
        )
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
        self.pair_pursuit_credit = np.zeros_like(self.f_advt)
        self.pair_pursuit_quality = np.zeros_like(self.f_advt)
        self.pair_to_triplet_transition_score = np.zeros_like(self.f_advt)
        self.pair_transition_delay = np.zeros_like(self.f_advt)
        self.triplet_capture_quality = np.zeros_like(self.f_advt)
        self.capture_matched_count = np.zeros_like(
            self.dones_env,
            dtype=np.float32,
        )
        self.capture_to_win_episode_success_gate = np.zeros_like(
            self.dones_env,
            dtype=np.float32,
        )
        self.failed_episode_capture_count = np.zeros_like(
            self.dones_env,
            dtype=np.float32,
        )
        self.capture_outcome_diagnostics = np.zeros(
            (
                self.episode_length,
                self.buffer_size,
                CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH,
            ),
            dtype=np.float32,
        )
        self.positive_reward_step = np.zeros_like(
            self.dones_env,
            dtype=np.float32,
        )
        self.positive_reward_without_capture = np.zeros_like(
            self.dones_env,
            dtype=np.float32,
        )
        self.offset0_candidate_count = np.zeros_like(
            self.dones_env,
            dtype=np.float32,
        )
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
        self.pair_pursuit_credit[:, episode_idx] = 0.0
        self.pair_pursuit_quality[:, episode_idx] = 0.0
        self.pair_to_triplet_transition_score[:, episode_idx] = 0.0
        self.pair_transition_delay[:, episode_idx] = 0.0
        self.triplet_capture_quality[:, episode_idx] = 0.0
        self.capture_matched_count[:, episode_idx] = 0.0
        self.capture_to_win_episode_success_gate[:, episode_idx] = 0.0
        self.failed_episode_capture_count[:, episode_idx] = 0.0
        self.capture_outcome_diagnostics[:, episode_idx] = 0.0
        self.positive_reward_step[:, episode_idx] = 0.0
        self.positive_reward_without_capture[:, episode_idx] = 0.0
        self.offset0_candidate_count[:, episode_idx] = 0.0
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
        # dones_env[t] is post-action termination. The graph/action at t is a
        # valid training transition; only t+1 (padding after an early terminal)
        # must be excluded. This must use the same previous-done convention as
        # the recurrent replay generator below.
        previous_done_ref = build_previous_done_sequence(dones_env_ref)
        valid_factor = (
            (factor_size > 0)
            & (factor_alive_count == factor_size)
            & (previous_done_ref[..., None] < 0.5)
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
        stored_graph_advantage = write_graph_advantage_sequence(
            storage=self.advantage,
            ready_storage=self.graph_advantage_ready,
            episode_indices=episode_idx,
            graph_advantage=graph_adv,
            valid_transition=valid_graph_transition,
        )
        graph_advantage_readback = self.advantage[:, episode_idx, 0]
        graph_ready_readback = self.graph_advantage_ready[:, episode_idx, 0]
        if not np.array_equal(
                graph_ready_readback, valid_graph_transition):
            raise RuntimeError(
                "graph advantage ready mask does not match valid transitions"
            )
        storage_error = np.abs(
            graph_advantage_readback - stored_graph_advantage
        )
        if np.any(storage_error > 0.0):
            raise RuntimeError(
                "graph advantage replay write/readback mismatch: max_error={}"
                .format(float(storage_error.max()))
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

        pair_factor_mask = valid_factor & (factor_size == 2)
        triplet_factor_mask = valid_factor & (factor_size == 3)
        graph_success_gate = np.ones_like(team_rewards_ref, dtype=np.float32)

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
            graph_success_gate = success_gate.copy()
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

        capture_counts_ref = np.take(
            self.capture_counts,
            episode_idx,
            axis=1,
        )[..., 0]
        success_now_ref = np.take(
            self.success_now,
            episode_idx,
            axis=1,
        )[..., 0]
        capture_factor_matches_ref = np.take(
            self.capture_factor_matches,
            episode_idx,
            axis=1,
        )
        capture_candidate_only_matches_ref = np.take(
            self.capture_candidate_only_matches,
            episode_idx,
            axis=1,
        )
        capture_identity_candidates_ref = np.take(
            self.capture_identity_candidates,
            episode_idx,
            axis=1,
        )[..., 0]
        if capture_counts_ref.shape != team_rewards_ref.shape:
            raise RuntimeError(
                "AdjBuffer capture event axes are not [time, episode]: "
                "got {}, expected {}".format(
                    capture_counts_ref.shape,
                    team_rewards_ref.shape,
                )
            )
        if success_now_ref.shape != team_rewards_ref.shape:
            raise RuntimeError(
                "AdjBuffer success event axes are not [time, episode]: "
                "got {}, expected {}".format(
                    success_now_ref.shape,
                    team_rewards_ref.shape,
                )
            )

        pair_credit_diagnostics = compute_capture_anchored_pair_credit(
            current_adj=current_adj,
            valid_factor=valid_factor,
            factor_size=factor_size,
            capture_counts=capture_counts_ref,
            capture_factor_match=capture_factor_matches_ref,
            valid_graph_transition=valid_graph_transition,
            dones_env=dones_env_ref,
            team_rewards=team_rewards_ref,
            window=self.adj_pair_pursuit_credit_window,
            gamma=self.gamma,
        )
        pair_pursuit_credit = np.zeros_like(local_adv, dtype=np.float32)
        pair_pursuit_quality = pair_credit_diagnostics[
            "pair_pursuit_quality"
        ].astype(np.float32, copy=False)
        pair_to_triplet_transition_score = pair_credit_diagnostics[
            "pair_to_triplet_transition_score"
        ].astype(np.float32, copy=False)
        pair_transition_delay = pair_credit_diagnostics[
            "pair_transition_delay"
        ].astype(np.float32, copy=False)
        # Legacy storage/batch field name is retained for compatibility. The
        # value is definition-v5 identity-matched capture-factor quality and
        # may target an exact order-2 pair or an order-3 participant subgraph.
        triplet_capture_quality = pair_credit_diagnostics[
            "capture_factor_quality"
        ].astype(np.float32, copy=False)
        capture_matched_count = pair_credit_diagnostics[
            "capture_matched_count"
        ].astype(np.float32, copy=False)
        positive_reward_step = pair_credit_diagnostics[
            "positive_reward_step"
        ].astype(np.float32, copy=False)
        positive_reward_without_capture = pair_credit_diagnostics[
            "positive_reward_without_capture"
        ].astype(np.float32, copy=False)
        offset0_candidate_count = pair_credit_diagnostics[
            "offset0_candidate_count"
        ].astype(np.float32, copy=False)
        pair_to_triplet_transition_score *= pair_factor_mask.astype(
            np.float32
        )
        if (
            self.use_adj_pair_triplet_complementary_credit
            and self.adj_pair_pursuit_credit_coef > 0.0
        ):
            pair_pursuit_credit = (
                pair_to_triplet_transition_score
                * float(self.adj_pair_pursuit_credit_coef)
            )
            if self.adj_pair_pursuit_credit_cap > 0.0:
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
                    * float(self.adj_pair_pursuit_credit_cap)
                )
                pair_pursuit_credit = np.clip(
                    pair_pursuit_credit,
                    0.0,
                    credit_cap,
                )

        capture_to_win_triplet_credit = np.zeros_like(
            local_adv,
            dtype=np.float32,
        )
        capture_to_win_quality_gate = np.zeros_like(
            local_adv,
            dtype=np.float32,
        )
        capture_outcome_diagnostics = np.zeros(
            (
                self.episode_length,
                reference_idx.size,
                CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH,
            ),
            dtype=np.float32,
        )
        outcome_diagnostics = compute_capture_to_win_outcome_gate(
            success_now=success_now_ref,
            capture_counts=capture_counts_ref,
            valid_graph_transition=valid_graph_transition,
        )
        episode_success = outcome_diagnostics["episode_success"]
        capture_to_win_episode_success_gate = outcome_diagnostics[
            "episode_success_gate"
        ]
        failed_episode_capture_count = outcome_diagnostics[
            "failed_episode_capture_count"
        ]
        if (
            self.use_adj_capture_to_win_credit
            and self.adj_capture_to_win_credit_coef > 0.0
        ):
            # A capture-heavy failed episode can have a high shaped return.
            # Capture-to-win credit must therefore use the real environment
            # success event, never standardized team reward.
            # Keep pair->capture and capture->win as separate causal stages.
            # The old gate reused delayed positive-return/future-overlap
            # evidence, so capture-heavy failed episodes could still be
            # labelled as capture-to-win.  Compare successful and failed real
            # capture episodes directly so failures are not silently treated
            # as missing data.  Legacy future-match settings remain readable
            # but no longer alter this outcome definition.
            all_identity_capture_quality = np.concatenate(
                [
                    triplet_capture_quality,
                    capture_candidate_only_matches_ref,
                ],
                axis=2,
            )
            contrastive_outcome = (
                compute_capture_to_win_triplet_outcome_advantage(
                    episode_success=episode_success,
                    triplet_capture_quality=all_identity_capture_quality,
                )
            )
            all_identity_outcome_gate = contrastive_outcome[
                "triplet_outcome_advantage"
            ]
            capture_to_win_quality_gate = all_identity_outcome_gate[
                :, :, :self.num_factor
            ]
            capture_episode_count = int(
                contrastive_outcome["capture_episode_count"]
            )
            successful_capture_episode_count = int(
                contrastive_outcome[
                    "successful_capture_episode_count"
                ]
            )
            failed_capture_episode_count = int(
                contrastive_outcome["failed_capture_episode_count"]
            )
            mixed_outcome = float(
                successful_capture_episode_count > 0
                and failed_capture_episode_count > 0
            )
            single_success_outcome = float(
                successful_capture_episode_count > 0
                and failed_capture_episode_count == 0
            )
            single_failure_outcome = float(
                failed_capture_episode_count > 0
                and successful_capture_episode_count == 0
            )
            no_capture_window = float(capture_episode_count == 0)
            global_diagnostics = np.asarray(
                [
                    contrastive_outcome[
                        "capture_episode_success_rate"
                    ],
                    capture_episode_count,
                    successful_capture_episode_count,
                    failed_capture_episode_count,
                    mixed_outcome,
                    single_success_outcome,
                    single_failure_outcome,
                    no_capture_window,
                ],
                dtype=np.float32,
            )
            capture_identity_event_count = capture_counts_ref.sum(axis=0)
            capture_identity_matched_event_count = (
                triplet_capture_quality.sum(axis=(0, 2))
            )
            if np.any(
                    capture_identity_matched_event_count
                    > capture_identity_event_count + 1e-5):
                raise RuntimeError(
                    "identity-matched capture mass exceeds real capture count"
                )
            capture_identity_unmatched_event_count = np.maximum(
                capture_identity_event_count
                - capture_identity_matched_event_count,
                0.0,
            )
            capture_identity_candidate_factor_count = (
                capture_identity_candidates_ref.sum(axis=0)
            )
            episode_label_count = contrastive_outcome[
                "capture_triplet_count_per_episode"
            ]
            episode_gate_total = all_identity_outcome_gate.sum(axis=(0, 2))
            capture_episode_mask = contrastive_outcome[
                "capture_episode_mask"
            ]
            label_gate_correlation = 0.0
            label_gate_correlation_valid = 0.0
            if int(capture_episode_mask.sum()) >= 2:
                corr_labels = episode_label_count[capture_episode_mask]
                corr_gates = episode_gate_total[capture_episode_mask]
                if (
                    float(corr_labels.std()) > 1e-8
                    and float(corr_gates.std()) > 1e-8
                ):
                    label_gate_correlation = float(
                        np.corrcoef(corr_labels, corr_gates)[0, 1]
                    )
                    label_gate_correlation_valid = 1.0
            successful_capture_mask = capture_episode_mask & episode_success
            failed_capture_mask = capture_episode_mask & ~episode_success

            def _masked_mean(values, mask):
                return float(values[mask].mean()) if np.any(mask) else 0.0

            attribution_diagnostics = np.asarray(
                [
                    label_gate_correlation,
                    label_gate_correlation_valid,
                    _masked_mean(episode_label_count, successful_capture_mask),
                    _masked_mean(episode_label_count, failed_capture_mask),
                    _masked_mean(episode_gate_total, successful_capture_mask),
                    _masked_mean(episode_gate_total, failed_capture_mask),
                ],
                dtype=np.float32,
            )
            # Store one diagnostic row per completed episode, never one row
            # per transition. This keeps p_success and class counts truly
            # episode-weighted even when episode lengths differ.
            for ep_pos in range(reference_idx.size):
                valid_steps = np.flatnonzero(
                    valid_graph_transition[:, ep_pos]
                )
                if valid_steps.size == 0:
                    continue
                diagnostic_step = int(valid_steps[0])
                capture_outcome_diagnostics[
                    valid_steps, ep_pos, :8
                ] = global_diagnostics
                capture_outcome_diagnostics[
                    diagnostic_step, ep_pos, 8
                ] = contrastive_outcome[
                    "capture_triplet_count_per_episode"
                ][ep_pos]
                capture_outcome_diagnostics[
                    diagnostic_step, ep_pos, 9
                ] = contrastive_outcome[
                    "episode_outcome_advantage"
                ][ep_pos]
                capture_outcome_diagnostics[
                    diagnostic_step, ep_pos, 10
                ] = 1.0
                capture_episode_mask = contrastive_outcome[
                    "capture_episode_mask"
                ]
                if np.any(capture_episode_mask):
                    window_raw_outcome_mean = float(
                        contrastive_outcome[
                            "episode_outcome_advantage"
                        ][capture_episode_mask].mean()
                    )
                else:
                    window_raw_outcome_mean = 0.0
                capture_outcome_diagnostics[
                    valid_steps, ep_pos, 11
                ] = window_raw_outcome_mean
                capture_outcome_diagnostics[
                    valid_steps, ep_pos, 12
                ] = float(all_identity_outcome_gate.sum())
                capture_outcome_diagnostics[
                    valid_steps, ep_pos, 13
                ] = float(np.abs(all_identity_outcome_gate).sum())
                capture_outcome_diagnostics[
                    valid_steps, ep_pos, 24:30
                ] = attribution_diagnostics
                capture_outcome_diagnostics[
                    diagnostic_step, ep_pos, 20
                ] = capture_identity_event_count[ep_pos]
                capture_outcome_diagnostics[
                    diagnostic_step, ep_pos, 21
                ] = capture_identity_matched_event_count[ep_pos]
                capture_outcome_diagnostics[
                    diagnostic_step, ep_pos, 22
                ] = capture_identity_unmatched_event_count[ep_pos]
                capture_outcome_diagnostics[
                    diagnostic_step, ep_pos, 23
                ] = capture_identity_candidate_factor_count[ep_pos]

            scaled_outcome_credit = scale_capture_to_win_outcome_credit(
                triplet_outcome_advantage=capture_to_win_quality_gate,
                graph_advantage=graph_adv,
                valid_graph_transition=valid_graph_transition,
                coefficient=self.adj_capture_to_win_credit_coef,
                cap=self.adj_capture_to_win_credit_cap,
                return_diagnostics=True,
            )
            capture_to_win_triplet_credit = scaled_outcome_credit["credit"]
            capture_outcome_diagnostics[..., 14][valid_graph_transition] = (
                scaled_outcome_credit["preclip_mean"]
            )
            capture_outcome_diagnostics[..., 15][valid_graph_transition] = (
                scaled_outcome_credit["preclip_std"]
            )
            capture_outcome_diagnostics[..., 16][valid_graph_transition] = (
                scaled_outcome_credit["preclip_max"]
            )
            capture_outcome_diagnostics[..., 17][valid_graph_transition] = (
                scaled_outcome_credit["preclip_min"]
            )
            capture_outcome_diagnostics[..., 18][valid_graph_transition] = (
                scaled_outcome_credit["positive_clip_fraction"]
            )
            capture_outcome_diagnostics[..., 19][valid_graph_transition] = (
                scaled_outcome_credit["negative_clip_fraction"]
            )

        combined_adv = (
            self.adj_return_adv_coef * graph_factor_adv
            + self.adj_factor_adv_coef * local_adv
            + delayed_triplet_credit
            + pair_pursuit_credit
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
        self.capture_outcome_diagnostics[
            :, episode_idx, :
        ] = capture_outcome_diagnostics
        pair_pursuit_credit_values = self.pair_pursuit_credit[..., 0]
        pair_pursuit_credit_values[:, episode_idx, :] = pair_pursuit_credit
        pair_pursuit_quality_values = self.pair_pursuit_quality[..., 0]
        pair_pursuit_quality_values[:, episode_idx, :] = pair_pursuit_quality
        pair_to_triplet_transition_values = (
            self.pair_to_triplet_transition_score[..., 0]
        )
        pair_to_triplet_transition_values[:, episode_idx, :] = (
            pair_to_triplet_transition_score
        )
        pair_transition_delay_values = self.pair_transition_delay[..., 0]
        pair_transition_delay_values[:, episode_idx, :] = pair_transition_delay
        triplet_capture_quality_values = self.triplet_capture_quality[..., 0]
        triplet_capture_quality_values[:, episode_idx, :] = (
            triplet_capture_quality
        )
        capture_matched_count_values = self.capture_matched_count[..., 0]
        capture_matched_count_values[:, episode_idx] = capture_matched_count
        episode_success_gate_values = (
            self.capture_to_win_episode_success_gate[..., 0]
        )
        episode_success_gate_values[:, episode_idx] = (
            capture_to_win_episode_success_gate
        )
        failed_episode_capture_values = (
            self.failed_episode_capture_count[..., 0]
        )
        failed_episode_capture_values[:, episode_idx] = (
            failed_episode_capture_count
        )
        positive_reward_step_values = self.positive_reward_step[..., 0]
        positive_reward_step_values[:, episode_idx] = positive_reward_step
        positive_reward_without_capture_values = (
            self.positive_reward_without_capture[..., 0]
        )
        positive_reward_without_capture_values[:, episode_idx] = (
            positive_reward_without_capture
        )
        offset0_candidate_count_values = self.offset0_candidate_count[..., 0]
        offset0_candidate_count_values[:, episode_idx] = (
            offset0_candidate_count
        )

        return idx

    def insert(self, num_insert_episodes, obs, share_obs, acts, rewards, dones, dones_env, avail_acts, adj=None,
               prob_adj=None, qtot=None, f_v=None, f_q=None, rnn_obs=None,
               capture_counts=None, success_now=None,
               capture_factor_matches=None,
               capture_candidate_only_matches=None,
               capture_candidate_behavior=None,
               capture_identity_candidates=None):
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

        expected_event_shape = (
            self.episode_length,
            num_insert_episodes,
            1,
        )
        if capture_counts is None:
            capture_counts = np.zeros(expected_event_shape, dtype=np.float32)
        if success_now is None:
            success_now = np.zeros(expected_event_shape, dtype=np.float32)
        capture_counts = np.asarray(capture_counts, dtype=np.float32)
        success_now = np.asarray(success_now, dtype=np.float32)
        if capture_counts.ndim == 2:
            capture_counts = capture_counts[..., None]
        if success_now.ndim == 2:
            success_now = success_now[..., None]
        if capture_counts.shape != expected_event_shape:
            raise ValueError(
                "capture_counts must have shape {}, got {}".format(
                    expected_event_shape,
                    capture_counts.shape,
                )
            )
        if success_now.shape != expected_event_shape:
            raise ValueError(
                "success_now must have shape {}, got {}".format(
                    expected_event_shape,
                    success_now.shape,
                )
            )
        capture_counts = np.where(
            np.isfinite(capture_counts),
            np.maximum(capture_counts, 0.0),
            0.0,
        ).astype(np.float32, copy=False)
        success_now = (
            np.isfinite(success_now) & (success_now > 0.0)
        ).astype(np.float32)
        expected_factor_match_shape = (
            self.episode_length,
            num_insert_episodes,
            self.num_factor,
        )
        if capture_factor_matches is None:
            if self.use_adj_capture_to_win_credit:
                raise RuntimeError(
                    "capture-to-win credit requires identity-matched "
                    "capture_factor_matches"
                )
            capture_factor_matches = np.zeros(
                expected_factor_match_shape,
                dtype=np.float32,
            )
        capture_factor_matches = np.asarray(
            capture_factor_matches,
            dtype=np.float32,
        )
        if capture_factor_matches.shape != expected_factor_match_shape:
            raise ValueError(
                "capture_factor_matches must have shape {}, got {}".format(
                    expected_factor_match_shape,
                    capture_factor_matches.shape,
                )
            )
        if not np.isfinite(capture_factor_matches).all() or np.any(
                capture_factor_matches < 0.0):
            raise ValueError(
                "capture_factor_matches must be finite and non-negative"
            )
        expected_candidate_match_shape = (
            self.episode_length,
            num_insert_episodes,
            self.num_candidate_factor,
        )
        if capture_candidate_only_matches is None:
            if self.use_adj_capture_to_win_credit:
                raise RuntimeError(
                    "capture-to-win credit requires exact candidate-only "
                    "identity matches"
                )
            capture_candidate_only_matches = np.zeros(
                expected_candidate_match_shape,
                dtype=np.float32,
            )
        capture_candidate_only_matches = np.asarray(
            capture_candidate_only_matches,
            dtype=np.float32,
        )
        if capture_candidate_only_matches.shape != expected_candidate_match_shape:
            raise ValueError(
                "capture_candidate_only_matches must have shape {}, got {}"
                .format(
                    expected_candidate_match_shape,
                    capture_candidate_only_matches.shape,
                )
            )
        if (
                not np.isfinite(capture_candidate_only_matches).all()
                or np.any(capture_candidate_only_matches < 0.0)):
            raise ValueError(
                "capture_candidate_only_matches must be finite and "
                "non-negative"
            )
        expected_candidate_behavior_shape = (
            self.episode_length,
            num_insert_episodes,
            self.num_candidate_factor,
            4,
        )
        if capture_candidate_behavior is None:
            if self.use_adj_capture_to_win_credit:
                raise RuntimeError(
                    "capture-to-win candidate supervision requires rollout "
                    "candidate score/rank/mask/policy-version metadata"
                )
            capture_candidate_behavior = np.zeros(
                expected_candidate_behavior_shape,
                dtype=np.float32,
            )
        capture_candidate_behavior = np.asarray(
            capture_candidate_behavior,
            dtype=np.float32,
        )
        if capture_candidate_behavior.shape != expected_candidate_behavior_shape:
            raise ValueError(
                "capture_candidate_behavior must have shape {}, got {}"
                .format(
                    expected_candidate_behavior_shape,
                    capture_candidate_behavior.shape,
                )
            )
        if not np.isfinite(capture_candidate_behavior).all():
            raise ValueError("capture_candidate_behavior must be finite")
        behavior_score = capture_candidate_behavior[..., 0]
        behavior_rank = capture_candidate_behavior[..., 1]
        behavior_valid = capture_candidate_behavior[..., 2]
        behavior_version = capture_candidate_behavior[..., 3]
        if (
                np.any(behavior_score < 0.0)
                or np.any(behavior_rank < 0.0)
                or np.any(behavior_version < 0.0)
                or np.any((behavior_valid != 0.0) & (behavior_valid != 1.0))):
            raise ValueError(
                "candidate behavior score/rank/version must be non-negative "
                "and valid_mask must be binary"
            )
        candidate_target = capture_candidate_only_matches > 0.0
        if np.any(candidate_target & (behavior_valid <= 0.0)):
            raise RuntimeError(
                "candidate-only capture identity targets an invalid rollout "
                "candidate"
            )
        if np.any(candidate_target & (behavior_rank < 1.0)):
            raise RuntimeError(
                "candidate-only capture identity has no positive canonical "
                "behavior rank"
            )
        if capture_identity_candidates is None:
            capture_identity_candidates = np.zeros(
                expected_event_shape,
                dtype=np.float32,
            )
        capture_identity_candidates = np.asarray(
            capture_identity_candidates,
            dtype=np.float32,
        )
        if capture_identity_candidates.ndim == 2:
            capture_identity_candidates = (
                capture_identity_candidates[..., None]
            )
        if capture_identity_candidates.shape != expected_event_shape:
            raise ValueError(
                "capture_identity_candidates must have shape {}, got {}"
                .format(
                    expected_event_shape,
                    capture_identity_candidates.shape,
                )
            )
        if not np.isfinite(capture_identity_candidates).all() or np.any(
                capture_identity_candidates < 0.0):
            raise ValueError(
                "capture_identity_candidates must be finite and non-negative"
            )

        if self.current_i + num_insert_episodes <= self.buffer_size:
            idx_range = np.arange(self.current_i, self.current_i + num_insert_episodes)
        else:
            num_left_episodes = self.current_i + num_insert_episodes - self.buffer_size
            idx_range = np.concatenate((np.arange(self.current_i, self.buffer_size), np.arange(num_left_episodes)))

        if self.use_same_share_obs:
            # remove agent dimension since all agents share centralized observation
            share_obs = share_obs[:, :, 0]

        overwritten_mask = self.episode_generation[idx_range] >= 0
        self.outcome_slot_overwrite_count += int(np.sum(overwritten_mask))
        if np.any(
                self.outcome_support_used[idx_range]
                & ~overwritten_mask):
            self.outcome_invalid_used_state_count += int(np.sum(
                self.outcome_support_used[idx_range] & ~overwritten_mask
            ))
            raise RuntimeError(
                "unused adjacency replay slot carries outcome support state"
            )
        generations = np.arange(
            self._next_episode_generation,
            self._next_episode_generation + len(idx_range),
            dtype=np.int64,
        )
        self._next_episode_generation += len(idx_range)
        self.outcome_generation_update_count += len(idx_range)
        self.episode_generation[idx_range] = generations
        self.outcome_support_used[idx_range] = False
        # A circular-buffer overwrite cannot inherit the prior episode's graph
        # confidence. compute_advantage() repopulates both arrays before replay.
        self.advantage[:, idx_range, 0] = 0.0
        self.graph_advantage_ready[:, idx_range, 0] = False
        occupied_generations = self.episode_generation[
            self.episode_generation >= 0
        ]
        if np.unique(occupied_generations).size != occupied_generations.size:
            self.outcome_generation_conflict_count += 1
            raise RuntimeError(
                "occupied adjacency replay slots share an episode generation"
            )
        # Insertion between PPO epochs would invalidate the cohort identity;
        # make such a state visible rather than silently reusing stale slots.
        self._cached_outcome_support_round = None
        self._cached_outcome_support_signature = None
        self._cached_outcome_support_selection = None

        self.obs[:, idx_range] = obs.copy()
        self.share_obs[:, idx_range] = share_obs.copy()
        self.acts[:, idx_range] = acts.copy()
        self.rewards[:, idx_range] = rewards.copy()
        self.dones[:, idx_range] = dones.copy()
        self.dones_env[:, idx_range] = dones_env.copy()
        self.capture_counts[:, idx_range] = capture_counts.copy()
        self.success_now[:, idx_range] = success_now.copy()
        self.capture_factor_matches[:, idx_range] = (
            capture_factor_matches.copy()
        )
        self.capture_candidate_only_matches[:, idx_range] = (
            capture_candidate_only_matches.copy()
        )
        self.capture_candidate_behavior[:, idx_range] = (
            capture_candidate_behavior.copy()
        )
        self.capture_identity_candidates[:, idx_range] = (
            capture_identity_candidates.copy()
        )
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
            recent_episode_window=0,
            outcome_support_round=None):
        """
        只从 filled_i 范围内采样，避免未填充 episode 污染邻接训练。
        """
        base_episode_indices = self._recent_episode_indices(
            recent_episode_window
        )
        valid_slot_count = int(self.filled_i)
        end = int(self.current_i) % int(self.buffer_size)
        recency_order = (
            (end - 1 - np.arange(valid_slot_count, dtype=np.int64))
            % int(self.buffer_size)
        )
        if np.any(self.episode_generation[recency_order] < 0):
            raise RuntimeError(
                "occupied adjacency replay slots are missing episode "
                "generation identities"
            )
        # Outcome support is a property of completed, identity-matched capture
        # episodes.  Do not infer the class from the already scaled credit:
        # graph confidence can legitimately turn a labelled episode's final
        # credit into zero, which would make sampling and the centered
        # baseline use different populations.
        active_capture_episode_mask = np.any(
            self.triplet_capture_quality[..., 0] > 0.0,
            axis=(0, 2),
        )
        candidate_capture_episode_mask = np.any(
            self.capture_candidate_only_matches > 0.0,
            axis=(0, 2),
        )
        capture_episode_mask = (
            active_capture_episode_mask | candidate_capture_episode_mask
        )
        successful_episode_mask = np.any(
            self.success_now[..., 0] > 0.0,
            axis=0,
        )
        positive_episode_mask = (
            capture_episode_mask & successful_episode_mask
        )
        negative_episode_mask = (
            capture_episode_mask & ~successful_episode_mask
        )
        if outcome_support_round is None:
            outcome_support_round = self._next_outcome_support_round
            self._next_outcome_support_round += 1
        outcome_support_round = int(outcome_support_round)
        support_signature = (
            tuple(base_episode_indices.tolist()),
            tuple(recency_order.tolist()),
            tuple(self.episode_generation[recency_order].tolist()),
            positive_episode_mask.tobytes(),
            negative_episode_mask.tobytes(),
        )
        cached_selection_reused = 0.0
        if self._cached_outcome_support_round == outcome_support_round:
            if self._cached_outcome_support_signature != support_signature:
                raise RuntimeError(
                    "adjacency replay content changed within one outcome "
                    "support round"
                )
            contrast_selection = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in
                self._cached_outcome_support_selection.items()
            }
            cached_selection_reused = 1.0
        else:
            contrast_selection = select_outcome_contrast_complete_episodes(
                base_episode_indices,
                positive_episode_mask,
                negative_episode_mask,
                recency_order,
                positive_eligible_mask=(
                    positive_episode_mask & ~self.outcome_support_used
                ),
                negative_eligible_mask=(
                    negative_episode_mask & ~self.outcome_support_used
                ),
            )
            supplemented = contrast_selection[
                "supplemented_episode_indices"
            ]
            if supplemented.size:
                if np.any(self.outcome_support_used[supplemented]):
                    raise RuntimeError(
                        "outcome support episode was reused across adjacency "
                        "updates"
                    )
                self.outcome_support_used[supplemented] = True
            self._cached_outcome_support_round = outcome_support_round
            self._cached_outcome_support_signature = support_signature
            self._cached_outcome_support_selection = {
                key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in contrast_selection.items()
            }
        episode_indices = contrast_selection["episode_indices"]
        valid_episodes = int(episode_indices.size)
        self.last_sample_base_episode_count = int(base_episode_indices.size)
        self.last_sample_episode_count = valid_episodes
        self.last_sample_recent_window = int(recent_episode_window or 0)
        self.last_sample_episode_indices = episode_indices.copy()
        self.last_sample_outcome_contrast_augmented_count = int(
            contrast_selection["augmented_count"]
        )
        self.last_sample_outcome_positive_available = float(
            contrast_selection["positive_available"]
        )
        self.last_sample_outcome_negative_available = float(
            contrast_selection["negative_available"]
        )
        self.last_sample_outcome_positive_episode_count = int(
            contrast_selection["positive_selected_count"]
        )
        self.last_sample_outcome_negative_episode_count = int(
            contrast_selection["negative_selected_count"]
        )
        self.last_sample_outcome_class_complete = float(
            contrast_selection["class_complete"]
        )
        self.last_sample_outcome_support_exhausted = float(
            contrast_selection["support_exhausted"]
        )
        self.last_sample_outcome_credit_enabled = float(
            contrast_selection["outcome_credit_enabled"]
        )
        self.last_sample_outcome_cached_selection_reused = float(
            cached_selection_reused
        )
        self.last_sample_outcome_support_round = int(outcome_support_round)
        self.last_sample_outcome_cross_update_reuse_count = 0
        self.last_sample_outcome_positive_available_count = int(
            contrast_selection["positive_available_count"]
        )
        self.last_sample_outcome_negative_available_count = int(
            contrast_selection["negative_available_count"]
        )
        self.last_sample_outcome_base_positive_count = int(
            contrast_selection["base_positive_count"]
        )
        self.last_sample_outcome_base_negative_count = int(
            contrast_selection["base_negative_count"]
        )
        self.last_sample_outcome_augmented_positive_count = int(
            contrast_selection["augmented_positive_count"]
        )
        self.last_sample_outcome_augmented_negative_count = int(
            contrast_selection["augmented_negative_count"]
        )
        recency_age = np.full(self.buffer_size, -1, dtype=np.int64)
        recency_age[recency_order] = np.arange(
            recency_order.size,
            dtype=np.int64,
        )
        supplemented = contrast_selection["supplemented_episode_indices"]
        base_ages = recency_age[base_episode_indices]
        supplemented_ages = recency_age[supplemented]
        self.last_sample_outcome_base_age_mean = float(
            np.mean(base_ages) if base_ages.size else np.nan
        )
        self.last_sample_outcome_base_age_max = float(
            np.max(base_ages) if base_ages.size else np.nan
        )
        self.last_sample_outcome_augmented_age_mean = float(
            np.mean(supplemented_ages)
            if supplemented_ages.size else np.nan
        )
        self.last_sample_outcome_augmented_age_max = float(
            np.max(supplemented_ages)
            if supplemented_ages.size else np.nan
        )
        positive_supplement = supplemented[
            positive_episode_mask[supplemented]
        ]
        negative_supplement = supplemented[
            negative_episode_mask[supplemented]
        ]
        self.last_sample_outcome_positive_support_generation = float(
            self.episode_generation[positive_supplement[0]]
            if positive_supplement.size else -1
        )
        self.last_sample_outcome_negative_support_generation = float(
            self.episode_generation[negative_supplement[0]]
            if negative_supplement.size else -1
        )
        self.last_sample_outcome_positive_support_age = float(
            recency_age[positive_supplement[0]]
            if positive_supplement.size else np.nan
        )
        self.last_sample_outcome_negative_support_age = float(
            recency_age[negative_supplement[0]]
            if negative_supplement.size else np.nan
        )
        occupied_outcome_mask = (
            positive_episode_mask | negative_episode_mask
        )
        occupied_outcome_count = int(np.sum(occupied_outcome_mask))
        used_outcome_count = int(np.sum(
            self.outcome_support_used & occupied_outcome_mask
        ))
        self.last_sample_outcome_support_used_count = used_outcome_count
        self.last_sample_outcome_support_used_fraction = float(
            used_outcome_count / float(max(occupied_outcome_count, 1))
        )
        self.last_sample_outcome_generation_update_count = int(
            self.outcome_generation_update_count
        )
        self.last_sample_outcome_slot_overwrite_count = int(
            self.outcome_slot_overwrite_count
        )
        self.last_sample_outcome_generation_conflict_count = int(
            self.outcome_generation_conflict_count
        )
        self.last_sample_outcome_invalid_used_state_count = int(
            self.outcome_invalid_used_state_count
        )

        def _cohort_success_rate(indices):
            indices = np.asarray(indices, dtype=np.int64).reshape(-1)
            positive_count = int(np.sum(positive_episode_mask[indices]))
            negative_count = int(np.sum(negative_episode_mask[indices]))
            labelled_count = positive_count + negative_count
            return (
                float(positive_count / float(labelled_count))
                if labelled_count > 0 else np.nan
            )

        full_buffer_baseline = _cohort_success_rate(recency_order)
        base_cohort_baseline = _cohort_success_rate(base_episode_indices)
        trained_cohort_baseline = _cohort_success_rate(episode_indices)
        self.last_sample_outcome_full_buffer_baseline = full_buffer_baseline
        self.last_sample_outcome_base_cohort_baseline = base_cohort_baseline
        self.last_sample_outcome_trained_cohort_baseline = (
            trained_cohort_baseline
        )
        self.last_sample_outcome_full_trained_baseline_gap = float(
            abs(trained_cohort_baseline - full_buffer_baseline)
            if np.isfinite(trained_cohort_baseline)
            and np.isfinite(full_buffer_baseline)
            else np.nan
        )
        self.last_sample_outcome_trained_capture_episode_count = int(
            np.sum(positive_episode_mask[episode_indices])
            + np.sum(negative_episode_mask[episode_indices])
        )
        self.last_sample_outcome_cohort_centered_sum = 0.0
        self.last_sample_outcome_cohort_center_error = 0.0
        self.last_sample_outcome_cohort_center_valid = 0.0
        self.last_sample_outcome_positive_gate_episode_count = 0
        self.last_sample_outcome_negative_gate_episode_count = 0
        self.last_sample_outcome_positive_credit_episode_count = 0
        self.last_sample_outcome_negative_credit_episode_count = 0
        self.last_sample_outcome_signed_scaling_version = 3.0
        self.last_sample_outcome_graph_advantage_source_ready_fraction = 0.0
        self.last_sample_outcome_graph_confidence_mean = 0.0
        self.last_sample_outcome_graph_confidence_std = 0.0
        self.last_sample_outcome_graph_confidence_p50 = 0.0
        self.last_sample_outcome_graph_confidence_p95 = 0.0
        self.last_sample_outcome_graph_confidence_max = 0.0
        self.last_sample_outcome_positive_graph_confidence_mean = 0.0
        self.last_sample_outcome_positive_graph_confidence_max = 0.0
        self.last_sample_outcome_negative_graph_confidence_mean = 0.0
        self.last_sample_outcome_negative_graph_confidence_max = 0.0
        self.last_sample_outcome_graph_advantage_positive_fraction = 0.0
        self.last_sample_outcome_graph_advantage_negative_fraction = 0.0
        self.last_sample_outcome_graph_advantage_zero_fraction = 0.0
        self.last_sample_outcome_positive_zero_confidence_fraction = 0.0
        self.last_sample_outcome_negative_zero_confidence_fraction = 0.0
        self.last_sample_outcome_gate_to_credit_drop_fraction = 0.0
        self.last_sample_outcome_preclip_positive_mass = 0.0
        self.last_sample_outcome_preclip_negative_mass = 0.0
        self.last_sample_outcome_postclip_positive_mass = 0.0
        self.last_sample_outcome_postclip_negative_mass = 0.0
        self.last_sample_outcome_positive_clip_fraction = 0.0
        self.last_sample_outcome_negative_clip_fraction = 0.0

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
        dones_seq = np.all(
            obs_seq <= -0.999, axis=-1, keepdims=True
        ).astype(np.float32)
        dones = (
            dones_seq.transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_agents, 1)
        )

        # Keep environment termination semantics from the original shifted
        # dones_env path; only per-slot activity is reconstructed from obs.
        dones_env_seq = np.take(
            self.dones_env,
            episode_indices,
            axis=1,
        )
        previous_dones_env_seq = build_previous_done_sequence(dones_env_seq)
        dones_env = (
            previous_dones_env_seq.transpose(1, 0, 2)
            .reshape(batch_size, 1)
        )

        adj_seq_full = np.take(self.adj[:-1], episode_indices, axis=1)
        previous_adj_seq_full = build_previous_adjacency_sequence(
            adj_seq_full
        )
        prob_adj_seq = np.take(
            self.prob_adj[:-1],
            episode_indices,
            axis=1,
        )
        rnn_obs_seq = np.take(self.rnn_obs[:-1], episode_indices, axis=1)
        adj = adj_seq_full.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        previous_adj = (
            previous_adj_seq_full
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_agents, -1)
        )
        prob_adj = prob_adj_seq.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)
        rnn_obs = rnn_obs_seq.transpose(1, 0, 2, 3).reshape(batch_size, self.num_agents, -1)

        advantage = np.take(self.advantage, episode_indices, axis=1)
        graph_advantage_ready = np.take(
            self.graph_advantage_ready,
            episode_indices,
            axis=1,
        )
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
        pair_pursuit_credit = np.take(
            self.pair_pursuit_credit,
            episode_indices,
            axis=1,
        )
        pair_pursuit_quality = np.take(
            self.pair_pursuit_quality,
            episode_indices,
            axis=1,
        )
        pair_to_triplet_transition_score = np.take(
            self.pair_to_triplet_transition_score,
            episode_indices,
            axis=1,
        )
        triplet_capture_quality = np.take(
            self.triplet_capture_quality,
            episode_indices,
            axis=1,
        )
        candidate_only_capture_quality = np.take(
            self.capture_candidate_only_matches,
            episode_indices,
            axis=1,
        ).astype(np.float32, copy=False)
        candidate_behavior = np.take(
            self.capture_candidate_behavior,
            episode_indices,
            axis=1,
        ).astype(np.float32, copy=False)
        candidate_identity_delta = np.zeros_like(
            candidate_only_capture_quality,
            dtype=np.float32,
        )
        pair_transition_delay = np.take(
            self.pair_transition_delay,
            episode_indices,
            axis=1,
        )
        capture_counts = np.take(
            self.capture_counts,
            episode_indices,
            axis=1,
        )
        capture_matched_count = np.take(
            self.capture_matched_count,
            episode_indices,
            axis=1,
        )
        capture_to_win_episode_success_gate = np.take(
            self.capture_to_win_episode_success_gate,
            episode_indices,
            axis=1,
        )
        failed_episode_capture_count = np.take(
            self.failed_episode_capture_count,
            episode_indices,
            axis=1,
        )
        capture_outcome_diagnostics = np.take(
            self.capture_outcome_diagnostics,
            episode_indices,
            axis=1,
        )
        positive_reward_step = np.take(
            self.positive_reward_step,
            episode_indices,
            axis=1,
        )
        positive_reward_without_capture = np.take(
            self.positive_reward_without_capture,
            episode_indices,
            axis=1,
        )
        offset0_candidate_count = np.take(
            self.offset0_candidate_count,
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
            & (previous_dones_env_seq[..., 0, None] < 0.5)
        )

        # The centered outcome baseline must be defined over the episodes that
        # actually reach this optimizer update.  compute_advantage() maintains
        # a full-buffer diagnostic population, whereas replay support may add
        # or remove episodes before training.  Reusing the stored full-buffer
        # gate here would make a nominally complete 1-positive/1-negative
        # cohort carry a non-zero raw centered sum whenever the full-buffer
        # success rate differs from 0.5.
        if bool(contrast_selection["outcome_credit_enabled"]):
            selected_success = np.any(
                np.take(self.success_now, episode_indices, axis=1)[..., 0]
                > 0.0,
                axis=0,
            )
            selected_active_capture_quality = (
                triplet_capture_quality[..., 0].astype(
                    np.float32,
                    copy=False,
                )
            )
            # Every real event appears in exactly one branch: an exact active
            # factor when selected, otherwise its exact canonical candidate.
            # Concatenating before centering gives both branches one common
            # final-optimizer-cohort baseline without duplicating event mass.
            selected_capture_quality = np.concatenate(
                [
                    selected_active_capture_quality,
                    candidate_only_capture_quality,
                ],
                axis=2,
            )
            selected_contrast = (
                compute_capture_to_win_triplet_outcome_advantage(
                    episode_success=selected_success,
                    triplet_capture_quality=selected_capture_quality,
                )
            )
            selected_positive_count = int(
                selected_contrast["successful_capture_episode_count"]
            )
            selected_negative_count = int(
                selected_contrast["failed_capture_episode_count"]
            )
            if selected_positive_count <= 0 or selected_negative_count <= 0:
                raise RuntimeError(
                    "outcome support marked a cohort complete without both "
                    "real completed capture outcomes"
                )
            cohort_gate_all = selected_contrast[
                "triplet_outcome_advantage"
            ].astype(np.float32, copy=False)
            cohort_gate = cohort_gate_all[:, :, :self.num_factor]
            cohort_candidate_gate = cohort_gate_all[:, :, self.num_factor:]
            cohort_episode_total = cohort_gate_all.sum(axis=(0, 2))
            cohort_capture_mask = selected_contrast["capture_episode_mask"]
            cohort_centered_sum = float(
                cohort_episode_total[cohort_capture_mask].sum()
            )
            cohort_center_error = abs(cohort_centered_sum)
            if cohort_center_error > 1e-5:
                raise RuntimeError(
                    "sampled capture-outcome cohort is not centered: {}"
                    .format(cohort_centered_sum)
                )
            valid_graph_transition = np.any(valid_factor, axis=2)
            labelled_graph_transition = np.any(
                np.abs(cohort_gate) > 0.0,
                axis=2,
            )
            cohort_graph_advantage = require_ready_graph_advantage(
                graph_advantage=advantage[..., 0],
                ready_mask=graph_advantage_ready[..., 0],
                labelled_transition=labelled_graph_transition,
            )
            labelled_count = int(np.sum(labelled_graph_transition))
            self.last_sample_outcome_graph_advantage_source_ready_fraction = (
                float(np.sum(
                    graph_advantage_ready[..., 0]
                    & labelled_graph_transition
                )) / float(max(labelled_count, 1))
            )
            scaled_cohort_credit = scale_capture_to_win_outcome_credit(
                triplet_outcome_advantage=cohort_gate,
                graph_advantage=cohort_graph_advantage,
                valid_graph_transition=valid_graph_transition,
                coefficient=self.adj_capture_to_win_credit_coef,
                cap=self.adj_capture_to_win_credit_cap,
                return_diagnostics=True,
            )
            candidate_identity_delta = (
                cohort_candidate_gate
                * float(self.adj_capture_to_win_credit_coef)
            ).astype(np.float32, copy=False)
            if self.adj_capture_to_win_credit_cap > 0.0:
                candidate_identity_delta = np.clip(
                    candidate_identity_delta,
                    -float(self.adj_capture_to_win_credit_cap),
                    float(self.adj_capture_to_win_credit_cap),
                )
            capture_to_win_quality_gate = cohort_gate[..., None]
            capture_to_win_triplet_credit = scaled_cohort_credit[
                "credit"
            ][..., None]
            self.last_sample_outcome_cohort_centered_sum = (
                cohort_centered_sum
            )
            self.last_sample_outcome_cohort_center_error = (
                cohort_center_error
            )
            self.last_sample_outcome_cohort_center_valid = 1.0
            self.last_sample_outcome_positive_gate_episode_count = int(
                np.sum(np.any(cohort_gate_all > 0.0, axis=(0, 2)))
            )
            self.last_sample_outcome_negative_gate_episode_count = int(
                np.sum(np.any(cohort_gate_all < 0.0, axis=(0, 2)))
            )
            self.last_sample_outcome_positive_credit_episode_count = int(
                np.sum(
                    np.any(
                        scaled_cohort_credit["credit"] > 0.0,
                        axis=(0, 2),
                    )
                    | np.any(candidate_identity_delta > 0.0, axis=(0, 2))
                )
            )
            self.last_sample_outcome_negative_credit_episode_count = int(
                np.sum(
                    np.any(
                        scaled_cohort_credit["credit"] < 0.0,
                        axis=(0, 2),
                    )
                    | np.any(candidate_identity_delta < 0.0, axis=(0, 2))
                )
            )
            self.last_sample_outcome_graph_confidence_mean = float(
                scaled_cohort_credit["graph_confidence_mean"]
            )
            self.last_sample_outcome_graph_confidence_std = float(
                scaled_cohort_credit["graph_confidence_std"]
            )
            self.last_sample_outcome_graph_confidence_p50 = float(
                scaled_cohort_credit["graph_confidence_p50"]
            )
            self.last_sample_outcome_graph_confidence_p95 = float(
                scaled_cohort_credit["graph_confidence_p95"]
            )
            self.last_sample_outcome_graph_confidence_max = float(
                scaled_cohort_credit["graph_confidence_max"]
            )
            self.last_sample_outcome_positive_graph_confidence_mean = float(
                scaled_cohort_credit["positive_graph_confidence_mean"]
            )
            self.last_sample_outcome_positive_graph_confidence_max = float(
                scaled_cohort_credit["positive_graph_confidence_max"]
            )
            self.last_sample_outcome_negative_graph_confidence_mean = float(
                scaled_cohort_credit["negative_graph_confidence_mean"]
            )
            self.last_sample_outcome_negative_graph_confidence_max = float(
                scaled_cohort_credit["negative_graph_confidence_max"]
            )
            self.last_sample_outcome_graph_advantage_positive_fraction = float(
                scaled_cohort_credit[
                    "labelled_graph_advantage_positive_fraction"
                ]
            )
            self.last_sample_outcome_graph_advantage_negative_fraction = float(
                scaled_cohort_credit[
                    "labelled_graph_advantage_negative_fraction"
                ]
            )
            self.last_sample_outcome_graph_advantage_zero_fraction = float(
                scaled_cohort_credit["labelled_graph_advantage_zero_fraction"]
            )
            self.last_sample_outcome_positive_zero_confidence_fraction = float(
                scaled_cohort_credit["positive_zero_confidence_fraction"]
            )
            self.last_sample_outcome_negative_zero_confidence_fraction = float(
                scaled_cohort_credit["negative_zero_confidence_fraction"]
            )
            self.last_sample_outcome_gate_to_credit_drop_fraction = float(
                scaled_cohort_credit["gate_to_credit_drop_fraction"]
            )
            self.last_sample_outcome_preclip_positive_mass = float(
                scaled_cohort_credit["preclip_positive_mass"]
            )
            self.last_sample_outcome_preclip_negative_mass = float(
                scaled_cohort_credit["preclip_negative_mass"]
            )
            self.last_sample_outcome_postclip_positive_mass = float(
                scaled_cohort_credit["postclip_positive_mass"]
            )
            self.last_sample_outcome_postclip_negative_mass = float(
                scaled_cohort_credit["postclip_negative_mass"]
            )
            self.last_sample_outcome_positive_clip_fraction = float(
                scaled_cohort_credit["positive_clip_fraction"]
            )
            self.last_sample_outcome_negative_clip_fraction = float(
                scaled_cohort_credit["negative_clip_fraction"]
            )
        else:
            # A one-sided optimizer cohort is not a valid contrast. Preserve
            # all other replay fields and graph/pair objectives, but suppress
            # only capture-outcome training inputs for this cohort.
            capture_to_win_triplet_credit.fill(0.0)
            capture_to_win_quality_gate.fill(0.0)
            candidate_identity_delta.fill(0.0)
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
        candidate_identity_delta = np.where(
            np.isfinite(candidate_identity_delta),
            candidate_identity_delta,
            0.0,
        ).astype(np.float32, copy=False)
        candidate_identity_deltas = (
            candidate_identity_delta[..., None]
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_candidate_factor, -1)
        )
        candidate_behaviors = (
            candidate_behavior
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_candidate_factor, 4)
        )
        capture_to_win_quality_gate = np.where(
            np.isfinite(capture_to_win_quality_gate),
            capture_to_win_quality_gate,
            0.0,
        ).astype(np.float32, copy=False)
        capture_to_win_quality_gate[~valid_factor[..., None]] = 0.0
        # The gate is a signed, centered outcome advantage.  Clipping it to
        # [0, 1] would silently erase all failed-capture evidence in replay.
        capture_to_win_quality_gate = np.clip(
            capture_to_win_quality_gate,
            -1.0,
            1.0,
        )
        capture_to_win_quality_gates = (
            capture_to_win_quality_gate
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        pair_pursuit_credit = np.where(
            np.isfinite(pair_pursuit_credit),
            pair_pursuit_credit,
            0.0,
        ).astype(np.float32, copy=False)
        pair_pursuit_credit[~valid_factor[..., None]] = 0.0
        pair_pursuit_credit = np.clip(pair_pursuit_credit, -5.0, 5.0)
        pair_pursuit_credits = (
            pair_pursuit_credit
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        pair_pursuit_quality = np.where(
            np.isfinite(pair_pursuit_quality),
            pair_pursuit_quality,
            0.0,
        ).astype(np.float32, copy=False)
        pair_pursuit_quality[~valid_factor[..., None]] = 0.0
        pair_pursuit_quality = np.clip(pair_pursuit_quality, 0.0, 1.0)
        pair_pursuit_qualities = (
            pair_pursuit_quality
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        pair_to_triplet_transition_score = np.where(
            np.isfinite(pair_to_triplet_transition_score),
            pair_to_triplet_transition_score,
            0.0,
        ).astype(np.float32, copy=False)
        pair_to_triplet_transition_score[~valid_factor[..., None]] = 0.0
        pair_to_triplet_transition_score = np.clip(
            pair_to_triplet_transition_score,
            0.0,
            1.0,
        )
        pair_to_triplet_transition_scores = (
            pair_to_triplet_transition_score
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        triplet_capture_quality = np.where(
            np.isfinite(triplet_capture_quality),
            triplet_capture_quality,
            0.0,
        ).astype(np.float32, copy=False)
        triplet_capture_quality[~valid_factor[..., None]] = 0.0
        triplet_capture_quality = np.maximum(
            triplet_capture_quality,
            0.0,
        )
        triplet_capture_qualities = (
            triplet_capture_quality
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )
        pair_transition_delay = np.where(
            np.isfinite(pair_transition_delay),
            np.maximum(pair_transition_delay, 0.0),
            0.0,
        ).astype(np.float32, copy=False)
        pair_transition_delay[~valid_factor[..., None]] = 0.0
        pair_transition_delays = (
            pair_transition_delay
            .transpose(1, 0, 2, 3)
            .reshape(batch_size, self.num_factor, -1)
        )

        def _flatten_graph_diagnostic(values):
            values = np.where(
                np.isfinite(values),
                np.maximum(values, 0.0),
                0.0,
            ).astype(np.float32, copy=False)
            return values.transpose(1, 0, 2).reshape(batch_size, -1)

        capture_counts_flat = _flatten_graph_diagnostic(capture_counts)
        capture_matched_count_flat = _flatten_graph_diagnostic(
            capture_matched_count
        )
        capture_to_win_episode_success_gate_flat = (
            _flatten_graph_diagnostic(
                capture_to_win_episode_success_gate
            )
        )
        failed_episode_capture_count_flat = _flatten_graph_diagnostic(
            failed_episode_capture_count
        )
        capture_outcome_diagnostics_flat = np.where(
            np.isfinite(capture_outcome_diagnostics),
            capture_outcome_diagnostics,
            0.0,
        ).astype(np.float32, copy=False)
        capture_outcome_diagnostics_flat = (
            capture_outcome_diagnostics_flat
            .transpose(1, 0, 2)
            .reshape(batch_size, CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH)
        )
        positive_reward_step_flat = _flatten_graph_diagnostic(
            positive_reward_step
        )
        positive_reward_without_capture_flat = _flatten_graph_diagnostic(
            positive_reward_without_capture
        )
        offset0_candidate_count_flat = _flatten_graph_diagnostic(
            offset0_candidate_count
        )

        # ``dones`` and ``dones_env`` were already flattened episode-major
        # above. Re-transposing them here used to be correct when they were
        # still sequence tensors, but now corrupts the axis contract (and for
        # ``dones`` raises "axes don't match array"). Fail loudly if a future
        # edit changes either layout.
        if dones.shape != (batch_size, self.num_agents, 1):
            raise RuntimeError(
                "flattened agent activity mask must have shape {}, got {}"
                .format((batch_size, self.num_agents, 1), dones.shape)
            )
        if dones_env.shape != (batch_size, 1):
            raise RuntimeError(
                "flattened previous-done mask must have shape {}, got {}"
                .format((batch_size, 1), dones_env.shape)
            )

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
            pair_pursuit_credit_batch = []
            pair_pursuit_quality_batch = []
            pair_to_triplet_transition_score_batch = []
            triplet_capture_quality_batch = []
            rnn_obs_batch = []
            pair_transition_delay_batch = []
            capture_counts_batch = []
            capture_matched_count_batch = []
            positive_reward_step_batch = []
            positive_reward_without_capture_batch = []
            offset0_candidate_count_batch = []
            previous_adj_batch = []
            capture_to_win_episode_success_gate_batch = []
            failed_episode_capture_count_batch = []
            capture_outcome_diagnostics_batch = []
            candidate_identity_delta_batch = []
            candidate_behavior_batch = []

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
                pair_pursuit_credit_batch.append(
                    pair_pursuit_credits[ind:ind + data_chunk_length]
                )
                pair_pursuit_quality_batch.append(
                    pair_pursuit_qualities[ind:ind + data_chunk_length]
                )
                pair_to_triplet_transition_score_batch.append(
                    pair_to_triplet_transition_scores[
                        ind:ind + data_chunk_length
                    ]
                )
                triplet_capture_quality_batch.append(
                    triplet_capture_qualities[ind:ind + data_chunk_length]
                )
                rnn_obs_batch.append(rnn_obs[ind:ind + data_chunk_length])
                pair_transition_delay_batch.append(
                    pair_transition_delays[ind:ind + data_chunk_length]
                )
                capture_counts_batch.append(
                    capture_counts_flat[ind:ind + data_chunk_length]
                )
                capture_matched_count_batch.append(
                    capture_matched_count_flat[ind:ind + data_chunk_length]
                )
                positive_reward_step_batch.append(
                    positive_reward_step_flat[ind:ind + data_chunk_length]
                )
                positive_reward_without_capture_batch.append(
                    positive_reward_without_capture_flat[
                        ind:ind + data_chunk_length
                    ]
                )
                offset0_candidate_count_batch.append(
                    offset0_candidate_count_flat[ind:ind + data_chunk_length]
                )
                previous_adj_batch.append(
                    previous_adj[ind:ind + data_chunk_length]
                )
                capture_to_win_episode_success_gate_batch.append(
                    capture_to_win_episode_success_gate_flat[
                        ind:ind + data_chunk_length
                    ]
                )
                failed_episode_capture_count_batch.append(
                    failed_episode_capture_count_flat[
                        ind:ind + data_chunk_length
                    ]
                )
                capture_outcome_diagnostics_batch.append(
                    capture_outcome_diagnostics_flat[
                        ind:ind + data_chunk_length
                    ]
                )
                candidate_identity_delta_batch.append(
                    candidate_identity_deltas[
                        ind:ind + data_chunk_length
                    ]
                )
                candidate_behavior_batch.append(
                    candidate_behaviors[ind:ind + data_chunk_length]
                )

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
                np.stack(pair_pursuit_credit_batch, axis=0),
                np.stack(pair_pursuit_quality_batch, axis=0),
                np.stack(pair_to_triplet_transition_score_batch, axis=0),
                np.stack(triplet_capture_quality_batch, axis=0),
                np.stack(rnn_obs_batch, axis=0),
                np.stack(pair_transition_delay_batch, axis=0),
                np.stack(capture_counts_batch, axis=0),
                np.stack(capture_matched_count_batch, axis=0),
                np.stack(positive_reward_without_capture_batch, axis=0),
                np.stack(offset0_candidate_count_batch, axis=0),
                np.stack(positive_reward_step_batch, axis=0),
                np.stack(previous_adj_batch, axis=0),
                np.stack(capture_to_win_episode_success_gate_batch, axis=0),
                np.stack(failed_episode_capture_count_batch, axis=0),
                np.stack(capture_outcome_diagnostics_batch, axis=0),
                np.stack(candidate_identity_delta_batch, axis=0),
                np.stack(candidate_behavior_batch, axis=0),
            )
