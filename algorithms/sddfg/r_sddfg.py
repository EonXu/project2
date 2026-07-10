import torch
import copy
from utils.util import soft_update, huber_loss, mse_loss, to_torch
import numpy as np
from utils.popart import PopArt

def _to_float(x):
    """兼容 torch 标量 / python float / numpy float"""
    if x is None:
        return 0.0
    if torch.is_tensor(x):
        return x.detach().cpu().item()
    try:
        return float(x)
    except Exception:
        return 0.0

def _nan_to_num_compat(x, nan=0.0, posinf=None, neginf=None):
    """
    兼容旧版本 PyTorch 的 nan_to_num。
    旧版 torch 没有 torch.nan_to_num，因此用 torch.where 手动替换 NaN/Inf。
    """
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

    out = x

    # 替换 NaN
    out = torch.where(
        torch.isnan(out),
        torch.full_like(out, float(nan)),
        out
    )

    # 替换 +Inf
    if posinf is not None:
        out = torch.where(
            torch.isinf(out) & (out > 0),
            torch.full_like(out, float(posinf)),
            out
        )

    # 替换 -Inf
    if neginf is not None:
        out = torch.where(
            torch.isinf(out) & (out < 0),
            torch.full_like(out, float(neginf)),
            out
        )

    return out


def _parameter_grad_norm(parameters):
    """Return the finite L2 norm of an explicit parameter collection."""
    total_sq = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = float(parameter.grad.detach().norm(2).cpu().item())
        total_sq += grad_norm * grad_norm
    return float(total_sq ** 0.5)


def _infer_inactive_dones_from_obs(obs_seq: torch.Tensor) -> torch.Tensor:
    """
    Infer fixed-capacity inactive slots from Wolfpack padded observations.

    In intra-episode dynamic Wolfpack, inactive / empty capacity slots are
    represented by an all -1 observation vector.  The replay buffer stores
    agent dones for the *next* state only, so the initial state s0 must be
    reconstructed from observations; otherwise the learner treats empty
    capacity slots as live agents at the beginning of every episode.

    Args:
        obs_seq: [B, T, N, obs_dim] or [B, N, obs_dim].
    Returns:
        Float mask with shape obs_seq[..., :1], where 1 means inactive/done.
    """
    obs_seq = to_torch(obs_seq)
    finite_obs = torch.where(torch.isfinite(obs_seq), obs_seq, torch.zeros_like(obs_seq))
    return torch.all(finite_obs <= -0.999, dim=-1, keepdim=True).float()


class R_SDDFG:
    def __init__(self, args, num_agents, policies, adj_network, policy_mapping_fn, device=torch.device("cuda:0"),
                 episode_length=25, vdn=False):
        """
        Trainer class for QMix with MLP policies. See parent class for more information.
        :param vdn: (bool) whether the algorithm in use is VDN.
        """
        self.args = args
        self.use_popart = self.args.use_popart
        self.use_value_active_masks = self.args.use_value_active_masks
        self.use_per = self.args.use_per
        self.per_eps = self.args.per_eps
        self.use_huber_loss = self.args.use_huber_loss
        self.huber_delta = self.args.huber_delta
        self.clip_param = self.args.clip_param
        self.adj_order_adv_coef = max(
            0.0,
            float(getattr(self.args, "adj_order_adv_coef", 0.0)),
        )
        self.adj_order_adv_positive_only = bool(
            getattr(self.args, "adj_order_adv_positive_only", False)
        )
        self.adj_order_adv_negative_coef = max(
            0.0,
            float(getattr(self.args, "adj_order_adv_negative_coef", 0.0)),
        )
        self.adj_order_adv_require_positive_graph_adv = bool(
            getattr(
                self.args,
                "adj_order_adv_require_positive_graph_adv",
                False,
            )
        )
        graph_gate_mode = str(
            getattr(self.args, "adj_order_adv_graph_gate_mode", "binary")
        ).lower()
        if graph_gate_mode not in ("binary", "soft"):
            graph_gate_mode = "binary"
        self.adj_order_adv_graph_gate_mode = graph_gate_mode
        self.adj_order_adv_graph_gate_scale = max(
            1e-6,
            float(getattr(self.args, "adj_order_adv_graph_gate_scale", 1.0)),
        )
        self.use_adj_triplet_graph_return_credit = bool(
            getattr(self.args, "use_adj_triplet_graph_return_credit", False)
        )
        self.adj_triplet_graph_return_credit_coef = max(
            0.0,
            float(
                getattr(
                    self.args,
                    "adj_triplet_graph_return_credit_coef",
                    0.0,
                )
            ),
        )
        self.adj_triplet_graph_return_credit_cap = max(
            0.0,
            float(
                getattr(
                    self.args,
                    "adj_triplet_graph_return_credit_cap",
                    0.0,
                )
            ),
        )
        self.adj_triplet_graph_return_credit_min_graph_adv = float(
            getattr(
                self.args,
                "adj_triplet_graph_return_credit_min_graph_adv",
                0.0,
            )
        )
        self.adj_triplet_graph_return_credit_raw_gate_scale = max(
            0.0,
            float(
                getattr(
                    self.args,
                    "adj_triplet_graph_return_credit_raw_gate_scale",
                    0.0,
                )
            ),
        )
        self.adj_triplet_graph_return_credit_require_delayed_gate = bool(
            getattr(
                self.args,
                "adj_triplet_graph_return_credit_require_delayed_gate",
                False,
            )
        )
        self.use_adj_delayed_triplet_credit = bool(
            getattr(self.args, "use_adj_delayed_triplet_credit", False)
        )
        self.adj_delayed_triplet_credit_coef = max(
            0.0,
            float(getattr(self.args, "adj_delayed_triplet_credit_coef", 0.0)),
        )
        self.adj_delayed_triplet_credit_window = max(
            0,
            int(getattr(self.args, "adj_delayed_triplet_credit_window", 0)),
        )
        self.adj_delayed_triplet_credit_cap = max(
            0.0,
            float(getattr(self.args, "adj_delayed_triplet_credit_cap", 0.0)),
        )
        self.adj_delayed_triplet_credit_min_reward = float(
            getattr(self.args, "adj_delayed_triplet_credit_min_reward", 0.0)
        )
        self.adj_delayed_triplet_credit_positive_only = bool(
            getattr(
                self.args,
                "adj_delayed_triplet_credit_positive_only",
                False,
            )
        )
        self.adj_delayed_triplet_credit_min_adv = max(
            0.0,
            float(getattr(self.args, "adj_delayed_triplet_credit_min_adv", 0.0)),
        )
        self.adj_delayed_triplet_credit_require_future_match = bool(
            getattr(
                self.args,
                "adj_delayed_triplet_credit_require_future_match",
                False,
            )
        )
        self.use_adj_delayed_triplet_success_gate = bool(
            getattr(self.args, "use_adj_delayed_triplet_success_gate", False)
        )
        self.adj_delayed_triplet_success_gate_min_adv = float(
            getattr(
                self.args,
                "adj_delayed_triplet_success_gate_min_adv",
                0.0,
            )
        )
        self.adj_delayed_triplet_success_gate_scale = max(
            1e-6,
            float(
                getattr(
                    self.args,
                    "adj_delayed_triplet_success_gate_scale",
                    1.0,
                )
            ),
        )
        self.adj_delayed_triplet_success_gate_floor = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        self.args,
                        "adj_delayed_triplet_success_gate_floor",
                        0.0,
                    )
                ),
            ),
        )
        self.adj_delayed_triplet_future_overlap_min_nodes = min(
            3,
            max(
                1,
                int(
                    getattr(
                        self.args,
                        "adj_delayed_triplet_future_overlap_min_nodes",
                        3,
                    )
                ),
            ),
        )
        self.adj_delayed_triplet_partial_match_weight = min(
            1.0,
            max(
                0.0,
                float(
                    getattr(
                        self.args,
                        "adj_delayed_triplet_partial_match_weight",
                        0.5,
                    )
                ),
            ),
        )
        self.use_adj_capture_to_win_credit = bool(
            getattr(self.args, "use_adj_capture_to_win_credit", False)
        )
        self.adj_capture_to_win_credit_coef = max(
            0.0,
            float(getattr(self.args, "adj_capture_to_win_credit_coef", 0.0)),
        )
        self.adj_capture_to_win_credit_min_outcome_adv = float(
            getattr(
                self.args,
                "adj_capture_to_win_credit_min_outcome_adv",
                0.5,
            )
        )
        self.adj_capture_to_win_credit_scale = max(
            1e-6,
            float(getattr(self.args, "adj_capture_to_win_credit_scale", 0.75)),
        )
        self.adj_capture_to_win_credit_cap = max(
            0.0,
            float(getattr(self.args, "adj_capture_to_win_credit_cap", 0.35)),
        )
        self.adj_capture_to_win_credit_require_future_match = bool(
            getattr(
                self.args,
                "adj_capture_to_win_credit_require_future_match",
                False,
            )
        )
        self.use_adj_ppo_stale_trust = bool(
            getattr(self.args, "use_adj_ppo_stale_trust", False)
        )
        self.adj_ppo_stale_trust_clip = max(
            0.0,
            float(
                getattr(
                    self.args,
                    "adj_ppo_stale_trust_clip",
                    self.clip_param,
                )
            ),
        )
        self.adj_ppo_stale_trust_scale = max(
            1e-6,
            float(getattr(self.args, "adj_ppo_stale_trust_scale", 0.25)),
        )
        self.adj_ppo_stale_trust_min_weight = max(
            0.0,
            min(
                1.0,
                float(
                    getattr(
                        self.args,
                        "adj_ppo_stale_trust_min_weight",
                        0.25,
                    )
                ),
            ),
        )
        self.use_vfunction = self.args.use_vfunction
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.lr = self.args.lr
        self.critic_lr = self.args.critic_lr
        self.adj_lr = self.args.adj_lr
        self.tau = self.args.tau
        self.opti_eps = self.args.opti_eps
        self.weight_decay = self.args.weight_decay
        self.episode_length = episode_length
        self.num_agents = num_agents
        self.highest_orders = self.args.highest_orders
        self.use_dyn_graph = self.args.use_dyn_graph
        self.num_factor = self.args.num_factor
        # ``entropy_coef`` was historically reused as the SDDFG graph entropy
        # weight.  In the intra-episode dynamic Wolfpack runs, run21 showed
        # that the legacy value (1e-3) keeps the graph distribution close to
        # uniform at 200k steps.  Keep the old fallback, but allow the graph
        # policy to use its own smaller coefficient.
        configured_adj_entropy_coef = float(
            getattr(self.args, "adj_entropy_coef", -1.0)
        )
        self.entropy_coef = (
            float(self.args.entropy_coef)
            if configured_adj_entropy_coef < 0.0
            else configured_adj_entropy_coef
        )
        self.adj_entropy_coef = float(self.entropy_coef)
        configured_entropy_final = float(
            getattr(self.args, "adj_entropy_coef_final", -1.0)
        )
        self.adj_entropy_coef_final = (
            self.adj_entropy_coef
            if configured_entropy_final < 0.0
            else configured_entropy_final
        )
        self.adj_entropy_anneal_steps = max(
            0,
            int(getattr(self.args, "adj_entropy_anneal_steps", 0)),
        )
        self._use_valuenorm = self.args.use_valuenorm
        self.adj_max_grad_norm = self.args.adj_max_grad_norm
        self.policies = policies
        self.policy_mapping_fn = policy_mapping_fn
        self.policy_ids = sorted(list(self.policies.keys()))
        self.policy_agents = {policy_id: sorted(
            [agent_id for agent_id in range(self.num_agents) if self.policy_mapping_fn(agent_id) == policy_id]) for
            policy_id in
            self.policies.keys()}
        if self._use_valuenorm:
            self.value_normalizer = {policy_id: PopArt(1, self.device).to(self.device) for policy_id in
                                     self.policies.keys()}

        multidiscrete_list = None
        if any([isinstance(policy.act_dim, np.ndarray) for policy in self.policies.values()]):
            # multidiscrete
            multidiscrete_list = [len(self.policies[p_id].act_dim) *
                                  len(self.policy_agents[p_id]) for p_id in self.policy_ids]

        # target policies/networks
        self.adj_network = adj_network
        self.target_policies = {p_id: copy.deepcopy(self.policies[p_id]) for p_id in self.policy_ids}

        self.policy_parameters = []
        for policy in self.policies.values():
            self.policy_parameters += policy.parameters()
        self.policy_optimizer = torch.optim.Adam(params=self.policy_parameters, lr=self.lr, eps=self.opti_eps)

        self.critic_fv_parameters = []
        self.critic_vtot_parameters = []
        self.critic_fv_optimizer = None
        self.critic_vtot_optimizer = None
        if self.use_vfunction:
            for policy in self.policies.values():
                self.critic_fv_parameters += policy.critic_fv_parameters()
                self.critic_vtot_parameters += policy.critic_vtot_parameters()
            if not self.critic_fv_parameters or not self.critic_vtot_parameters:
                raise RuntimeError(
                    "use_vfunction=True but SDDFG critic parameters are empty"
                )
            self.critic_fv_optimizer = torch.optim.Adam(
                params=self.critic_fv_parameters,
                lr=self.critic_lr,
                eps=self.opti_eps,
            )
            self.critic_vtot_optimizer = torch.optim.Adam(
                params=self.critic_vtot_parameters,
                lr=self.critic_lr,
                eps=self.opti_eps,
            )

        self.adj_parameters = []
        self.adj_parameters += self.adj_network.parameters()
        self.adj_optimizer = torch.optim.Adam(params=self.adj_parameters, lr=self.adj_lr, eps=self.opti_eps)

    def _set_optimizer_lr_with_floor(self, optimizer, init_lr, episode, episodes, floor_lr):
        """
        线性衰减，但不低于 floor_lr。
        """
        if episodes <= 0:
            lr = init_lr
        else:
            frac = 1.0 - float(episode) / float(episodes)
            frac = max(0.0, min(1.0, frac))
            lr = max(float(init_lr) * frac, float(floor_lr))

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        return lr

    def lr_decay(self, episode, episodes):
        """
        Decay policy/critic lr, optionally decay adj/GAT lr.

        关键变化：
        1. policy/critic 仍然随 --use_linear_lr_decay 衰减；
        2. adj/GAT 默认不衰减，避免后期 adj_lr 过低；
        3. 如果显式传 --use_adj_linear_lr_decay，则 adj/GAT 衰减但不低于 adj_lr_decay_floor。
        """
        policy_floor = float(getattr(self.args, "policy_lr_decay_floor", 1e-5))
        critic_floor = float(getattr(self.args, "critic_lr_decay_floor", 1e-5))
        use_adj_decay = bool(getattr(self.args, "use_adj_linear_lr_decay", False))

        # policy / critic 继续衰减，但设置 floor，避免后期完全学不动
        self._set_optimizer_lr_with_floor(
            self.policy_optimizer,
            self.lr,
            episode,
            episodes,
            policy_floor
        )
        if self.use_vfunction:
            self._set_optimizer_lr_with_floor(
                self.critic_fv_optimizer,
                self.critic_lr,
                episode,
                episodes,
                critic_floor
            )
            self._set_optimizer_lr_with_floor(
                self.critic_vtot_optimizer,
                self.critic_lr,
                episode,
                episodes,
                critic_floor
            )

        # adj/GAT 默认不跟随全局 linear decay
        if use_adj_decay:
            self.adj_lr_decay(episode)
        else:
            for param_group in self.adj_optimizer.param_groups:
                param_group["lr"] = float(self.adj_lr)

    def adj_lr_decay(self, env_step):
        """Decay only graph parameters on a fixed environment-step clock.

        Dynamic topology inference remains active after annealing; only the
        cumulative drift of the GAT/hyperedge parameters is reduced. The
        caller supplies an explicit environment-step horizon so experiments
        can either preserve a common prefix or anneal across the full run.
        """
        anneal_steps = max(
            1,
            int(getattr(self.args, "adj_lr_anneal_steps", 500000)),
        )
        adj_floor = float(getattr(self.args, "adj_lr_decay_floor", 2e-5))
        current_lr = self._set_optimizer_lr_with_floor(
            self.adj_optimizer,
            self.adj_lr,
            env_step,
            anneal_steps,
            adj_floor,
        )
        if self.adj_entropy_anneal_steps > 0:
            progress = min(
                max(
                    float(env_step)
                    / float(self.adj_entropy_anneal_steps),
                    0.0,
                ),
                1.0,
            )
            self.adj_entropy_coef = (
                self.entropy_coef
                + progress
                * (self.adj_entropy_coef_final - self.entropy_coef)
            )
        else:
            self.adj_entropy_coef = float(self.entropy_coef)
        return current_lr

    def train_policy_on_batch(self, batch, use_same_share_obs=None):
        """See parent class."""

        obs_batch, cent_obs_batch, \
            act_batch, rew_batch, \
            dones_batch, dones_env_batch, \
            avail_act_batch, adj, \
            prob_adj, idxes = batch
        dones_env_batch = to_torch(dones_env_batch[self.policy_ids[0]]).transpose(0, 1).to(**self.tpdv)

        # individual agent q values: each element is of shape (batch_size, 1)
        qs = []
        target_qs = []
        vtot = []
        target_vtot = []
        fv = []

        for p_id in self.policy_ids:

            policy = self.policies[p_id]
            target_policy = self.target_policies[p_id]
            # get data related to the policy id
            stored_next_dones = to_torch(dones_batch[p_id]).permute(1, 2, 0, 3).to(self.device)
            rewards = to_torch(rew_batch[p_id][0]).transpose(0, 1).to(**self.tpdv)  # [25,32,1]
            curr_obs_batch = to_torch(obs_batch[p_id]).transpose(0, 2)  # [3,26,32,18]
            curr_act_batch = to_torch(act_batch[p_id]).transpose(0, 2).to(**self.tpdv)  # [3,25,32,5]
            state_obs_batch = to_torch(cent_obs_batch[p_id]).to(**self.tpdv)
            adj = to_torch(adj[p_id])

            if avail_act_batch[p_id] is not None:
                curr_avail_act_batch = to_torch(avail_act_batch[p_id]).transpose(0, 2).to(self.device)
            else:
                curr_avail_act_batch = None

            act_dim = curr_act_batch.shape[3]
            step = rewards.shape[1] + 1
            batch_size = curr_obs_batch.shape[0]

            # Intra-episode dynamic fix:
            # replay stores agent dones as next-state masks.  For the current
            # state sequence [s0...sT], prepend the true s0 inactive mask
            # inferred from padded observations instead of all-zero dones.
            # OR with observation-derived masks for all states as a guard
            # against any stale buffer transition around join/recover events.
            obs_inactive_dones = _infer_inactive_dones_from_obs(curr_obs_batch).permute(1, 0, 2, 3).to(**self.tpdv)
            if obs_inactive_dones.shape[0] != stored_next_dones.shape[0] + 1:
                raise RuntimeError(
                    "SDDFG replay mask length mismatch: "
                    f"obs states={obs_inactive_dones.shape[0]}, "
                    f"stored next dones={stored_next_dones.shape[0]}"
                )
            dones = torch.cat((obs_inactive_dones[:1], stored_next_dones), dim=0).float()
            obs_inactive_dones = obs_inactive_dones.float()
            # Old PyTorch on the server does not provide torch.maximum.
            # This is elementwise max(a, b): keep a slot done if either replay
            # next-done or observation-derived inactive mask marks it inactive.
            dones = torch.where(dones >= obs_inactive_dones, dones, obs_inactive_dones)

            dones_batch = torch.cat((torch.zeros(batch_size, 1, 1).to(**self.tpdv), dones_env_batch), dim=1)
            bad_transitions_mask = dones_batch[:, :-1]
            active_transition_mask = (
                (1.0 - dones[:-1].float()).sum(dim=2).gt(0.0).float().transpose(0, 1)
            )
            valid_transition_mask = (1.0 - bad_transitions_mask) * active_transition_mask

            stacked_act_batch = torch.cat(list(curr_act_batch), dim=-2)  # [25,256,5]
            stacked_obs_batch = torch.cat(list(curr_obs_batch), dim=-2)  # [26,256,48]
            pol_prev_act_buffer_seq = torch.cat((torch.zeros(1, batch_size * self.num_agents, act_dim).to(**self.tpdv),
                                                 stacked_act_batch))  # [26,256,5]
            stacked_act_batch_ind = stacked_act_batch.max(dim=-1)[1]  # [25,256]

            alive_mask = (1.0 - dones.float()).clamp(0.0, 1.0)

            eye = torch.eye(
                self.num_agents,
                dtype=torch.int64,
                device=self.device
            ).view(1, 1, self.num_agents, self.num_agents).repeat(step, batch_size, 1, 1)
            eye = eye * alive_mask.long()

            if self.use_dyn_graph:
                adj_dyn = adj.to(self.device).long()

                factor_size = adj_dyn.float().sum(dim=2, keepdim=True)
                alive_count = (adj_dyn.float() * alive_mask).sum(dim=2, keepdim=True)
                factor_valid = ((factor_size > 0.0) & (alive_count == factor_size)).long()

                adj_dyn = adj_dyn * alive_mask.long() * factor_valid
                adj_input = torch.cat([adj_dyn, eye], dim=3).to(self.device)
            else:
                adj_input = eye.to(self.device)

            rnn_states_1 = policy.init_hidden(self.num_agents, batch_size)
            target_rnn_states = target_policy.init_hidden(self.num_agents, batch_size)

            dones_seq = dones.reshape(step, batch_size * self.num_agents, 1)

            rnn_obs_batch_1, _, no_sequence = policy.get_hidden_states(
                stacked_obs_batch, pol_prev_act_buffer_seq, rnn_states_1, dones=dones_seq
            )
            target_rnn_obs_batch, _, _ = target_policy.get_hidden_states(
                stacked_obs_batch, pol_prev_act_buffer_seq, target_rnn_states, dones=dones_seq
            )

            curr_act_batch_ind = stacked_act_batch_ind.reshape((step - 1) * batch_size, self.num_agents, -1)
            rnn_obs_q = rnn_obs_batch_1[:-1].reshape((step - 1) * batch_size, self.num_agents, -1)
            obs_q = to_torch(stacked_obs_batch[:-1]).reshape((step - 1) * batch_size, self.num_agents, -1)
            adj_input_q = adj_input[:-1].reshape((step - 1) * batch_size, self.num_agents, -1)
            dones_q = dones[:-1].reshape((step - 1) * batch_size, self.num_agents, -1)
            policy_qs = policy.get_q_values(obs_q, rnn_obs_q, curr_act_batch_ind, adj_input_q, no_sequence, dones_q)

            rnn_obs_qtot = rnn_obs_batch_1[1:].reshape((step - 1) * batch_size, self.num_agents, -1)
            obs_qtot = to_torch(stacked_obs_batch[1:]).reshape((step - 1) * batch_size, self.num_agents, -1)

            adj_input_qtot = adj_input[1:].reshape((step - 1) * batch_size, self.num_agents, -1)
            dones_qtot = dones[1:].reshape((step - 1) * batch_size, self.num_agents, -1)
            rnn_target_obs = target_rnn_obs_batch[1:].reshape((step - 1) * batch_size, self.num_agents, -1)
            curr_avail_act = curr_avail_act_batch.transpose(0, 1)[1:].reshape((step - 1) * batch_size, self.num_agents,
                                                                              -1)
            with torch.no_grad():
                greedy, _, _, _ = policy.get_actions(obs_qtot, rnn_obs_qtot, curr_avail_act, None, False,
                                                     adj_input_qtot, no_sequence, dones_qtot)
                curr_nact_batch_ind = torch.from_numpy(greedy).max(dim=-1)[1].to(self.device)
                target_policy_qs = target_policy.get_q_values(obs_qtot, rnn_target_obs,
                                                              curr_nact_batch_ind.unsqueeze(dim=-1), adj_input_qtot,
                                                              no_sequence, dones_qtot)

            qs.append(policy_qs.reshape(step - 1, batch_size).transpose(0, 1))
            target_qs.append(target_policy_qs.reshape(step - 1, batch_size).transpose(0, 1))
            if self.use_vfunction:
                rnn_critic = policy.init_hidden(1, batch_size)
                policy_v_tot = policy.get_vtot(state_obs_batch, rnn_critic).reshape(step, batch_size)
                target_rnn_critic = target_policy.init_hidden(1, batch_size)
                with torch.no_grad():
                    target_policy_v_tot = target_policy.get_vtot(
                        state_obs_batch,
                        target_rnn_critic
                    ).reshape(step, batch_size)
                policy_v = policy.get_v_values(rnn_obs_q.detach(),
                                               state_obs_batch[:-1].reshape((step - 1) * batch_size, 1, -1),
                                               adj_input_q, no_sequence, dones_q)
                '''with torch.no_grad():
                    target_policy_v =target_policy.get_v_values(rnn_target_obs,state_obs_batch[1:].reshape((step-1)*batch_size,1,-1), adj_input_qtot,no_sequence,dones_qtot)'''
                vtot.append(policy_v_tot[:-1].transpose(0, 1))
                target_vtot.append(target_policy_v_tot[1:].transpose(0, 1))
                fv.append(policy_v.reshape(step - 1, batch_size).transpose(0, 1))
                # target_fv.append(target_policy_v.reshape(step-1,batch_size).transpose(0,1))

        # combine the agent q value sequences to feed into mixer networks
        curr_Q_tot = torch.cat(qs, dim=-1).unsqueeze(-1)
        next_step_Q_tot = torch.cat(target_qs, dim=-1).unsqueeze(-1)

        # all agents must share reward, so get the reward sequence for an agent
        if self._use_valuenorm:
            Q_tot_targets = rewards + (1 - dones_env_batch) * self.args.gamma * \
                            self.value_normalizer[p_id].denormalize(next_step_Q_tot)
            Q_tot_targets = self.value_normalizer[p_id](Q_tot_targets)
        else:
            Q_tot_targets = rewards + (1 - dones_env_batch) * self.args.gamma * next_step_Q_tot
        error = (curr_Q_tot - Q_tot_targets.detach()) * valid_transition_mask
        valid_denom = valid_transition_mask.sum().clamp_min(1.0)

        if self.use_per:
            if self.use_huber_loss:
                loss = huber_loss(error, self.huber_delta).flatten()
            else:
                loss = mse_loss(error).flatten()
            loss = (loss * to_torch(importance_weights).to(**self.tpdv)).sum() / valid_denom
            new_priorities = error.abs().cpu().detach().numpy().flatten() + self.per_eps
        else:
            if self.use_huber_loss:
                loss = huber_loss(error, self.huber_delta).sum() / valid_denom
            else:
                loss = mse_loss(error).sum() / valid_denom
            new_priorities = None

        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite SDDFG policy loss")

        if self.use_vfunction:
            curr_v_tot = torch.cat(vtot, dim=-1).unsqueeze(-1)
            next_step_v_tot = torch.cat(target_vtot, dim=-1).unsqueeze(-1)
            v_tot_targets = rewards + (1 - dones_env_batch) * self.args.gamma * next_step_v_tot
            error_v = (curr_v_tot - v_tot_targets.detach()) * valid_transition_mask
            loss_v = mse_loss(error_v).sum() / valid_denom

            f_v_tot = torch.cat(fv, dim=-1).unsqueeze(-1)
            error_fv = (f_v_tot - curr_v_tot.detach()) * valid_transition_mask
            loss_fv = mse_loss(error_fv).sum() / valid_denom
            if not torch.isfinite(loss_v):
                raise FloatingPointError("non-finite SDDFG total-value loss")
            if not torch.isfinite(loss_fv):
                raise FloatingPointError("non-finite SDDFG factor-value loss")

        self.policy_optimizer.zero_grad()
        loss.backward()
        q_grad_parameters = []
        rnn_grad_parameters = []
        for policy in self.policies.values():
            rnn_grad_parameters += list(policy.rnn_network.parameters())
            for order in range(1, self.highest_orders + 1):
                q_grad_parameters += list(
                    policy.q_network[order].parameters()
                )
        q_network_grad_norm = _parameter_grad_norm(q_grad_parameters)
        rnn_network_grad_norm = _parameter_grad_norm(rnn_grad_parameters)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_parameters, self.args.max_grad_norm)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError("non-finite SDDFG policy gradient norm")
        self.policy_optimizer.step()
        if self.use_vfunction:
            self.critic_vtot_optimizer.zero_grad()
            loss_v.backward()
            vtot_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic_vtot_parameters, self.args.max_grad_norm
            )
            if not torch.isfinite(torch.as_tensor(vtot_grad_norm)):
                raise FloatingPointError("non-finite SDDFG total-value gradient norm")
            self.critic_vtot_optimizer.step()

            self.critic_fv_optimizer.zero_grad()
            loss_fv.backward()
            fv_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.critic_fv_parameters, self.args.max_grad_norm
            )
            if not torch.isfinite(torch.as_tensor(fv_grad_norm)):
                raise FloatingPointError("non-finite SDDFG factor-value gradient norm")
            self.critic_fv_optimizer.step()

        train_info = {}
        train_info['loss'] = _to_float(loss)
        train_info['loss_v'] = _to_float(loss_v) if self.use_vfunction else 0.0
        train_info['loss_fv'] = _to_float(loss_fv) if self.use_vfunction else 0.0
        train_info['policy_grad_norm'] = _to_float(grad_norm)
        train_info['q_network_grad_norm'] = q_network_grad_norm
        train_info['rnn_network_grad_norm'] = rnn_network_grad_norm
        train_info['vtot_grad_norm'] = _to_float(vtot_grad_norm) if self.use_vfunction else 0.0
        train_info['fv_grad_norm'] = _to_float(fv_grad_norm) if self.use_vfunction else 0.0
        train_info['valid_transition_ratio'] = _to_float(valid_transition_mask.mean())
        train_info['active_transition_ratio'] = _to_float(active_transition_mask.mean())
        train_info['q_tot_mean'] = _to_float(
            (curr_Q_tot.detach() * valid_transition_mask).sum()
            / valid_denom
        )
        train_info['q_target_mean'] = _to_float(
            (Q_tot_targets.detach() * valid_transition_mask).sum()
            / valid_denom
        )
        train_info['td_abs_mean'] = _to_float(
            error.detach().abs().sum() / valid_denom
        )
        train_info['policy_lr'] = float(
            self.policy_optimizer.param_groups[0]['lr']
        )
        train_info['policy_grad_was_clipped'] = float(
            _to_float(grad_norm) > float(self.args.max_grad_norm)
        )
        return train_info, new_priorities, idxes

    def train_adj_on_batch(self, batch, use_adj_init, use_same_share_obs=None):
        """See parent class."""

        if len(batch) >= 16:
            obs_batch, _share_obs_batch, dones_batch, \
                dones_env_batch, adj_batch, prob_adj_batch, \
                _advantages_batch, f_advts_batch, \
                delayed_triplet_credit_batch, \
                delayed_triplet_success_gate_batch, \
                delayed_triplet_future_match_batch, \
                delayed_triplet_future_exact_batch, \
                delayed_triplet_future_partial_batch, \
                capture_to_win_triplet_credit_batch, \
                capture_to_win_quality_gate_batch, \
                rnn_obs_batch = batch[:16]
        elif len(batch) >= 14:
            obs_batch, _share_obs_batch, dones_batch, \
                dones_env_batch, adj_batch, prob_adj_batch, \
                _advantages_batch, f_advts_batch, \
                delayed_triplet_credit_batch, \
                delayed_triplet_success_gate_batch, \
                delayed_triplet_future_match_batch, \
                delayed_triplet_future_exact_batch, \
                delayed_triplet_future_partial_batch, \
                rnn_obs_batch = batch[:14]
            capture_to_win_triplet_credit_batch = None
            capture_to_win_quality_gate_batch = None
        elif len(batch) >= 11:
            obs_batch, _share_obs_batch, dones_batch, \
                dones_env_batch, adj_batch, prob_adj_batch, \
                _advantages_batch, f_advts_batch, \
                delayed_triplet_credit_batch, \
                delayed_triplet_success_gate_batch, \
                rnn_obs_batch = batch[:11]
            delayed_triplet_future_match_batch = None
            delayed_triplet_future_exact_batch = None
            delayed_triplet_future_partial_batch = None
            capture_to_win_triplet_credit_batch = None
            capture_to_win_quality_gate_batch = None
        elif len(batch) >= 10:
            obs_batch, _share_obs_batch, dones_batch, \
                dones_env_batch, adj_batch, prob_adj_batch, \
                _advantages_batch, f_advts_batch, \
                delayed_triplet_credit_batch, rnn_obs_batch = batch[:10]
            delayed_triplet_success_gate_batch = None
            delayed_triplet_future_match_batch = None
            delayed_triplet_future_exact_batch = None
            delayed_triplet_future_partial_batch = None
            capture_to_win_triplet_credit_batch = None
            capture_to_win_quality_gate_batch = None
        else:
            obs_batch, _share_obs_batch, dones_batch, \
                dones_env_batch, adj_batch, prob_adj_batch, \
                _advantages_batch, f_advts_batch, rnn_obs_batch = batch
            delayed_triplet_credit_batch = None
            delayed_triplet_success_gate_batch = None
            delayed_triplet_future_match_batch = None
            delayed_triplet_future_exact_batch = None
            delayed_triplet_future_partial_batch = None
            capture_to_win_triplet_credit_batch = None
            capture_to_win_quality_gate_batch = None
        tarprob_adj = []
        adj_entropy = []
        adj = to_torch(adj_batch)  # [batch_size,step,num_agent,-1]
        batch_size = adj.shape[0] * adj.shape[1]
        adj = adj.reshape(batch_size, self.num_agents, -1).to(**self.tpdv)
        dones = to_torch(dones_batch).reshape(batch_size, self.num_agents, -1).to(self.device)
        dones_env = to_torch(dones_env_batch).reshape(batch_size, -1).to(**self.tpdv)
        prob_adj = to_torch(prob_adj_batch).reshape(batch_size, self.num_agents, -1).to(**self.tpdv)
        f_advts = to_torch(f_advts_batch).reshape(batch_size, self.num_factor, -1).to(**self.tpdv)
        if delayed_triplet_credit_batch is None:
            delayed_triplet_credit = torch.zeros_like(f_advts)
        else:
            delayed_triplet_credit = (
                to_torch(delayed_triplet_credit_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if delayed_triplet_success_gate_batch is None:
            delayed_triplet_success_gate = torch.zeros_like(f_advts)
        else:
            delayed_triplet_success_gate = (
                to_torch(delayed_triplet_success_gate_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if delayed_triplet_future_match_batch is None:
            delayed_triplet_future_match = torch.zeros_like(f_advts)
        else:
            delayed_triplet_future_match = (
                to_torch(delayed_triplet_future_match_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if delayed_triplet_future_exact_batch is None:
            delayed_triplet_future_exact = torch.zeros_like(f_advts)
        else:
            delayed_triplet_future_exact = (
                to_torch(delayed_triplet_future_exact_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if delayed_triplet_future_partial_batch is None:
            delayed_triplet_future_partial = torch.zeros_like(f_advts)
        else:
            delayed_triplet_future_partial = (
                to_torch(delayed_triplet_future_partial_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if capture_to_win_triplet_credit_batch is None:
            capture_to_win_triplet_credit = torch.zeros_like(f_advts)
        else:
            capture_to_win_triplet_credit = (
                to_torch(capture_to_win_triplet_credit_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if capture_to_win_quality_gate_batch is None:
            capture_to_win_quality_gate = torch.zeros_like(f_advts)
        else:
            capture_to_win_quality_gate = (
                to_torch(capture_to_win_quality_gate_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        rnn_obs = to_torch(rnn_obs_batch).reshape(batch_size, self.num_agents, -1).to(
            **self.tpdv)  # [batch_size,step-1,1]
        obs = to_torch(obs_batch).reshape(batch_size, self.num_agents, -1).to(**self.tpdv)

        for _ in self.policy_ids:
            target_prob_adj, entropy = self.adj_network.evaluate_prob(
                obs=obs,
                rnn_obs=rnn_obs,
                use_adj_init=use_adj_init,
                dones=dones.bool(),
                adj=adj
            )

            target_prob = torch.where(
                adj == 1,
                target_prob_adj,
                torch.ones_like(target_prob_adj, dtype=torch.float32)
            ).clamp_min(1e-8)

            adj_entropy.append(entropy)
            tarprob_adj.append(torch.log(target_prob))

        tarlog_prob_adj = torch.cat(tarprob_adj, dim=-1)
        adj_entropy_batch = torch.cat(adj_entropy, dim=-1).unsqueeze(-1)

        prob_adj_safe = torch.where(
            adj == 1,
            prob_adj,
            torch.ones_like(prob_adj, dtype=torch.float32)
        ).clamp_min(1e-8)
        log_prob_adj = torch.log(prob_adj_safe)

        active_masks = (1.0 - dones.float()).clamp(0.0, 1.0)  # [B, N, 1]

        adj_binary = (adj > 0.5).float()  # [B, N, F]
        factor_size = adj_binary.sum(dim=1, keepdim=True)  # [B, 1, F]
        alive_count = (adj_binary * active_masks).sum(dim=1, keepdim=True)

        # 只有 factor 中所有参与节点均存活，该 factor 才有效
        valid_factor_masks = ((factor_size > 0.0) & (alive_count == factor_size)).float()  # [B, 1, F]

        tarlog_prob_adj = tarlog_prob_adj * active_masks * valid_factor_masks
        log_prob_adj = log_prob_adj * active_masks * valid_factor_masks

        # 同步过滤 factor advantage
        f_advts = f_advts * valid_factor_masks.transpose(1, 2)

        # 后续 order 判定必须使用 valid_adj
        valid_adj = adj_binary * active_masks * valid_factor_masks

        # Each selected column is one pair/triplet factor action. The generator
        # stores the same categorical factor probability on every participating
        # node, so average node log-probability recovers the joint factor
        # log-probability. The previous order-specific square/cube logic counted
        # one factor multiple times and caused most adjacency updates to clip.
        factor_den = factor_size.squeeze(1).clamp_min(1.0)
        target_factor_logp = (
            (tarlog_prob_adj * valid_adj).sum(dim=1) / factor_den
        )
        behavior_factor_logp = (
            (log_prob_adj * valid_adj).sum(dim=1) / factor_den
        )
        factor_mask = valid_factor_masks.transpose(1, 2)
        factor_mask_2d = factor_mask.squeeze(-1)
        valid_factor_count = factor_mask.sum(
            dim=-2,
            keepdim=False,
        ).clamp_min(1.0)

        # The ordered factor sequence is one structured graph action. Its
        # probability is the product of the conditional factor probabilities,
        # so PPO must use the sum of their log-probabilities. Treating every
        # factor as an independent action gives the wrong objective and makes
        # roster sizes with more factor slots receive larger updates.
        target_graph_logp = (
            target_factor_logp * factor_mask_2d
        ).sum(dim=-1, keepdim=True)
        behavior_graph_logp = (
            behavior_factor_logp * factor_mask_2d
        ).sum(dim=-1, keepdim=True)
        graph_diff_log = (
            target_graph_logp - behavior_graph_logp
        ).clamp(min=-10.0, max=10.0)
        graph_imp_weights = torch.exp(graph_diff_log)
        graph_imp_weights = _nan_to_num_compat(
            graph_imp_weights,
            nan=1.0,
            posinf=1.0 + self.clip_param,
            neginf=1.0 - self.clip_param,
        )

        bad_transitions_mask = dones_env
        clipped_graph_imp_weights = torch.clamp(
            graph_imp_weights,
            1.0 - self.clip_param,
            1.0 + self.clip_param
        )

        factor_advantage = f_advts.squeeze(-1)
        graph_advantage = (
            (f_advts * factor_mask).sum(dim=-2)
            / valid_factor_count
        )
        surr1 = graph_imp_weights * graph_advantage
        surr2 = clipped_graph_imp_weights * graph_advantage
        per_transition_surr = torch.min(surr1, surr2)

        has_valid_factor = (
                factor_mask.sum(dim=-2) > 0
        ).float()

        transition_mask = (
                (1.0 - bad_transitions_mask)
                * has_valid_factor
        )
        graph_trust_weights = torch.ones_like(graph_imp_weights)
        if self.use_adj_ppo_stale_trust:
            # run37 showed that high-clamp early stop reliably stops after one
            # epoch, but the first epoch is already far outside the PPO trust
            # region.  That means the buffer is stale before any replay update
            # happens.  Down-weight those out-of-distribution graph decisions
            # instead of letting clipped stale samples dominate the gradient.
            graph_ratio_deviation = (
                graph_imp_weights.detach() - 1.0
            ).abs()
            graph_stale_excess = (
                graph_ratio_deviation
                - float(self.adj_ppo_stale_trust_clip)
            ).clamp_min(0.0)
            graph_trust_weights = torch.exp(
                -graph_stale_excess
                / float(self.adj_ppo_stale_trust_scale)
            )
            graph_trust_weights = torch.clamp(
                graph_trust_weights,
                min=float(self.adj_ppo_stale_trust_min_weight),
                max=1.0,
            ).detach()
        graph_loss_mask = transition_mask * graph_trust_weights

        graph_rl_loss = -(
                per_transition_surr * graph_loss_mask
        ).sum() / graph_loss_mask.sum().clamp_min(1.0)

        # ``AdjBuffer`` centers the auxiliary factor Q(-V) advantage within
        # each selected graph. Averaging factors into ``graph_advantage``
        # therefore cancels that signal exactly. Apply the centered residual
        # to each sequential conditional factor decision, while retaining the
        # graph-level PPO objective for the trajectory return advantage.
        factor_training_mask = (
            factor_mask_2d * (1.0 - bad_transitions_mask)
        )
        factor_order_2d = factor_size.squeeze(1)
        raw_local_factor_advantage = (
            factor_advantage - graph_advantage
        ) * factor_training_mask
        order_extra = torch.clamp(factor_order_2d - 2.0, min=0.0)
        positive_order_credit_weight = (
            1.0 + self.adj_order_adv_coef * order_extra
        )
        triplet_order_mask = order_extra > 0.0
        triplet_graph_return_credit = torch.zeros_like(
            raw_local_factor_advantage
        )
        triplet_graph_return_credit_gate = torch.ones_like(
            raw_local_factor_advantage
        )
        graph_return_credit_strength = torch.zeros_like(graph_advantage)
        credit_local_factor_advantage = raw_local_factor_advantage
        triplet_success_selective_gate = (
            delayed_triplet_success_gate.squeeze(-1).clamp(0.0, 1.0)
        )
        if self.use_adj_delayed_triplet_success_gate:
            gate_floor = float(self.adj_delayed_triplet_success_gate_floor)
            if 0.0 < gate_floor < 1.0:
                triplet_success_selective_gate = (
                    (triplet_success_selective_gate - gate_floor)
                    / (1.0 - gate_floor)
                ).clamp(0.0, 1.0)
            elif gate_floor >= 1.0:
                triplet_success_selective_gate = torch.zeros_like(
                    triplet_success_selective_gate
                )
        triplet_graph_return_success_gate = torch.ones_like(
            raw_local_factor_advantage
        )
        if self.adj_triplet_graph_return_credit_require_delayed_gate:
            triplet_graph_return_success_gate = triplet_success_selective_gate
        if (
            self.use_adj_triplet_graph_return_credit
            and self.adj_triplet_graph_return_credit_coef > 0.0
        ):
            # run42 reached high capture counts but its triplet marginal EMA
            # became increasingly negative.  The local factor residual alone
            # misses delayed graph-level payoff, so selected triplets in
            # positive-return graphs receive a bounded supplemental credit.
            graph_return_credit = torch.clamp(
                graph_advantage
                - float(self.adj_triplet_graph_return_credit_min_graph_adv),
                min=0.0,
            )
            graph_return_credit = graph_return_credit * transition_mask
            graph_return_abs_scale = (
                (graph_advantage.abs() * transition_mask).sum()
                / transition_mask.sum().clamp_min(1.0)
            ).clamp_min(1e-6)
            graph_return_credit = (
                graph_return_credit
                * float(self.adj_triplet_graph_return_credit_coef)
            )
            if self.adj_triplet_graph_return_credit_cap > 0.0:
                graph_return_credit_cap = (
                    graph_return_abs_scale
                    * float(self.adj_triplet_graph_return_credit_cap)
                )
                graph_return_credit_cap = (
                    torch.ones_like(graph_return_credit)
                    * graph_return_credit_cap
                )
                graph_return_credit = torch.where(
                    graph_return_credit > graph_return_credit_cap,
                    graph_return_credit_cap,
                    graph_return_credit,
                )
            graph_return_credit_strength = (
                graph_return_credit / graph_return_abs_scale
            ).clamp(0.0, 1.0)
            if self.adj_triplet_graph_return_credit_raw_gate_scale > 0.0:
                gate_scale = (
                    graph_return_abs_scale
                    * float(
                        self.adj_triplet_graph_return_credit_raw_gate_scale
                    )
                ).clamp_min(1e-6)
                triplet_graph_return_credit_gate = (
                    (
                        raw_local_factor_advantage
                        + gate_scale
                    )
                    / gate_scale
                ).clamp(0.0, 1.0)
            triplet_graph_return_credit = (
                graph_return_credit
                * triplet_order_mask.to(graph_return_credit.dtype)
                * triplet_graph_return_credit_gate
                * triplet_graph_return_success_gate
                * factor_training_mask
            )
            credit_local_factor_advantage = (
                raw_local_factor_advantage
                + triplet_graph_return_credit
            )
        positive_residual_mask = credit_local_factor_advantage > 0.0
        graph_promotion_strength = torch.ones_like(graph_advantage)
        if self.adj_order_adv_require_positive_graph_adv:
            # A triplet that is locally better than the average factor inside
            # a bad graph can still be a bad coordination choice.  The first
            # binary gate fixed that direction, but run34 showed it was too
            # noisy: promoted fractions collapsed while graph PPO clamp ratios
            # grew.  Use a continuous positive graph-advantage strength when
            # requested; negative-return graphs still get zero positive
            # triplet credit, while mildly positive graphs get a proportional
            # rather than all-or-nothing promotion.
            if self.adj_order_adv_graph_gate_mode == "soft":
                graph_adv_abs_scale = (
                    (graph_advantage.abs() * transition_mask).sum()
                    / transition_mask.sum().clamp_min(1.0)
                ).clamp_min(1e-6)
                graph_adv_abs_scale = (
                    graph_adv_abs_scale
                    * float(self.adj_order_adv_graph_gate_scale)
                ).clamp_min(1e-6)
                positive_graph_advantage = torch.clamp(
                    graph_advantage,
                    min=0.0,
                )
                graph_promotion_strength = (
                    positive_graph_advantage
                    / (
                        positive_graph_advantage
                        + graph_adv_abs_scale
                    )
                )
            else:
                graph_promotion_strength = (
                    graph_advantage > 0.0
                ).float()
        promoted_positive_adv_mask = (
            positive_residual_mask
            & (
                graph_promotion_strength
                > 0.0
            )
        )
        if self.adj_order_adv_positive_only:
            negative_order_credit_weight = (
                1.0
                + self.adj_order_adv_negative_coef
                * order_extra
            )
            # run32 still had a persistent positive order3 PPO loss even
            # though the structure ratio was already pair-heavy.  The gate
            # was partly reacting to negative triplet residuals that had
            # been multiplied by the same order-aware factor used to promote
            # useful triplets.  Split the two directions: positive triplet
            # residuals keep the strong promotion signal, while negative
            # residuals are suppressed at a configurable, usually smaller,
            # scale so the graph learner is driven by marginal gains instead
            # of amplified early triplet noise.
            triplet_positive_weight = (
                positive_order_credit_weight
                * graph_promotion_strength
            )
            positive_credit_weight = torch.where(
                triplet_order_mask,
                triplet_positive_weight,
                torch.ones_like(positive_order_credit_weight),
            )
            order_credit_weight = torch.where(
                positive_residual_mask,
                positive_credit_weight,
                negative_order_credit_weight,
            )
        else:
            order_credit_weight = positive_order_credit_weight
        order_credit_weight = torch.where(
            factor_training_mask > 0.0,
            order_credit_weight,
            torch.ones_like(order_credit_weight),
        )
        local_factor_advantage = (
            credit_local_factor_advantage
            * order_credit_weight
        )
        factor_diff_log = (
            target_factor_logp - behavior_factor_logp
        ).clamp(min=-10.0, max=10.0)
        factor_imp_weights = torch.exp(factor_diff_log)
        factor_imp_weights = _nan_to_num_compat(
            factor_imp_weights,
            nan=1.0,
            posinf=1.0 + self.clip_param,
            neginf=1.0 - self.clip_param,
        )
        clipped_factor_imp_weights = torch.clamp(
            factor_imp_weights,
            1.0 - self.clip_param,
            1.0 + self.clip_param,
        )
        factor_trust_weights = torch.ones_like(factor_imp_weights)
        if self.use_adj_ppo_stale_trust:
            factor_ratio_deviation = (
                factor_imp_weights.detach() - 1.0
            ).abs()
            factor_stale_excess = (
                factor_ratio_deviation
                - float(self.adj_ppo_stale_trust_clip)
            ).clamp_min(0.0)
            factor_trust_weights = torch.exp(
                -factor_stale_excess
                / float(self.adj_ppo_stale_trust_scale)
            )
            factor_trust_weights = torch.clamp(
                factor_trust_weights,
                min=float(self.adj_ppo_stale_trust_min_weight),
                max=1.0,
            ).detach()
        factor_loss_mask = factor_training_mask * factor_trust_weights
        factor_surr1 = factor_imp_weights * local_factor_advantage
        factor_surr2 = (
            clipped_factor_imp_weights * local_factor_advantage
        )
        factor_min_surr = torch.min(factor_surr1, factor_surr2)
        raw_factor_surr1 = (
            factor_imp_weights * raw_local_factor_advantage
        )
        raw_factor_surr2 = (
            clipped_factor_imp_weights * raw_local_factor_advantage
        )
        raw_factor_min_surr = torch.min(
            raw_factor_surr1,
            raw_factor_surr2,
        )
        credit_factor_surr1 = (
            factor_imp_weights * credit_local_factor_advantage
        )
        credit_factor_surr2 = (
            clipped_factor_imp_weights * credit_local_factor_advantage
        )
        credit_factor_min_surr = torch.min(
            credit_factor_surr1,
            credit_factor_surr2,
        )
        factor_rl_loss = -(
            factor_min_surr
            * factor_loss_mask
        ).sum() / factor_loss_mask.sum().clamp_min(1.0)
        rl_loss = graph_rl_loss + factor_rl_loss

        # 仅对环境未结束且在线的 agent 计算图熵。
        valid_env_masks = (1.0 - bad_transitions_mask).view(-1, 1, 1)
        valid_agent_masks = active_masks * valid_env_masks

        # 先对每个 transition 的在线 agent 求平均，再对有效 transition 求平均。
        active_count = active_masks.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

        entropy_per_transition = (
                (adj_entropy_batch * active_masks).sum(
                    dim=1, keepdim=True
                )
                / active_count
        )

        entropy_transition_mask = (
                (1.0 - bad_transitions_mask)
                .view(-1, 1, 1)
                * (active_count > 0).float()
        )

        entropy_loss = (
            entropy_per_transition * entropy_transition_mask
        ).sum() / entropy_transition_mask.sum().clamp_min(1.0)

        loss = rl_loss - self.adj_entropy_coef * entropy_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "non-finite SDDFG adjacency loss"
            )

        self.adj_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.adj_parameters, self.adj_max_grad_norm)
        self.adj_optimizer.step()

        with torch.no_grad():
            valid_transition_ratio = float((1.0 - bad_transitions_mask).mean().detach().cpu().item())
            valid_agent_ratio = float(valid_agent_masks.mean().detach().cpu().item())

            valid_factor_stats_mask = (
                factor_mask
                * (1.0 - bad_transitions_mask).view(-1, 1, 1)
            )
            valid_factor_stats_count = valid_factor_stats_mask.sum().clamp_min(1.0)
            valid_graph_stats_mask = transition_mask > 0.5
            clip_fraction = (
                (
                    (graph_imp_weights > 1.0 + self.clip_param)
                    | (graph_imp_weights < 1.0 - self.clip_param)
                ).float()
                * transition_mask
            ).sum() / transition_mask.sum().clamp_min(1.0)
            factor_clip_fraction = (
                (
                    (factor_imp_weights > 1.0 + self.clip_param)
                    | (factor_imp_weights < 1.0 - self.clip_param)
                ).float()
                * factor_training_mask
            ).sum() / factor_training_mask.sum().clamp_min(1.0)
            trusted_clip_fraction = (
                (
                    (
                        (graph_imp_weights > 1.0 + self.clip_param)
                        | (graph_imp_weights < 1.0 - self.clip_param)
                    ).float()
                    * graph_loss_mask
                ).sum()
                / graph_loss_mask.sum().clamp_min(1.0)
            )
            trusted_factor_clip_fraction = (
                (
                    (
                        (factor_imp_weights > 1.0 + self.clip_param)
                        | (factor_imp_weights < 1.0 - self.clip_param)
                    ).float()
                    * factor_loss_mask
                ).sum()
                / factor_loss_mask.sum().clamp_min(1.0)
            )
            graph_trust_weight_mean = (
                (graph_trust_weights * transition_mask).sum()
                / transition_mask.sum().clamp_min(1.0)
            )
            factor_trust_weight_mean = (
                (factor_trust_weights * factor_training_mask).sum()
                / factor_training_mask.sum().clamp_min(1.0)
            )
            graph_stale_ratio = (
                (
                    (
                        (graph_imp_weights.detach() - 1.0).abs()
                        > float(self.adj_ppo_stale_trust_clip)
                    ).float()
                    * transition_mask
                ).sum()
                / transition_mask.sum().clamp_min(1.0)
            )
            factor_stale_ratio = (
                (
                    (
                        (factor_imp_weights.detach() - 1.0).abs()
                        > float(self.adj_ppo_stale_trust_clip)
                    ).float()
                    * factor_training_mask
                ).sum()
                / factor_training_mask.sum().clamp_min(1.0)
            )
            order2_factor_mask = (
                (factor_order_2d == 2.0).float()
                * factor_training_mask
            )
            order3_factor_mask = (
                (factor_order_2d == 3.0).float()
                * factor_training_mask
            )
            order2_factor_count = order2_factor_mask.sum()
            order3_factor_count = order3_factor_mask.sum()
            selected_order_factor_count = (
                order2_factor_count + order3_factor_count
            ).clamp_min(1.0)
            order3_factor_fraction = (
                order3_factor_count / selected_order_factor_count
            )
            order2_factor_advantage_abs_mean = (
                (raw_local_factor_advantage.abs() * order2_factor_mask).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            order3_factor_advantage_abs_mean = (
                (raw_local_factor_advantage.abs() * order3_factor_mask).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            weighted_order3_factor_advantage_abs_mean = (
                (local_factor_advantage.abs() * order3_factor_mask).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            credit_order2_factor_advantage_abs_mean = (
                (
                    credit_local_factor_advantage.abs()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            credit_order3_factor_advantage_abs_mean = (
                (
                    credit_local_factor_advantage.abs()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            triplet_graph_return_credit_sum = (
                triplet_graph_return_credit
                * order3_factor_mask
            ).sum()
            triplet_graph_return_credit_mean = (
                triplet_graph_return_credit_sum
                / order3_factor_count.clamp_min(1.0)
            )
            triplet_graph_return_credit_active_fraction = (
                (
                    (triplet_graph_return_credit > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            triplet_graph_return_credit_gate_mean = (
                (
                    triplet_graph_return_credit_gate
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_credit_2d = delayed_triplet_credit.squeeze(-1)
            delayed_triplet_credit_mean = (
                (
                    delayed_triplet_credit_2d
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_credit_active_fraction = (
                (
                    (delayed_triplet_credit_2d.abs() > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_credit_positive_fraction = (
                (
                    (delayed_triplet_credit_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_credit_negative_fraction = (
                (
                    (delayed_triplet_credit_2d < 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_success_gate_2d = (
                delayed_triplet_success_gate.squeeze(-1)
            )
            delayed_triplet_success_gate_mean = (
                (
                    delayed_triplet_success_gate_2d
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_success_gate_active_fraction = (
                (
                    (delayed_triplet_success_gate_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_success_gate_selective_2d = (
                triplet_success_selective_gate
            )
            delayed_triplet_success_gate_selective_mean = (
                (
                    delayed_triplet_success_gate_selective_2d
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_success_gate_selective_fraction = (
                (
                    (
                        delayed_triplet_success_gate_selective_2d > 0.0
                    ).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            triplet_graph_return_success_gate_2d = (
                triplet_graph_return_success_gate
            )
            triplet_graph_return_success_gate_mean = (
                (
                    triplet_graph_return_success_gate_2d
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            triplet_graph_return_success_gate_active_fraction = (
                (
                    (triplet_graph_return_success_gate_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_future_match_2d = (
                delayed_triplet_future_match.squeeze(-1)
            )
            delayed_triplet_future_exact_2d = (
                delayed_triplet_future_exact.squeeze(-1)
            )
            delayed_triplet_future_partial_2d = (
                delayed_triplet_future_partial.squeeze(-1)
            )
            delayed_triplet_future_match_weight_mean = (
                (
                    delayed_triplet_future_match_2d
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_future_matched_fraction = (
                (
                    (delayed_triplet_future_match_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_future_exact_fraction = (
                (
                    (delayed_triplet_future_exact_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            delayed_triplet_future_partial_fraction = (
                (
                    (delayed_triplet_future_partial_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_2d = (
                capture_to_win_triplet_credit.squeeze(-1)
            )
            capture_to_win_quality_gate_2d = (
                capture_to_win_quality_gate.squeeze(-1)
            )
            capture_to_win_triplet_credit_mean = (
                (
                    capture_to_win_triplet_credit_2d
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_active_fraction = (
                (
                    (
                        capture_to_win_triplet_credit_2d.abs() > 0.0
                    ).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            capture_to_win_quality_gate_mean = (
                (
                    capture_to_win_quality_gate_2d
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            capture_to_win_quality_gate_active_fraction = (
                (
                    (capture_to_win_quality_gate_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            graph_return_credit_strength_mean = (
                (graph_return_credit_strength * transition_mask).sum()
                / transition_mask.sum().clamp_min(1.0)
            )
            order2_factor_loss_mask = order2_factor_mask * factor_trust_weights
            order3_factor_loss_mask = order3_factor_mask * factor_trust_weights
            order2_factor_loss_count = order2_factor_loss_mask.sum()
            order3_factor_loss_count = order3_factor_loss_mask.sum()
            order2_factor_rl_loss = -(
                factor_min_surr * order2_factor_loss_mask
            ).sum() / order2_factor_loss_count.clamp_min(1.0)
            order3_factor_rl_loss = -(
                factor_min_surr * order3_factor_loss_mask
            ).sum() / order3_factor_loss_count.clamp_min(1.0)
            raw_order2_factor_rl_loss = -(
                raw_factor_min_surr * order2_factor_loss_mask
            ).sum() / order2_factor_loss_count.clamp_min(1.0)
            raw_order3_factor_rl_loss = -(
                raw_factor_min_surr * order3_factor_loss_mask
            ).sum() / order3_factor_loss_count.clamp_min(1.0)
            credit_order2_factor_rl_loss = -(
                credit_factor_min_surr * order2_factor_loss_mask
            ).sum() / order2_factor_loss_count.clamp_min(1.0)
            credit_order3_factor_rl_loss = -(
                credit_factor_min_surr * order3_factor_loss_mask
            ).sum() / order3_factor_loss_count.clamp_min(1.0)
            order2_positive_adv_fraction = (
                (
                    (raw_local_factor_advantage > 0.0).float()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            order3_positive_adv_fraction = (
                (
                    (raw_local_factor_advantage > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            credit_order2_positive_adv_fraction = (
                (
                    (credit_local_factor_advantage > 0.0).float()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            credit_order3_positive_adv_fraction = (
                (
                    (credit_local_factor_advantage > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            order2_promoted_adv_fraction = (
                (
                    promoted_positive_adv_mask.float()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            order3_promoted_adv_fraction = (
                (
                    promoted_positive_adv_mask.float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            order2_promotion_strength_mean = (
                (
                    graph_promotion_strength
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            order3_promotion_strength_mean = (
                (
                    graph_promotion_strength
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            order3_promoted_adv_weighted_fraction = (
                (
                    graph_promotion_strength
                    * positive_residual_mask.float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            advantage_triplet_credit_info = {}
            if hasattr(self.adj_network, "update_factor_credit_memory"):
                advantage_triplet_credit_info = (
                    self.adj_network.update_factor_credit_memory(
                        valid_adj.detach(),
                        credit_local_factor_advantage.detach(),
                        graph_advantage.detach(),
                        factor_training_mask.detach(),
                    )
                )

            valid_imp_weights = graph_imp_weights[
                valid_graph_stats_mask
            ]
            valid_factor_imp_weights = factor_imp_weights[
                factor_training_mask > 0.5
            ]
            valid_target_probs = torch.exp(target_factor_logp)[
                valid_factor_stats_mask.squeeze(-1) > 0.5
            ]
            valid_behavior_probs = torch.exp(behavior_factor_logp)[
                valid_factor_stats_mask.squeeze(-1) > 0.5
            ]
            if valid_imp_weights.numel() == 0:
                imp_weight_mean = torch.tensor(1.0, device=self.device)
                imp_weight_max = torch.tensor(1.0, device=self.device)
                imp_weight_std = torch.tensor(0.0, device=self.device)
                target_prob_mean = torch.tensor(0.0, device=self.device)
                behavior_prob_mean = torch.tensor(0.0, device=self.device)
                positive_adv_fraction = torch.tensor(0.0, device=self.device)
            else:
                imp_weight_mean = valid_imp_weights.mean()
                imp_weight_max = valid_imp_weights.max()
                imp_weight_std = (
                    valid_imp_weights.std()
                    if valid_imp_weights.numel() > 1
                    else torch.tensor(0.0, device=self.device)
                )
                target_prob_mean = valid_target_probs.mean()
                behavior_prob_mean = valid_behavior_probs.mean()
                positive_adv_fraction = (
                    (graph_advantage > 0.0).float()
                    * transition_mask
                ).sum() / transition_mask.sum().clamp_min(1.0)

            if valid_factor_imp_weights.numel() == 0:
                factor_imp_weight_mean = torch.tensor(
                    1.0, device=self.device
                )
                factor_imp_weight_max = torch.tensor(
                    1.0, device=self.device
                )
            else:
                factor_imp_weight_mean = valid_factor_imp_weights.mean()
                factor_imp_weight_max = valid_factor_imp_weights.max()

            current_adj_lr = self.adj_optimizer.param_groups[0]["lr"]

            order3_credit_gate = 1.0
            if hasattr(self.adj_network, "update_order3_credit_gate"):
                order3_credit_gate = self.adj_network.update_order3_credit_gate(
                    _to_float(raw_order3_factor_rl_loss),
                    _to_float(raw_order2_factor_rl_loss),
                )
            order3_credit_loss_ema = float(
                getattr(self.adj_network, "order3_credit_loss_ema", 0.0)
            )
            order3_credit_margin_ema = float(
                getattr(self.adj_network, "order3_credit_margin_ema", 0.0)
            )

        train_info = {}
        train_info['advantage'] = _to_float(f_advts.mean())
        train_info['f_advts_abs_mean'] = _to_float(f_advts.abs().mean())
        train_info['graph_advantage_abs_mean'] = _to_float(
            (graph_advantage.abs() * transition_mask).sum()
            / transition_mask.sum().clamp_min(1.0)
        )
        train_info['clamp_ratio'] = _to_float(clip_fraction)
        train_info['factor_clamp_ratio'] = _to_float(
            factor_clip_fraction
        )
        train_info['trusted_clamp_ratio'] = _to_float(
            trusted_clip_fraction
        )
        train_info['trusted_factor_clamp_ratio'] = _to_float(
            trusted_factor_clip_fraction
        )
        train_info['adj_graph_trust_weight_mean'] = _to_float(
            graph_trust_weight_mean
        )
        train_info['adj_factor_trust_weight_mean'] = _to_float(
            factor_trust_weight_mean
        )
        train_info['adj_graph_stale_ratio'] = _to_float(graph_stale_ratio)
        train_info['adj_factor_stale_ratio'] = _to_float(factor_stale_ratio)
        train_info['use_adj_ppo_stale_trust'] = float(
            self.use_adj_ppo_stale_trust
        )
        train_info['adj_ppo_stale_trust_clip'] = float(
            self.adj_ppo_stale_trust_clip
        )
        train_info['adj_ppo_stale_trust_scale'] = float(
            self.adj_ppo_stale_trust_scale
        )
        train_info['adj_ppo_stale_trust_min_weight'] = float(
            self.adj_ppo_stale_trust_min_weight
        )
        train_info['imp_weight_mean'] = _to_float(imp_weight_mean)
        train_info['imp_weight_max'] = _to_float(imp_weight_max)
        train_info['imp_weight_std'] = _to_float(imp_weight_std)
        train_info['factor_imp_weight_mean'] = _to_float(
            factor_imp_weight_mean
        )
        train_info['factor_imp_weight_max'] = _to_float(
            factor_imp_weight_max
        )
        train_info['target_factor_prob_mean'] = _to_float(target_prob_mean)
        train_info['behavior_factor_prob_mean'] = _to_float(behavior_prob_mean)
        train_info['positive_adv_fraction'] = _to_float(positive_adv_fraction)
        train_info['valid_transition_ratio'] = valid_transition_ratio
        train_info['valid_agent_ratio'] = valid_agent_ratio
        train_info['rl_loss'] = _to_float(rl_loss)
        train_info['graph_rl_loss'] = _to_float(graph_rl_loss)
        train_info['factor_rl_loss'] = _to_float(factor_rl_loss)
        train_info['factor_advantage_abs_mean'] = _to_float(
            (local_factor_advantage.abs() * factor_training_mask).sum()
            / factor_training_mask.sum().clamp_min(1.0)
        )
        train_info['raw_factor_advantage_abs_mean'] = _to_float(
            (raw_local_factor_advantage.abs() * factor_training_mask).sum()
            / factor_training_mask.sum().clamp_min(1.0)
        )
        train_info['order3_factor_fraction'] = _to_float(
            order3_factor_fraction
        )
        train_info['order2_factor_advantage_abs_mean'] = _to_float(
            order2_factor_advantage_abs_mean
        )
        train_info['order3_factor_advantage_abs_mean'] = _to_float(
            order3_factor_advantage_abs_mean
        )
        train_info['weighted_order3_factor_advantage_abs_mean'] = _to_float(
            weighted_order3_factor_advantage_abs_mean
        )
        train_info['credit_order2_factor_advantage_abs_mean'] = _to_float(
            credit_order2_factor_advantage_abs_mean
        )
        train_info['credit_order3_factor_advantage_abs_mean'] = _to_float(
            credit_order3_factor_advantage_abs_mean
        )
        train_info['triplet_graph_return_credit_mean'] = _to_float(
            triplet_graph_return_credit_mean
        )
        train_info['triplet_graph_return_credit_active_fraction'] = _to_float(
            triplet_graph_return_credit_active_fraction
        )
        train_info['triplet_graph_return_credit_gate_mean'] = _to_float(
            triplet_graph_return_credit_gate_mean
        )
        train_info['delayed_triplet_credit_mean'] = _to_float(
            delayed_triplet_credit_mean
        )
        train_info['delayed_triplet_credit_active_fraction'] = _to_float(
            delayed_triplet_credit_active_fraction
        )
        train_info['delayed_triplet_credit_positive_fraction'] = _to_float(
            delayed_triplet_credit_positive_fraction
        )
        train_info['delayed_triplet_credit_negative_fraction'] = _to_float(
            delayed_triplet_credit_negative_fraction
        )
        train_info['delayed_triplet_success_gate_mean'] = _to_float(
            delayed_triplet_success_gate_mean
        )
        train_info['delayed_triplet_success_gate_active_fraction'] = _to_float(
            delayed_triplet_success_gate_active_fraction
        )
        train_info['delayed_triplet_success_gate_selective_mean'] = _to_float(
            delayed_triplet_success_gate_selective_mean
        )
        train_info[
            'delayed_triplet_success_gate_selective_fraction'
        ] = _to_float(delayed_triplet_success_gate_selective_fraction)
        train_info['triplet_graph_return_success_gate_mean'] = _to_float(
            triplet_graph_return_success_gate_mean
        )
        train_info[
            'triplet_graph_return_success_gate_active_fraction'
        ] = _to_float(triplet_graph_return_success_gate_active_fraction)
        train_info['delayed_triplet_future_match_weight_mean'] = _to_float(
            delayed_triplet_future_match_weight_mean
        )
        train_info['delayed_triplet_future_matched_fraction'] = _to_float(
            delayed_triplet_future_matched_fraction
        )
        train_info['delayed_triplet_future_exact_fraction'] = _to_float(
            delayed_triplet_future_exact_fraction
        )
        train_info['delayed_triplet_future_partial_fraction'] = _to_float(
            delayed_triplet_future_partial_fraction
        )
        train_info['capture_to_win_triplet_credit_mean'] = _to_float(
            capture_to_win_triplet_credit_mean
        )
        train_info['capture_to_win_triplet_credit_active_fraction'] = _to_float(
            capture_to_win_triplet_credit_active_fraction
        )
        train_info['capture_to_win_quality_gate_mean'] = _to_float(
            capture_to_win_quality_gate_mean
        )
        train_info['capture_to_win_quality_gate_active_fraction'] = _to_float(
            capture_to_win_quality_gate_active_fraction
        )
        train_info['graph_return_credit_strength_mean'] = _to_float(
            graph_return_credit_strength_mean
        )
        train_info['order2_factor_rl_loss'] = _to_float(
            order2_factor_rl_loss
        )
        train_info['order3_factor_rl_loss'] = _to_float(
            order3_factor_rl_loss
        )
        train_info['raw_order2_factor_rl_loss'] = _to_float(
            raw_order2_factor_rl_loss
        )
        train_info['raw_order3_factor_rl_loss'] = _to_float(
            raw_order3_factor_rl_loss
        )
        train_info['raw_o3_minus_o2_factor_rl_loss'] = _to_float(
            raw_order3_factor_rl_loss - raw_order2_factor_rl_loss
        )
        train_info['credit_order2_factor_rl_loss'] = _to_float(
            credit_order2_factor_rl_loss
        )
        train_info['credit_order3_factor_rl_loss'] = _to_float(
            credit_order3_factor_rl_loss
        )
        train_info['credit_o3_minus_o2_factor_rl_loss'] = _to_float(
            credit_order3_factor_rl_loss - credit_order2_factor_rl_loss
        )
        train_info['order2_positive_adv_fraction'] = _to_float(
            order2_positive_adv_fraction
        )
        train_info['order3_positive_adv_fraction'] = _to_float(
            order3_positive_adv_fraction
        )
        train_info['credit_order2_positive_adv_fraction'] = _to_float(
            credit_order2_positive_adv_fraction
        )
        train_info['credit_order3_positive_adv_fraction'] = _to_float(
            credit_order3_positive_adv_fraction
        )
        train_info['order2_promoted_adv_fraction'] = _to_float(
            order2_promoted_adv_fraction
        )
        train_info['order3_promoted_adv_fraction'] = _to_float(
            order3_promoted_adv_fraction
        )
        train_info['order2_promotion_strength_mean'] = _to_float(
            order2_promotion_strength_mean
        )
        train_info['order3_promotion_strength_mean'] = _to_float(
            order3_promotion_strength_mean
        )
        train_info['order3_promoted_adv_weighted_fraction'] = _to_float(
            order3_promoted_adv_weighted_fraction
        )
        train_info['adj_order_adv_coef'] = float(self.adj_order_adv_coef)
        train_info['adj_order_adv_positive_only'] = float(
            self.adj_order_adv_positive_only
        )
        train_info['adj_order_adv_negative_coef'] = float(
            self.adj_order_adv_negative_coef
        )
        train_info['adj_order_adv_require_positive_graph_adv'] = float(
            self.adj_order_adv_require_positive_graph_adv
        )
        train_info['adj_order_adv_graph_gate_soft'] = float(
            self.adj_order_adv_graph_gate_mode == "soft"
        )
        train_info['adj_order_adv_graph_gate_scale'] = float(
            self.adj_order_adv_graph_gate_scale
        )
        train_info['use_adj_triplet_graph_return_credit'] = float(
            self.use_adj_triplet_graph_return_credit
        )
        train_info['adj_triplet_graph_return_credit_coef'] = float(
            self.adj_triplet_graph_return_credit_coef
        )
        train_info['adj_triplet_graph_return_credit_cap'] = float(
            self.adj_triplet_graph_return_credit_cap
        )
        train_info['adj_triplet_graph_return_credit_min_graph_adv'] = float(
            self.adj_triplet_graph_return_credit_min_graph_adv
        )
        train_info['adj_triplet_graph_return_credit_raw_gate_scale'] = float(
            self.adj_triplet_graph_return_credit_raw_gate_scale
        )
        train_info[
            'adj_triplet_graph_return_credit_require_delayed_gate'
        ] = float(
            self.adj_triplet_graph_return_credit_require_delayed_gate
        )
        train_info['use_adj_delayed_triplet_credit'] = float(
            self.use_adj_delayed_triplet_credit
        )
        train_info['adj_delayed_triplet_credit_coef'] = float(
            self.adj_delayed_triplet_credit_coef
        )
        train_info['adj_delayed_triplet_credit_window'] = float(
            self.adj_delayed_triplet_credit_window
        )
        train_info['adj_delayed_triplet_credit_cap'] = float(
            self.adj_delayed_triplet_credit_cap
        )
        train_info['adj_delayed_triplet_credit_min_reward'] = float(
            self.adj_delayed_triplet_credit_min_reward
        )
        train_info['adj_delayed_triplet_credit_positive_only'] = float(
            self.adj_delayed_triplet_credit_positive_only
        )
        train_info['adj_delayed_triplet_credit_min_adv'] = float(
            self.adj_delayed_triplet_credit_min_adv
        )
        train_info['adj_delayed_triplet_credit_require_future_match'] = float(
            self.adj_delayed_triplet_credit_require_future_match
        )
        train_info['use_adj_delayed_triplet_success_gate'] = float(
            self.use_adj_delayed_triplet_success_gate
        )
        train_info['adj_delayed_triplet_success_gate_min_adv'] = float(
            self.adj_delayed_triplet_success_gate_min_adv
        )
        train_info['adj_delayed_triplet_success_gate_scale'] = float(
            self.adj_delayed_triplet_success_gate_scale
        )
        train_info['adj_delayed_triplet_success_gate_floor'] = float(
            self.adj_delayed_triplet_success_gate_floor
        )
        train_info['adj_delayed_triplet_future_overlap_min_nodes'] = float(
            self.adj_delayed_triplet_future_overlap_min_nodes
        )
        train_info['adj_delayed_triplet_partial_match_weight'] = float(
            self.adj_delayed_triplet_partial_match_weight
        )
        train_info['use_adj_capture_to_win_credit'] = float(
            self.use_adj_capture_to_win_credit
        )
        train_info['adj_capture_to_win_credit_coef'] = float(
            self.adj_capture_to_win_credit_coef
        )
        train_info['adj_capture_to_win_credit_min_outcome_adv'] = float(
            self.adj_capture_to_win_credit_min_outcome_adv
        )
        train_info['adj_capture_to_win_credit_scale'] = float(
            self.adj_capture_to_win_credit_scale
        )
        train_info['adj_capture_to_win_credit_cap'] = float(
            self.adj_capture_to_win_credit_cap
        )
        train_info['adj_capture_to_win_credit_require_future_match'] = float(
            self.adj_capture_to_win_credit_require_future_match
        )
        for credit_key, credit_value in advantage_triplet_credit_info.items():
            train_info[credit_key] = float(credit_value)
        train_info['use_adj_advantage_triplet_scorer'] = float(
            bool(
                getattr(
                    self.adj_network,
                    "use_advantage_triplet_scorer",
                    False,
                )
            )
        )
        train_info['use_adj_triplet_credit_direct_rank'] = float(
            bool(
                getattr(
                    self.adj_network,
                    "use_triplet_credit_direct_rank",
                    False,
                )
            )
        )
        train_info['adj_triplet_credit_rank_coef'] = float(
            getattr(self.adj_network, "triplet_credit_rank_coef", 0.0)
        )
        train_info['adj_triplet_credit_min_multiplier'] = float(
            getattr(self.adj_network, "triplet_credit_min_multiplier", 1.0)
        )
        train_info['adj_triplet_credit_max_multiplier'] = float(
            getattr(self.adj_network, "triplet_credit_max_multiplier", 1.0)
        )
        train_info['adj_triplet_credit_negative_rank_scale'] = float(
            getattr(
                self.adj_network,
                "triplet_credit_negative_rank_scale",
                1.0,
            )
        )
        train_info['adj_triplet_credit_min_positive_fraction'] = float(
            getattr(
                self.adj_network,
                "triplet_credit_min_positive_fraction",
                0.0,
            )
        )
        train_info['adv_triplet_score_multiplier_mean'] = float(
            getattr(
                self.adj_network,
                "last_adv_triplet_score_multiplier_mean",
                1.0,
            )
        )
        train_info['adv_triplet_score_multiplier_min'] = float(
            getattr(
                self.adj_network,
                "last_adv_triplet_score_multiplier_min",
                1.0,
            )
        )
        train_info['adv_triplet_score_multiplier_max'] = float(
            getattr(
                self.adj_network,
                "last_adv_triplet_score_multiplier_max",
                1.0,
            )
        )
        train_info['adv_triplet_score_marginal_mean'] = float(
            getattr(
                self.adj_network,
                "last_adv_triplet_score_marginal_mean",
                0.0,
            )
        )
        train_info['adv_triplet_score_positive_fraction'] = float(
            getattr(
                self.adj_network,
                "last_adv_triplet_score_positive_fraction",
                0.0,
            )
        )
        train_info['adv_triplet_negative_scaled_fraction'] = float(
            getattr(
                self.adj_network,
                "last_adv_triplet_negative_scaled_fraction",
                0.0,
            )
        )
        train_info['adj_order3_credit_gate_current'] = float(
            order3_credit_gate
        )
        train_info['adj_order3_credit_loss_ema'] = float(
            order3_credit_loss_ema
        )
        train_info['adj_order3_credit_margin_ema'] = float(
            order3_credit_margin_ema
        )
        train_info['adj_order3_relative_credit_gate'] = float(
            bool(
                getattr(
                    self.adj_network,
                    "use_relative_order3_credit_gate",
                    False,
                )
            )
        )
        train_info['entropy_loss'] = _to_float(entropy_loss)
        train_info['adj_entropy_coef_initial'] = float(self.entropy_coef)
        train_info['adj_entropy_coef'] = float(self.adj_entropy_coef)
        train_info['grad_norm'] = _to_float(grad_norm)
        train_info['adj_lr'] = float(current_adj_lr)

        return train_info, None, None

    def hard_target_updates(self):
        """Hard update the target networks."""
        for policy_id in self.policy_ids:
            self.target_policies[policy_id].load_state(
                self.policies[policy_id])

    def soft_target_updates(self):
        """Soft update the target networks."""
        for policy_id in self.policy_ids:
            soft_update(
                self.target_policies[policy_id], self.policies[policy_id], self.tau)
            if self.use_vfunction:
                target_policy = self.target_policies[policy_id]
                source_policy = self.policies[policy_id]
                soft_update(
                    target_policy.rnn_critic_network,
                    source_policy.rnn_critic_network,
                    self.tau,
                )
                soft_update(
                    target_policy.vtot_network,
                    source_policy.vtot_network,
                    self.tau,
                )
                for order in range(1, self.highest_orders + 1):
                    soft_update(
                        target_policy.v_network[order],
                        source_policy.v_network[order],
                        self.tau,
                    )

    def prep_training(self):
        """See parent class."""
        self.adj_network.train()
        for p_id in self.policy_ids:
            self.policies[p_id].rnn_network.train()
            self.policies[p_id].rnn_critic_network.train()
            self.target_policies[p_id].rnn_network.train()
            self.target_policies[p_id].rnn_critic_network.train()
            if self.use_vfunction:
                self.policies[p_id].vtot_network.train()
                self.target_policies[p_id].vtot_network.train()

            for num_orders in range(1, self.highest_orders + 1):
                self.policies[p_id].q_network[num_orders].train()
                self.target_policies[p_id].q_network[num_orders].train()
                if self.use_vfunction:
                    self.policies[p_id].v_network[num_orders].train()
                    self.target_policies[p_id].v_network[num_orders].train()

    def prep_rollout(self):
        """See parent class."""

        self.adj_network.eval()
        for p_id in self.policy_ids:
            self.policies[p_id].rnn_network.eval()
            self.policies[p_id].rnn_critic_network.eval()
            self.target_policies[p_id].rnn_network.eval()
            self.target_policies[p_id].rnn_critic_network.eval()
            if self.use_vfunction:
                self.policies[p_id].vtot_network.eval()
                self.target_policies[p_id].vtot_network.eval()

            for num_orders in range(1, self.highest_orders + 1):
                self.policies[p_id].q_network[num_orders].eval()
                self.target_policies[p_id].q_network[num_orders].eval()
                if self.use_vfunction:
                    self.policies[p_id].v_network[num_orders].eval()
                    self.target_policies[p_id].v_network[num_orders].eval()
