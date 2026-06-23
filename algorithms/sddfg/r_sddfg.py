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
        self.entropy_coef = self.args.entropy_coef
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
        for policy in self.policies.values():
            self.critic_fv_parameters += policy.critic_fv_parameters()
        self.critic_fv_optimizer = torch.optim.Adam(params=self.critic_fv_parameters, lr=self.critic_lr,
                                                    eps=self.opti_eps)

        self.critic_vtot_parameters = []
        for policy in self.policies.values():
            self.critic_vtot_parameters += policy.critic_vtot_parameters()
        self.critic_vtot_optimizer = torch.optim.Adam(params=self.critic_vtot_parameters, lr=self.critic_lr,
                                                      eps=self.opti_eps)

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
        adj_floor = float(getattr(self.args, "adj_lr_decay_floor", 2e-5))
        use_adj_decay = bool(getattr(self.args, "use_adj_linear_lr_decay", False))

        # policy / critic 继续衰减，但设置 floor，避免后期完全学不动
        self._set_optimizer_lr_with_floor(
            self.policy_optimizer,
            self.lr,
            episode,
            episodes,
            policy_floor
        )
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
            self._set_optimizer_lr_with_floor(
                self.adj_optimizer,
                self.adj_lr,
                episode,
                episodes,
                adj_floor
            )
        else:
            for param_group in self.adj_optimizer.param_groups:
                param_group["lr"] = float(self.adj_lr)

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
            dones = torch.cat((obs_inactive_dones[:1], stored_next_dones), dim=0)
            dones = torch.maximum(dones.float(), obs_inactive_dones.float())

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
        train_info['vtot_grad_norm'] = _to_float(vtot_grad_norm) if self.use_vfunction else 0.0
        train_info['fv_grad_norm'] = _to_float(fv_grad_norm) if self.use_vfunction else 0.0
        train_info['valid_transition_ratio'] = _to_float(valid_transition_mask.mean())
        train_info['active_transition_ratio'] = _to_float(active_transition_mask.mean())
        return train_info, new_priorities, idxes

    def train_adj_on_batch(self, batch, use_adj_init, use_same_share_obs=None):
        """See parent class."""

        obs_batch, _share_obs_batch, dones_batch, \
            dones_env_batch, adj_batch, prob_adj_batch, \
            _advantages_batch, f_advts_batch, rnn_obs_batch = batch
        tarprob_adj = []
        adj_entropy = []
        adj = to_torch(adj_batch)  # [batch_size,step,num_agent,-1]
        batch_size = adj.shape[0] * adj.shape[1]
        adj = adj.reshape(batch_size, self.num_agents, -1).to(**self.tpdv)
        dones = to_torch(dones_batch).reshape(batch_size, self.num_agents, -1).to(self.device)
        dones_env = to_torch(dones_env_batch).reshape(batch_size, -1).to(**self.tpdv)
        prob_adj = to_torch(prob_adj_batch).reshape(batch_size, self.num_agents, -1).to(**self.tpdv)
        f_advts = to_torch(f_advts_batch).reshape(batch_size, self.num_factor, -1).to(**self.tpdv)
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

        max_log_ratio = 10.0

        if self.highest_orders == 3:
            sort_tar_proadj = torch.topk(tarlog_prob_adj, k=self.highest_orders, dim=1, largest=False)[0].to(
                self.device)
            sort_proadj = torch.topk(log_prob_adj, k=self.highest_orders, dim=1, largest=False)[0].to(self.device)

            idx1 = torch.tensor([[[2], [1], [0]]], device=self.device)
            idx2 = torch.tensor([[[1], [2], [0]]], device=self.device)
            idx_order2 = (valid_adj.sum(-2) == 1).unsqueeze(1)

            log_tar_1 = torch.where(idx_order2, sort_tar_proadj * idx1, sort_tar_proadj).sum(-2)
            log_tar_2 = torch.where(idx_order2, sort_tar_proadj * idx2, sort_tar_proadj).sum(-2)
            log_1 = torch.where(idx_order2, sort_proadj * idx1, sort_proadj).sum(-2)
            log_2 = torch.where(idx_order2, sort_proadj * idx2, sort_proadj).sum(-2)

            log_tar_1 = log_tar_1.clamp(min=-max_log_ratio, max=max_log_ratio)
            log_tar_2 = log_tar_2.clamp(min=-max_log_ratio, max=max_log_ratio)
            log_1 = log_1.clamp(min=-max_log_ratio, max=max_log_ratio)
            log_2 = log_2.clamp(min=-max_log_ratio, max=max_log_ratio)

            denom = (torch.exp(log_1) + torch.exp(log_2)).clamp_min(1e-8)
            imp_weights = (torch.exp(log_tar_1) + torch.exp(log_tar_2)) / denom
            imp_weights = _nan_to_num_compat(
                imp_weights,
                nan=1.0,
                posinf=1.0 + self.clip_param,
                neginf=1.0 - self.clip_param
            )

            imp_weights_multinomial = torch.where(
                valid_adj.sum(-2) == 1,
                imp_weights * imp_weights * imp_weights,
                imp_weights
            ).unsqueeze(-1)

        elif self.highest_orders == 2:
            diff_log = (tarlog_prob_adj.sum(-2) - log_prob_adj.sum(-2)).clamp(
                min=-max_log_ratio,
                max=max_log_ratio
            )
            imp_weights = torch.exp(diff_log)
            imp_weights = _nan_to_num_compat(
                imp_weights,
                nan=1.0,
                posinf=1.0 + self.clip_param,
                neginf=1.0 - self.clip_param
            )
            imp_weights_multinomial = torch.where(
                valid_adj.sum(-2) == 1,
                imp_weights * imp_weights,
                imp_weights
            ).unsqueeze(-1)

        else:
            diff_log = (tarlog_prob_adj.sum(-2) - log_prob_adj.sum(-2)).clamp(
                min=-max_log_ratio,
                max=max_log_ratio
            )
            imp_weights = torch.exp(diff_log)
            imp_weights = _nan_to_num_compat(
                imp_weights,
                nan=1.0,
                posinf=1.0 + self.clip_param,
                neginf=1.0 - self.clip_param
            )
            imp_weights_multinomial = imp_weights.unsqueeze(-1)

        bad_transitions_mask = dones_env
        clamp_imp_weights = torch.clamp(
            imp_weights_multinomial,
            1.0 - self.clip_param,
            1.0 + self.clip_param
        )

        surr1 = imp_weights_multinomial * f_advts
        surr2 = clamp_imp_weights * f_advts
        factor_mask = valid_factor_masks.transpose(1, 2)

        clipped_surr = torch.min(surr1, surr2)
        clipped_surr = clipped_surr * factor_mask

        valid_factor_count = factor_mask.sum(
            dim=-2, keepdim=False
        ).clamp_min(1.0)

        per_transition_surr = (
                clipped_surr.sum(dim=-2)
                / valid_factor_count
        )

        has_valid_factor = (
                factor_mask.sum(dim=-2) > 0
        ).float()

        transition_mask = (
                (1.0 - bad_transitions_mask)
                * has_valid_factor
        )

        rl_loss = -(
                per_transition_surr * transition_mask
        ).sum() / transition_mask.sum().clamp_min(1.0)

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

        loss = rl_loss - self.entropy_coef * entropy_loss
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

            clip_fraction = (
                    (imp_weights_multinomial > 1.0 + self.clip_param) |
                    (imp_weights_multinomial < 1.0 - self.clip_param)
            ).float().mean()

            current_adj_lr = self.adj_optimizer.param_groups[0]["lr"]

        train_info = {}
        train_info['advantage'] = _to_float(f_advts.mean())
        train_info['f_advts_abs_mean'] = _to_float(f_advts.abs().mean())
        train_info['clamp_ratio'] = _to_float(clip_fraction)
        train_info['imp_weight_mean'] = _to_float(imp_weights_multinomial.mean())
        train_info['imp_weight_max'] = _to_float(imp_weights_multinomial.max())
        train_info['imp_weight_std'] = _to_float(imp_weights_multinomial.std())
        train_info['valid_transition_ratio'] = valid_transition_ratio
        train_info['valid_agent_ratio'] = valid_agent_ratio
        train_info['rl_loss'] = _to_float(rl_loss)
        train_info['entropy_loss'] = _to_float(entropy_loss)
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
