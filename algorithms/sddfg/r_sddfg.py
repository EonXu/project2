import torch
import copy
from utils.util import soft_update, huber_loss, mse_loss, to_torch
import numpy as np
from utils.popart import PopArt
from utils.pair_credit import CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH

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


def compute_capture_outcome_factor_ppo_loss(
        factor_imp_weights,
        clipped_factor_imp_weights,
        capture_outcome_local_delta,
        factor_loss_mask,
        transition_mask):
    """Return the target-local outcome PPO term with transition normalization.

    Capture outcome mass is normalized per episode before it reaches the
    trainer.  Dividing its sparse factor surrogate by *all* selected factors
    makes the same episode-level supervision shrink whenever unrelated factor
    slots are added.  The graph PPO objective is transition averaged, so this
    auxiliary factor term uses the same transition denominator while summing
    the already event/episode-normalized target-factor mass.

    Outcome sign is preserved by the standard PPO min surrogate.  The helper
    never changes reward, Q targets, priorities, or non-target factor values.
    """
    tensors = (
        factor_imp_weights,
        clipped_factor_imp_weights,
        capture_outcome_local_delta,
        factor_loss_mask,
    )
    reference_shape = factor_imp_weights.shape
    if any(tensor.shape != reference_shape for tensor in tensors[1:]):
        raise ValueError(
            "outcome factor PPO tensors must share shape {}, got {}".format(
                reference_shape,
                [tuple(tensor.shape) for tensor in tensors],
            )
        )
    if transition_mask.dim() == 1:
        transition_mask = transition_mask.unsqueeze(-1)
    if (
            transition_mask.dim() != 2
            or transition_mask.shape[0] != reference_shape[0]
            or transition_mask.shape[1] != 1):
        raise ValueError(
            "transition_mask must have shape [{}, 1], got {}".format(
                reference_shape[0], tuple(transition_mask.shape)
            )
        )
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise FloatingPointError("non-finite outcome factor PPO input")
    if not bool(torch.isfinite(transition_mask).all().item()):
        raise FloatingPointError("non-finite outcome transition mask")
    target_mask = capture_outcome_local_delta.abs() > 0.0
    if bool(torch.any(target_mask & (factor_loss_mask <= 0.0)).item()):
        raise RuntimeError(
            "capture outcome delta reached a padded or invalid factor"
        )
    valid_transition_count = transition_mask.sum()
    if (
            bool(torch.any(target_mask).item())
            and not bool((valid_transition_count > 0.0).item())):
        raise RuntimeError(
            "capture outcome delta has no valid transition denominator"
        )

    surr1 = factor_imp_weights * capture_outcome_local_delta
    surr2 = clipped_factor_imp_weights * capture_outcome_local_delta
    min_surr = torch.min(surr1, surr2)
    denominator = valid_transition_count.clamp_min(1.0)
    masked_surr = min_surr * factor_loss_mask
    loss = -masked_surr.sum() / denominator
    positive_loss = -(
        masked_surr * (capture_outcome_local_delta > 0.0).float()
    ).sum() / denominator
    negative_loss = -(
        masked_surr * (capture_outcome_local_delta < 0.0).float()
    ).sum() / denominator
    target_count = (
        target_mask.float() * (factor_loss_mask > 0.0).float()
    ).sum()
    valid_factor_count = (
        (factor_loss_mask > 0.0).float().sum()
    )
    factors_per_transition = (
        valid_factor_count / denominator
    )
    return {
        "loss": loss,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "min_surr": min_surr,
        "target_count": target_count,
        "valid_transition_count": valid_transition_count,
        "factors_per_transition": factors_per_transition,
    }


def compute_capture_candidate_identity_loss(
        candidate_scores,
        candidate_identity_delta,
        candidate_valid_mask,
        transition_mask):
    """Target-local loss for exact real captures absent from the active graph.

    Candidate scores are positive *relative selection weights*.  They are not
    independent Bernoulli probabilities: every graph-selection slot samples
    from a conditionally normalized candidate set.  Definition version 3
    therefore supervises the canonical valid-catalog distribution

        ``P(c | C_t) = score_c / sum(j in C_t, score_j)``

    with the signed conditional objective

        ``loss = -sum(delta * log(P(c | C_t))) / valid_graph_transitions``.

    For a single target in a transition, the derivative with respect to a
    candidate log-weight is ``-delta * (1[c == target] - P(c | C_t))``.
    Positive outcomes raise the target relative to its competitors; negative
    outcomes lower it.  Multiplying every valid score by the same constant
    leaves both the loss and gradients unchanged, so the auxiliary objective
    cannot improve itself through global score-scale drift.  Competitors get
    only the implicit normalization gradient: their explicit identity delta
    remains exactly zero.  No behavior importance ratio is used because these
    candidates were explicitly *not* active graph actions.
    """
    if (
            candidate_scores.shape != candidate_identity_delta.shape
            or candidate_scores.shape != candidate_valid_mask.shape):
        raise ValueError(
            "candidate score/delta/mask shapes must match: {}, {}, {}"
            .format(
                tuple(candidate_scores.shape),
                tuple(candidate_identity_delta.shape),
                tuple(candidate_valid_mask.shape),
            )
        )
    if transition_mask.dim() == 1:
        transition_mask = transition_mask.unsqueeze(-1)
    if (
            transition_mask.dim() != 2
            or transition_mask.shape[0] != candidate_scores.shape[0]
            or transition_mask.shape[1] != 1):
        raise ValueError(
            "candidate transition_mask must have shape [{}, 1], got {}"
            .format(candidate_scores.shape[0], tuple(transition_mask.shape))
        )
    for tensor in (
            candidate_scores,
            candidate_identity_delta,
            candidate_valid_mask,
            transition_mask):
        if not bool(torch.isfinite(tensor).all().item()):
            raise FloatingPointError("non-finite candidate identity loss input")
    if bool(torch.any(candidate_scores < 0.0).item()):
        raise RuntimeError("candidate scores must be non-negative")
    target_mask = candidate_identity_delta.abs() > 0.0
    if bool(torch.any(target_mask & (candidate_scores <= 0.0)).item()):
        raise RuntimeError(
            "candidate-only outcome delta requires a strictly positive "
            "selection weight"
        )
    if bool(torch.any(target_mask & (candidate_valid_mask <= 0.0)).item()):
        raise RuntimeError(
            "candidate-only outcome delta targets an invalid candidate"
        )
    if bool(torch.any(target_mask & (transition_mask <= 0.0)).item()):
        raise RuntimeError(
            "candidate-only outcome delta targets a padded or invalid "
            "transition"
        )
    if bool(torch.any(
            (candidate_valid_mask > 0.0)
            & (transition_mask > 0.0)
            & (candidate_scores <= 0.0)
    ).item()):
        raise RuntimeError(
            "valid conditional candidate set contains a non-positive score"
        )
    valid_transition_count = transition_mask.sum()
    if (
            bool(torch.any(target_mask).item())
            and not bool((valid_transition_count > 0.0).item())):
        raise RuntimeError(
            "candidate-only outcome delta has no valid transition denominator"
        )
    effective_valid = (
        (candidate_valid_mask > 0.0)
        & (transition_mask > 0.0)
    )
    target_transition = target_mask.any(dim=1)
    valid_candidate_count = effective_valid.long().sum(dim=1)
    if bool(torch.any(target_transition & (valid_candidate_count <= 0)).item()):
        raise RuntimeError(
            "candidate-only outcome target has no valid conditional "
            "candidate set"
        )

    # Evaluate the conditional log probability in log space.  Invalid entries
    # receive a finite sentinel only inside logsumexp and are zeroed before any
    # objective or diagnostic reduction.  Valid scores are strictly positive
    # by construction; target scores were checked explicitly above.
    safe_valid_score = torch.where(
        effective_valid,
        candidate_scores,
        torch.ones_like(candidate_scores),
    )
    log_selection_weight = torch.log(safe_valid_score)
    masked_log_weight = torch.where(
        effective_valid,
        log_selection_weight,
        torch.full_like(log_selection_weight, -1.0e30),
    )
    log_normalizer = torch.logsumexp(
        masked_log_weight,
        dim=1,
        keepdim=True,
    )
    log_selection_probability = torch.where(
        effective_valid,
        log_selection_weight - log_normalizer,
        torch.zeros_like(log_selection_weight),
    )
    selection_probability = (
        torch.exp(log_selection_probability)
        * effective_valid.to(candidate_scores.dtype)
    )
    positive_weight = candidate_identity_delta.clamp_min(0.0)
    negative_weight = (-candidate_identity_delta).clamp_min(0.0)
    positive_objective = -(positive_weight * log_selection_probability)
    negative_objective = negative_weight * log_selection_probability
    effective_mask = candidate_valid_mask * transition_mask
    denominator = valid_transition_count.clamp_min(1.0)
    positive_loss = (positive_objective * effective_mask).sum() / denominator
    negative_loss = (negative_objective * effective_mask).sum() / denominator
    loss = positive_loss + negative_loss
    positive_target_mask = (
        (candidate_identity_delta > 0.0).float() * effective_mask
    )
    negative_target_mask = (
        (candidate_identity_delta < 0.0).float() * effective_mask
    )
    return {
        "loss": loss,
        "positive_loss": positive_loss,
        "negative_loss": negative_loss,
        "target_count": (
            target_mask.float() * (effective_mask > 0.0).float()
        ).sum(),
        "valid_transition_count": valid_transition_count,
        "positive_mass": positive_weight.sum(),
        "negative_mass": negative_weight.sum(),
        "positive_score_mean": (
            (candidate_scores * positive_target_mask).sum()
            / positive_target_mask.sum().clamp_min(1.0)
        ),
        "negative_score_mean": (
            (candidate_scores * negative_target_mask).sum()
            / negative_target_mask.sum().clamp_min(1.0)
        ),
        "selection_probability": selection_probability,
        "log_selection_probability": log_selection_probability,
        "positive_selection_probability_mean": (
            (selection_probability * positive_target_mask).sum()
            / positive_target_mask.sum().clamp_min(1.0)
        ),
        "negative_selection_probability_mean": (
            (selection_probability * negative_target_mask).sum()
            / negative_target_mask.sum().clamp_min(1.0)
        ),
        # Retain the old bounded-score diagnostic for CSV compatibility only.
        # Definition version 2 never consumes it as a probability or loss.
        "positive_legacy_bounded_score_mean": (
            (
                (candidate_scores / (1.0 + candidate_scores))
                * positive_target_mask
            ).sum()
            / positive_target_mask.sum().clamp_min(1.0)
        ),
        "negative_legacy_bounded_score_mean": (
            (
                (candidate_scores / (1.0 + candidate_scores))
                * negative_target_mask
            ).sum()
            / negative_target_mask.sum().clamp_min(1.0)
        ),
        "positive_log_score_mean": (
            (log_selection_weight * positive_target_mask).sum()
            / positive_target_mask.sum().clamp_min(1.0)
        ),
        "negative_log_score_mean": (
            (log_selection_weight * negative_target_mask).sum()
            / negative_target_mask.sum().clamp_min(1.0)
        ),
        "positive_log_probability_mean": (
            (log_selection_probability * positive_target_mask).sum()
            / positive_target_mask.sum().clamp_min(1.0)
        ),
        "negative_log_probability_mean": (
            (log_selection_probability * negative_target_mask).sum()
            / negative_target_mask.sum().clamp_min(1.0)
        ),
    }


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
        self.use_adj_pair_triplet_complementary_credit = bool(
            getattr(
                self.args,
                "use_adj_pair_triplet_complementary_credit",
                False,
            )
        )
        self.adj_pair_pursuit_credit_coef = max(
            0.0,
            float(getattr(self.args, "adj_pair_pursuit_credit_coef", 0.0)),
        )
        self.adj_pair_pursuit_credit_window = max(
            1,
            int(getattr(self.args, "adj_pair_pursuit_credit_window", 20)),
        )
        self.adj_pair_pursuit_credit_cap = max(
            0.0,
            float(getattr(self.args, "adj_pair_pursuit_credit_cap", 0.20)),
        )
        self.adj_pair_pursuit_credit_min_reward = float(
            getattr(self.args, "adj_pair_pursuit_credit_min_reward", 0.0)
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

        pair_transition_delay_batch = None
        capture_counts_batch = None
        capture_matched_count_batch = None
        positive_reward_without_capture_batch = None
        offset0_candidate_count_batch = None
        positive_reward_step_batch = None
        previous_adj_batch = None
        capture_to_win_episode_success_gate_batch = None
        failed_episode_capture_count_batch = None
        capture_outcome_diagnostics_batch = None
        candidate_identity_delta_batch = None
        candidate_behavior_batch = None
        if len(batch) >= 27:
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
                pair_pursuit_credit_batch, \
                pair_pursuit_quality_batch, \
                pair_to_triplet_transition_score_batch, \
                triplet_capture_quality_batch, \
                rnn_obs_batch, \
                pair_transition_delay_batch, \
                capture_counts_batch, \
                capture_matched_count_batch, \
                positive_reward_without_capture_batch, \
                offset0_candidate_count_batch, \
                positive_reward_step_batch, \
                previous_adj_batch = batch[:27]
            if len(batch) >= 29:
                capture_to_win_episode_success_gate_batch, \
                    failed_episode_capture_count_batch = batch[27:29]
            if len(batch) >= 30:
                capture_outcome_diagnostics_batch = batch[29]
            if len(batch) >= 31:
                candidate_identity_delta_batch = batch[30]
            if len(batch) >= 32:
                candidate_behavior_batch = batch[31]
        elif len(batch) >= 26:
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
                pair_pursuit_credit_batch, \
                pair_pursuit_quality_batch, \
                pair_to_triplet_transition_score_batch, \
                triplet_capture_quality_batch, \
                rnn_obs_batch, \
                pair_transition_delay_batch, \
                capture_counts_batch, \
                capture_matched_count_batch, \
                positive_reward_without_capture_batch, \
                offset0_candidate_count_batch, \
                positive_reward_step_batch = batch[:26]
        elif len(batch) >= 20:
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
                pair_pursuit_credit_batch, \
                pair_pursuit_quality_batch, \
                pair_to_triplet_transition_score_batch, \
                triplet_capture_quality_batch, \
                rnn_obs_batch = batch[:20]
        elif len(batch) >= 16:
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
            pair_pursuit_credit_batch = None
            pair_pursuit_quality_batch = None
            pair_to_triplet_transition_score_batch = None
            triplet_capture_quality_batch = None
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
            pair_pursuit_credit_batch = None
            pair_pursuit_quality_batch = None
            pair_to_triplet_transition_score_batch = None
            triplet_capture_quality_batch = None
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
            pair_pursuit_credit_batch = None
            pair_pursuit_quality_batch = None
            pair_to_triplet_transition_score_batch = None
            triplet_capture_quality_batch = None
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
            pair_pursuit_credit_batch = None
            pair_pursuit_quality_batch = None
            pair_to_triplet_transition_score_batch = None
            triplet_capture_quality_batch = None
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
            pair_pursuit_credit_batch = None
            pair_pursuit_quality_batch = None
            pair_to_triplet_transition_score_batch = None
            triplet_capture_quality_batch = None
        tarprob_adj = []
        adj_entropy = []
        adj = to_torch(adj_batch)  # [batch_size,step,num_agent,-1]
        batch_size = adj.shape[0] * adj.shape[1]
        adj = adj.reshape(batch_size, self.num_agents, -1).to(**self.tpdv)
        dones = to_torch(dones_batch).reshape(batch_size, self.num_agents, -1).to(self.device)
        dones_env = to_torch(dones_env_batch).reshape(batch_size, -1).to(**self.tpdv)
        prob_adj = to_torch(prob_adj_batch).reshape(batch_size, self.num_agents, -1).to(**self.tpdv)
        previous_adj = None
        if previous_adj_batch is not None:
            previous_adj = (
                to_torch(previous_adj_batch)
                .reshape(batch_size, self.num_agents, -1)
                .to(**self.tpdv)
            )
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
        if pair_pursuit_credit_batch is None:
            pair_pursuit_credit = torch.zeros_like(f_advts)
        else:
            pair_pursuit_credit = (
                to_torch(pair_pursuit_credit_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if pair_pursuit_quality_batch is None:
            pair_pursuit_quality = torch.zeros_like(f_advts)
        else:
            pair_pursuit_quality = (
                to_torch(pair_pursuit_quality_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if pair_to_triplet_transition_score_batch is None:
            pair_to_triplet_transition_score = torch.zeros_like(f_advts)
        else:
            pair_to_triplet_transition_score = (
                to_torch(pair_to_triplet_transition_score_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        if triplet_capture_quality_batch is None:
            triplet_capture_quality = torch.zeros_like(f_advts)
        else:
            triplet_capture_quality = (
                to_torch(triplet_capture_quality_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )
        num_candidate_factor = (
            self.num_agents * (self.num_agents - 1) // 2
            + self.num_agents * (self.num_agents - 1)
            * (self.num_agents - 2) // 6
        )
        if candidate_identity_delta_batch is None:
            candidate_identity_delta = torch.zeros(
                batch_size,
                num_candidate_factor,
                1,
                device=self.device,
                dtype=f_advts.dtype,
            )
        else:
            candidate_identity_delta = (
                to_torch(candidate_identity_delta_batch)
                .reshape(batch_size, num_candidate_factor, -1)
                .to(**self.tpdv)
            )
        if candidate_behavior_batch is None:
            candidate_behavior = torch.zeros(
                batch_size,
                num_candidate_factor,
                4,
                device=self.device,
                dtype=f_advts.dtype,
            )
        else:
            candidate_behavior = (
                to_torch(candidate_behavior_batch)
                .reshape(batch_size, num_candidate_factor, 4)
                .to(**self.tpdv)
            )
        if not bool(torch.isfinite(candidate_behavior).all().item()):
            raise FloatingPointError(
                "non-finite rollout candidate behavior metadata"
            )
        if pair_transition_delay_batch is None:
            pair_transition_delay = torch.zeros_like(f_advts)
        else:
            pair_transition_delay = (
                to_torch(pair_transition_delay_batch)
                .reshape(batch_size, self.num_factor, -1)
                .to(**self.tpdv)
            )

        def _graph_diagnostic_tensor(values):
            if values is None:
                return torch.zeros_like(dones_env)
            return (
                to_torch(values)
                .reshape(batch_size, -1)
                .to(**self.tpdv)
            )

        capture_counts = _graph_diagnostic_tensor(capture_counts_batch)
        capture_matched_count = _graph_diagnostic_tensor(
            capture_matched_count_batch
        )
        positive_reward_without_capture = _graph_diagnostic_tensor(
            positive_reward_without_capture_batch
        )
        offset0_candidate_count = _graph_diagnostic_tensor(
            offset0_candidate_count_batch
        )
        positive_reward_step = _graph_diagnostic_tensor(
            positive_reward_step_batch
        )
        capture_to_win_episode_success_gate = _graph_diagnostic_tensor(
            capture_to_win_episode_success_gate_batch
        )
        failed_episode_capture_count = _graph_diagnostic_tensor(
            failed_episode_capture_count_batch
        )
        if capture_outcome_diagnostics_batch is None:
            capture_outcome_diagnostics = torch.zeros(
                (batch_size, CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH),
                device=self.device,
                dtype=f_advts.dtype,
            )
        else:
            capture_outcome_diagnostics = (
                to_torch(capture_outcome_diagnostics_batch)
                .reshape(batch_size, -1)
                .to(**self.tpdv)
            )
            if (
                capture_outcome_diagnostics.shape[-1]
                != CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH
            ):
                raise RuntimeError(
                    "capture outcome diagnostics must have width {}, got {}"
                    .format(
                        CAPTURE_OUTCOME_DIAGNOSTIC_WIDTH,
                        capture_outcome_diagnostics.shape[-1],
                    )
                )
            if not bool(torch.isfinite(capture_outcome_diagnostics).all().item()):
                raise FloatingPointError(
                    "non-finite capture outcome diagnostics in adjacency batch"
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
                adj=adj,
                previous_adj=previous_adj,
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
        candidate_identity_scores, candidate_identity_valid_mask = (
            self.adj_network.evaluate_candidate_identity_scores(
                rnn_obs=rnn_obs,
                dones=dones.bool(),
            )
        )
        if candidate_identity_scores.shape != candidate_identity_delta[..., 0].shape:
            raise RuntimeError(
                "candidate identity replay/catalog shape mismatch: {} vs {}"
                .format(
                    tuple(candidate_identity_delta[..., 0].shape),
                    tuple(candidate_identity_scores.shape),
                )
            )
        candidate_identity_current_rank = (
            self.adj_network.canonical_candidate_ranks(
                candidate_identity_scores,
                candidate_identity_valid_mask,
            ).to(candidate_identity_scores.dtype)
        )

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
        capture_factor_order_mask = (
            ((factor_order_2d == 2.0) | (factor_order_2d == 3.0))
            & (factor_training_mask > 0.0)
        )
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
        # Outcome credit has an exact capture-factor identity.  Keep it out of
        # AdjBuffer's shared f_advt tensor and inject it only after the graph
        # mean has been removed.  Otherwise one credited capture factor changes the
        # graph mean and silently broadcasts the opposite residual to every
        # unrelated pair/triplet in the same transition.
        capture_outcome_local_delta = (
            capture_to_win_triplet_credit.squeeze(-1)
            * factor_training_mask
        )
        if bool(torch.any(
                (capture_outcome_local_delta.abs() > 0.0)
                & ~capture_factor_order_mask
        ).item()):
            raise RuntimeError(
                "capture outcome credit leaked into a non-pair/non-triplet "
                "or invalid factor"
            )
        # Keep the capture-outcome objective separate from generic order credit.
        # Its event/episode mass is already normalized and its identity is exact;
        # mixing it into the shared residual makes later order weighting and the
        # all-factor loss denominator change that mass.
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
        base_local_factor_advantage = (
            credit_local_factor_advantage
            * order_credit_weight
        )
        capture_outcome_weighted_delta = capture_outcome_local_delta
        local_factor_advantage = (
            base_local_factor_advantage
            + capture_outcome_weighted_delta
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
        base_factor_surr1 = (
            factor_imp_weights * base_local_factor_advantage
        )
        base_factor_surr2 = (
            clipped_factor_imp_weights * base_local_factor_advantage
        )
        base_factor_min_surr = torch.min(
            base_factor_surr1,
            base_factor_surr2,
        )
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
        base_factor_rl_loss = -(
            base_factor_min_surr
            * factor_loss_mask
        ).sum() / factor_loss_mask.sum().clamp_min(1.0)
        capture_outcome_loss_info = compute_capture_outcome_factor_ppo_loss(
            factor_imp_weights=factor_imp_weights,
            clipped_factor_imp_weights=clipped_factor_imp_weights,
            capture_outcome_local_delta=capture_outcome_weighted_delta,
            factor_loss_mask=factor_loss_mask,
            transition_mask=transition_mask,
        )
        capture_outcome_factor_loss_contribution = (
            capture_outcome_loss_info["loss"]
        )
        capture_outcome_positive_factor_loss_contribution = (
            capture_outcome_loss_info["positive_loss"]
        )
        capture_outcome_negative_factor_loss_contribution = (
            capture_outcome_loss_info["negative_loss"]
        )
        capture_outcome_factor_loss_target_count = (
            capture_outcome_loss_info["target_count"]
        )
        capture_outcome_factor_loss_valid_transition_count = (
            capture_outcome_loss_info["valid_transition_count"]
        )
        capture_outcome_factor_loss_factors_per_transition = (
            capture_outcome_loss_info["factors_per_transition"]
        )
        capture_outcome_factor_loss_normalization_version = 2.0
        candidate_identity_loss_info = compute_capture_candidate_identity_loss(
            candidate_scores=candidate_identity_scores,
            candidate_identity_delta=candidate_identity_delta[..., 0],
            candidate_valid_mask=candidate_identity_valid_mask,
            transition_mask=transition_mask,
        )
        candidate_target_mask = candidate_identity_delta[..., 0].abs() > 0.0
        behavior_candidate_score = candidate_behavior[..., 0]
        behavior_candidate_rank = candidate_behavior[..., 1]
        behavior_candidate_valid = candidate_behavior[..., 2]
        behavior_candidate_version = candidate_behavior[..., 3]
        if bool(torch.any(
                candidate_target_mask & (behavior_candidate_score <= 0.0)
        ).item()):
            raise RuntimeError(
                "candidate-only outcome target has no positive rollout "
                "selection weight"
            )
        if bool(torch.any(candidate_target_mask & (behavior_candidate_valid <= 0.0)).item()):
            raise RuntimeError(
                "candidate-only outcome target was invalid under the rollout "
                "candidate mask"
            )
        if bool(torch.any(candidate_target_mask & (behavior_candidate_rank < 1.0)).item()):
            raise RuntimeError(
                "candidate-only outcome target has no rollout canonical rank"
            )
        if not bool(torch.all(
                behavior_candidate_version
                == torch.round(behavior_candidate_version)
        ).item()):
            raise RuntimeError("candidate graph-policy version must be integral")
        current_candidate_policy_version = int(getattr(
            self.adj_network,
            "candidate_policy_version",
            0,
        ))
        candidate_policy_age = (
            float(current_candidate_policy_version)
            - behavior_candidate_version
        )
        if bool(torch.any(candidate_target_mask & (candidate_policy_age < 0.0)).item()):
            raise RuntimeError(
                "candidate replay metadata comes from a future graph-policy "
                "version"
            )
        candidate_target_float = candidate_target_mask.float()
        candidate_positive_target = (
            candidate_identity_delta[..., 0] > 0.0
        ).float()
        candidate_negative_target = (
            candidate_identity_delta[..., 0] < 0.0
        ).float()
        candidate_target_denominator = candidate_target_float.sum().clamp_min(1.0)
        candidate_positive_denominator = candidate_positive_target.sum().clamp_min(1.0)
        candidate_negative_denominator = candidate_negative_target.sum().clamp_min(1.0)
        capture_candidate_identity_behavior_score_mean = (
            (behavior_candidate_score * candidate_target_float).sum()
            / candidate_target_denominator
        )
        capture_candidate_identity_behavior_rank_mean = (
            (behavior_candidate_rank * candidate_target_float).sum()
            / candidate_target_denominator
        )
        capture_candidate_identity_positive_behavior_score_mean = (
            (behavior_candidate_score * candidate_positive_target).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_behavior_score_mean = (
            (behavior_candidate_score * candidate_negative_target).sum()
            / candidate_negative_denominator
        )
        capture_candidate_identity_positive_behavior_rank_mean = (
            (behavior_candidate_rank * candidate_positive_target).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_behavior_rank_mean = (
            (behavior_candidate_rank * candidate_negative_target).sum()
            / candidate_negative_denominator
        )
        capture_candidate_identity_positive_current_rank_mean = (
            (candidate_identity_current_rank * candidate_positive_target).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_current_rank_mean = (
            (candidate_identity_current_rank * candidate_negative_target).sum()
            / candidate_negative_denominator
        )
        behavior_effective_valid = (
            (behavior_candidate_valid > 0.0)
            & (transition_mask > 0.0)
        )
        behavior_safe_score = torch.where(
            behavior_effective_valid,
            behavior_candidate_score,
            torch.ones_like(behavior_candidate_score),
        )
        behavior_log_score = torch.log(behavior_safe_score)
        behavior_masked_log_score = torch.where(
            behavior_effective_valid,
            behavior_log_score,
            torch.full_like(behavior_log_score, -1.0e30),
        )
        behavior_log_probability = torch.where(
            behavior_effective_valid,
            behavior_log_score - torch.logsumexp(
                behavior_masked_log_score,
                dim=1,
                keepdim=True,
            ),
            torch.zeros_like(behavior_log_score),
        )
        behavior_probability = (
            torch.exp(behavior_log_probability)
            * behavior_effective_valid.to(behavior_candidate_score.dtype)
        )
        current_probability = candidate_identity_loss_info[
            "selection_probability"
        ]
        current_log_probability = candidate_identity_loss_info[
            "log_selection_probability"
        ]
        capture_candidate_identity_positive_behavior_probability_mean = (
            (behavior_probability * candidate_positive_target).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_behavior_probability_mean = (
            (behavior_probability * candidate_negative_target).sum()
            / candidate_negative_denominator
        )
        candidate_safe_current_score = torch.where(
            candidate_target_mask,
            candidate_identity_scores,
            torch.ones_like(candidate_identity_scores),
        )
        candidate_safe_behavior_score = torch.where(
            candidate_target_mask,
            behavior_candidate_score,
            torch.ones_like(behavior_candidate_score),
        )
        candidate_log_score_change = (
            torch.log(candidate_safe_current_score)
            - torch.log(candidate_safe_behavior_score)
        )
        capture_candidate_identity_positive_log_score_change_mean = (
            (candidate_log_score_change * candidate_positive_target).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_log_score_change_mean = (
            (candidate_log_score_change * candidate_negative_target).sum()
            / candidate_negative_denominator
        )
        candidate_safe_current_log_probability = torch.where(
            candidate_target_mask,
            current_log_probability,
            torch.zeros_like(current_log_probability),
        )
        candidate_safe_behavior_log_probability = torch.where(
            candidate_target_mask,
            behavior_log_probability,
            torch.zeros_like(behavior_log_probability),
        )
        candidate_log_probability_change = (
            candidate_safe_current_log_probability
            - candidate_safe_behavior_log_probability
        )
        capture_candidate_identity_positive_log_probability_change_mean = (
            (
                candidate_log_probability_change
                * candidate_positive_target
            ).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_log_probability_change_mean = (
            (
                candidate_log_probability_change
                * candidate_negative_target
            ).sum()
            / candidate_negative_denominator
        )
        capture_candidate_identity_positive_score_improved_fraction = (
            (
                (candidate_log_score_change > 0.0).float()
                * candidate_positive_target
            ).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_score_reduced_fraction = (
            (
                (candidate_log_score_change < 0.0).float()
                * candidate_negative_target
            ).sum()
            / candidate_negative_denominator
        )
        capture_candidate_identity_positive_probability_improved_fraction = (
            (
                (candidate_log_probability_change > 0.0).float()
                * candidate_positive_target
            ).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_probability_reduced_fraction = (
            (
                (candidate_log_probability_change < 0.0).float()
                * candidate_negative_target
            ).sum()
            / candidate_negative_denominator
        )
        capture_candidate_identity_positive_rank_improved_fraction = (
            (
                (
                    candidate_identity_current_rank
                    < behavior_candidate_rank
                ).float()
                * candidate_positive_target
            ).sum()
            / candidate_positive_denominator
        )
        capture_candidate_identity_negative_rank_reduced_fraction = (
            (
                (
                    candidate_identity_current_rank
                    > behavior_candidate_rank
                ).float()
                * candidate_negative_target
            ).sum()
            / candidate_negative_denominator
        )
        capture_candidate_identity_behavior_valid_fraction = (
            (behavior_candidate_valid * candidate_target_float).sum()
            / candidate_target_denominator
        )
        capture_candidate_identity_policy_age_mean = (
            (candidate_policy_age * candidate_target_float).sum()
            / candidate_target_denominator
        )
        capture_candidate_identity_policy_age_max = torch.where(
            candidate_target_mask,
            candidate_policy_age,
            torch.zeros_like(candidate_policy_age),
        ).max()
        capture_candidate_identity_loss_contribution = (
            candidate_identity_loss_info["loss"]
        )
        capture_candidate_identity_positive_loss_contribution = (
            candidate_identity_loss_info["positive_loss"]
        )
        capture_candidate_identity_negative_loss_contribution = (
            candidate_identity_loss_info["negative_loss"]
        )
        capture_candidate_identity_target_count = (
            candidate_identity_loss_info["target_count"]
        )
        capture_candidate_identity_positive_mass = (
            candidate_identity_loss_info["positive_mass"]
        )
        capture_candidate_identity_negative_mass = (
            candidate_identity_loss_info["negative_mass"]
        )
        capture_candidate_identity_positive_score_mean = (
            candidate_identity_loss_info["positive_score_mean"]
        )
        capture_candidate_identity_negative_score_mean = (
            candidate_identity_loss_info["negative_score_mean"]
        )
        capture_candidate_identity_positive_probability_mean = (
            candidate_identity_loss_info[
                "positive_selection_probability_mean"
            ]
        )
        capture_candidate_identity_negative_probability_mean = (
            candidate_identity_loss_info[
                "negative_selection_probability_mean"
            ]
        )
        capture_candidate_identity_positive_legacy_bounded_score_mean = (
            candidate_identity_loss_info[
                "positive_legacy_bounded_score_mean"
            ]
        )
        capture_candidate_identity_negative_legacy_bounded_score_mean = (
            candidate_identity_loss_info[
                "negative_legacy_bounded_score_mean"
            ]
        )
        capture_candidate_identity_positive_log_score_mean = (
            candidate_identity_loss_info["positive_log_score_mean"]
        )
        capture_candidate_identity_negative_log_score_mean = (
            candidate_identity_loss_info["negative_log_score_mean"]
        )
        capture_candidate_identity_positive_log_probability_mean = (
            candidate_identity_loss_info["positive_log_probability_mean"]
        )
        capture_candidate_identity_negative_log_probability_mean = (
            candidate_identity_loss_info["negative_log_probability_mean"]
        )
        candidate_valid_transition_mask = (
            (candidate_identity_valid_mask > 0.0)
            & (transition_mask > 0.0)
        )
        candidate_valid_scores = candidate_identity_scores[
            candidate_valid_transition_mask
        ]
        if candidate_valid_scores.numel() > 0:
            capture_candidate_identity_valid_score_mean = (
                candidate_valid_scores.mean()
            )
            capture_candidate_identity_valid_score_min = (
                candidate_valid_scores.min()
            )
            capture_candidate_identity_valid_score_max = (
                candidate_valid_scores.max()
            )
        else:
            capture_candidate_identity_valid_score_mean = (
                candidate_identity_scores.new_tensor(0.0)
            )
            capture_candidate_identity_valid_score_min = (
                candidate_identity_scores.new_tensor(0.0)
            )
            capture_candidate_identity_valid_score_max = (
                candidate_identity_scores.new_tensor(0.0)
            )
        factor_rl_loss = (
            base_factor_rl_loss
            + capture_outcome_factor_loss_contribution
            + capture_candidate_identity_loss_contribution
        )
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
        current_candidate_policy_version = int(getattr(
            self.adj_network,
            "candidate_policy_version",
            0,
        ))
        if current_candidate_policy_version < 0:
            raise RuntimeError("candidate graph-policy version is negative")
        self.adj_network.candidate_policy_version = (
            current_candidate_policy_version + 1
        )

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
            capture_factor_order_mask_float = (
                order2_factor_mask + order3_factor_mask
            ).clamp(max=1.0)
            capture_factor_order_count = (
                capture_factor_order_mask_float.sum()
            )
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
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_abs_mean = (
                (
                    capture_to_win_triplet_credit_2d.abs()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_active_fraction = (
                (
                    (
                        capture_to_win_triplet_credit_2d.abs() > 0.0
                    ).float()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_positive_fraction = (
                (
                    (capture_to_win_triplet_credit_2d > 0.0).float()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_negative_fraction = (
                (
                    (capture_to_win_triplet_credit_2d < 0.0).float()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_positive_mass = (
                (
                    torch.clamp(
                        capture_to_win_triplet_credit_2d,
                        min=0.0,
                    )
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_triplet_credit_negative_mass = (
                (
                    torch.clamp(
                        -capture_to_win_triplet_credit_2d,
                        min=0.0,
                    )
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_outcome_local_delta_positive_mass = (
                (
                    torch.clamp(
                        capture_outcome_weighted_delta,
                        min=0.0,
                    )
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_outcome_local_delta_negative_mass = (
                (
                    torch.clamp(
                        -capture_outcome_weighted_delta,
                        min=0.0,
                    )
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_outcome_local_delta_active_fraction = (
                (
                    (capture_outcome_weighted_delta.abs() > 0.0).float()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_identity_target_mask_float = (
                (capture_to_win_triplet_credit_2d.abs() > 0.0).float()
                * capture_factor_order_mask_float
            )
            capture_identity_target_count = (
                capture_identity_target_mask_float.sum()
            )
            capture_outcome_identity_order2_delta_abs_mass = (
                capture_outcome_weighted_delta.abs() * order2_factor_mask
            ).sum()
            capture_outcome_identity_order3_delta_abs_mass = (
                capture_outcome_weighted_delta.abs() * order3_factor_mask
            ).sum()
            capture_identity_target_order2_fraction = (
                (capture_identity_target_mask_float * order2_factor_mask).sum()
                / capture_identity_target_count.clamp_min(1.0)
            )
            capture_identity_target_order3_fraction = (
                (capture_identity_target_mask_float * order3_factor_mask).sum()
                / capture_identity_target_count.clamp_min(1.0)
            )
            capture_outcome_non_target_delta_abs_max = (
                capture_outcome_weighted_delta.abs()
                * (capture_identity_target_mask_float <= 0.0).float()
                * capture_factor_order_mask_float
            ).max()
            capture_to_win_quality_gate_mean = (
                (
                    capture_to_win_quality_gate_2d
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_quality_gate_abs_mean = (
                (
                    capture_to_win_quality_gate_2d.abs()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_quality_gate_active_fraction = (
                (
                    (capture_to_win_quality_gate_2d.abs() > 0.0).float()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_quality_gate_positive_fraction = (
                (
                    (capture_to_win_quality_gate_2d > 0.0).float()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_quality_gate_negative_fraction = (
                (
                    (capture_to_win_quality_gate_2d < 0.0).float()
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_quality_gate_positive_mass = (
                (
                    torch.clamp(
                        capture_to_win_quality_gate_2d,
                        min=0.0,
                    )
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            capture_to_win_quality_gate_negative_mass = (
                (
                    torch.clamp(
                        -capture_to_win_quality_gate_2d,
                        min=0.0,
                    )
                    * capture_factor_order_mask_float
                ).sum()
                / capture_factor_order_count.clamp_min(1.0)
            )
            pair_pursuit_credit_2d = pair_pursuit_credit.squeeze(-1)
            pair_pursuit_quality_2d = pair_pursuit_quality.squeeze(-1)
            pair_to_triplet_transition_score_2d = (
                pair_to_triplet_transition_score.squeeze(-1)
            )
            triplet_capture_quality_2d = triplet_capture_quality.squeeze(-1)
            pair_pursuit_credit_mean = (
                (
                    pair_pursuit_credit_2d
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            pair_pursuit_credit_active_fraction = (
                (
                    (pair_pursuit_credit_2d.abs() > 0.0).float()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            valid_pair_credit_values = pair_pursuit_credit_2d[
                order2_factor_mask > 0.0
            ].abs()
            if valid_pair_credit_values.numel() > 0:
                pair_pursuit_credit_std = valid_pair_credit_values.std(
                    unbiased=False
                )
                pair_pursuit_credit_max = valid_pair_credit_values.max()
                pair_pursuit_credit_nonzero_count = (
                    valid_pair_credit_values > 0.0
                ).float().sum()
                nonzero_pair_credit_values = valid_pair_credit_values[
                    valid_pair_credit_values > 0.0
                ]
                pair_credit_top1_count = max(
                    1,
                    int(np.ceil(
                        float(nonzero_pair_credit_values.numel()) * 0.01
                    )),
                )
                if nonzero_pair_credit_values.numel() > 0:
                    pair_credit_top1_mass_fraction = (
                        torch.topk(
                            nonzero_pair_credit_values,
                            k=pair_credit_top1_count,
                        )[0].sum()
                        / nonzero_pair_credit_values.sum().clamp_min(1e-8)
                    )
                else:
                    pair_credit_top1_mass_fraction = torch.zeros(
                        (), device=self.device, dtype=f_advts.dtype
                    )
            else:
                pair_pursuit_credit_std = torch.zeros(
                    (), device=self.device, dtype=f_advts.dtype
                )
                pair_pursuit_credit_max = torch.zeros(
                    (), device=self.device, dtype=f_advts.dtype
                )
                pair_pursuit_credit_nonzero_count = torch.zeros(
                    (), device=self.device, dtype=f_advts.dtype
                )
                pair_credit_top1_mass_fraction = torch.zeros(
                    (), device=self.device, dtype=f_advts.dtype
                )
            pair_pursuit_quality_mean = (
                (
                    pair_pursuit_quality_2d
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            pair_pursuit_quality_active_fraction = (
                (
                    (pair_pursuit_quality_2d > 0.0).float()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            pair_to_triplet_transition_score_mean = (
                (
                    pair_to_triplet_transition_score_2d
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            pair_to_triplet_transition_active_fraction = (
                (
                    (
                        pair_to_triplet_transition_score_2d > 0.0
                    ).float()
                    * order2_factor_mask
                ).sum()
                / order2_factor_count.clamp_min(1.0)
            )
            triplet_capture_quality_mean = (
                (
                    triplet_capture_quality_2d
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            triplet_capture_quality_active_fraction = (
                (
                    (triplet_capture_quality_2d > 0.0).float()
                    * order3_factor_mask
                ).sum()
                / order3_factor_count.clamp_min(1.0)
            )
            pair_transition_delay_2d = pair_transition_delay.squeeze(-1)
            pair_transition_delay_mask = (
                (pair_transition_delay_2d > 0.0).float()
                * order2_factor_mask
            )
            pair_transition_delay_count = (
                pair_transition_delay_mask.sum()
            )
            transition_delay_mean = (
                (
                    pair_transition_delay_2d
                    * pair_transition_delay_mask
                ).sum()
                / pair_transition_delay_count.clamp_min(1.0)
            )
            if bool((pair_transition_delay_count > 0.0).item()):
                valid_transition_delays = pair_transition_delay_2d[
                    pair_transition_delay_mask > 0.0
                ]
                transition_delay_min = valid_transition_delays.min()
                transition_delay_max = valid_transition_delays.max()
            else:
                transition_delay_min = torch.zeros(
                    (), device=self.device, dtype=f_advts.dtype
                )
                transition_delay_max = torch.zeros(
                    (), device=self.device, dtype=f_advts.dtype
                )

            capture_counts_safe = torch.clamp(capture_counts, min=0.0)
            capture_matched_count_nonnegative = torch.clamp(
                capture_matched_count, min=0.0
            )
            # Keep compatibility with the older server PyTorch, which does not
            # provide torch.minimum. This is elementwise min(matched, captures)
            # and also prevents a malformed diagnostic count from exceeding the
            # number of real capture events.
            capture_matched_count_safe = torch.where(
                capture_matched_count_nonnegative <= capture_counts_safe,
                capture_matched_count_nonnegative,
                capture_counts_safe,
            )
            capture_count_total = capture_counts_safe.sum()
            capture_matched_count_value = capture_matched_count_safe.sum()
            unmatched_capture_count_value = (
                capture_count_total - capture_matched_count_value
            ).clamp_min(0.0)
            capture_matched_fraction = (
                capture_matched_count_value
                / capture_count_total.clamp_min(1.0)
            )
            unmatched_capture_fraction = (
                unmatched_capture_count_value
                / capture_count_total.clamp_min(1.0)
            )
            failed_episode_capture_count_value = (
                torch.clamp(
                    failed_episode_capture_count,
                    min=0.0,
                ).sum()
            )
            failed_episode_capture_fraction = (
                failed_episode_capture_count_value
                / capture_count_total.clamp_min(1.0)
            )
            capture_to_win_capture_success_fraction = (
                (
                    capture_count_total - failed_episode_capture_count_value
                ).clamp_min(0.0)
                / capture_count_total.clamp_min(1.0)
            )
            capture_to_win_episode_success_fraction = (
                (
                    capture_to_win_episode_success_gate
                    * transition_mask.reshape(batch_size, -1)
                ).sum()
                / transition_mask.sum().clamp_min(1.0)
            )
            outcome_episode_mask = torch.clamp(
                capture_outcome_diagnostics[:, 10],
                min=0.0,
                max=1.0,
            )
            outcome_global_mask = (
                transition_mask.reshape(batch_size, -1)
                .max(dim=1)
                .values
            )
            def _outcome_episode_mean(column, mask=outcome_episode_mask):
                return (
                    capture_outcome_diagnostics[:, column] * mask
                ).sum() / mask.sum().clamp_min(1.0)

            def _outcome_global_mean(column):
                return (
                    capture_outcome_diagnostics[:, column]
                    * outcome_global_mask
                ).sum() / outcome_global_mask.sum().clamp_min(1.0)

            capture_outcome_baseline_mean = _outcome_global_mean(0)
            capture_outcome_capture_episode_count_mean = (
                _outcome_global_mean(1)
            )
            capture_outcome_success_episode_count_mean = (
                _outcome_global_mean(2)
            )
            capture_outcome_failure_episode_count_mean = (
                _outcome_global_mean(3)
            )
            capture_outcome_mixed_window_fraction = _outcome_global_mean(4)
            capture_outcome_single_success_window_fraction = (
                _outcome_global_mean(5)
            )
            capture_outcome_single_failure_window_fraction = (
                _outcome_global_mean(6)
            )
            capture_outcome_no_capture_window_fraction = (
                _outcome_global_mean(7)
            )
            sampled_capture_episode_mask = (
                outcome_episode_mask
                * (capture_outcome_diagnostics[:, 8] > 0.0).float()
            )
            capture_outcome_triplet_labels_per_episode_mean = (
                _outcome_episode_mean(
                    8,
                    mask=sampled_capture_episode_mask,
                )
            )
            if bool((sampled_capture_episode_mask.sum() > 0.0).item()):
                capture_outcome_triplet_labels_per_episode_max = (
                    capture_outcome_diagnostics[
                        sampled_capture_episode_mask > 0.0,
                        8,
                    ].max()
                )
            else:
                capture_outcome_triplet_labels_per_episode_max = torch.zeros(
                    (), device=self.device, dtype=f_advts.dtype
                )
            capture_outcome_raw_episode_advantage_mean = (
                _outcome_episode_mean(
                    9,
                    mask=sampled_capture_episode_mask,
                )
            )
            capture_outcome_raw_episode_advantage_abs_mean = (
                (
                    capture_outcome_diagnostics[:, 9].abs()
                    * sampled_capture_episode_mask
                ).sum()
                / sampled_capture_episode_mask.sum().clamp_min(1.0)
            )
            capture_outcome_window_raw_centered_mean = (
                _outcome_global_mean(11)
            )
            capture_outcome_window_expanded_gate_sum = (
                _outcome_global_mean(12)
            )
            capture_outcome_window_expanded_gate_abs_sum = (
                _outcome_global_mean(13)
            )
            capture_outcome_window_center_error_ratio = (
                capture_outcome_window_expanded_gate_sum.abs()
                / capture_outcome_window_expanded_gate_abs_sum.clamp_min(1e-8)
            )
            capture_to_win_credit_preclip_mean = _outcome_global_mean(14)
            capture_to_win_credit_preclip_std = _outcome_global_mean(15)
            capture_to_win_credit_preclip_max = _outcome_global_mean(16)
            capture_to_win_credit_preclip_min = _outcome_global_mean(17)
            capture_to_win_credit_positive_clip_fraction = (
                _outcome_global_mean(18)
            )
            capture_to_win_credit_negative_clip_fraction = (
                _outcome_global_mean(19)
            )
            capture_identity_event_count_mean = _outcome_episode_mean(20)
            capture_identity_matched_event_count_mean = (
                _outcome_episode_mean(21)
            )
            capture_identity_unmatched_event_count_mean = (
                _outcome_episode_mean(22)
            )
            capture_identity_candidate_factor_count_mean = (
                _outcome_episode_mean(23)
            )
            capture_identity_match_fraction = (
                capture_identity_matched_event_count_mean
                / capture_identity_event_count_mean.clamp_min(1e-8)
            )
            capture_identity_candidates_per_matched_event = (
                capture_identity_candidate_factor_count_mean
                / capture_identity_matched_event_count_mean.clamp_min(1e-8)
            )
            capture_outcome_label_gate_correlation = _outcome_global_mean(24)
            capture_outcome_label_gate_correlation_valid = (
                _outcome_global_mean(25)
            )
            capture_outcome_success_labels_mean = _outcome_global_mean(26)
            capture_outcome_failure_labels_mean = _outcome_global_mean(27)
            capture_outcome_success_gate_total_mean = _outcome_global_mean(28)
            capture_outcome_failure_gate_total_mean = _outcome_global_mean(29)
            positive_reward_step_count_value = positive_reward_step.sum()
            positive_reward_without_capture_count_value = (
                positive_reward_without_capture.sum()
            )
            positive_reward_without_capture_fraction = (
                positive_reward_without_capture_count_value
                / positive_reward_step_count_value.clamp_min(1.0)
            )
            positive_reward_step_fraction = (
                positive_reward_step_count_value
                / transition_mask.sum().clamp_min(1.0)
            )
            offset0_candidate_count_value = (
                offset0_candidate_count.sum()
            )
            offset0_candidate_fraction = (
                (
                    (offset0_candidate_count > 0.0).float()
                    * transition_mask.reshape(batch_size, -1)
                ).sum()
                / transition_mask.sum().clamp_min(1.0)
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
        train_info['capture_to_win_triplet_credit_abs_mean'] = _to_float(
            capture_to_win_triplet_credit_abs_mean
        )
        train_info['capture_to_win_triplet_credit_active_fraction'] = _to_float(
            capture_to_win_triplet_credit_active_fraction
        )
        train_info['capture_to_win_triplet_credit_positive_fraction'] = _to_float(
            capture_to_win_triplet_credit_positive_fraction
        )
        train_info['capture_to_win_triplet_credit_negative_fraction'] = _to_float(
            capture_to_win_triplet_credit_negative_fraction
        )
        train_info['capture_to_win_triplet_credit_positive_mass'] = _to_float(
            capture_to_win_triplet_credit_positive_mass
        )
        train_info['capture_to_win_triplet_credit_negative_mass'] = _to_float(
            capture_to_win_triplet_credit_negative_mass
        )
        train_info['capture_outcome_local_delta_positive_mass'] = _to_float(
            capture_outcome_local_delta_positive_mass
        )
        train_info['capture_outcome_local_delta_negative_mass'] = _to_float(
            capture_outcome_local_delta_negative_mass
        )
        train_info['capture_outcome_local_delta_active_fraction'] = _to_float(
            capture_outcome_local_delta_active_fraction
        )
        train_info['capture_outcome_factor_loss_contribution'] = _to_float(
            capture_outcome_factor_loss_contribution
        )
        train_info[
            'capture_outcome_positive_factor_loss_contribution'
        ] = _to_float(capture_outcome_positive_factor_loss_contribution)
        train_info[
            'capture_outcome_negative_factor_loss_contribution'
        ] = _to_float(capture_outcome_negative_factor_loss_contribution)
        train_info[
            'capture_outcome_factor_loss_target_count'
        ] = _to_float(capture_outcome_factor_loss_target_count)
        train_info[
            'capture_outcome_factor_loss_valid_transition_count'
        ] = _to_float(capture_outcome_factor_loss_valid_transition_count)
        train_info[
            'capture_outcome_factor_loss_factors_per_transition'
        ] = _to_float(capture_outcome_factor_loss_factors_per_transition)
        train_info[
            'capture_outcome_factor_loss_normalization_version'
        ] = float(capture_outcome_factor_loss_normalization_version)
        train_info['capture_candidate_identity_loss_contribution'] = _to_float(
            capture_candidate_identity_loss_contribution
        )
        train_info[
            'capture_candidate_identity_positive_loss_contribution'
        ] = _to_float(
            capture_candidate_identity_positive_loss_contribution
        )
        train_info[
            'capture_candidate_identity_negative_loss_contribution'
        ] = _to_float(
            capture_candidate_identity_negative_loss_contribution
        )
        train_info['capture_candidate_identity_target_count'] = _to_float(
            capture_candidate_identity_target_count
        )
        train_info['capture_candidate_identity_positive_mass'] = _to_float(
            capture_candidate_identity_positive_mass
        )
        train_info['capture_candidate_identity_negative_mass'] = _to_float(
            capture_candidate_identity_negative_mass
        )
        train_info['capture_candidate_identity_positive_score_mean'] = _to_float(
            capture_candidate_identity_positive_score_mean
        )
        train_info['capture_candidate_identity_negative_score_mean'] = _to_float(
            capture_candidate_identity_negative_score_mean
        )
        train_info[
            'capture_candidate_identity_positive_probability_mean'
        ] = _to_float(capture_candidate_identity_positive_probability_mean)
        train_info[
            'capture_candidate_identity_negative_probability_mean'
        ] = _to_float(capture_candidate_identity_negative_probability_mean)
        train_info[
            'capture_candidate_identity_positive_legacy_bounded_score_mean'
        ] = _to_float(
            capture_candidate_identity_positive_legacy_bounded_score_mean
        )
        train_info[
            'capture_candidate_identity_negative_legacy_bounded_score_mean'
        ] = _to_float(
            capture_candidate_identity_negative_legacy_bounded_score_mean
        )
        # Version 3 uses the conditional probability over the current valid
        # canonical candidate catalog.  The legacy s/(1+s) diagnostic now has
        # an explicit field name and is never consumed by the objective.
        train_info['capture_candidate_identity_loss_definition_version'] = 3.0
        train_info['capture_candidate_identity_score_semantics_version'] = 3.0
        train_info[
            'capture_candidate_identity_positive_log_score_mean'
        ] = _to_float(capture_candidate_identity_positive_log_score_mean)
        train_info[
            'capture_candidate_identity_negative_log_score_mean'
        ] = _to_float(capture_candidate_identity_negative_log_score_mean)
        train_info[
            'capture_candidate_identity_positive_log_probability_mean'
        ] = _to_float(
            capture_candidate_identity_positive_log_probability_mean
        )
        train_info[
            'capture_candidate_identity_negative_log_probability_mean'
        ] = _to_float(
            capture_candidate_identity_negative_log_probability_mean
        )
        train_info[
            'capture_candidate_identity_valid_score_mean'
        ] = _to_float(capture_candidate_identity_valid_score_mean)
        train_info[
            'capture_candidate_identity_valid_score_min'
        ] = _to_float(capture_candidate_identity_valid_score_min)
        train_info[
            'capture_candidate_identity_valid_score_max'
        ] = _to_float(capture_candidate_identity_valid_score_max)
        train_info['capture_candidate_identity_behavior_score_mean'] = _to_float(
            capture_candidate_identity_behavior_score_mean
        )
        train_info['capture_candidate_identity_behavior_rank_mean'] = _to_float(
            capture_candidate_identity_behavior_rank_mean
        )
        train_info[
            'capture_candidate_identity_positive_behavior_score_mean'
        ] = _to_float(
            capture_candidate_identity_positive_behavior_score_mean
        )
        train_info[
            'capture_candidate_identity_negative_behavior_score_mean'
        ] = _to_float(
            capture_candidate_identity_negative_behavior_score_mean
        )
        train_info[
            'capture_candidate_identity_positive_behavior_probability_mean'
        ] = _to_float(
            capture_candidate_identity_positive_behavior_probability_mean
        )
        train_info[
            'capture_candidate_identity_negative_behavior_probability_mean'
        ] = _to_float(
            capture_candidate_identity_negative_behavior_probability_mean
        )
        train_info[
            'capture_candidate_identity_positive_behavior_rank_mean'
        ] = _to_float(
            capture_candidate_identity_positive_behavior_rank_mean
        )
        train_info[
            'capture_candidate_identity_negative_behavior_rank_mean'
        ] = _to_float(
            capture_candidate_identity_negative_behavior_rank_mean
        )
        train_info[
            'capture_candidate_identity_positive_current_rank_mean'
        ] = _to_float(
            capture_candidate_identity_positive_current_rank_mean
        )
        train_info[
            'capture_candidate_identity_negative_current_rank_mean'
        ] = _to_float(
            capture_candidate_identity_negative_current_rank_mean
        )
        train_info[
            'capture_candidate_identity_positive_log_score_change_mean'
        ] = _to_float(
            capture_candidate_identity_positive_log_score_change_mean
        )
        train_info[
            'capture_candidate_identity_negative_log_score_change_mean'
        ] = _to_float(
            capture_candidate_identity_negative_log_score_change_mean
        )
        train_info[
            'capture_candidate_identity_positive_log_probability_change_mean'
        ] = _to_float(
            capture_candidate_identity_positive_log_probability_change_mean
        )
        train_info[
            'capture_candidate_identity_negative_log_probability_change_mean'
        ] = _to_float(
            capture_candidate_identity_negative_log_probability_change_mean
        )
        train_info[
            'capture_candidate_identity_positive_score_improved_fraction'
        ] = _to_float(
            capture_candidate_identity_positive_score_improved_fraction
        )
        train_info[
            'capture_candidate_identity_negative_score_reduced_fraction'
        ] = _to_float(
            capture_candidate_identity_negative_score_reduced_fraction
        )
        train_info[
            'capture_candidate_identity_positive_probability_improved_fraction'
        ] = _to_float(
            capture_candidate_identity_positive_probability_improved_fraction
        )
        train_info[
            'capture_candidate_identity_negative_probability_reduced_fraction'
        ] = _to_float(
            capture_candidate_identity_negative_probability_reduced_fraction
        )
        train_info[
            'capture_candidate_identity_positive_rank_improved_fraction'
        ] = _to_float(
            capture_candidate_identity_positive_rank_improved_fraction
        )
        train_info[
            'capture_candidate_identity_negative_rank_reduced_fraction'
        ] = _to_float(
            capture_candidate_identity_negative_rank_reduced_fraction
        )
        train_info[
            'capture_candidate_identity_behavior_valid_fraction'
        ] = _to_float(capture_candidate_identity_behavior_valid_fraction)
        train_info['capture_candidate_identity_policy_age_mean'] = _to_float(
            capture_candidate_identity_policy_age_mean
        )
        train_info['capture_candidate_identity_policy_age_max'] = _to_float(
            capture_candidate_identity_policy_age_max
        )
        train_info['capture_candidate_identity_policy_version'] = float(
            current_candidate_policy_version
        )
        train_info['capture_identity_factor_credit_active_fraction'] = _to_float(
            capture_to_win_triplet_credit_active_fraction
        )
        train_info['capture_identity_factor_credit_positive_mass'] = _to_float(
            capture_to_win_triplet_credit_positive_mass
        )
        train_info['capture_identity_factor_credit_negative_mass'] = _to_float(
            capture_to_win_triplet_credit_negative_mass
        )
        train_info['capture_outcome_identity_order2_delta_abs_mass'] = _to_float(
            capture_outcome_identity_order2_delta_abs_mass
        )
        train_info['capture_outcome_identity_order3_delta_abs_mass'] = _to_float(
            capture_outcome_identity_order3_delta_abs_mass
        )
        train_info['capture_identity_target_order2_fraction'] = _to_float(
            capture_identity_target_order2_fraction
        )
        train_info['capture_identity_target_order3_fraction'] = _to_float(
            capture_identity_target_order3_fraction
        )
        train_info['capture_outcome_non_target_delta_abs_max'] = _to_float(
            capture_outcome_non_target_delta_abs_max
        )
        train_info['capture_to_win_quality_gate_mean'] = _to_float(
            capture_to_win_quality_gate_mean
        )
        train_info['capture_to_win_quality_gate_abs_mean'] = _to_float(
            capture_to_win_quality_gate_abs_mean
        )
        train_info['capture_to_win_quality_gate_active_fraction'] = _to_float(
            capture_to_win_quality_gate_active_fraction
        )
        train_info['capture_to_win_quality_gate_positive_fraction'] = _to_float(
            capture_to_win_quality_gate_positive_fraction
        )
        train_info['capture_to_win_quality_gate_negative_fraction'] = _to_float(
            capture_to_win_quality_gate_negative_fraction
        )
        train_info['capture_to_win_quality_gate_positive_mass'] = _to_float(
            capture_to_win_quality_gate_positive_mass
        )
        train_info['capture_to_win_quality_gate_negative_mass'] = _to_float(
            capture_to_win_quality_gate_negative_mass
        )
        train_info['capture_to_win_outcome_contrastive'] = 1.0
        train_info['capture_to_win_quality_gate_definition_version'] = 5.0
        train_info['capture_outcome_baseline_mean'] = _to_float(
            capture_outcome_baseline_mean
        )
        train_info['capture_outcome_capture_episode_count_mean'] = _to_float(
            capture_outcome_capture_episode_count_mean
        )
        train_info['capture_outcome_success_episode_count_mean'] = _to_float(
            capture_outcome_success_episode_count_mean
        )
        train_info['capture_outcome_failure_episode_count_mean'] = _to_float(
            capture_outcome_failure_episode_count_mean
        )
        train_info['capture_outcome_mixed_window_fraction'] = _to_float(
            capture_outcome_mixed_window_fraction
        )
        train_info['capture_outcome_single_success_window_fraction'] = _to_float(
            capture_outcome_single_success_window_fraction
        )
        train_info['capture_outcome_single_failure_window_fraction'] = _to_float(
            capture_outcome_single_failure_window_fraction
        )
        train_info['capture_outcome_no_capture_window_fraction'] = _to_float(
            capture_outcome_no_capture_window_fraction
        )
        train_info['capture_outcome_triplet_labels_per_episode_mean'] = _to_float(
            capture_outcome_triplet_labels_per_episode_mean
        )
        train_info['capture_outcome_triplet_labels_per_episode_max'] = _to_float(
            capture_outcome_triplet_labels_per_episode_max
        )
        train_info['capture_outcome_raw_episode_advantage_mean'] = _to_float(
            capture_outcome_raw_episode_advantage_mean
        )
        train_info['capture_outcome_raw_episode_advantage_abs_mean'] = _to_float(
            capture_outcome_raw_episode_advantage_abs_mean
        )
        train_info['capture_outcome_window_raw_centered_mean'] = _to_float(
            capture_outcome_window_raw_centered_mean
        )
        train_info['capture_outcome_window_expanded_gate_sum'] = _to_float(
            capture_outcome_window_expanded_gate_sum
        )
        train_info['capture_outcome_window_expanded_gate_abs_sum'] = _to_float(
            capture_outcome_window_expanded_gate_abs_sum
        )
        train_info['capture_outcome_window_center_error_ratio'] = _to_float(
            capture_outcome_window_center_error_ratio
        )
        train_info['capture_to_win_credit_preclip_mean'] = _to_float(
            capture_to_win_credit_preclip_mean
        )
        train_info['capture_to_win_credit_preclip_std'] = _to_float(
            capture_to_win_credit_preclip_std
        )
        train_info['capture_to_win_credit_preclip_max'] = _to_float(
            capture_to_win_credit_preclip_max
        )
        train_info['capture_to_win_credit_preclip_min'] = _to_float(
            capture_to_win_credit_preclip_min
        )
        train_info['capture_to_win_credit_positive_clip_fraction'] = _to_float(
            capture_to_win_credit_positive_clip_fraction
        )
        train_info['capture_to_win_credit_negative_clip_fraction'] = _to_float(
            capture_to_win_credit_negative_clip_fraction
        )
        train_info['capture_identity_event_count_mean'] = _to_float(
            capture_identity_event_count_mean
        )
        train_info['capture_identity_matched_event_count_mean'] = _to_float(
            capture_identity_matched_event_count_mean
        )
        train_info['capture_identity_unmatched_event_count_mean'] = _to_float(
            capture_identity_unmatched_event_count_mean
        )
        train_info['capture_identity_candidate_factor_count_mean'] = _to_float(
            capture_identity_candidate_factor_count_mean
        )
        train_info['capture_identity_match_fraction'] = _to_float(
            capture_identity_match_fraction
        )
        train_info['capture_identity_candidates_per_matched_event'] = _to_float(
            capture_identity_candidates_per_matched_event
        )
        train_info['capture_outcome_label_gate_correlation'] = _to_float(
            capture_outcome_label_gate_correlation
        )
        train_info['capture_outcome_label_gate_correlation_valid'] = _to_float(
            capture_outcome_label_gate_correlation_valid
        )
        train_info['capture_outcome_success_labels_mean'] = _to_float(
            capture_outcome_success_labels_mean
        )
        train_info['capture_outcome_failure_labels_mean'] = _to_float(
            capture_outcome_failure_labels_mean
        )
        train_info['capture_outcome_success_gate_total_mean'] = _to_float(
            capture_outcome_success_gate_total_mean
        )
        train_info['capture_outcome_failure_gate_total_mean'] = _to_float(
            capture_outcome_failure_gate_total_mean
        )
        train_info['pair_pursuit_credit_mean'] = _to_float(
            pair_pursuit_credit_mean
        )
        train_info['pair_pursuit_credit_active_fraction'] = _to_float(
            pair_pursuit_credit_active_fraction
        )
        train_info['pair_credit_active_fraction'] = _to_float(
            pair_pursuit_credit_active_fraction
        )
        train_info['pair_pursuit_credit_std'] = _to_float(
            pair_pursuit_credit_std
        )
        train_info['pair_pursuit_credit_max'] = _to_float(
            pair_pursuit_credit_max
        )
        train_info['pair_pursuit_credit_nonzero_count'] = _to_float(
            pair_pursuit_credit_nonzero_count
        )
        train_info['pair_credit_top1_mass_fraction'] = _to_float(
            pair_credit_top1_mass_fraction
        )
        train_info['pair_pursuit_quality_mean'] = _to_float(
            pair_pursuit_quality_mean
        )
        train_info['pair_pursuit_quality_active_fraction'] = _to_float(
            pair_pursuit_quality_active_fraction
        )
        train_info['pair_to_triplet_transition_score_mean'] = _to_float(
            pair_to_triplet_transition_score_mean
        )
        train_info[
            'pair_to_triplet_transition_active_fraction'
        ] = _to_float(pair_to_triplet_transition_active_fraction)
        train_info['triplet_capture_quality_mean'] = _to_float(
            triplet_capture_quality_mean
        )
        train_info['triplet_capture_quality_active_fraction'] = _to_float(
            triplet_capture_quality_active_fraction
        )
        train_info['transition_delay_mean'] = _to_float(
            transition_delay_mean
        )
        train_info['transition_delay_min'] = _to_float(
            transition_delay_min
        )
        train_info['transition_delay_max'] = _to_float(
            transition_delay_max
        )
        train_info['capture_event_count'] = _to_float(capture_count_total)
        train_info['capture_matched_count'] = _to_float(
            capture_matched_count_value
        )
        train_info['unmatched_capture_count'] = _to_float(
            unmatched_capture_count_value
        )
        train_info['capture_matched_fraction'] = _to_float(
            capture_matched_fraction
        )
        train_info['unmatched_capture_fraction'] = _to_float(
            unmatched_capture_fraction
        )
        train_info['failed_episode_capture_count'] = _to_float(
            failed_episode_capture_count_value
        )
        train_info['failed_episode_capture_fraction'] = _to_float(
            failed_episode_capture_fraction
        )
        train_info['capture_to_win_capture_success_fraction'] = _to_float(
            capture_to_win_capture_success_fraction
        )
        train_info['capture_to_win_episode_success_fraction'] = _to_float(
            capture_to_win_episode_success_fraction
        )
        train_info['positive_reward_without_capture_fraction'] = _to_float(
            positive_reward_without_capture_fraction
        )
        train_info['positive_reward_step_count'] = _to_float(
            positive_reward_step_count_value
        )
        train_info['positive_reward_without_capture_count'] = _to_float(
            positive_reward_without_capture_count_value
        )
        train_info['positive_reward_step_fraction'] = _to_float(
            positive_reward_step_fraction
        )
        train_info['offset0_candidate_count'] = _to_float(
            offset0_candidate_count_value
        )
        train_info['offset0_candidate_fraction'] = _to_float(
            offset0_candidate_fraction
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
        train_info['use_adj_pair_triplet_complementary_credit'] = float(
            self.use_adj_pair_triplet_complementary_credit
        )
        train_info['adj_pair_pursuit_credit_coef'] = float(
            self.adj_pair_pursuit_credit_coef
        )
        train_info['adj_pair_pursuit_credit_window'] = float(
            self.adj_pair_pursuit_credit_window
        )
        train_info['adj_pair_pursuit_credit_cap'] = float(
            self.adj_pair_pursuit_credit_cap
        )
        train_info['adj_pair_pursuit_credit_min_reward'] = float(
            self.adj_pair_pursuit_credit_min_reward
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
