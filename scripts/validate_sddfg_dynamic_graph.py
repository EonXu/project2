#!/usr/bin/env python
"""Fast server-side checks for the SDDFG dynamic graph training path."""

from __future__ import print_function

import os
import sys
from types import SimpleNamespace

# Match the training entrypoint before importing torch.  Strict deterministic
# CUDA linear layers require this CuBLAS workspace mode on CUDA >= 10.2.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from gym.spaces import Box, Discrete

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from algorithms.sddfg.algorithm.adj_generator import Adj_Generator
from algorithms.sddfg.algorithm.agent_q_function import AgentQFunction
from algorithms.sddfg.algorithm.rSDDFGPolicy import (
    R_SDDFGPolicy,
    scatter_add,
)
from algorithms.sddfg.r_sddfg import R_SDDFG
from utils.adj_buffer import AdjPolicyBuffer


def validate_adj_buffer():
    """Exercise the exact single-episode axis case that previously crashed."""
    episode_length = 8
    num_agents = 6
    num_factor = 6
    active_counts = [4, 4, 2, 3, 5, 3, 4, 6, 6]

    for num_episodes in (1, 4):
        buffer = AdjPolicyBuffer(
            buffer_size=4,
            episode_length=episode_length,
            num_agents=num_agents,
            num_factor=num_factor,
            obs_space=Box(
                low=-1.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32,
            ),
            share_obs_space=Box(
                low=-1.0,
                high=1.0,
                shape=(8,),
                dtype=np.float32,
            ),
            act_space=Discrete(7),
            use_same_share_obs=True,
            use_avail_acts=False,
            gamma=0.97,
            adj_return_adv_coef=1.0,
            adj_factor_adv_coef=0.25,
            use_adj_delayed_triplet_credit=True,
            adj_delayed_triplet_credit_coef=0.25,
            adj_delayed_triplet_credit_window=3,
            adj_delayed_triplet_credit_positive_only=True,
            adj_delayed_triplet_credit_min_adv=0.25,
            adj_delayed_triplet_credit_require_future_match=True,
            use_adj_delayed_triplet_success_gate=True,
            adj_delayed_triplet_success_gate_min_adv=0.50,
            adj_delayed_triplet_success_gate_scale=0.75,
            adj_delayed_triplet_success_gate_floor=0.25,
            adj_delayed_triplet_future_overlap_min_nodes=2,
            adj_delayed_triplet_partial_match_weight=0.50,
            use_adj_capture_to_win_credit=True,
            adj_capture_to_win_credit_coef=0.15,
            adj_capture_to_win_credit_min_outcome_adv=0.50,
            adj_capture_to_win_credit_scale=0.75,
            adj_capture_to_win_credit_cap=0.25,
            adj_capture_to_win_credit_require_future_match=True,
            seed=11,
        )
        buffer.filled_i = num_episodes
        buffer.current_i = num_episodes % buffer.buffer_size
        buffer.obs[:, :num_episodes] = -1.0
        # AdjPolicyBuffer deliberately initializes every unused transition as
        # terminal.  A populated episode must explicitly mark its non-final
        # transitions as live, exactly as runner insertion does.
        buffer.dones_env[:, :num_episodes, 0] = 0.0

        for episode_idx in range(num_episodes):
            for step, active_count in enumerate(active_counts):
                buffer.obs[step, episode_idx, :active_count] = (
                    0.01 * float(episode_idx + step + 1)
                )

            for step in range(episode_length):
                active_count = active_counts[step]
                buffer.rewards[step, episode_idx, :, 0] = (
                    0.01 * float(episode_idx + step + 1)
                )
                factor_slot = 0
                for first in range(0, active_count, 2):
                    second = first + 1
                    if second >= active_count:
                        second = 0
                    buffer.adj[
                        step,
                        episode_idx,
                        [first, second],
                        factor_slot,
                    ] = 1
                    factor_slot += 1
                # Exercise factor-local credit even when the single-episode
                # return control variate is zero.
                buffer.f_q[
                    step,
                    episode_idx,
                    :factor_slot,
                    0,
                ] = np.arange(factor_slot, dtype=np.float32)
            buffer.dones_env[-1, episode_idx, 0] = 1.0

        buffer.compute_advantage(np.arange(num_episodes))
        computed = buffer.f_advt[:, :num_episodes]
        assert computed.shape == (
            episode_length,
            num_episodes,
            num_factor,
            1,
        )
        assert np.isfinite(computed).all()
        assert np.abs(computed).sum() > 0.0
        recent_samples = list(
            buffer.sample_inds(
                data_chunk_length=4,
                num_mini_batch=1,
                recent_episode_window=2,
            )
        )
        assert len(recent_samples) == 1
        expected_recent = min(num_episodes, 2)
        assert buffer.last_sample_episode_count == expected_recent
        assert buffer.last_sample_episode_indices.size == expected_recent
        assert recent_samples[0][0].shape[0] >= 1
        assert len(recent_samples[0]) >= 16
        assert np.isfinite(recent_samples[0][13]).all()
        assert np.isfinite(recent_samples[0][14]).all()


def validate_scatter_gradient():
    src = torch.randn(7, 5, requires_grad=True)
    index = torch.tensor([0, 1, 0, 2, 1, 2, 0], dtype=torch.long)
    output = scatter_add(src, index, dim=0, dim_size=3)
    assert tuple(output.shape) == (3, 5)
    expected = torch.zeros_like(output)
    for src_idx, dst_idx in enumerate(index.tolist()):
        expected[dst_idx] = expected[dst_idx] + src[src_idx].detach()
    assert bool(torch.allclose(output.detach(), expected, atol=1e-6))
    output.sum().backward()
    assert src.grad is not None
    assert bool(torch.isfinite(src.grad).all().item())


def validate_deterministic_cuda_scatter():
    """Exercise the non-atomic fallback used by deterministic CUDA training."""
    if not torch.cuda.is_available():
        return False

    has_strict_api = (
        hasattr(torch, "use_deterministic_algorithms")
        and hasattr(torch, "are_deterministic_algorithms_enabled")
    )
    was_enabled = (
        bool(torch.are_deterministic_algorithms_enabled())
        if has_strict_api else False
    )
    was_cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    try:
        # Old server PyTorch has no strict API.  rSDDFGPolicy deliberately
        # treats this cuDNN flag as the compatibility signal for its
        # deterministic scatter/gather implementations.
        torch.backends.cudnn.deterministic = True
        if has_strict_api:
            torch.use_deterministic_algorithms(True)
        src = torch.randn(11, 5, device="cuda", requires_grad=True)
        index = torch.tensor(
            [0, 2, 1, 2, 0, 3, 1, 3, 2, 0, 3],
            dtype=torch.long,
            device="cuda",
        )
        output = scatter_add(src, index, dim=0, dim_size=4)
        expected = torch.zeros(4, 5, dtype=src.dtype)
        for src_idx, dst_idx in enumerate(index.cpu().tolist()):
            expected[dst_idx] += src[src_idx].detach().cpu()
        assert bool(
            torch.allclose(
                output.detach().cpu(),
                expected,
                atol=1e-6,
            )
        )
        output.sum().backward()
        assert src.grad is not None
        assert bool(torch.isfinite(src.grad).all().item())
    finally:
        if has_strict_api:
            torch.use_deterministic_algorithms(was_enabled)
        torch.backends.cudnn.deterministic = was_cudnn_deterministic
    return True


def validate_factor_q_gradient():
    args = SimpleNamespace(
        use_ReLU=True,
        use_orthogonal=True,
        gain=0.01,
    )
    for order in (1, 2, 3):
        action_output = 7 if order == 1 else 3 * 7
        network = AgentQFunction(
            args=args,
            obs_dim=4,
            input_dim=64,
            num_orders=order,
            act_dim=action_output,
            device=torch.device("cpu"),
        )
        obs = torch.randn(5, order * 4)
        hidden = torch.randn(5, order * 64)
        output = network(obs, hidden, no_sequence=False)
        assert bool(torch.isfinite(output).all().item())
        output.pow(2).mean().backward()
        grads = [
            parameter.grad
            for parameter in network.parameters()
            if parameter.grad is not None
        ]
        assert grads
        assert all(bool(torch.isfinite(grad).all().item()) for grad in grads)
        assert sum(float(grad.abs().sum().item()) for grad in grads) > 0.0


def validate_trainer_without_critics():
    class MinimalPolicy(object):
        def __init__(self):
            self.act_dim = 7
            self.module = torch.nn.Linear(4, 7)

        def parameters(self):
            return list(self.module.parameters())

        def critic_fv_parameters(self):
            return []

        def critic_vtot_parameters(self):
            return []

    args = SimpleNamespace(
        use_popart=False,
        use_value_active_masks=False,
        use_per=False,
        per_eps=1e-6,
        use_huber_loss=False,
        huber_delta=10.0,
        clip_param=0.2,
        use_vfunction=False,
        lr=5e-4,
        critic_lr=5e-4,
        adj_lr=3e-4,
        tau=0.005,
        opti_eps=1e-5,
        weight_decay=0.0,
        highest_orders=3,
        use_dyn_graph=True,
        num_factor=6,
        entropy_coef=0.01,
        adj_entropy_coef=0.002,
        use_valuenorm=False,
        adj_max_grad_norm=0.5,
        adj_lr_anneal_steps=200000,
        adj_lr_decay_floor=2e-5,
        adj_entropy_coef_final=0.0,
        adj_entropy_anneal_steps=200000,
    )
    trainer = R_SDDFG(
        args=args,
        num_agents=6,
        policies={"policy_0": MinimalPolicy()},
        adj_network=torch.nn.Linear(4, 4),
        policy_mapping_fn=lambda _agent_id: "policy_0",
        device=torch.device("cpu"),
        episode_length=8,
    )
    assert trainer.critic_fv_optimizer is None
    assert trainer.critic_vtot_optimizer is None
    initial_adj_lr = trainer.adj_optimizer.param_groups[0]["lr"]
    midpoint_adj_lr = trainer.adj_lr_decay(100000)
    midpoint_entropy_coef = trainer.adj_entropy_coef
    final_adj_lr = trainer.adj_lr_decay(200000)
    final_entropy_coef = trainer.adj_entropy_coef
    assert abs(initial_adj_lr - 3e-4) < 1e-12
    assert abs(midpoint_adj_lr - 1.5e-4) < 1e-12
    assert abs(final_adj_lr - 2e-5) < 1e-12
    assert abs(midpoint_entropy_coef - 0.001) < 1e-12
    assert abs(final_entropy_coef) < 1e-12


def validate_full_policy_gradient(device=None):
    if device is None:
        device = torch.device("cpu")
    args = SimpleNamespace(
        hidden_size=64,
        lamda=0.0,
        num_rank=3,
        num_factor=6,
        prev_act_inp=False,
        highest_orders=3,
        use_vfunction=False,
        epsilon_start=1.0,
        epsilon_finish=0.05,
        epsilon_anneal_time=500000,
        use_ReLU=True,
        use_orthogonal=True,
        use_feature_normalization=True,
        gain=0.01,
        seed=13,
        msg_iterations=4,
        msg_normalized=True,
        msg_anytime=True,
    )
    num_agents = 6
    batch_size = 3
    obs_dim = 10
    act_dim = 7
    policy = R_SDDFGPolicy(
        config={
            "args": args,
            "device": device,
            "num_agents": num_agents,
        },
        policy_config={
            "obs_space": Box(
                low=-1.0,
                high=1.0,
                shape=(obs_dim,),
                dtype=np.float32,
            ),
            "act_space": Discrete(act_dim),
            "cent_obs_dim": 20,
        },
        train=True,
    )

    obs = torch.randn(
        batch_size,
        num_agents,
        obs_dim,
        device=device,
    )
    initial_hidden = torch.zeros(
        batch_size * num_agents,
        args.hidden_size,
        device=device,
    )
    _, hidden, _ = policy.get_hidden_states(
        obs.reshape(batch_size * num_agents, obs_dim),
        prev_actions=None,
        rnn_states=initial_hidden,
        dones=torch.zeros(
            batch_size * num_agents,
            1,
            device=device,
        ),
    )
    hidden = hidden.reshape(batch_size, num_agents, args.hidden_size)

    total_factors = args.num_factor + num_agents
    adj = torch.zeros(
        batch_size,
        num_agents,
        total_factors,
        dtype=torch.long,
    )
    dynamic_factors = (
        (0, 1),
        (2, 3),
        (4, 5),
        (0, 2, 4),
        (1, 3),
        (2, 4, 5),
    )
    for factor_idx, nodes in enumerate(dynamic_factors):
        adj[:, list(nodes), factor_idx] = 1
    for agent_idx in range(num_agents):
        adj[:, agent_idx, args.num_factor + agent_idx] = 1
    # Build the small static fixture on CPU.  PyTorch 1.8 on Windows has an
    # intermittent CUDA advanced-assignment assertion when two validation
    # processes share a device; the policy itself only consumes this tensor.
    adj = adj.to(device)

    action_indices = torch.randint(
        low=0,
        high=act_dim,
        size=(batch_size, num_agents, 1),
        device=device,
    )
    q_tot = policy.get_q_values(
        obs_batch=obs,
        rnn_q_states_batch=hidden,
        action_batch=action_indices,
        adj_input=adj,
        no_sequence=False,
        dones=torch.zeros(
            batch_size,
            num_agents,
            1,
            device=device,
        ),
    )
    assert tuple(q_tot.shape) == (batch_size,)
    assert bool(torch.isfinite(q_tot).all().item())
    q_tot.pow(2).mean().backward()

    rnn_grad = sum(
        float(parameter.grad.abs().sum().item())
        for parameter in policy.rnn_network.parameters()
        if parameter.grad is not None
    )
    q_grad = 0.0
    for order in range(1, args.highest_orders + 1):
        q_grad += sum(
            float(parameter.grad.abs().sum().item())
            for parameter in policy.q_network[order].parameters()
            if parameter.grad is not None
        )
    assert rnn_grad > 0.0
    assert q_grad > 0.0


def validate_deterministic_cuda_full_policy_gradient():
    if not torch.cuda.is_available():
        return False

    has_strict_api = (
        hasattr(torch, "use_deterministic_algorithms")
        and hasattr(torch, "are_deterministic_algorithms_enabled")
    )
    was_enabled = (
        bool(torch.are_deterministic_algorithms_enabled())
        if has_strict_api else False
    )
    was_cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    try:
        torch.backends.cudnn.deterministic = True
        if has_strict_api:
            torch.use_deterministic_algorithms(True)
        validate_full_policy_gradient(torch.device("cuda"))
    finally:
        if has_strict_api:
            torch.use_deterministic_algorithms(was_enabled)
        torch.backends.cudnn.deterministic = was_cudnn_deterministic
    return True


def main():
    torch.manual_seed(7)
    np.random.seed(7)

    args = SimpleNamespace(
        max_player_num=6,
        num_factor=6,
        highest_orders=3,
        sparsity=0.3,
        seed=7,
        hidden_size=64,
        gat_heads=4,
        gat_negative_slope=0.2,
        gat_hyperedge_hidden=64,
        adj_order3_bonus=1.35,
        adj_order3_bonus_start=1.0,
        adj_order3_bonus_anneal_steps=100,
        adj_sampling_temperature_start=1.0,
        adj_sampling_temperature_final=0.35,
        adj_sampling_temperature_anneal_steps=100,
        adj_min_order3_ratio_start=0.5,
        adj_min_order3_ratio_final=0.72,
        adj_min_order3_ratio_anneal_steps=100,
        adj_max_order3_ratio_start=0.82,
        adj_max_order3_ratio_final=0.82,
        adj_max_order3_ratio_anneal_steps=0,
        adj_greedy_sample_prob_start=0.0,
        adj_greedy_sample_prob_final=0.75,
        adj_greedy_sample_prob_anneal_steps=100,
        adj_greedy_sample_prob_cap=0.50,
        adj_order3_quota_mode="soft",
        adj_order3_soft_quota_coef=2.5,
        adj_triplet_feature_mode="synergy",
        adj_triplet_balance_coef=0.75,
        use_adj_advantage_triplet_scorer=True,
        adj_triplet_credit_ema_alpha=0.05,
        adj_triplet_credit_score_coef=0.50,
        adj_triplet_credit_score_scale=0.05,
        use_adj_triplet_credit_direct_rank=True,
        adj_triplet_credit_rank_coef=1.25,
        adj_triplet_credit_min_multiplier=0.70,
        adj_triplet_credit_max_multiplier=2.50,
        adj_triplet_credit_negative_rank_scale=0.25,
        adj_triplet_credit_min_positive_fraction=0.45,
        adj_triplet_negative_graph_penalty=0.50,
        adj_order3_quota_score_floor=0.45,
        adj_min_pair_ratio=0.20,
        adj_order_adv_coef=0.75,
        adj_order_adv_positive_only=True,
        adj_order_adv_negative_coef=0.20,
        adj_order_adv_require_positive_graph_adv=True,
        use_adj_triplet_graph_return_credit=True,
        adj_triplet_graph_return_credit_coef=0.25,
        adj_triplet_graph_return_credit_cap=0.35,
        adj_triplet_graph_return_credit_min_graph_adv=0.0,
        adj_triplet_graph_return_credit_raw_gate_scale=0.75,
        adj_triplet_graph_return_credit_require_delayed_gate=True,
        use_adj_delayed_triplet_credit=True,
        adj_delayed_triplet_credit_coef=0.25,
        adj_delayed_triplet_credit_window=20,
        adj_delayed_triplet_credit_cap=0.75,
        adj_delayed_triplet_credit_min_reward=0.0,
        adj_delayed_triplet_credit_positive_only=True,
        adj_delayed_triplet_credit_min_adv=0.25,
        adj_delayed_triplet_credit_require_future_match=True,
        use_adj_delayed_triplet_success_gate=True,
        adj_delayed_triplet_success_gate_min_adv=0.50,
        adj_delayed_triplet_success_gate_scale=0.75,
        adj_delayed_triplet_success_gate_floor=0.10,
        adj_delayed_triplet_future_overlap_min_nodes=2,
        adj_delayed_triplet_partial_match_weight=0.35,
        use_adj_capture_to_win_credit=True,
        adj_capture_to_win_credit_coef=0.15,
        adj_capture_to_win_credit_min_outcome_adv=0.50,
        adj_capture_to_win_credit_scale=0.75,
        adj_capture_to_win_credit_cap=0.25,
        adj_capture_to_win_credit_require_future_match=True,
        use_adj_order3_credit_gate=True,
        use_adj_order3_relative_credit_gate=True,
        adj_order3_credit_gate_loss_scale=0.004,
        adj_order3_credit_gate_margin=0.0,
        adj_order3_credit_gate_min_scale=0.70,
        adj_order3_credit_gate_ema_alpha=0.10,
        adj_order3_credit_gate_max_delta=0.05,
        adj_ppo_clip_stop_ratio=0.35,
        adj_ppo_factor_clip_stop_ratio=0.35,
        adj_ppo_min_epochs=1,
        use_adj_ppo_stale_trust=True,
        adj_ppo_stale_trust_clip=0.20,
        adj_ppo_stale_trust_scale=0.25,
        adj_ppo_stale_trust_min_weight=0.25,
        adj_recent_episode_window=4,
        use_adj_dynamic_recent_window=True,
        adj_recent_episode_window_min=1,
        adj_recent_window_stale_threshold=0.35,
        adj_recent_window_factor_stale_threshold=0.30,
        adj_recent_window_shrink_patience=1,
        adj_recent_window_recover_patience=2,
        adj_recent_window_recover_stale_threshold=0.28,
        adj_recent_window_recover_factor_stale_threshold=0.24,
        adj_recent_window_severe_margin=0.20,
        adj_recent_episode_window_emergency=1,
        adj_recent_window_emergency_stale_threshold=0.40,
        adj_recent_window_emergency_factor_stale_threshold=0.25,
        adj_hidden_dim=64,
        epsilon_start=1.0,
        epsilon_finish=0.05,
        adj_anneal_time=500000,
        require_connected_adj=True,
    )
    device = torch.device("cpu")
    graph = Adj_Generator(
        args=args,
        obs_dim=1,
        state_dim=1,
        act_dim=1,
        device=device,
    )

    active_counts = [2, 3, 4, 5, 6]
    batch_size = len(active_counts)
    rnn_obs = torch.randn(
        batch_size,
        args.max_player_num,
        args.hidden_size,
        device=device,
    )
    dones = torch.ones(
        batch_size,
        args.max_player_num,
        1,
        dtype=torch.bool,
        device=device,
    )
    for batch_idx, active_count in enumerate(active_counts):
        dones[batch_idx, :active_count] = False

    behavior_prob, adj, entropy = graph.sample(
        obs=None,
        rnn_obs=rnn_obs,
        use_adj_init=True,
        dones=dones,
        explore=True,
        t_env=500000,
    )
    assert abs(graph.current_order3_bonus - 1.35) < 1e-12
    assert abs(graph.current_sampling_temperature - 0.35) < 1e-12
    assert abs(graph.current_min_order3_ratio - 0.72) < 1e-12
    assert abs(graph.current_max_order3_ratio - 0.82) < 1e-12
    assert abs(
        graph.current_greedy_sample_prob
        - args.adj_greedy_sample_prob_cap
    ) < 1e-12
    assert abs(graph.current_order3_credit_gate - 1.0) < 1e-12
    target_prob, target_entropy = graph.evaluate_prob(
        obs=None,
        rnn_obs=rnn_obs,
        use_adj_init=True,
        dones=dones,
        adj=adj,
    )

    assert tuple(adj.shape) == (
        batch_size,
        args.max_player_num,
        args.num_factor,
    )
    for tensor in (
        behavior_prob,
        target_prob,
        entropy,
        target_entropy,
    ):
        assert bool(torch.isfinite(tensor).all().item())

    order_counts = {2: 0, 3: 0}
    for batch_idx, active_count in enumerate(active_counts):
        active = torch.zeros(args.max_player_num, dtype=torch.bool)
        active[:active_count] = True
        selected = adj[batch_idx].bool()
        assert not bool(selected[~active].any().item())
        degree = selected.long().sum(dim=1)
        assert bool((degree[active] > 0).all().item())

        # Coverage alone is insufficient for global message passing. Verify
        # that all active nodes belong to one factor-graph component.
        co_membership = (
            selected.float()
            @ selected.float().transpose(0, 1)
        ) > 0.0
        reachability = co_membership.clone()
        reachability = reachability | torch.eye(
            args.max_player_num,
            dtype=torch.bool,
        )
        for _ in range(args.max_player_num):
            reachability = reachability | (
                reachability.float() @ reachability.float() > 0.0
            )
        active_idx = torch.where(active)[0]
        assert bool(
            reachability[active_idx][:, active_idx].all().item()
        )

        orders = selected.long().sum(dim=0)
        valid_orders = orders[orders > 0]
        assert bool(
            ((valid_orders == 2) | (valid_orders == 3)).all().item()
        )
        for order in (2, 3):
            order_counts[order] += int(
                (valid_orders == order).long().sum().item()
            )
        if active_count >= 3:
            expected_triplets = int(
                np.ceil(
                    graph.current_min_order3_ratio
                    * float(valid_orders.numel())
                    - 1e-8
                )
            )
            # The lower band cannot require more triplets than exist for the
            # active roster (e.g. only one unique triplet exists for 3 agents).
            max_feasible_triplets = 0
            if active_count >= 3:
                max_feasible_triplets = int(
                    active_count
                    * (active_count - 1)
                    * (active_count - 2)
                    // 6
                )
            expected_triplets = min(
                expected_triplets,
                max_feasible_triplets,
            )
            max_triplets = int(
                np.floor(
                    graph.current_max_order3_ratio
                    * float(valid_orders.numel())
                    + 1e-8
                )
            )
            max_triplets = max(expected_triplets, max_triplets)
            actual_triplets = int(
                (valid_orders == 3).long().sum().item()
            )
            if graph.order3_quota_mode == "hard":
                assert actual_triplets >= expected_triplets
            assert actual_triplets <= max_triplets
        assert bool((target_prob[batch_idx][selected] > 0.0).all().item())

    factor_mask_for_credit = (
        adj.float().sum(dim=1) > 1.5
    ).float()
    synthetic_local_credit = torch.linspace(
        -0.2,
        0.2,
        steps=batch_size * args.num_factor,
        device=device,
    ).reshape(batch_size, args.num_factor)
    synthetic_graph_credit = torch.ones(
        batch_size,
        1,
        device=device,
    )
    credit_info = graph.update_factor_credit_memory(
        adj.float(),
        synthetic_local_credit,
        synthetic_graph_credit,
        factor_mask_for_credit,
    )
    assert credit_info["adv_triplet_credit_triplet_updates"] > 0
    assert credit_info["adv_triplet_credit_seen_ratio"] > 0.0
    assert "adv_triplet_marginal_positive_fraction" in credit_info
    graph.sample(
        obs=None,
        rnn_obs=rnn_obs,
        use_adj_init=True,
        dones=dones,
        explore=True,
        t_env=500000,
    )
    assert np.isfinite(graph.last_adv_triplet_score_multiplier_mean)
    assert graph.last_adv_triplet_score_multiplier_mean > 0.0
    assert np.isfinite(graph.last_adv_triplet_score_multiplier_min)
    assert np.isfinite(graph.last_adv_triplet_score_multiplier_max)
    assert graph.last_adv_triplet_score_multiplier_min > 0.0
    assert graph.last_adv_triplet_score_multiplier_max > 0.0
    assert np.isfinite(graph.last_adv_triplet_score_positive_fraction)
    assert np.isfinite(graph.last_adv_triplet_negative_scaled_fraction)
    assert bool(args.use_adj_triplet_graph_return_credit)
    assert args.adj_triplet_graph_return_credit_coef > 0.0
    assert args.adj_triplet_graph_return_credit_cap > 0.0
    assert args.adj_triplet_graph_return_credit_raw_gate_scale > 0.0
    assert bool(args.adj_triplet_graph_return_credit_require_delayed_gate)
    assert bool(args.use_adj_delayed_triplet_credit)
    assert args.adj_delayed_triplet_credit_coef > 0.0
    assert args.adj_delayed_triplet_credit_window > 0
    assert bool(args.adj_delayed_triplet_credit_positive_only)
    assert args.adj_delayed_triplet_credit_min_adv > 0.0
    assert bool(args.adj_delayed_triplet_credit_require_future_match)
    assert bool(args.use_adj_delayed_triplet_success_gate)
    assert args.adj_delayed_triplet_success_gate_min_adv > 0.0
    assert args.adj_delayed_triplet_success_gate_scale > 0.0
    assert args.adj_delayed_triplet_success_gate_floor > 0.0
    assert args.adj_delayed_triplet_future_overlap_min_nodes == 2
    assert args.adj_delayed_triplet_partial_match_weight > 0.0
    assert bool(args.use_adj_capture_to_win_credit)
    assert args.adj_capture_to_win_credit_coef > 0.0
    assert args.adj_capture_to_win_credit_min_outcome_adv > 0.0
    assert args.adj_capture_to_win_credit_scale > 0.0
    assert args.adj_capture_to_win_credit_cap > 0.0
    assert bool(args.adj_capture_to_win_credit_require_future_match)
    assert args.adj_recent_episode_window_min == 1
    assert args.adj_recent_episode_window_emergency == 1
    assert args.adj_recent_window_emergency_stale_threshold > 0.0
    assert args.adj_recent_window_emergency_factor_stale_threshold > 0.0

    selected_float = adj.float()
    loss = -(
        torch.log(target_prob.clamp_min(1e-8)) * selected_float
    ).sum() / selected_float.sum().clamp_min(1.0)
    loss.backward()

    finite_grad_count = 0
    for parameter in graph.parameters():
        if parameter.grad is None:
            continue
        assert bool(torch.isfinite(parameter.grad).all().item())
        finite_grad_count += 1
    assert finite_grad_count > 0
    gate_after_bad_credit = graph.update_order3_credit_gate(
        0.004,
        0.0,
    )
    assert gate_after_bad_credit < 1.0
    graph.sample(
        obs=None,
        rnn_obs=rnn_obs,
        use_adj_init=True,
        dones=dones,
        explore=True,
        t_env=500000,
    )
    assert graph.current_min_order3_ratio < 0.72
    assert graph.current_greedy_sample_prob < 0.75

    validate_adj_buffer()
    validate_scatter_gradient()
    deterministic_cuda_scatter = validate_deterministic_cuda_scatter()
    validate_factor_q_gradient()
    validate_trainer_without_critics()
    validate_full_policy_gradient()
    deterministic_cuda_full_policy = (
        validate_deterministic_cuda_full_policy_gradient()
    )

    print("SDDFG dynamic graph validation passed")
    print("active_counts={}".format(active_counts))
    print("order2_factors={}".format(order_counts[2]))
    print("order3_factors={}".format(order_counts[3]))
    print(
        "current_min_order3_ratio={}".format(
            graph.current_min_order3_ratio
        )
    )
    print(
        "current_max_order3_ratio={}".format(
            graph.current_max_order3_ratio
        )
    )
    print(
        "current_greedy_sample_prob={}".format(
            graph.current_greedy_sample_prob
        )
    )
    print(
        "greedy_sample_prob_cap={}".format(
            graph.greedy_sample_prob_cap
        )
    )
    print("order3_quota_mode={}".format(graph.order3_quota_mode))
    print(
        "order3_soft_quota_coef={}".format(
            graph.order3_soft_quota_coef
        )
    )
    print(
        "triplet_balance_coef={}".format(
            graph.triplet_balance_coef
        )
    )
    print("triplet_feature_mode={}".format(graph.triplet_feature_mode))
    print(
        "use_advantage_triplet_scorer={}".format(
            graph.use_advantage_triplet_scorer
        )
    )
    print(
        "triplet_credit_score_coef={}".format(
            graph.triplet_credit_score_coef
        )
    )
    print(
        "triplet_credit_score_scale={}".format(
            graph.triplet_credit_score_scale
        )
    )
    print(
        "use_triplet_credit_direct_rank={}".format(
            graph.use_triplet_credit_direct_rank
        )
    )
    print(
        "triplet_credit_rank_coef={}".format(
            graph.triplet_credit_rank_coef
        )
    )
    print(
        "triplet_credit_multiplier_bounds=[{},{}]".format(
            graph.triplet_credit_min_multiplier,
            graph.triplet_credit_max_multiplier,
        )
    )
    print(
        "triplet_credit_negative_rank_scale={}".format(
            graph.triplet_credit_negative_rank_scale
        )
    )
    print(
        "triplet_credit_min_positive_fraction={}".format(
            graph.triplet_credit_min_positive_fraction
        )
    )
    print(
        "adv_triplet_credit_seen_ratio={}".format(
            credit_info["adv_triplet_credit_seen_ratio"]
        )
    )
    print(
        "adv_triplet_marginal_positive_fraction={}".format(
            credit_info["adv_triplet_marginal_positive_fraction"]
        )
    )
    print(
        "adv_triplet_score_multiplier_mean={}".format(
            graph.last_adv_triplet_score_multiplier_mean
        )
    )
    print(
        "adv_triplet_score_multiplier_min={}".format(
            graph.last_adv_triplet_score_multiplier_min
        )
    )
    print(
        "adv_triplet_score_multiplier_max={}".format(
            graph.last_adv_triplet_score_multiplier_max
        )
    )
    print(
        "adv_triplet_score_positive_fraction={}".format(
            graph.last_adv_triplet_score_positive_fraction
        )
    )
    print(
        "adv_triplet_negative_scaled_fraction={}".format(
            graph.last_adv_triplet_negative_scaled_fraction
        )
    )
    print(
        "adj_order_adv_positive_only={}".format(
            args.adj_order_adv_positive_only
        )
    )
    print(
        "adj_order_adv_negative_coef={}".format(
            args.adj_order_adv_negative_coef
        )
    )
    print(
        "adj_order_adv_require_positive_graph_adv={}".format(
            args.adj_order_adv_require_positive_graph_adv
        )
    )
    print(
        "use_adj_triplet_graph_return_credit={}".format(
            args.use_adj_triplet_graph_return_credit
        )
    )
    print(
        "adj_triplet_graph_return_credit_coef={}".format(
            args.adj_triplet_graph_return_credit_coef
        )
    )
    print(
        "adj_triplet_graph_return_credit_cap={}".format(
            args.adj_triplet_graph_return_credit_cap
        )
    )
    print(
        "adj_triplet_graph_return_credit_min_graph_adv={}".format(
            args.adj_triplet_graph_return_credit_min_graph_adv
        )
    )
    print(
        "adj_triplet_graph_return_credit_raw_gate_scale={}".format(
            args.adj_triplet_graph_return_credit_raw_gate_scale
        )
    )
    print(
        "adj_triplet_graph_return_credit_require_delayed_gate={}".format(
            args.adj_triplet_graph_return_credit_require_delayed_gate
        )
    )
    print(
        "use_adj_delayed_triplet_credit={}".format(
            args.use_adj_delayed_triplet_credit
        )
    )
    print(
        "adj_delayed_triplet_credit_coef={}".format(
            args.adj_delayed_triplet_credit_coef
        )
    )
    print(
        "adj_delayed_triplet_credit_window={}".format(
            args.adj_delayed_triplet_credit_window
        )
    )
    print(
        "adj_delayed_triplet_credit_cap={}".format(
            args.adj_delayed_triplet_credit_cap
        )
    )
    print(
        "adj_delayed_triplet_credit_min_reward={}".format(
            args.adj_delayed_triplet_credit_min_reward
        )
    )
    print(
        "adj_delayed_triplet_credit_positive_only={}".format(
            args.adj_delayed_triplet_credit_positive_only
        )
    )
    print(
        "adj_delayed_triplet_credit_min_adv={}".format(
            args.adj_delayed_triplet_credit_min_adv
        )
    )
    print(
        "adj_delayed_triplet_credit_require_future_match={}".format(
            args.adj_delayed_triplet_credit_require_future_match
        )
    )
    print(
        "use_adj_delayed_triplet_success_gate={}".format(
            args.use_adj_delayed_triplet_success_gate
        )
    )
    print(
        "adj_delayed_triplet_success_gate_min_adv={}".format(
            args.adj_delayed_triplet_success_gate_min_adv
        )
    )
    print(
        "adj_delayed_triplet_success_gate_scale={}".format(
            args.adj_delayed_triplet_success_gate_scale
        )
    )
    print(
        "adj_delayed_triplet_success_gate_floor={}".format(
            args.adj_delayed_triplet_success_gate_floor
        )
    )
    print(
        "adj_delayed_triplet_future_overlap_min_nodes={}".format(
            args.adj_delayed_triplet_future_overlap_min_nodes
        )
    )
    print(
        "adj_delayed_triplet_partial_match_weight={}".format(
            args.adj_delayed_triplet_partial_match_weight
        )
    )
    print(
        "use_adj_capture_to_win_credit={}".format(
            args.use_adj_capture_to_win_credit
        )
    )
    print(
        "adj_capture_to_win_credit_coef={}".format(
            args.adj_capture_to_win_credit_coef
        )
    )
    print(
        "adj_capture_to_win_credit_min_outcome_adv={}".format(
            args.adj_capture_to_win_credit_min_outcome_adv
        )
    )
    print(
        "adj_capture_to_win_credit_scale={}".format(
            args.adj_capture_to_win_credit_scale
        )
    )
    print(
        "adj_capture_to_win_credit_cap={}".format(
            args.adj_capture_to_win_credit_cap
        )
    )
    print(
        "adj_capture_to_win_credit_require_future_match={}".format(
            args.adj_capture_to_win_credit_require_future_match
        )
    )
    print(
        "current_order3_credit_gate={}".format(
            graph.current_order3_credit_gate
        )
    )
    print(
        "order3_credit_loss_ema={}".format(
            graph.order3_credit_loss_ema
        )
    )
    print(
        "order3_credit_margin_ema={}".format(
            graph.order3_credit_margin_ema
        )
    )
    print(
        "relative_order3_credit_gate={}".format(
            graph.use_relative_order3_credit_gate
        )
    )
    print(
        "order3_credit_gate_max_delta={}".format(
            graph.order3_credit_gate_max_delta
        )
    )
    print("adj_ppo_clip_stop_ratio={}".format(args.adj_ppo_clip_stop_ratio))
    print(
        "adj_ppo_factor_clip_stop_ratio={}".format(
            args.adj_ppo_factor_clip_stop_ratio
        )
    )
    print("adj_ppo_min_epochs={}".format(args.adj_ppo_min_epochs))
    print("use_adj_ppo_stale_trust={}".format(args.use_adj_ppo_stale_trust))
    print(
        "adj_ppo_stale_trust_clip={}".format(
            args.adj_ppo_stale_trust_clip
        )
    )
    print(
        "adj_ppo_stale_trust_scale={}".format(
            args.adj_ppo_stale_trust_scale
        )
    )
    print(
        "adj_ppo_stale_trust_min_weight={}".format(
            args.adj_ppo_stale_trust_min_weight
        )
    )
    print(
        "adj_recent_episode_window={}".format(
            args.adj_recent_episode_window
        )
    )
    print(
        "use_adj_dynamic_recent_window={}".format(
            args.use_adj_dynamic_recent_window
        )
    )
    print(
        "adj_recent_episode_window_min={}".format(
            args.adj_recent_episode_window_min
        )
    )
    print(
        "adj_recent_window_stale_threshold={}".format(
            args.adj_recent_window_stale_threshold
        )
    )
    print(
        "adj_recent_window_factor_stale_threshold={}".format(
            args.adj_recent_window_factor_stale_threshold
        )
    )
    print(
        "adj_recent_window_shrink_patience={}".format(
            args.adj_recent_window_shrink_patience
        )
    )
    print(
        "adj_recent_window_recover_patience={}".format(
            args.adj_recent_window_recover_patience
        )
    )
    print(
        "adj_recent_window_recover_stale_threshold={}".format(
            args.adj_recent_window_recover_stale_threshold
        )
    )
    print(
        "adj_recent_window_recover_factor_stale_threshold={}".format(
            args.adj_recent_window_recover_factor_stale_threshold
        )
    )
    print(
        "adj_recent_window_severe_margin={}".format(
            args.adj_recent_window_severe_margin
        )
    )
    print(
        "adj_recent_episode_window_emergency={}".format(
            args.adj_recent_episode_window_emergency
        )
    )
    print(
        "adj_recent_window_emergency_stale_threshold={}".format(
            args.adj_recent_window_emergency_stale_threshold
        )
    )
    print(
        "adj_recent_window_emergency_factor_stale_threshold={}".format(
            args.adj_recent_window_emergency_factor_stale_threshold
        )
    )
    print("finite_grad_tensors={}".format(finite_grad_count))
    print("adj_buffer_axis_and_return_test=passed")
    print("scatter_gradient_test=passed")
    print(
        "deterministic_cuda_scatter_test={}".format(
            "passed" if deterministic_cuda_scatter else "skipped"
        )
    )
    print("factor_q_gradient_test=passed")
    print("trainer_without_critics_test=passed")
    print("full_policy_gradient_test=passed")
    print(
        "deterministic_cuda_full_policy_gradient_test={}".format(
            "passed" if deterministic_cuda_full_policy else "skipped"
        )
    )


if __name__ == "__main__":
    main()
