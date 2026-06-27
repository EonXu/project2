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
            adj_factor_adv_coef=0.0,
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
        if num_episodes > 1:
            assert np.abs(computed).sum() > 0.0


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
    """Exercise the non-atomic fallback used by strict CUDA training."""
    if (
        not torch.cuda.is_available()
        or not hasattr(torch, "use_deterministic_algorithms")
        or not hasattr(torch, "are_deterministic_algorithms_enabled")
    ):
        return False

    was_enabled = bool(
        torch.are_deterministic_algorithms_enabled()
    )
    try:
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
        torch.use_deterministic_algorithms(was_enabled)
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
        use_valuenorm=False,
        adj_max_grad_norm=0.5,
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
    if (
        not torch.cuda.is_available()
        or not hasattr(torch, "use_deterministic_algorithms")
        or not hasattr(torch, "are_deterministic_algorithms_enabled")
    ):
        return False

    was_enabled = bool(
        torch.are_deterministic_algorithms_enabled()
    )
    try:
        torch.use_deterministic_algorithms(True)
        validate_full_policy_gradient(torch.device("cuda"))
    finally:
        torch.use_deterministic_algorithms(was_enabled)
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
        adj_hidden_dim=64,
        epsilon_start=1.0,
        epsilon_finish=0.05,
        adj_anneal_time=500000,
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

        orders = selected.long().sum(dim=0)
        valid_orders = orders[orders > 0]
        assert bool(
            ((valid_orders == 2) | (valid_orders == 3)).all().item()
        )
        for order in (2, 3):
            order_counts[order] += int(
                (valid_orders == order).long().sum().item()
            )
        assert bool((target_prob[batch_idx][selected] > 0.0).all().item())

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
