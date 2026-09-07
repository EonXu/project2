#!/usr/bin/env python
from __future__ import print_function

import os
import re


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAUNCHER_PATH = os.path.join(
    SCRIPT_DIR, "train_wolfpack_sddfg_intra_episode_dynamic.sh"
)
CONFIG_PATH = os.path.join(SCRIPT_DIR, "train", "train_wolfpack.py")
REPLAY_PATH = os.path.join(
    SCRIPT_DIR, "debug_pair_transaction_replay_preflight.py"
)
REPLAY_FIXTURE_PATH = os.path.join(
    SCRIPT_DIR, "fixtures", "run116_transaction_replays.json"
)
FORBIDDEN_FORWARDER_PATH = os.path.join(
    SCRIPT_DIR, "train", "train_wolfpack_sddfg_intra_episode_dynamic.sh"
)


def _read(path):
    with open(path, "r", encoding="utf-8") as stream:
        return stream.read()


def _require(pattern, text, description):
    if re.search(pattern, text, flags=re.MULTILINE) is None:
        raise AssertionError("missing launcher contract: {}".format(description))


def test_experiment_launcher_defaults_to_bounded_ttl4():
    launcher = _read(LAUNCHER_PATH)
    _require(
        r'pair_bounded_pending_evidence='
        r'"\$\{PAIR_BOUNDED_PENDING_EVIDENCE:-1\}"',
        launcher,
        "bounded pending default must be enabled",
    )
    _require(
        r'elif \[\[ "\$\{pair_bounded_pending_evidence\}" == "1" \]\]; then'
        r'\s+pair_pending_max_adj_updates=4',
        launcher,
        "enabled experiment default must resolve TTL=4",
    )
    _require(
        r'--pair_bounded_pending_evidence\s+'
        r'--pair_pending_max_adj_updates '
        r'"\$\{pair_pending_max_adj_updates\}"',
        launcher,
        "resolved values must reach the training command",
    )


def test_explicit_off_path_and_fail_loud_contract_are_preserved():
    launcher = _read(LAUNCHER_PATH)
    _require(
        r'else\s+pair_pending_max_adj_updates=0\s+fi',
        launcher,
        "explicit disabled path must resolve TTL=0",
    )
    _require(
        r'pair_bounded_pending_evidence\}" == "0" && '
        r'"\$\{pair_pending_max_adj_updates\}" -ne 0',
        launcher,
        "disabled/nonzero-TTL mismatch must fail loudly",
    )
    config = _read(CONFIG_PATH)
    _require(
        r'--pair_bounded_pending_evidence[\s\S]*?action="store_true"'
        r'[\s\S]*?default=False',
        config,
        "library/parser bounded-pending default must remain off",
    )
    _require(
        r'--pair_pending_max_adj_updates[\s\S]*?type=int'
        r'[\s\S]*?default=0',
        config,
        "library/parser TTL default must remain zero",
    )


def test_formal_launcher_forwards_arguments_and_runs_preflight():
    launcher = _read(LAUNCHER_PATH)
    _require(
        r'if \[ "\$#" -ge 2 \]; then[\s\S]*?'
        r'num_env_steps="\$\{3:-\$\{NUM_ENV_STEPS:-60000\}\}"[\s\S]*?'
        r'else[\s\S]*?seed="\$\{SEED:-1\}"[\s\S]*?'
        r'num_env_steps="\$\{1:-\$\{NUM_ENV_STEPS:-60000\}\}"',
        launcher,
        "formal launcher must accept one positional steps argument and the legacy form",
    )
    for relative_path in (
        "scripts/train_wolfpack_sddfg_intra_episode_dynamic.sh",
        "scripts/debug_pair_pending_launcher_contract.py",
        "scripts/debug_pair_optimizer_transaction_diagnostics.py",
        "scripts/debug_pair_transaction_replay_preflight.py",
        "scripts/fixtures/run116_transaction_replays.json",
        "utils/pair_direction.py",
    ):
        if relative_path not in launcher:
            raise AssertionError(
                "launcher source manifest omits {}".format(relative_path)
            )
    _require(
        r'"\$\{python_bin\}" '
        r'"\$\{script_dir\}/debug_pair_pending_launcher_contract\.py"'
        r'\s+\\?\s*2>&1 \| tee -a "\$\{console_log\}"',
        launcher,
        "production preflight must execute the launcher contract",
    )
    _require(
        r'"\$\{python_bin\}" '
        r'"\$\{script_dir\}/debug_pair_transaction_replay_preflight\.py"'
        r'\s+\\?\s*2>&1 \| tee -a "\$\{console_log\}"',
        launcher,
        "production preflight must execute run116 transaction replay",
    )
    replay_position = launcher.index(
        '"${script_dir}/debug_pair_transaction_replay_preflight.py"'
    )
    training_position = launcher.index(
        '"${script_dir}/train/train_wolfpack.py"'
    )
    if replay_position >= training_position:
        raise AssertionError(
            "transaction replay must finish before the training process starts"
        )
    if not os.path.isfile(REPLAY_PATH):
        raise AssertionError("transaction replay preflight file is missing")
    if not os.path.isfile(REPLAY_FIXTURE_PATH):
        raise AssertionError("run116 transaction replay fixture is missing")
    if os.path.exists(FORBIDDEN_FORWARDER_PATH):
        raise AssertionError("forbidden duplicate training forwarder exists")
    _require(
        r'CUDA_VISIBLE_DEVICES="\$\{gpu\}" '
        r'"\$\{python_bin\}" "\$\{script_dir\}/train/train_wolfpack\.py"'
        r'[\s\S]*?--seed "\$\{seed\}"'
        r'[\s\S]*?--num_env_steps "\$\{num_env_steps\}"',
        launcher,
        "formal launcher must forward seed, GPU, and training budget",
    )


def main():
    tests = (
        test_experiment_launcher_defaults_to_bounded_ttl4,
        test_explicit_off_path_and_fail_loud_contract_are_preserved,
        test_formal_launcher_forwards_arguments_and_runs_preflight,
    )
    for test in tests:
        test()
        print("PASS {}".format(test.__name__))
    print("PASS all {} pair-pending launcher contract tests".format(len(tests)))


if __name__ == "__main__":
    main()
