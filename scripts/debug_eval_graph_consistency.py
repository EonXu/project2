#!/usr/bin/env python
"""Lightweight checks for separating action and graph evaluation modes."""

from __future__ import print_function

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.graph_sampling import resolve_graph_sampling_mode


def main():
    cases = [
        # Training follows the caller's exploration mode and uses train RNG.
        ((True, True, False, True), (True, False)),
        ((False, True, False, True), (False, False)),
        # Consistent eval explores only the graph and uses isolated eval RNG.
        ((False, False, False, True), (True, True)),
        # Legacy eval remains deterministic when the mechanism is disabled.
        ((False, False, False, False), (False, False)),
        # Warmup is never treated as evaluation even if training_episode=False.
        ((True, False, True, True), (True, False)),
    ]
    for inputs, expected in cases:
        actual = resolve_graph_sampling_mode(*inputs)
        if actual != expected:
            raise AssertionError(
                "mode {} resolved to {}, expected {}".format(
                    inputs, actual, expected
                )
            )
        print("PASS mode={} -> {}".format(inputs, actual))

    print("PASS all {} eval-graph consistency tests".format(len(cases)))


if __name__ == "__main__":
    main()
