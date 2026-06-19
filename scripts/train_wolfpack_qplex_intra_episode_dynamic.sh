#!/usr/bin/env bash
set -euo pipefail

# Usage: bash scripts/train_wolfpack_qplex_intra_episode_dynamic.sh \
#          [seed] [gpu] [num_env_steps]
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
exec bash "${script_dir}/train_wolfpack_value_baseline_intra_episode_dynamic.sh" qplex "$@"

