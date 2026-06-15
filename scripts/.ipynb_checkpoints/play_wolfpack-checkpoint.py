import os
import sys
import time
import csv
import ast
import argparse
import numpy as np
import torch
import importlib.util

# =========================================================
# 0) 永远把项目根目录加入 sys.path（稳定解决 envs / algorithms 导入）
# =========================================================
def add_project_root():
    here = os.path.abspath(os.path.dirname(__file__))          # .../SDDFG/scripts
    root = os.path.abspath(os.path.join(here, ".."))           # .../SDDFG
    if root not in sys.path:
        sys.path.insert(0, root)
    return root

PROJECT_ROOT = add_project_root()


# =========================================================
# 1) 兜底式加载 WolfpackPenaltyOpen：
#    - 先尝试正常 import：envs.wolfpack.wolfpack_penalty_open
#    - 若失败：按文件路径动态 import（不依赖 envs 是否是 package）
# =========================================================
def load_wolfpack_env_class(project_root: str):
    try:
        from envs.Wolfpack.wolfpack_penalty_open import WolfpackPenaltyOpen
        return WolfpackPenaltyOpen
    except Exception as e1:
        # 动态加载
        env_file = os.path.join(project_root, "envs", "Wolfpack", "wolfpack_penalty_open.py")
        if not os.path.exists(env_file):
            raise FileNotFoundError(
                f"无法正常 import envs.wolfpack.wolfpack_penalty_open，且未找到文件：{env_file}\n"
                f"正常 import 的错误：{repr(e1)}"
            )

        spec = importlib.util.spec_from_file_location("wolfpack_penalty_open_dyn", env_file)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        if not hasattr(mod, "WolfpackPenaltyOpen"):
            raise AttributeError(f"{env_file} 中未找到 WolfpackPenaltyOpen 类。")
        return getattr(mod, "WolfpackPenaltyOpen")


WolfpackPenaltyOpen = load_wolfpack_env_class(PROJECT_ROOT)


# =========================================================
# 2) Policy（按你当前版本：R_DDFGPolicy）
# =========================================================
from algorithms.ddfg.algorithm.rDDFGPolicy import R_DDFGPolicy


def _find_upwards(start_path: str, target_name: str, max_up: int = 8):
    p = os.path.abspath(start_path)
    if os.path.isfile(p):
        p = os.path.dirname(p)
    cur = p
    for _ in range(max_up):
        cand = os.path.join(cur, target_name)
        if os.path.exists(cand):
            return cand
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return None


def _smart_cast(v: str):
    if v is None:
        return None
    s = str(v).strip()
    if s == "":
        return s
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    try:
        return ast.literal_eval(s)
    except Exception:
        pass
    try:
        if "." in s:
            return float(s)
        return int(s)
    except Exception:
        return s


def load_config_csv(config_path: str) -> argparse.Namespace:
    cfg = {}
    with open(config_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if len(r) >= 2]

    # 跳过表头
    if rows:
        h0 = rows[0][0].strip().lower()
        h1 = rows[0][1].strip().lower()
        if (h0 in ("parameter", "param", "key", "name")) and (h1 in ("value", "val", "v")):
            rows = rows[1:]

    for r in rows:
        k = r[0].strip()
        v = r[1].strip()
        if k:
            cfg[k] = _smart_cast(v)

    ns = argparse.Namespace()
    for k, v in cfg.items():
        setattr(ns, k, v)
    return ns


def _load_state_dict_if_exists(module: torch.nn.Module, path: str, device: torch.device):
    if not os.path.exists(path):
        print(f"[WARN] missing: {path}")
        return False
    sd = torch.load(path, map_location=device)
    module.load_state_dict(sd)
    print(f"[OK] loaded: {os.path.basename(path)}")
    return True


def parse_args():
    ap = argparse.ArgumentParser("Play one episode with a trained DDFG wolfpack policy")
    ap.add_argument("--model_dir", type=str, required=True,
                    help="例如 scripts/results/.../run13(算法加入mask)/models/policy_0")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--record", action="store_true",
                    help="若无GUI，建议开启录制（需要 env.render() 返回 rgb_array）")
    ap.add_argument("--out", type=str, default="wolfpack_episode.mp4",
                    help="录制输出文件名（--record 时有效）")
    return ap.parse_args()


def main():
    cli = parse_args()

    cfg_path = _find_upwards(cli.model_dir, "config.csv")
    if cfg_path is None:
        raise FileNotFoundError(
            f"未能从 {cli.model_dir} 向上找到 config.csv。"
            f"请确认 model_dir 指向 runXX/models/policy_0 并且 runXX 下有 config.csv"
        )
    all_args = load_config_csv(cfg_path)

    use_cuda = cli.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device(cli.device if use_cuda else "cpu")

    # ---- env params（尽量从 config 读取；没有就用合理默认）----
    max_player_num = int(getattr(all_args, "max_player_num", getattr(all_args, "num_agents", 5)))
    episode_len = int(getattr(all_args, "episode_length", getattr(all_args, "max_time_steps", 200)))

    render_mode = "rgb_array" if cli.record else "human"

    env = WolfpackPenaltyOpen(
        map_size=int(getattr(all_args, "map_size", 20)),
        max_player_num=max_player_num,
        init_num_players=int(getattr(all_args, "num_agents", max_player_num)),
        min_agents=int(getattr(all_args, "min_agents", 2)),
        close_penalty=float(getattr(all_args, "close_penalty", 0.0)),
        shaping_reward=float(getattr(all_args, "shaping_reward", 0.1)),
        max_time_steps=episode_len,
        render_mode=render_mode
    )

    reset_out = env.reset()
    if len(reset_out) != 3:
        raise RuntimeError(f"env.reset() 返回格式不符合预期: len={len(reset_out)}")
    obs, share_obs, avail_actions = reset_out

    # ---- policy ----
    obs_space = env.observation_space
    cent_obs_space = getattr(env, "share_observation_space", None)
    if cent_obs_space is None:
        raise AttributeError("env 缺少 share_observation_space（DDFG 需要）。")
    act_space = env.action_space

    policy = R_DDFGPolicy(
        args=all_args,
        obs_space=obs_space,
        cent_obs_space=cent_obs_space,
        act_space=act_space,
        device=device
    )
    policy.eval()

    # ---- load weights ----
    candidates = [
        ("q_network_1", "q_network_1.pt"),
        ("q_network_2", "q_network_2.pt"),
        ("q_network_3", "q_network_3.pt"),
        ("v_network_1", "v_network_1.pt"),
        ("v_network_2", "v_network_2.pt"),
        ("v_network_3", "v_network_3.pt"),
        ("rnn_network", "rnn_network.pt"),
        ("vtot_network", "vtot_network.pt"),
        ("adj_network", "adj_network.pt"),
    ]
    for attr, fname in candidates:
        if hasattr(policy, attr):
            _load_state_dict_if_exists(getattr(policy, attr), os.path.join(cli.model_dir, fname), device)

    # ---- rollout one episode ----
    n_agents = max_player_num
    n_actions = int(act_space.n) if hasattr(act_space, "n") else int(getattr(all_args, "n_actions", 7))
    hidden_dim = int(getattr(policy, "rnn_hidden_dim", 64))

    rnn_states = np.zeros((n_agents, hidden_dim), dtype=np.float32)
    last_actions = np.zeros((n_agents, n_actions), dtype=np.float32)
    dones = np.zeros((1, n_agents, 1), dtype=np.bool_)

    max_steps = int(cli.max_steps) if cli.max_steps is not None else episode_len

    frames = []
    t = 0
    done = False
    print("\n[INFO] Playing 1 episode...\n")

    while (not done) and (t < max_steps):
        with torch.no_grad():
            # 兼容 get_actions 是否支持 dones 参数
            try:
                actions, _, rnn_states = policy.get_actions(
                    obs=np.expand_dims(obs, 0),
                    share_obs=np.expand_dims(share_obs, 0),
                    available_actions=np.expand_dims(avail_actions, 0),
                    rnn_states=rnn_states,
                    last_actions=last_actions,
                    explore=False,
                    dones=torch.tensor(dones).to(device)
                )
            except TypeError:
                actions, _, rnn_states = policy.get_actions(
                    obs=np.expand_dims(obs, 0),
                    share_obs=np.expand_dims(share_obs, 0),
                    available_actions=np.expand_dims(avail_actions, 0),
                    rnn_states=rnn_states,
                    last_actions=last_actions,
                    explore=False
                )

        actions = np.asarray(actions).squeeze(0)  # (N,)

        step_out = env.step(actions)
        if len(step_out) != 6:
            raise RuntimeError(f"env.step() 返回格式不符合预期: len={len(step_out)}")
        obs, share_obs, rewards, env_dones, infos, avail_actions = step_out

        # render / record
        if hasattr(env, "render"):
            img = env.render()
            if cli.record and img is not None:
                frames.append(img)

        if not cli.record:
            time.sleep(1.0 / max(1, cli.fps))

        last_actions = np.eye(n_actions, dtype=np.float32)[actions]
        env_dones = np.asarray(env_dones).reshape(1, n_agents, 1).astype(np.bool_)
        dones = env_dones
        done = bool(np.all(env_dones))
        t += 1

    print(f"\n[INFO] Episode finished at t={t}")

    if cli.record:
        if len(frames) == 0:
            print("[WARN] 未收集到任何帧。请确认 env.render() 在 rgb_array 模式下会返回图像。")
        else:
            try:
                import imageio.v2 as imageio
                imageio.mimsave(cli.out, frames, fps=cli.fps)
                print(f"[OK] Saved video to: {cli.out}")
            except Exception as e:
                print(f"[ERROR] 保存视频失败：{repr(e)}（可安装 imageio: pip install imageio）")

    if hasattr(env, "close"):
        env.close()


if __name__ == "__main__":
    main()
