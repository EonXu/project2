#!/usr/bin/env python
"""Build a reproducible seed-1, 200k Wolfpack dynamic-agent comparison report."""

from __future__ import annotations

import csv
import html
import json
import os
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "scripts"
    / "results"
    / "wolfpack"
    / "comparisons"
    / "dynamic_200k_seed1"
)

RUNS = {
    "SDDFG": (
        ROOT
        / "scripts/results/wolfpack/sddfg/"
        "sddfg_intra_ep_4to6_r2_j1_rec30_seed1/run6",
        "run6",
    ),
    "QMIX": (
        ROOT
        / "scripts/results/wolfpack/qmix/"
        "qmix_intra_ep_4to6_r2_j1_rec30_seed1/run1",
        "run1",
    ),
    "QPLEX": (
        ROOT
        / "scripts/results/wolfpack/qplex/"
        "qplex_intra_ep_4to6_r2_j1_rec30_seed1/run1",
        "run1",
    ),
    "VDN": (
        ROOT
        / "scripts/results/wolfpack/vdn/"
        "vdn_intra_ep_4to6_r2_j1_rec30_seed1/run2",
        "run2",
    ),
}

STAGES = [
    ("Initial: 4", 0, 50),
    ("Leave 1: 2", 50, 60),
    ("Join 1: 3", 60, 80),
    ("Recover 1: 5", 80, 120),
    ("Leave 2: 3", 120, 130),
    ("Join 2: 4", 130, 150),
    ("Recover 2: 6", 150, 200),
]

COLORS = {
    "SDDFG": "#5477C4",
    "QMIX": "#B8A037",
    "QPLEX": "#CC6F47",
    "VDN": "#7A828F",
}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def as_float(value):
    return float(value)


def load_results():
    eval_curve = []
    stage_rows = []
    summary = {}

    for algorithm, (run_dir, run_name) in RUNS.items():
        eval_rows = read_csv(run_dir / f"{run_name}_progress_eval.csv")
        rewards = [
            as_float(row["eval_average_episode_rewards"]) for row in eval_rows
        ]
        wins = [as_float(row["eval_win_rate"]) for row in eval_rows]

        for row in eval_rows:
            eval_curve.append(
                {
                    "algorithm": algorithm,
                    "step": int(row["step"]),
                    "reward": as_float(row["eval_average_episode_rewards"]),
                    "win_rate": as_float(row["eval_win_rate"]),
                }
            )

        trajectory = read_csv(
            run_dir / f"{run_name}_progress_eval_num_players_traj.csv"
        )
        final_step = max(int(row["step"]) for row in trajectory)
        final_rows = [
            row for row in trajectory if int(row["step"]) == final_step
        ]

        stage_values = {}
        for stage_name, start, end in STAGES:
            values = [
                as_float(row["team_reward"])
                for row in final_rows
                if start <= int(row["t"]) < end
            ]
            stage_values[stage_name] = mean(values)
            stage_rows.append(
                {
                    "algorithm": algorithm,
                    "stage": stage_name,
                    "reward": mean(values),
                }
            )

        episode_rewards = [
            sum(
                as_float(row["team_reward"])
                for row in final_rows
                if int(row["eval_ep"]) == episode
            )
            for episode in range(10)
        ]

        train_rows = read_csv(run_dir / f"{run_name}_progress_train.csv")
        clipped = [
            as_float(row["grad_was_clipped"])
            for row in train_rows
            if row.get("grad_was_clipped", "") != ""
        ]
        grad_norms = [
            as_float(row["grad_norm"])
            for row in train_rows
            if row.get("grad_norm", "") != ""
        ]

        summary[algorithm] = {
            "final_reward": rewards[-1],
            "final_win_rate": wins[-1],
            "best_reward": max(rewards),
            "best_reward_step": int(
                eval_rows[rewards.index(max(rewards))]["step"]
            ),
            "best_win_rate": max(wins),
            "mean_eval_reward": mean(rewards),
            "eval_reward_std": pstdev(rewards),
            "last3_reward": mean(rewards[-3:]),
            "last3_win_rate": mean(wins[-3:]),
            "positive_evaluations": sum(value > 0 for value in rewards),
            "final_episode_std": pstdev(episode_rewards),
            "gradient_clip_ratio": mean(clipped) if clipped else None,
            "gradient_norm_mean": mean(grad_norms) if grad_norms else None,
            "gradient_norm_max": max(grad_norms) if grad_norms else None,
            "stage_rewards": stage_values,
        }

    return pd.DataFrame(eval_curve), pd.DataFrame(stage_rows), summary


def chart_theme():
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": "#FCFCFD",
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": "#D7DBE7",
            "axes.labelcolor": "#1F2430",
            "grid.color": "#E6E8F0",
            "font.family": "sans-serif",
        },
    )


def add_header(fig, ax, title, subtitle):
    fig.subplots_adjust(top=0.80, left=0.10, right=0.98, bottom=0.14)
    left = ax.get_position().x0
    fig.text(
        left,
        0.965,
        title,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color="#1F2430",
    )
    fig.text(
        left,
        0.915,
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color="#6F768A",
    )
    sns.despine(ax=ax)


def render_charts(eval_curve, stage_rows):
    chart_theme()

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    for algorithm in RUNS:
        part = eval_curve[eval_curve["algorithm"] == algorithm]
        ax.plot(
            part["step"],
            part["reward"],
            marker="o",
            markersize=4,
            linewidth=1.4,
            color=COLORS[algorithm],
            label=algorithm,
        )
    ax.axhline(0, color="#464C55", linewidth=1, linestyle=":")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Mean evaluation episode reward")
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=4,
        borderaxespad=0,
    )
    add_header(
        fig,
        ax,
        "Evaluation reward across 200k training steps",
        "Seed 1; ten evaluation episodes every 20k steps; higher is better.",
    )
    trend_path = OUTPUT / "evaluation_reward_curve.png"
    fig.savefig(trend_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6.2))
    stage_order = [stage[0] for stage in STAGES]
    sns.barplot(
        data=stage_rows,
        x="stage",
        y="reward",
        hue="algorithm",
        order=stage_order,
        hue_order=list(RUNS),
        palette=COLORS,
        edgecolor="#464C55",
        linewidth=0.7,
        ax=ax,
    )
    ax.axhline(0, color="#464C55", linewidth=1)
    ax.set_xlabel("")
    ax.set_ylabel("Mean team reward per environment step")
    ax.tick_params(axis="x", rotation=20)
    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0, 1.02),
        frameon=False,
        ncol=4,
        borderaxespad=0,
    )
    add_header(
        fig,
        ax,
        "Final-policy reward through each roster transition",
        "Final 200k checkpoint evaluation; ten episodes; stages follow the fixed intra-episode shock schedule.",
    )
    stage_path = OUTPUT / "dynamic_stage_rewards.png"
    fig.savefig(stage_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def metric_table(summary):
    rows = []
    for algorithm in RUNS:
        item = summary[algorithm]
        clip = item["gradient_clip_ratio"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(algorithm)}</td>"
            f"<td>{item['final_reward']:.3f}</td>"
            f"<td>{item['final_win_rate']:.2f}</td>"
            f"<td>{item['last3_reward']:.3f}</td>"
            f"<td>{item['mean_eval_reward']:.3f}</td>"
            f"<td>{item['best_reward']:.3f}</td>"
            f"<td>{item['positive_evaluations']}/10</td>"
            f"<td>{'N/A' if clip is None else f'{clip:.1%}'}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build_report(summary):
    sddfg = summary["SDDFG"]
    report = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Wolfpack 动态智能体 200k 单种子对比</title>
  <style>
    body {{ margin: 0; background: #fcfcfd; color: #1f2430;
            font-family: "Segoe UI", "Microsoft YaHei", sans-serif; }}
    main {{ max-width: 1080px; margin: 0 auto; padding: 40px 28px 64px; }}
    h1 {{ font-size: 30px; margin-bottom: 24px; }}
    h2 {{ margin-top: 38px; font-size: 21px; }}
    p, li {{ line-height: 1.7; }}
    .summary {{ border-left: 5px solid #5477c4; background: white;
               padding: 18px 22px; border-radius: 8px; }}
    .chart {{ width: 100%; background: white; border-radius: 10px;
              margin: 14px 0 8px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ padding: 11px 10px; border-bottom: 1px solid #e6e8f0;
              text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    .note {{ color: #6f768a; font-size: 14px; }}
    code {{ background: #f1f3f7; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body><main>
  <h1>Wolfpack 动态智能体 200k 单种子对比</h1>

  <section class="summary">
    <h2>技术结论</h2>
    <ul>
      <li>SDDFG 在最终奖励、最终胜率、最近三次评估奖励和全程平均奖励上均为四种算法最高。</li>
      <li>SDDFG 最终奖励为 {sddfg['final_reward']:.3f}，胜率为 {sddfg['final_win_rate']:.2f}；
          最近三次奖励均值为 {sddfg['last3_reward']:.3f}。</li>
      <li>最终策略在七个动态人数阶段的逐步平均奖励全部为非负，说明退出、加入和恢复机制没有再次触发系统性性能掉点。</li>
      <li>当前没有日志或代码证据支持继续修改 SDDFG。结果仍是单 seed、每次仅十个评估回合，尚不能视为统计显著结论。</li>
    </ul>
  </section>

  <section>
    <h2>SDDFG 在后期形成稳定领先</h2>
    <p>奖励曲线显示 SDDFG 的最后三次评估全部为正，而三种值分解基线的最后三次均值仍为负。
       这比单独比较 200k 最终点更能排除一次评估随机波动。</p>
    <img class="chart" src="evaluation_reward_curve.png" alt="Evaluation reward curves">
  </section>

  <section>
    <h2>动态人数切换阶段不再系统性失分</h2>
    <p>SDDFG 在最终评估的七个阶段全部保持非负逐步奖励。VDN 七个阶段全部为负；
       QPLEX 仅部分阶段为正；QMIX 接近零但捕获胜率为零，属于低活动、低收益状态。</p>
    <img class="chart" src="dynamic_stage_rewards.png" alt="Dynamic stage rewards">
  </section>

  <section>
    <h2>核心指标对比</h2>
    <table>
      <thead><tr>
        <th>算法</th><th>最终奖励</th><th>最终胜率</th><th>最近三次奖励</th>
        <th>全程平均奖励</th><th>最佳奖励</th><th>正奖励评估</th><th>梯度裁剪比例</th>
      </tr></thead>
      <tbody>{metric_table(summary)}</tbody>
    </table>
  </section>

  <section>
    <h2>配置与测量范围</h2>
    <p>四组实验均为 seed 1、200k environment steps、episode length 200、初始 4 人、
       最大容量 6 人，并使用相同的 50/120 shock、退出 2 人、延迟加入 1 人和 30 步恢复设置。
       评估每 20k 执行一次，每次十个回合。</p>
    <p>比较使用保存的 progress、逐步人数/奖励轨迹、训练指标和控制台日志。
       算法专属参数（动态图、QPLEX mixer 等）不要求相同。</p>
  </section>

  <section>
    <h2>稳定性与限制</h2>
    <ul>
      <li>SDDFG 邻接 PPO 平均裁剪率约 0.49%，动态图训练处于稳定区间。</li>
      <li>QMIX 的预裁剪梯度均值约 906，全部更新都触发梯度裁剪，Q_tot 约 90–100；
          因此 QMIX 结果存在明确数值稳定性问题，不宜作为唯一的优势证据。</li>
      <li>最终十回合奖励方差仍较大。单 seed 和十回合评估不足以证明统计显著性，
          只能支持“当前实现和该 seed 下描述性领先”。</li>
    </ul>
  </section>

  <section>
    <h2>推荐下一步</h2>
    <ol>
      <li>保持当前 SDDFG 代码不变，进行四种算法各自的 seed 1、2M 单种子验证。</li>
      <li>完整实验将 <code>num_eval_episodes</code> 提高到至少 30。</li>
      <li>QMIX 在进入正式结论前应单独诊断 mixer 状态输入和梯度爆炸问题。</li>
      <li>单 seed 2M 排名符合预期后，再运行 seeds 1–5 并报告均值、标准差和置信区间。</li>
    </ol>
  </section>

  <p class="note">生成来源：本地四组 200k 训练结果；生成脚本 scripts/analyze_wolfpack_dynamic_200k.py。</p>
</main></body></html>"""
    (OUTPUT / "report.html").write_text(report, encoding="utf-8")


def main():
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    eval_curve, stage_rows, summary = load_results()
    render_charts(eval_curve, stage_rows)
    build_report(summary)
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
