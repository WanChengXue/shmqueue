"""从 rl 训练 TensorBoard 事件文件绘制诊断图。

用法:
    /tmp/tbenv/bin/python docs/plot_figures.py <events_file> <out_dir>

生成 4 张 PNG, 佐证 shmqueue README §1.1 / §7 / §8 的论点:
  fig1_queue_depth.png      —— 队列始终为空 (数据供给 << GPU 消耗)
  fig2_gpu_time_breakdown.png —— GPU 时间占比 (algo 仅 ~9%, queue_get ~80%)
  fig3_update_rate_vs_ceiling.png —— 每分钟更新次数 vs 理论上限 (达成率)
  fig4_sampler_throughput.png —— worker 采样耗时 (数据慢的根因)
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorboard.backend.event_processing import event_accumulator

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110

CEILING_PER_MIN = 400.0  # 理论上限: 零数据开销 10min 4000 次 ≈ 400/min


def load(events_file):
    ea = event_accumulator.EventAccumulator(events_file, size_guidance={"scalars": 0})
    ea.Reload()
    rows = []
    t0 = None
    for tag in ea.Tags()["scalars"]:
        for ev in ea.Scalars(tag):
            if t0 is None:
                t0 = ev.wall_time
            rows.append((ev.wall_time - t0, ev.step, tag, ev.value))
    df = pd.DataFrame(rows, columns=["t_sec", "step", "tag", "value"])
    df["t"] = df["t_sec"] / 60.0
    return df


def series(df, tag):
    s = df[df.tag == tag].sort_values("t").copy()
    return s


def fig1_queue_depth(df, out):
    """8 个队列的 depth 时序 —— 证明队列始终为空。"""
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, 8))

    ax = axes[0]
    all_zero = True
    for i in range(8):
        s = series(df, f"learn_queue_server/queue_{i}/depth")
        if s.value.max() > 0:
            all_zero = False
        ax.plot(s.t, s.value, label=f"queue_{i}", color=colors[i], lw=1.2)
    ax.set_ylabel("队列深度 (slots)")
    ax.set_title("8 个共享内存队列的实时深度 —— 全程为 0")
    ax.legend(ncol=8, fontsize=8, loc="upper center", frameon=False)
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    txt = "所有队列 depth ≡ 0\n数据一入队即被消费 / 根本无数据入队" if all_zero else ""
    ax.text(0.02, 0.5, txt, transform=ax.transAxes, fontsize=11,
            color="crimson", va="center",
            bbox=dict(boxstyle="round", fc="white", ec="crimson", alpha=0.9))

    ax = axes[1]
    for i in range(8):
        s = series(df, f"learn_queue_server/queue_{i}/utilization_pct")
        ax.plot(s.t, s.value, label=f"queue_{i}", color=colors[i], lw=1.2)
    ax.set_ylabel("利用率 (%)")
    ax.set_xlabel("训练时间 (分钟)")
    ax.set_title("队列利用率 —— 全程 0%")
    ax.legend(ncol=8, fontsize=8, loc="upper center", frameon=False)

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig2_gpu_time_breakdown(df, out):
    """各 stage 时间占比 —— GPU 90% 时间在等数据。"""
    stages = [
        ("algo_step_ratio_per_min", "algo (GPU 算力)", "#2ca02c"),
        ("queue_get_ratio_per_min", "queue_get (等数据)", "#d62728"),
        ("h2d_ratio_per_min", "h2d (CPU→GPU)", "#1f77b4"),
        ("logger_ratio_per_min", "logger", "#9467bd"),
        ("model_send_ratio_per_min", "model_send", "#ff7f0e"),
        ("metrics_send_ratio_per_min", "metrics_send", "#8c564b"),
    ]
    s_total = series(df, "learn_server_timing/ppo/total_ratio_per_min")
    s_update = series(df, "learn_server_timing/ppo/update_times_per_min")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    # 堆叠面积图, 各 stage 用线性插值对齐到 total 时间轴
    bottoms = np.zeros(len(s_total))
    xs = s_total.t.values
    for tag, label, color in stages:
        s = series(df, f"learn_server_timing/ppo/{tag}")
        vals = np.interp(xs, s.t.values, s.value.values, left=0, right=0)
        ax1.fill_between(xs, bottoms, bottoms + vals, label=label,
                         color=color, alpha=0.85)
        bottoms += vals
    ax1.plot(xs, s_total.value.values, "k--", lw=1.2, label="total (busy)")
    ax1.set_ylabel("每分钟时间占比 (≤1 = 满载)")
    ax1.set_ylim(0, max(1.1, bottoms.max() + 0.1))
    ax1.set_title("learner 各阶段时间占比 —— GPU algo 仅 ~9%, queue_get ~80% (等数据)")
    ax1.axhline(1.0, color="gray", ls=":", lw=1)
    ax1.legend(ncol=4, fontsize=8, loc="upper center", frameon=False)

    # 下方: 每分钟更新次数
    ax2.plot(s_update.t, s_update.value, color="crimson", lw=1.3)
    ax2.set_ylabel("update/min")
    ax2.set_xlabel("训练时间 (分钟)")
    ax2.set_title("每分钟模型更新次数")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig3_update_rate_vs_ceiling(df, out):
    """每分钟更新次数 vs 理论上限 —— 达成率。"""
    s = series(df, "learn_server_timing/ppo/update_times_per_min")
    roll = s.value.rolling(5, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(s.t, s.value, color="#9ecae1", lw=1, label="实测 (raw)")
    ax.plot(s.t, roll, color="#08519c", lw=2, label="实测 (5点滑动均值)")
    ax.axhline(CEILING_PER_MIN, color="crimson", ls="--", lw=2,
               label=f"理论上限 ≈ {CEILING_PER_MIN:.0f}/min (零数据开销 10min 4000 次)")
    ax.fill_between(s.t, 0, s.value, color="#9ecae1", alpha=0.3)

    mean_v = s.value.mean()
    rate = mean_v / CEILING_PER_MIN * 100
    ax.axhline(mean_v, color="#08519c", ls=":", lw=1.5,
               label=f"实测均值 {mean_v:.0f}/min (达成率 {rate:.0f}%)")
    ax.set_xlabel("训练时间 (分钟)")
    ax.set_ylabel("每分钟更新次数 (update/min)")
    ax.set_title(f"吞吐达成率 —— 平均 {mean_v:.0f}/min vs 理论 {CEILING_PER_MIN:.0f}/min = {rate:.0f}%")
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_ylim(0, CEILING_PER_MIN * 1.15)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def fig4_sampler_throughput(df, out):
    """worker 采样耗时 + data_server recv/pushed —— 数据慢的根因。"""
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=False)

    # (a) sample_time_per_episode 分布
    ax = axes[0]
    st = df[df.tag == "sampler/worker/ppo/sample_time_per_episode"].value.values
    st = st[st < np.percentile(st, 99)]  # 截尾去极端值可视化
    ax.hist(st, bins=80, color="#6baed6", edgecolor="white")
    ax.axvline(st.mean(), color="crimson", ls="--", lw=2,
               label=f"均值 {st.mean():.2f}s / episode")
    ax.set_xlabel("单 episode 采样耗时 (s)")
    ax.set_ylabel("episode 数")
    ax.set_title("worker 单 episode 采样耗时分布 —— 数据生产端的真实节奏")
    ax.legend(frameon=False)
    ax.set_xlim(0, st.max())

    # (b) data_server recv ratio & pushed_batch 时序
    ax = axes[1]
    s_recv = series(df, "learn_data_server/ppo/recv/ratio")
    s_push = series(df, "learn_data_server/ppo/pushed_batch")
    ax.plot(s_recv.t, s_recv.value, color="#e6550d", lw=1.2, label="recv/周期 (条)")
    ax.plot(s_push.t, s_push.value, color="#756bb1", lw=1.2, label="pushed_batch/周期")
    ax.set_xlabel("训练时间 (分钟)")
    ax.set_ylabel("每周期条数")
    ax.set_title("data_server 接收 / 入队速率")
    ax.legend(frameon=False, ncol=2)

    # (c) queue_rejects (应为 0 —— 队列从不拒绝, 因为从来不满)
    ax = axes[2]
    s_rej = series(df, "learn_data_server/ppo/queue_rejects")
    s_active = series(df, "sampler/active_worker/count")
    ax.plot(s_rej.t, s_rej.value, color="#31a354", lw=1.5, label="queue_rejects (满队列丢批)")
    ax.set_ylabel("rejects/周期", color="#31a354")
    ax.tick_params(axis="y", labelcolor="#31a354")
    ax.set_xlabel("训练时间 (分钟)")
    ax2 = ax.twinx()
    ax2.plot(s_active.t, s_active.value, color="#636363", lw=1.2, alpha=0.7,
             label="活跃 worker 数")
    ax2.set_ylabel("活跃 worker 数", color="#636363")
    ax.set_title("queue_rejects ≡ 0 (队列从不满) + 活跃 worker 数")
    ax.set_ylim(-0.5, max(1, s_rej.value.max() + 1))

    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    events_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tfevents_1gpu.bin"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "docs/figures"
    os.makedirs(out_dir, exist_ok=True)
    df = load(events_file)
    fig1_queue_depth(df, os.path.join(out_dir, "fig1_queue_depth.png"))
    fig2_gpu_time_breakdown(df, os.path.join(out_dir, "fig2_gpu_time_breakdown.png"))
    fig3_update_rate_vs_ceiling(df, os.path.join(out_dir, "fig3_update_rate_vs_ceiling.png"))
    fig4_sampler_throughput(df, os.path.join(out_dir, "fig4_sampler_throughput.png"))


if __name__ == "__main__":
    main()
