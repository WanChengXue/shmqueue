# shmqueue

**高吞吐共享内存消息队列中间件** —— 单机多进程之间以最高吞吐交换批量数据（numpy 数组 / 字节）。

---

## 1. 背景与动机

本项目从一个强化学习训练框架（rl_framework）的 learner 侧数据通路中抽象而来。原框架中存在一条 **data → queue → gpu** 链路：

```
DataServer 进程:  recv(ZMQ) → 解压 → replay buffer → 采样 → fast-encode → push 到共享内存队列
                                                                          │
                                   POSIX SharedMemoryQueue (环形缓冲 + fcntl 锁) ◀── 跨进程
                                                                          │
LearnerServer 进程: prefetch worker 线程 pop_raw → fast-decode(零拷贝) → pin_memory → deque → H2D → GPU 训练
```

### 1.1 worker – queue – learner 对应关系

多 worker（sampler，CPU）经 ZMQ 把 trajectory 推给 data_server；data_server 采样后 fast-encode 入 shmqueue；learner（GPU）的 prefetch worker 从 queue pop、pin、送 GPU。对应关系：

- **worker → data_server**：ZMQ PULL，**不绑定**——任一 worker 的数据可被任一 data_server 接收（负载均衡）
- **data_server → queue**：**1 : 1**——每 GPU 有 `data_server_number_per_device`(=8) 个 data_server，每个独占一个 shmqueue（`{exp}_queue_{flat_idx}`）
- **queue → learner**：**N : 1**——每 GPU 1 个 learner，消费该 GPU 的全部 8 个 queue（8 个 prefetch worker 线程，一 queue 一消费者，无 fcntl 锁争用）
- **多 GPU / 多机**：每 GPU 重复该结构；learner 间走 DDP（NCCL）同步梯度，与 shmqueue 无关

```mermaid
graph LR
    subgraph W["N workers (sampler, CPU)"]
      W0["worker-0"]
      W1["worker-1"]
      WN["worker-N"]
    end
    subgraph D["data_server × 8 (per GPU)"]
      DS0["ds_0"]
      DS1["ds_1"]
      DS7["ds_7"]
    end
    subgraph Q["shmqueue × 8 (POSIX shm)"]
      Q0["queue_0"]
      Q1["queue_1"]
      Q7["queue_7"]
    end
    L["learner (GPU)<br/>8 prefetch workers"]

    W0 -.->|"ZMQ PUSH<br/>trajectory"| DS0
    W1 -.->|"ZMQ"| DS1
    WN -.->|"ZMQ"| DS7

    DS0 -->|"fast-encode<br/>push_raw"| Q0
    DS1 --> Q1
    DS7 --> Q7

    Q0 -->|"pop_raw<br/>decode<br/>pin_memory"| L
    Q1 --> L
    Q7 --> L
```

> 图中虚线（worker → data_server 的 ZMQ）在 shmqueue 范围**之外**；实线（data_server → queue → learner）是 shmqueue 覆盖的 producer/consumer 数据通路。

这条通路的核心机制 —— **POSIX 共享内存环形队列 + 零拷贝 flat-binary 编解码 + producer/consumer 流水线** —— 本质是通用的进程间高吞吐数据交换，与 RL 业务（environment / algorithm / model / replay buffer）无关。但它原本深耦合在 RL 框架里：

- `DataServer` 依赖 `get_environment` / `create_buffer` / `get_data_class`
- `LearnerServer` 依赖 `get_algorithm_cls` / `create_model` / DDP
- 无法脱离 `rl_framework` 独立运行、独立 benchmark、独立优化

**shmqueue 把这条数据通路的核心抽成独立中间件**，去掉一切 RL 依赖，使其可独立部署、可量化基准、可针对性优化，目标是 **producer 端与 consumer 端之间的数据交换吞吐最大化**。

> **范围边界**：shmqueue 只覆盖 data-server ↔ learner-server 之间的共享内存队列段。worker → data-server 的 ZMQ 收发**不在**本中间件范围内（那是上层应用的事）。两端"连接"到 shmqueue，shmqueue 不关心数据怎么来、到哪去。

---

## 2. 定位

一个**单机、跨进程、零拷贝、高吞吐**的消息队列中间件，面向**单台机器内**需要大批量数据交换的场景：

- 多个 producer 进程往同一队列写批量数据（采样、推理结果、传感器数据…）
- 多个 consumer 进程从队列读（GPU 训练、落盘、下游处理…）
- 数据以 numpy 数组为主（科学计算 / ML 场景的典型负载）
- 追求**吞吐优先**，延迟其次

**不适用场景**：跨机器通信（需配合 ZMQ/gRPC 等网络层，由上层组合）；强持久化（数据驻留内存，进程退出即失）；事务/路由等复杂 MQ 语义。

---

## 3. 设计目标

### 3.1 核心目标：吞吐最大化

producer → consumer 的稳态吞吐（msg/s 与 MiB/s）在以下配置下达到硬件理论上限：
- 1P1C（单生产单消费）
- NP-MC（多生产多消费，N 个 producer、M 个 consumer 共享同一队列或一组队列）

衡量指标：
- **端到端吞吐**：consumer 每秒收到的数据量（MiB/s、batch/s）
- **队列开销占比**：push+pop+锁 总耗时占端到端延迟的比例（目标 < 10%）
- **背压行为**：producer 快于 consumer 时的丢批率 / 阻塞策略

### 3.2 功能性需求

| 需求 | 说明 |
|------|------|
| **跨进程** | producer 与 consumer 是独立进程（fork/spawn），无父子关系要求；通过命名 POSIX 共享内存段寻址 |
| **零拷贝** | consumer 侧 `decode(copy=False)` 返回指向共享内存的 view，避免数据复制；producer 侧 `tobytes()` 单次拷贝入队 |
| **fast 编解码** | 对 `dict[str, numpy.ndarray]` 用 flat-binary 格式（4B 长度头 + JSON meta + 拼接裸字节），比 pickle 快 5-10× |
| **pin_memory 流水线** | consumer 后台线程把 numpy → `torch.from_numpy().pin_memory()`（C++ ATen，释放 GIL），主线程后续 `.cuda(non_blocking=True)` 走真异步 DMA |
| **满队列背压** | `push_raw` 满时返回 `False`；支持 `soft_limit`（深度超阈值即拒写）；可选阻塞/超时模式 |
| **监控** | Monitor 进程 attach 队列，定时报告 depth / pushed / popped / utilization |
| **可选 metrics 出口** | 监控指标可经 ZMQ PUSH 转发到外部日志服务（可选，不强制依赖 pyzmq） |
| **优雅生命周期** | 首个 producer/supervisor `create=True` 创建并拥有 shm 段，退出时 `unlink`；其余 attach，60s 重试等待段就绪 |

### 3.3 非功能性需求

- **零外部硬依赖**：核心（queue + codec + producer + consumer）仅依赖 `numpy`；prefetch 的 `pin_memory` 路径在 import torch 时启用，无 torch 时降级为不 pin；监控的 ZMQ 转发在 import pyzmq 时启用。
- **不依赖任何框架**：不 import rl_framework / 不读 RL config；所有参数（队列名、maxsize、slot_size、ip:port）由构造函数传入。
- **可独立运行**：`import shmqueue` 在干净 venv（仅装 numpy）即可工作。
- **可量化**：自带 benchmark，能脱离任何业务上下文测吞吐/延迟。

---

## 4. API 设计

```python
from shmqueue import Producer, Consumer, Monitor

# ---------- producer 端（原 data server 侧） ----------
p = Producer.connect(
    name="exp_queue_0",
    maxsize=400,
    slot_size=512 * 1024 * 1024,   # 512 MiB / slot
    create=True,                    # 首个创建者拥有并负责 unlink
)
# push 一个 batch（dict[str, ndarray]）—— 自动 fast-encode 后入队
p.push({"obs": obs_array, "action": action_array, "reward": reward_scalar})
# 或跳过 encode，直推已序列化字节
p.push_raw(raw_bytes)

# ---------- consumer 端（原 learner server 侧） ----------
c = Consumer.connect(
    name="exp_queue_0",
    maxsize=400,
    slot_size=512 * 1024 * 1024,   # 必须与 producer 一致
)                                   # attach（create=False，60s 重试）

# 同步模式
batch = c.pop()                     # → dict[str, ndarray]，空时 None

# 流水线模式（推荐高吞吐）
c.start_prefetch(pin_memory=True, workers=8, qsize=50)
batch = c.next(timeout=5.0)         # 从 prefetch deque 取，已 pin 好
c.stop_prefetch()

# ---------- 监控 ----------
m = Monitor.attach("exp_queue_0")
m.run()                             # 阻塞，每 5s 打印 depth/pushed/popped/util
```

**队列寻址与多队列**：队列由 `name` 标识（POSIX shm 段名）。多 GPU / 多 data_server 场景用一组队列 `{prefix}_queue_{idx}`，由上层编排（与原 rl 的 `experiment_name_queue_{flat_idx}` 一致，但命名逻辑上移到调用方）。

---

## 5. 模块结构

```
shmqueue/
├── README.md                 # 本文档（需求说明）
├── pyproject.toml
├── shmqueue/
│   ├── __init__.py           # 导出 Producer/Consumer/SharedMemoryQueue/Monitor
│   ├── queue.py              # SharedMemoryQueue: POSIX shm ring buffer + fcntl flock
│   ├── codec.py              # encode_batch / decode_batch (flat-binary, 零拷贝)
│   ├── producer.py           # Producer: connect + push/push_raw
│   ├── consumer.py           # Consumer: connect + pop / start_prefetch / next
│   ├── monitor.py            # Monitor: attach + 周期性 depth/pushed/popped
│   └── base.py               # LogSink: 可选 ZMQ PUSH metrics 转发
├── benchmarks/
│   ├── bench_throughput.py   # 1P1C / NP-MC 端到端 msg/s, MiB/s
│   ├── bench_codec.py        # encode/decode μs/batch vs pickle
│   ├── bench_latency.py      # push→pop 单条 p50/p99
│   └── data_source.py        # 测试数据源: mock / zmq / file (见 §8.0)
└── tests/
    ├── test_queue.py         # 基础 push/pop、满队列、多进程 attach、codec 往返
    ├── test_zmq_source.py    # ZMQ 网络发送 → shmqueue 链路
    └── test_file_source.py   # 本地文件 IO 读取 → shmqueue 链路
```

---

## 6. 源映射（从 rl_framework 抽取关系）

| shmqueue 文件 | rl_framework 源 | 抽取内容 | 剥离的 RL 依赖 |
|---------------|-----------------|---------|----------------|
| `queue.py` | `learner/shared_memory_queue.py` | `SharedMemoryQueue` 整类 + `compute_auto_maxsize` | 无（本就纯通用） |
| `codec.py` | `learner/data_server.py::_encode_batch_fast` + `learner/learner_server.py::_decode_batch_fast` | flat-binary 编解码 | 无 |
| `producer.py` | `learner/data_server.py` 的 `_connect_to_queue_server` + `_batch_builder` 的 push 段 | Producer API | replay buffer / env / data_model |
| `consumer.py` | `learner/learner_server.py` 的 `_connect_queue_server` + `_prefetch_worker` + `_recursive_pin` | Consumer + prefetch 流水线 | algo / model / DDP / env |
| `monitor.py` | `learner/queue_server.py` | Monitor | BaseServer 的 config 耦合 |
| `base.py` | `learner/base_server.py` | ZMQ PUSH log 转发 → `LogSink` | config_utils |

---

## 7. 优化路线（实现后逐步推进，量化对比）

搬运后**先保持与原 rl 行为一致**（保证正确性 + 可回灌验证），再按下列方向优化，每个优化都由 benchmark 量化：

1. **锁优化（首要）**：当前 `SharedMemoryQueue` 每次 push/pop 都 `fcntl.flock(LOCK_EX)` 整个 ring，多 producer / 多 consumer 时全局串行争用。
   - 方向 A：lock-free，meta（write/read/count）用原子 CAS 更新
   - 方向 B：per-slot 锁分片，降低争用粒度
   - 基准：1P1C / 4P4C 吞吐对比

2. **编解码**：`encode_batch` 已用 `tobytes()` 零拷贝拼接、`decode_batch` 已 `np.frombuffer(copy=False)`。
   - 方向：header 用 struct 替代 JSON；支持 torch tensor 直传免 numpy 中转；支持直接写入 shm slot 免一次中间 bytes

3. **prefetch 流水线**：`pin_memory` 释放 GIL 是已有优势。
   - 方向：调 `workers` 数与 `prefetch_qsize`；测 consumer 饥饿率；prefetch 直接 H2D（绕过 deque）

4. **背压策略**：`push_raw` 满即返回 False。
   - 方向：阻塞/超时模式；producer 侧自适应采样率

---

## 8. 验证计划

### 8.0 测试数据源（三选一，统一接口）

benchmark 与测试的 **producer 端数据来源**支持三种模式，用 `--source mock|zmq|file` 切换，统一 `DataSource` 抽象（`__iter__` 产出 `dict[str, ndarray]` batch）：

| 模式 | 说明 | 模拟的真实场景 |
|------|------|----------------|
| `mock` | 内存中即时生成随机 numpy batch | 纯压测队列/codec 上限，排除 IO 干扰 |
| `zmq` | 启动 ZMQ PULL socket，从外部 ZMQ PUSH 接收 batch（字节/lz4 压缩） | 模拟真实 worker → data_server 的网络输入（多 worker 经网络推数据） |
| `file` | 从本地文件读取预先生成的 batch（`.npy` / `.npz` / pickle） | 模拟从磁盘加载离线数据集 / 回放录制数据 |

- `mock`：基准对照组，衡量队列本身吞吐天花板
- `zmq`：测**网络接收 + 入队**的端到端，反映真实多 worker 推流下队列是否成为瓶颈
- `file`：测**磁盘 IO + 入队**，反映离线数据回放场景

benchmark 默认对三种数据源各跑一遍，报告各自吞吐，便于定位瓶颈在队列、在网络、还是在磁盘。

> ZMQ 数据源依赖 `pyzmq`（已在 `monitor` optional 依赖中，复用）；file 数据源仅依赖 numpy。benchmark 启动时若缺可选依赖则跳过该模式并告警，不报错。

### 8.1 单测 `python -m pytest tests/ -v`

- `test_queue.py`：fast codec 往返一致性；多进程 producer/consumer 数据正确；满队列 `push_raw` 返回 False、`soft_limit` 生效；`qsize/total_pushed/total_popped` 计数正确
- `test_zmq_source.py`：起一个 ZMQ PUSH 进程推 N 个 batch → producer 经 `ZmqSource` 接收 → push 入 shmqueue → consumer pop 校验内容一致
- `test_file_source.py`：预写 N 个 batch 到临时目录 → producer 经 `FileSource` 读取 → push 入 shmqueue → consumer pop 校验内容一致

### 8.2 吞吐基准 `python benchmarks/bench_throughput.py`

```
--source mock|zmq|file    # 数据源（默认三者各跑）
--producers N             # producer 进程数
--consumers M             # consumer 进程数
--batch-size MiB          # 单 batch 大小
--duration S              # 采样时长（秒）
```

输出：各数据源 × 各 P/C 配置的 msg/s、MiB/s、队列开销占比。对比原 rl `DATA_SERVER_PROFILE` 的 `recv/pushed_batch` 量级，确认抽取无损。

### 8.3 延迟基准 `python benchmarks/bench_latency.py`

push→pop 单条 p50/p99（mock 源，排除 IO）。

### 8.4 独立性

干净 venv（仅 numpy）`python -c "import shmqueue"` 成功；`pytest tests/test_queue.py tests/test_file_source.py` 通过（zmq 测试在缺 pyzmq 时 skip）。

---

## 9. 状态

- [x] 仓库结构 + 需求文档（README）
- [ ] `queue.py` 实现
- [ ] `codec.py` 实现
- [ ] `producer.py` / `consumer.py` 实现
- [ ] `monitor.py` / `base.py` 实现
- [ ] 单测 + benchmark
- [ ] 与原 rl 链路吞吐对比验证
