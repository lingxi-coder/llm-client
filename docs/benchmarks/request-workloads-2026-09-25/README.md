# 请求工作负载优化 / Request workload optimizations

环境：macOS arm64、Rust 1.94.0、release、默认 features。每个 worker 128 个样本；同一台机器串行交替运行“基线 → 优化版”三轮。表格为三轮各自 p50/p95 的中位数。请求耗时只覆盖准备阶段，终点是进入 mock transport，不包含供应商网络或生成时间。

Environment: macOS arm64, Rust 1.94.0, release, default features. Each worker records 128 samples. Baseline and optimized binaries ran serially in alternating order for three rounds; figures are medians of per-run percentiles. Request timings end at entry to a mock transport and exclude provider latency.

基线是本轮实施开始时冻结的 dirty 工作区副本，已包含共享 client、内置目录缓存及上轮单文本快路径。两份二进制使用相同 benchmark 和 Cargo.lock；基线源文件临时保存在 `/private/tmp/llm-perf-implementation-baseline`，不是一个已提交的 git revision。该副本不属于仓库，清理临时目录后无法仅凭提交号重建它。

The baseline is a frozen dirty-worktree copy from the start of this implementation, already including shared clients, the builtin catalog cache, and the earlier single-text fast path. Both binaries use the same benchmark and Cargo.lock. The baseline source copy is temporary, not a committed revision, so the CSV files preserve the measured evidence rather than a commit-addressable baseline.

| 工作负载 / Workload | Workers | Before p50 (ms) | After p50 (ms) | p50 change | Before → after p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 简单文本 / Simple text | 1 | 0.0083 | 0.0070 | -16.1% | 0.0101 → 0.0087 |
| 128 × 4 KiB 历史 / History | 1 | 0.2787 | 0.2575 | -7.6% | 0.3686 → 0.3142 |
| 64 个工具，每个 20 字段 / Tools | 1 | 0.8944 | 0.2131 | -76.2% | 0.9586 → 0.2383 |
| 200 字段 schema / Schema | 1 | 1.5276 | 0.3150 | -79.4% | 1.6450 → 0.3683 |
| 200 字段 schema / Schema | 16 | 2.1394 | 0.3205 | -85.0% | 12.6376 → 1.7902 |
| 200 字段 schema / Schema | 64 | 2.7603 | 0.3734 | -86.5% | 137.7548 → 1.9250 |
| 列出 1,000 个模型 / Chat listing | 1 | 1.6762 | 0.3708 | -77.9% | 1.8644 → 0.4677 |
| 查询 1,000 个配置行 / Configured rows | 1 | 11.9450 | 3.9796 | -66.7% | 14.0930 → 4.4075 |

并发测试为每个 worker 创建独立的 current-thread Tokio runtime，计时含线程本地分配计数器开销。64 worker 会超额订阅 CPU，尾延迟尤其受调度影响；这些数值不是服务端 SLA。每种请求先预热一次，因此 schema 行衡量缓存热路径。首次编译仍然需要执行。

Each worker has its own current-thread Tokio runtime. Measurements include thread-local allocation accounting. The 64-worker case oversubscribes CPU and its tail is scheduler-sensitive, not a service SLA. Requests are warmed before sampling; schema figures measure reuse after compilation.

## 分配 / Allocations

以下是单 worker 的每请求分配。字节数表示累计申请量，realloc 按新大小计，不是峰值内存或 RSS。

Per-request allocations for one worker. Bytes are cumulative allocation requests, including the new size of reallocations, not peak memory or RSS.

| Workload | Allocations before → after | Allocated bytes before → after |
| --- | ---: | ---: |
| 简单文本 / Simple text | 83 → 83 | 31,446 → 10,841 |
| 128 × 4 KiB 历史 / History | 985 → 985 | 1,722,881 → 1,228,457 |
| 64 个工具，每个 20 字段 / Tools | 11,939 → 1,318 | 1,226,971 → 547,971 |
| 200 字段 schema / Schema | 17,191 → 1,695 | 1,451,206 → 181,792 |

配置查询在 blocking worker 执行，所以 CSV 中该行的分配只覆盖测量线程，不能据此推断整个查询的分配量。配置行测试使用一个包含 1,000 行的初始 profile，未设置配置目录；它测量继承行的物化，不包含磁盘写入。

Configured-row work runs on a blocking worker, so its allocation columns cover only the measuring thread. That case materializes 1,000 inherited rows without a configuration directory and does not measure disk writes.

## 附件与 tokenizer / Attachments and tokenizers

- 确定性测试验证：显式启用内容复用后，8 个并发同引用请求执行 8 次权限/可用性检查、1 次原始字节读取、1 次上传。默认 resolver 仍逐请求读取，且不会用缓存掩盖权限或文件缺失错误。测试也验证最多 4 个并发任务、消息顺序、取消后的锁释放及 Qwen 清理。没有使用真实供应商网络测量附件加速。
- Deterministic tests establish 8 authorization checks, 1 content read, and 1 upload for 8 concurrent opted-in requests. Existing resolvers retain per-request reads. Concurrency bounds, order, cancellation, and Qwen cleanup are covered; no live-provider attachment speedup is claimed.
- [tokenizers.csv](tokenizers.csv) 在同一进程中交替执行 encode/encode_fast，各 11 次，加载、输入生成、预热和完整 token IDs 一致性检查不计时。五套捆绑资产全部保持 IDs 相同。多数负载改善约 0–6%；GLM 工具 schema 样例约慢 0.3%，接近测量噪声，不宣称所有负载都有明显提速。
- Tokenizer measurements alternate encode and encode_fast in the same process, 11 samples each, excluding loading, input construction, warmup, and ID equality checks. Improvements are mostly 0–6%; the GLM tool-schema case is about 0.3% slower, near measurement noise.

## 验证与复现 / Verification and reproduction

全量 `cargo test --all-features --locked --offline` 通过（本地 HTTP 测试需要允许 loopback sockets）。最后一次 schema 锁粒度调整又通过 6 项缓存单元测试、12 项 structured-output 集成测试。最终 all-targets/all-features Clippy `-D warnings`、格式和 diff 检查通过。

The full all-features offline test suite passed with local loopback sockets enabled. The final schema lock adjustment was then checked by six cache unit tests and twelve structured-output integration tests. Final strict Clippy passed for all targets and features.

```sh
cargo run --release --locked --offline --example performance_workloads -- --iterations 128 --parallel
cargo run --release --all-features --locked --offline --example tokenizer_count_bench -- 11
```

源码：[performance_workloads.rs](../../../examples/performance_workloads.rs)、[tokenizer_count_bench.rs](../../../examples/tokenizer_count_bench.rs)。直接 schema_compile/cached_validate 行是编译器参考成本，未使用 client 缓存，不能把两个不同操作的比值当成请求提速倍数。

原始数据 / Raw data: [before 1](before-round1.csv), [after 1](after-round1.csv), [before 2](before-round2.csv), [after 2](after-round2.csv), [before 3](before-round3.csv), [after 3](after-round3.csv), [summary](summary.csv).

Cargo.lock SHA-256: `6a35711043ea65419814654ecfa98e679750c3ef47ffb2b7ab4dc4331b8d7350`.
