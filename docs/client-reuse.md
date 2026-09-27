# 共享 Client、配置更新与性能

[English](client-reuse.en.md)

应用启动时构建一个长期使用的 `LlmClient`，并把它的 `clone()` 交给并行任务。Clone 只共享运行资源和配置发布槽，不重新解析目录、创建 HTTP 连接池或复制模型配置。`model`、`thinking.effort`、`service_tier`（Fast）和凭证都是请求参数，不是 client 的复用 key。只有区域、transport／代理／TLS、扩展服务或配置来源需要独立时，才构建另一套 client。

## 初始化和动态配置

`builtin_catalog()` 返回进程内只解析一次的只读内置目录。需要编辑目录时，`builtin_providers()` 返回它的独立副本。`build()` 返回只读请求句柄；需要动态配置时，使用同步的 `build_managed()` 取得请求句柄和配置管理器：

```rust,no_run
use lingxi_llm_client::{builtin_catalog, LlmClientBuilder};
use lingxi_llm_client::protocol::Region;

# async fn example() -> Result<(), Box<dyn std::error::Error>> {
let (client, config) = LlmClientBuilder::new(builtin_catalog()?)?
    .with_region(Region::International)
    .build_managed()?;
config.set_config_dir("./config").await?;

let worker_client = client.clone();
// 可将 worker_client 移入一个并行任务；此处不会新建连接池。
assert_eq!(worker_client.region(), client.region());
config.set_tracked_models("openai", ["gpt-4.1-mini".to_owned()]).await?;
// 已存在的 client 和它的 clone 在后续操作中自动使用新配置。
# Ok(())
# }
```

`ClientConfigManager` 的修改、持久化、目录同步、账户源重新绑定和管理查询均为异步 `&self` 方法。它独立串行化配置写入，文件锁和 I/O 在 blocking 线程池执行；请求等待网络期间不持有配置锁。管理器本身不实现 `Clone`，需要跨管理任务共享时使用 `Arc<ClientConfigManager>`。丢弃管理器不会使 client 失效，client 继续使用最后一次发布的配置。

配置发布同时安装连接、模型索引、账户源绑定和缓存代际。新请求使用新状态，在途请求保持原状态，包括 failover、账户批量查询、流解码及文件清理。验证或写盘失败不发布新状态；取消尚未写盘的操作可终止提交。一旦开始写盘，即使调用方取消等待，后台 worker 仍完成持久化结果的内存安装和发布。

`prepare_provider_sync(...).await` → `operation.fetch().await` → `apply_provider_sync(...).await` 支持独立调度目录请求。Fetch 不持有管理锁，提交时重读并合并文件；切换配置目录会使之前准备的同步结果失效。

## 固定一次操作链的配置

长期保存的 live 服务句柄，每次异步操作首次执行时捕获最新快照；同步查询在入口捕获。若需要“路由 → 预估 → 请求 → 计价”使用同一套配置，先保存 `client.snapshot()`：

```rust,no_run
use lingxi_llm_client::{LlmClient, RequestOptions};
use lingxi_llm_client::protocol::{ChatRequest, PricingContext, Submission, Usage};

async fn run(client: &LlmClient, request: &ChatRequest, options: &RequestOptions)
    -> Result<(), Box<dyn std::error::Error>>
{
    let snapshot = client.snapshot();
    let route = snapshot.resolve_in(&request.model, Some("openai"))?;
    let assumed_usage = Usage { input_tokens: 1000, output_tokens: 500, ..Default::default() };
    let pricing = PricingContext {
        service_tier: request.service_tier,
        input_tokens: Some(1000),
        ..Default::default()
    };
    let estimate = snapshot.estimate_cost(&route, &assumed_usage, &pricing)?;
    let response = snapshot.chat().complete_in("openai", request, options).await?;
    let actual = snapshot.estimate_actual_cost(&route, &response, Submission::Interactive)?;
    println!("estimated: {estimate:?}; actual: {actual:?}");
    Ok(())
}
```

`ClientSnapshot` 提供全部服务和路由、token、价格、账户查询。`profiles()` 与 `provider()` 的借用查询移到 snapshot；需要保留返回的引用时，先把 snapshot 绑定到局部变量。Live client 的列表和价格便利方法返回 owned 结果，但每次调用独立捕获配置，不能用它恢复历史价格。旧 snapshot 保留旧账户源绑定，直到该 snapshot 及其操作结束；长时间保存快照会延长旧配置的生命周期。

## 缓存边界

HTTP、codec、认证实现和附件上传缓存由 client clone 共享。凭证仍逐请求提供，不写入配置。上传缓存与并发上传去重使用相同的完整键，包含 endpoint、稳定文件账户 scope、附件版本、配置 namespace 和 profile generation。缺少稳定 `file_account_scope` 时不跨请求复用；不同账户不能因共享 client 而共享上传结果。

切换配置目录或更改相关连接配置会使后续请求使用新的缓存代际。旧上传完成或旧请求报告文件 404，只能更新旧代缓存；Qwen 限速继续按 endpoint 与稳定账户身份共享。Qwen 自动清理的上传不复用。宿主的附件 resolver 仍负责原始内容读取和访问检查；不缓存模型响应或账户余额，也不改变现有重试策略。

`AttachmentResolver::validate_content_reuse` 默认返回 `false`，已有实现仍在每次请求调用 `resolve`。宿主可在这个方法中检查当前调用者的访问权限、附件仍然可用以及完整引用对应同一不可变内容，然后返回 `true`，显式允许复用原始字节。该检查在缓存命中、未命中及等待并发读取之前都会执行；返回错误不会回退到缓存。原始内容缓存由同一 client 的 clones 共享，按完整附件引用隔离，最多保存 256 项、合计 64 MiB。首次入缓存会把有效字节复制到独立的紧凑缓冲，当前请求和后续命中共享这份缓冲，避免小切片长期持有大块底层内存。缓存允许跨配置快照复用源字节，远端文件缓存仍按账户和配置代际隔离。

单次请求先完成全部附件元数据、总大小和重复引用一致性校验，再以最多 4 个任务并行读取或上传不同附件。相同附件仍去重，消息顺序、总 deadline、Qwen 限速与取消后的文件清理保持原契约。

结构化输出的 schema 编译结果按内容复用，最多保留 16 个条目及 4 MiB 序列化 schema 源内容；此限制不等于编译器对象的实际内存大小。命中同一缓存条目的并发初始化合并，失败结果也受同样容量限制。模型能力、strict 子集及 profile 原生字段冲突仍按当前请求检查。缓存匹配区分 `0.0` 和 `-0.0` 等长度不同的数字表示。请求继续执行 1 MiB schema 上限；仅用于本地响应验证的更大 schema 不进入缓存。

## 迁移和测量

- 把逐任务 `build()` 改为启动时构建，任务中 `client.clone()`。
- 把 `let mut client = ...build()?` 和 client 上的配置 API 改为 `let (client, config) = ...build_managed()?`，管理操作使用 `config.method(...).await`。
- 把 `client.profiles()` / `client.provider(name)` 改为局部 snapshot 上的借用查询。`deleted_builtin_profiles()`、`tracked_models()`、`configured_models()` 改由 config 异步返回 owned 数据。
- 需要稳定版本的多步调用统一使用一个 snapshot；直接使用 live client 的操作会在开始时读取最新配置。

运行不访问真实 provider 的 release 基准：

```sh
cargo run --release --example client_reuse_bench -- --iterations 1024
```

基准分别测量内置目录冷／热加载、可编辑目录复制、构建、clone、快照捕获和路由解析；1／16／64 个并发 worker 的 mock 请求报告本地准备和整体耗时分位数、分配次数及字节数。添加 `--mixed` 可单独测量与持久化配置更新并发的请求。CSV 输出包含 `p50_ns`、`p95_ns`、`allocations_per_sample` 和 `allocated_bytes_per_sample`；字节数是累计申请量（realloc 计新大小），不是 RSS。逐线程请求计数排除配置写入线程；准备阶段截至进入 mock transport，总耗时还包含 mock 响应解码。比较时使用相同机器、构建选项和工作树内容；旧版本的共享路径以 `Arc<LlmClient>` 为基线，避免把原本已可共享的场景误报为新增收益。数字反映本地准备成本，不代表真实 provider 的生成速度，也不作为依赖机器速度的 CI 阈值。


### 2026-09-25 第一轮本地测量

环境：macOS 15.7.8、arm64、Rust 1.94.0，release、默认 features，每个 worker 1,024 次迭代。停止并行构建和测试后，按“基线 → 优化版”串行交替运行三轮；下表是各轮 p50／p95 的中位数，完整记录及 min／max 见[汇总 CSV](benchmarks/client-reuse-2026-09-25/summary.csv)。基线来自本次改造前的 dirty 工作树副本，复制时竞态带入的本轮账户／缓存改动已恢复为原接口，其余既有改动保留。

| 操作 | 基线 p50 | 优化后 p50 | 分配次数：基线 → 优化后 |
| --- | ---: | ---: | ---: |
| 热加载可编辑内置目录 | 15.453 ms | 0.418 ms | 113,067 → 10,510 |
| 构建单 profile client（mock transport） | 3.333 µs | 3.292 µs | 84 → 82 |
| 共享句柄 clone + drop | 4 ns（`Arc<LlmClient>`） | 5 ns（`LlmClient`） | 0 → 0 |
| 捕获 snapshot + drop | — | 22 ns | 0 |

可编辑目录热加载约快 37 倍，累计申请字节从 34,237,122 降为 1,102,702；首次调用仍需解析目录。只读 `builtin_catalog()` 热访问为零分配。句柄和只读目录操作每个采样批量执行 128 次，再折算单次时间，以减少计时器分辨率影响；这些纳秒结果只表示量级。

已有共享 client 的 mock 请求准备热路径有小幅额外开销：

| 并发 worker | 基线 p50 / p95（µs） | 优化后 p50 / p95（µs） |
| --- | ---: | ---: |
| 1 | 9.667 / 11.584 | 10.791 / 12.833 |
| 16 | 10.291 / 21.375 | 11.333 / 21.625 |
| 64 | 10.500 / 20.792 | 11.917 / 22.417 |

准备 p50 增加约 1.0–1.4 µs；每次仍为 95 次分配，累计申请字节由 34,130 变为 34,354。16 并发 p95 的三轮范围为基线 14.500–21.416 µs、优化后 15.583–22.833 µs，尾延迟受调度影响。本次收益集中在避免反复解析目录、复用初始化结果，以及配置更新不阻塞在途请求，不宣称已共享请求本身提速。

另一次 CPU 空闲时的[混合运行](benchmarks/client-reuse-2026-09-25/optimized-mixed.csv)完成 19 次持久化配置提交，1／16／64 并发准备 p95 分别为 12.292／21.833／22.500 µs。它验证此负载下可并行运行，不代替确定性的并发契约测试。原始记录：基线[第 1 轮](benchmarks/client-reuse-2026-09-25/baseline-round1.csv)、[第 2 轮](benchmarks/client-reuse-2026-09-25/baseline-round2.csv)、[第 3 轮](benchmarks/client-reuse-2026-09-25/baseline-round3.csv)；优化后[第 1 轮](benchmarks/client-reuse-2026-09-25/optimized-round1.csv)、[第 2 轮](benchmarks/client-reuse-2026-09-25/optimized-round2.csv)、[第 3 轮](benchmarks/client-reuse-2026-09-25/optimized-round3.csv)。


### 后续请求工作负载优化

后续实现覆盖 schema 编译复用、文本和工具 schema 借用序列化、附件有界并发、token 计数与目录物化。三轮对比、16/64 worker 结果及测量边界见[工作负载测量和原始 CSV](benchmarks/request-workloads-2026-09-25/README.md)。可运行 `cargo run --release --example performance_workloads -- --iterations 128 --parallel` 在本机复测。
