# Architecture validation / 架构验收

The refactor preserves the dirty workspace that existed before implementation. No commit or publication was made. Per the final requirement, v1 configuration and usage-only response formats are rejected; no migration baseline or v1 backup code remains.

本次是在原有未提交改动上增量实施，没有提交或发布。按最终要求，不再兼容 v1 配置与只有 usage 数值的旧响应；没有迁移基线或 v1 备份逻辑。

## Verification / 验证

| Check | Result |
| --- | --- |
| Default features | 529 unique unit/integration tests + 17 doctests passed |
| All features | 533 unique unit/integration tests + 17 doctests passed |
| Downstream bilingual examples and public extension traits | 44 doctests + 1 integration test passed |
| Five individual tokenizer features | Each compiled with `--no-default-features` |
| Default dependency graph | No `tokenizers`, `onig`, `onig_sys`, `tiktoken-rs`, or `xz2` normal dependencies |
| Formatting, Clippy | fmt and default/all-feature Clippy with warnings denied passed |
| Rustdoc | Default and all features, warnings denied, passed |
| Standalone package | Default package verification and extracted all-feature compilation passed; 5 XZ assets and 4 license files present; no agent-api crate or runtime/build state |

Counts exclude the subprocess duplicate in the configuration-directory regression. Cross-codec contracts cover all nine protocol families with 1-byte, mixed and coalesced chunks, complete/stream text and usage parity, truncation, valid events preceding errors, and terminal resource release. Existing tests continue to cover native reasoning/signatures, tool identities, modality order, file replay, failover and cancellation. New configuration tests cover inherited fields, override reset, full replacement, filtered fallback, duplicate wire rows, directory reordering/recovery, observations, concurrent updates and version rejection. Deadline tests include custom transports that ignore timeouts, stalled bodies/RPC, remaining page budgets, partial reports, and unordered account scheduling.

测试计数排除了配置目录回归中的子进程重复项。契约测试覆盖九个协议族、单字节／混合／合并分片、完整与流式语义、截断、错误前事件保留和终态连接释放。原有 reasoning、签名、工具身份、多模态顺序、文件重放、failover 和取消测试继续通过。配置新增测试覆盖字段继承、覆盖重置、完整替换、备用快照、重复 wire 行、目录重排／恢复、账户观测、并发更新与版本拒绝。Deadline 测试覆盖忽略 timeout 的 transport、停滞响应体／RPC、分页剩余预算、部分报告和账户并发调度。

## Size and memory / 体积与内存

Measured on macOS 15.7.8, arm64, Rust 1.94.0 (`4a4ef493e`), with Cargo's default release profile. The before snapshot is the original dirty workspace, not Git HEAD. Its source tar SHA-256 is `1f0721837da663ee5590f27182c27d6e77e4cc9946594b8e404d652c9f79b484`.

同一 macOS 15.7.8 / arm64 / Rust 1.94.0 环境，使用默认 release 配置。重构前取自原始工作区快照（含已有改动），不是 Git HEAD。

| Measurement | Before | After (default features) | Change |
| --- | --- | --- | --- |
| Release crate archive / rlib | 24,480,184 bytes (23.35 MiB) | 14,970,912 bytes (14.28 MiB) | -38.8% |
| Release benchmark executable | 3,271,456 bytes (3.12 MiB) | 3,517,024 bytes (3.35 MiB) | +7.5% |
| Attachment peak RSS | 160,235,520 bytes (152.81 MiB) | 70,369,280 bytes (67.11 MiB) | -56.1% |

The workload resolves a 32 MiB PDF, uploads it once, then makes a second completion using the same account scope and attachment revision. HTTP is mocked; no provider or credential is used. RSS is a single observed peak from `/usr/bin/time -l`; it is not an allocation trace or a live-service latency measurement. The rlib is smaller while the sample executable is slightly larger; downstream executable sizes depend on which APIs are linked. Build times are not compared because build-cache states differ.

负载为 32 MiB PDF：首次上传后，以同一账户 scope 和 revision 再请求一次，命中缓存。HTTP 完全模拟，不使用真实服务或凭证。RSS 为 `/usr/bin/time -l` 的单次峰值观测，不是分配跟踪或服务延迟测试。rlib 缩小，示例二进制略有增加；下游二进制体积取决于实际链接的 API。构建缓存状态不同，因此不比较构建耗时。

```sh
cargo build --release --locked --offline --example architecture_probe
/usr/bin/time -l target/release/examples/architecture_probe
cargo test --locked --offline
cargo test --locked --offline --all-features
cargo test --manifest-path tests/downstream-docs/Cargo.toml --locked --offline --target-dir target
cargo clippy --locked --offline --all-targets --all-features -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --locked --offline --all-features --no-deps
cargo package --locked --offline --allow-dirty
```

Source archives retain all tokenizer assets and their licenses. Features only control compilation and embedding. Validation here uses fixtures and local HTTP servers; live provider calls were not run.

源码包保留全部 tokenizer 资产与许可证，feature 只控制编译和嵌入。本次验证使用 fixture 与本地 HTTP 服务，没有运行真实供应商请求。
