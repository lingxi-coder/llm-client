# llm-client

从 Lingxi 拆分的 Rust LLM 客户端，crate 名称保留为 `lingxi-llm-client`。
支持 provider 配置、模型路由、故障转移、流式响应和费用计算，内置 OpenAI、Anthropic、Gemini 及托管平台的 wire codec。
HTTP/WebSocket 传输由调用方通过 `Transport` 提供；凭证按请求传入。

## 项目结构

- `src/`：客户端、认证、协议编解码、流解析与模型目录。
- `data/providers/`：内置 provider 配置，由 `build.rs` 打包。
- `tests/`：集成测试及模拟传输。
- `crates/agent-api/`：已有的共享数据类型依赖，随仓库保留以兼容现有 API。

本仓库不依赖原 Lingxi 工作区的本地路径，也不包含 agent runtime 或 CLI。
`lingxi-agent-api` 当前保留完整类型集合；尚未裁剪其中的 agent 专用类型。

## 开发

使用 `rust-toolchain.toml` 指定的 Rust 1.94.0：

```sh
cargo test --workspace --locked
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --locked -- -D warnings
```

测试使用模拟传输，不需要真实 LLM API key。
客户端接入方式可参考 `tests/support/mod.rs` 和 `tests/failover.rs`。
内置模型目录与价格是静态快照，需要随 provider 的变更维护。

## 在其他 Rust 项目中使用

```toml
[dependencies]
lingxi-llm-client = { git = "https://github.com/lingxi-coder/llm-client", branch = "main" }
lingxi-agent-api = { git = "https://github.com/lingxi-coder/llm-client", branch = "main" }
```

共享类型通过 `lingxi_agent_api::protocol` 导入。两个依赖应指向同一 Git revision；
需要可复现构建时使用固定 `rev`。

## 来源

以原 Lingxi 工作目录中的 `crates/llm-client` 和 `crates/agent-api` 当前快照建立新历史，
包含拆分时尚未提交的相关修改。原仓库当时 HEAD 为
`df36a9b173a43783f944b5bf88c0d23e36dd3739`，不是该快照的完整版本标识。
保留原包的 `MIT OR Apache-2.0` 许可证声明。
