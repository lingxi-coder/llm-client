# 目录维护与发布

## 更新静态目录

`data/providers/*.toml` 是随 crate 打包的静态快照，`build.rs` 只负责从目录生成文件列表。本仓库没有原工作区的 `scripts/vendor-catalog.py`，模型和价格更新需直接编辑这些 TOML 文件。

1. 从 provider 官方模型目录和价格文档核对模型 ID、能力、上下文窗口、发布日期、token 费率与生效时间。记录所依据的链接和核对日期，不能确认的价格保持缺失，不要推测。
2. 保留第一个 `[[model]]` 之前手工维护的连接地址、协议、认证、计费与 provider 展示信息；在其后更新模型块。TOML 中表头之后的裸键会绑定到当前表，新增路由字段必须放在第一个表头之前。
3. 检查模型别名、搜索适配器、峰值时段和 region 专用连接。不要把实时目录返回的模型自动合并到静态快照；宿主需要自行合并并重建客户端。
4. 运行 `cargo test --workspace --locked`、`cargo fmt --all -- --check`、`cargo clippy --workspace --all-targets --locked -- -D warnings` 与 `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked`。价格变化另用实际 provider 文档人工核对，测试不能证明线上费率仍然有效。

## 发布 0.1.0

两个 crate 共享 `0.1.0` 版本；`lingxi-llm-client` 依赖 crates.io 上同版本的 `lingxi-agent-api`。发布需要有两个 crate 名称的发布权限和可用的 crates.io 网络连接。发布前检查 `Cargo.lock`、许可证、README、crate 归档内容和上述 CI 结果；本仓库没有自动发布工作流。

1. 在发布环境运行 `cargo package -p lingxi-agent-api --locked` 和 `cargo package -p lingxi-llm-client --list --locked`，检查打包内容。后一个命令只列出文件，尚不能验证从 registry 解析共享 crate。
2. 先运行 `cargo publish -p lingxi-agent-api --dry-run --locked`，确认后发布 `cargo publish -p lingxi-agent-api --locked`。
3. 等待 crates.io 索引可解析 `lingxi-agent-api = "0.1.0"`，再运行 `cargo package -p lingxi-llm-client --locked` 和 `cargo publish -p lingxi-llm-client --dry-run --locked`；确认后发布 `cargo publish -p lingxi-llm-client --locked`。
4. 如需线上验收，用测试账户分别检查真实 provider 请求与 HTTPS 信任链。离线测试和本地 HTTP 测试无法覆盖这两项。

上述命令说明发布步骤；维护或 CI 不会自动执行 `cargo publish`。
