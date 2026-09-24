# 目录维护与发布

[English](maintenance.en.md)

## 更新静态目录

`data/providers/*.toml` 是随 crate 打包的静态快照，`build.rs` 只负责从目录生成文件列表。本仓库没有原工作区的 `scripts/vendor-catalog.py`，模型和价格更新需直接编辑这些 TOML 文件。

1. 从 provider 官方模型目录和价格文档核对模型 ID、能力、上下文窗口、发布日期、token 费率与生效时间。记录所依据的链接和核对日期，不能确认的价格保持缺失，不要推测。
2. 保留第一个 `[[model]]` 之前手工维护的连接地址、协议、认证、计费与 provider 展示信息；在其后更新模型块。TOML 中表头之后的裸键会绑定到当前表，新增路由字段必须放在第一个表头之前。
3. 检查模型别名、搜索适配器、峰值时段和 region 专用连接。实时目录同步只更新宿主的本地配置，不改写仓库静态快照；维护静态目录仍需核对上游资料。缺失的能力字段保留未知，明确的 `false` 才写为不支持。
4. 运行 `cargo test --locked`、`cargo test --manifest-path tests/downstream-docs/Cargo.toml --doc --locked --offline --target-dir target`、`cargo fmt --all -- --check`、`cargo clippy --all-targets --locked -- -D warnings` 与 `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --locked`。独立下游工程仅用文档声明的直接依赖编译中英文示例。价格变化另用实际 provider 文档人工核对，测试不能证明线上费率仍然有效。

第一方 Anthropic 预设的 `id` 使用官方 wire ID；旧聚合平台名称只保存在 `aliases` 中用于本地解析。更新时核对[官方 ID 规则](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions)和[退役表](https://platform.claude.com/docs/en/about-claude/model-deprecations)，不要把聚合平台 ID 去掉前缀后直接发送到官方端点。2026-09-23 已移除第一方退役的 Haiku 3、Opus 4/4.1 和 Sonnet 4；聚合平台自己的条目保持不变。

架构调整还需运行 `cargo test --all-features --locked`，逐个检查 `tokenizer-*` feature，并验证默认与全部 feature 的 Clippy／Rustdoc。`tests/downstream-docs` 同时验证中英文指南和外部实现的 Transport、WireCodec、StreamDecoder、AccountUsageSource。使用 `cargo tree --no-default-features --edges normal` 检查默认依赖隔离。[验收记录](architecture-validation.md)提供体积、内存负载与命令。

更新目录时保留各模型的 `features` 和经过核验的 `pricing.rules`：effort 不作为单价维度，fast 仅使用该模型的官方倍率或独立费率；保留适用计费桶、币种、上下文区间、有效时间、来源及核验日期。未核实的 fast 价格留空。

## 发布 0.1.0

本仓库只发布 `lingxi-llm-client`，协议类型包含在该 crate 内。发布需要该 crate 的发布权限和可用的 crates.io 网络连接。发布前检查锁文件、许可证、README、归档内容及 CI 结果；本仓库没有自动发布工作流。

1. 运行 `cargo package --list --locked` 检查归档清单。应包含协议源码、指南、provider 预设和内置 tokenizer 资源，并排除本地运行状态及构建产物。
2. 运行 `cargo package --locked` 验证完整独立归档及包内构建，再运行 `cargo publish --dry-run --locked`。本地验证尚未提交的改动时，可为 `cargo package` 添加 `--allow-dirty`；发布验证使用干净检出。
3. 发布审查完成后执行 `cargo publish --locked`，无需预先发布配套协议 crate。
4. 如需线上验收，用测试账户分别检查真实 provider 请求与 HTTPS 信任链。离线测试和本地 HTTP 测试无法覆盖这两项。

上述命令描述手动发布流程；维护和 CI 不会自动发布。
