# 会话增量续接验证记录

日期：2026-11-03（America/Los_Angeles）。基线：`a30a3a96907a7e3847af4aa999e7ddf7b4590dcf` 加当前工作区实现。

最终组合命令退出码 0：

```sh
CARGO_TARGET_DIR=/Users/luolingfeng/lingxi/.build/llm-continuation \
CARGO_BUILD_JOBS=2 CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 \
CARGO_INCREMENTAL=0 cargo test --offline --features responses-websocket \
  --lib --test conversation_session --test continuation --test responses_session \
  --test presets --test gemini_interactions_codec --test interactions --test families \
  --test prepared_calls --test shared_client --test failover --test openai_wire \
  --test anthropic_wire --test anthropic_conversation --test anthropic_message_controls
```

| 测试组 | 通过数 |
| --- | ---: |
| library（含 WebSocket 特性） | 625 |
| anthropic_conversation | 24 |
| anthropic_message_controls | 18 |
| anthropic_wire | 29 |
| continuation | 3 |
| conversation_session（新增） | 23 |
| failover | 43 |
| families | 38 |
| gemini_interactions_codec | 12 |
| interactions | 12 |
| openai_wire | 38 |
| prepared_calls | 33 |
| presets | 33 |
| responses_session | 19 |
| shared_client | 4 |
| 合计 | 954 |

新增测试覆盖三轮增量、HTTP 与首个 WS 错误事件恢复、恢复上限、每次发送准入和响应记账、未知结果不重试、账户/transport/历史/固定参数变化、取消与旧流隔离、显式分支、自动缓存、Anthropic 工具结果、refusal/truncation/fallback、组织设置不支持 thread 时降级、Gemini 终止帧省略 steps、智谱存储和单对象 content、UTF-16/原始 JSON、Content-Length 与准备后修改拒绝。

修改的 Rust 文件通过 rustfmt 检查；`git diff --check` 通过。最初有五项现有代理测试被沙箱禁止绑定本机端口；允许 `127.0.0.1` 临时端口后，单独重跑及最终组合回归均通过。最终完整日志：`/private/tmp/llm-continuation-verified.log`。

这是 SDK 源码和离线合同验证，不是付费模型、真实账户、生产包或性能验收。未修改 Harness 的固定 Git 依赖 pin，未推送或部署。宿主需在发布 SDK 修订后接入 `ConversationSession`；原有普通 chat 调用不会自动获得新的 HTTP/Anthropic 增量行为。

接口与 provider 边界见[使用文档](conversation-continuation.md)。
