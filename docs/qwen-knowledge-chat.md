# Qwen Knowledge Chat

`QwenKnowledgeService::knowledge_chat` 调用阿里云百炼已发布的 Knowledge Q&A 服务，并逐帧返回其原生 SSE 输出。它与低层知识库检索及 Knowledge Search 是不同接口：本服务通过已发布的 `agent_id` 执行托管问答流程，不创建、发布或管理 agent。

接口只记录在北京工作区：`POST https://{workspaceId}.cn-beijing.maas.aliyuncs.com/api/v2/apps/knowledge/chat`。调用方须先在控制台创建并发布 Knowledge Q&A 服务，再用当前 profile、账号、区域和 workspace 构造 `QwenKnowledgeChatRef`。客户端会在发送前检查引用作用域，并且每个请求只发送一次。

```rust,no_run
use lingxi_llm_client::qwen_knowledge::{
    QwenKnowledgeChatMessage, QwenKnowledgeChatRequest, QwenKnowledgeService,
};

async fn ask(service: &QwenKnowledgeService<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let published = service.scope().knowledge_chat_ref("aid-your-published-service")?;
    let request = QwenKnowledgeChatRequest::new(
        published,
        [QwenKnowledgeChatMessage::user_text("如何配置 API Key？")],
    );
    let mut stream = service.knowledge_chat(&request).await?;
    while let Some(event) = stream.next_event().await? {
        if event.is_complete() {
            // Native final frame is available from event.native().
        }
    }
    Ok(())
}
```

该服务没有会话存储，也不会返回 `session_id`。每次调用都要由调用方提供完整、有序的 `messages` 历史。除了 user/assistant 文本和图文消息，类型还支持把先前轮次的 assistant `tool_calls` 与对应 tool 返回原样带入下一轮。Knowledge Chat 中的 planning、tool calling、tool return 和 generation 数据保存在 `event.native()`；tool function 名称及 JSON 字符串参数只是托管服务的历史内容，本库不会在本机解析或执行它们。

消息可用 `QwenKnowledgeChatMessage::user_parts` / `assistant_parts` 加入文本与 `image_url`。图片地址必须是带 host 的 HTTP(S) URL；客户端不会获取图片，也不检查远端网络可达性。可选 `input.request_id` 通过 `with_request_id` 设置。

可选 session files 使用 `with_session_files`，最多 10 个。每个 `QwenKnowledgeFileRef` 必须属于当前 workspace/account/profile/endpoint。控制台还须启用 in-chat file parsing；官方接口要求这些 ID 来自 `addFile`。常见的短期会话文件工作流使用 `SESSION_FILE` 类别，但 Knowledge Chat 页面没有把类别写成此接口的硬性请求限制，所以本库只验证引用作用域，不按类别或 ID 前缀拦截。调用方负责确认注册文件可供该已发布服务使用。本方法不上传文件、不创建文件，也不下载 URL。

`with_cache_control(true)` 会原样传递 `parameters.agent_options.enable_cache_control`。开启它不保证本次缓存命中；是否命中及费用取决于服务配置、模型、前缀和平台返回的 usage。本库保留 provider 原生 usage，不根据该选项推算缓存或费用。

必须读取到 `output.choices[0].finish_reason` 为 `stop` 的帧并正常读到 SSE EOF 才能把结果视为完成。该终帧会延迟到 EOF 后才返回并标记 `is_complete() == true`。没有 stop 的正常 EOF、晚到 provider error、超限或断流都会返回错误；已返回的中间帧仍是 provider 原始数据。请求体上限 8 MiB、单个 SSE 事件上限 8 MiB、响应流上限 64 MiB 和 120 秒 deadline 是本地保护值，不是服务商公布的业务限额。模块不自动重试，因为请求可能已到达托管服务。

当前接口契约和多轮工具历史格式参考阿里云一方文档：[Knowledge Chat](https://help.aliyun.com/en/model-studio/knowledgechat)、[Multi-turn Conversations: Correctly Passing Tool Call History](https://help.aliyun.com/en/model-studio/rag/multi-turn-chat) 和 [addFile](https://help.aliyun.com/en/model-studio/api-bailian-2023-12-29-addfile)。本模块的测试使用 mock transport，不会发送在线请求。
