# Qwen Knowledge Chat

`QwenKnowledgeService::knowledge_chat` calls an already-published Alibaba Cloud Model Studio Knowledge Q&A service and returns its native SSE frames. This is separate from direct knowledge-index retrieval and Knowledge Search: it invokes the hosted Q&A flow through a published `agent_id`; it does not create, publish, or manage agents.

The documented route is Beijing-only: `POST https://{workspaceId}.cn-beijing.maas.aliyuncs.com/api/v2/apps/knowledge/chat`. Publish a Knowledge Q&A service in the console first, then bind its ID to a `QwenKnowledgeChatRef` for the current profile, account, region, and workspace. The client checks the reference scope before sending and sends each request once.

```rust,no_run
use lingxi_llm_client::providers::qwen::knowledge::{
    QwenKnowledgeChatMessage, QwenKnowledgeChatRequest, QwenKnowledgeService,
};

async fn ask(request_options: &lingxi_llm_client::RequestOptions, service: &QwenKnowledgeService<'_>) -> Result<(), Box<dyn std::error::Error>> {
    let published = service.scope().knowledge_chat_ref("aid-your-published-service")?;
    let request = QwenKnowledgeChatRequest::new(
        published,
        [QwenKnowledgeChatMessage::user_text("How do I configure an API key?")],
    );
    let mut stream = service.knowledge_chat(&request, request_options).await?;
    while let Some(event) = stream.next_event().await? {
        if event.is_complete() {
            // The native final frame is available from event.native().
        }
    }
    Ok(())
}
```

The service does not persist conversation state or return a `session_id`. The caller supplies the complete, ordered `messages` history on each request. In addition to user/assistant text and multimodal messages, the typed API supports carrying previous assistant `tool_calls` and their corresponding tool returns into the next turn. Planning, tool-calling, tool-return, and generation frames remain available through `event.native()`. Hosted tool names and JSON-string arguments are history data; this crate does not parse or execute them locally.

Use `QwenKnowledgeChatMessage::user_parts` or `assistant_parts` for text and `image_url` content. Image URLs must be HTTP(S) URLs with a host. The client does not fetch images or check remote reachability. The optional `input.request_id` is set with `with_request_id`.

Pass optional session files with `with_session_files`, up to 10. Each `QwenKnowledgeFileRef` must match the current workspace, account, profile, and endpoint. In-chat file parsing must also be enabled in the console; the API says these IDs come from `addFile`. The common short-lived session-file flow uses the `SESSION_FILE` category, but the Knowledge Chat page does not state that category as a hard request constraint. The client therefore checks reference scope without imposing a category or ID-prefix rule. The caller must ensure registered files are available to the published service. This method does not upload or create files and does not download URLs.

`with_cache_control(true)` forwards `parameters.agent_options.enable_cache_control`. Enabling it does not guarantee a cache hit; hits and cost depend on the service configuration, model, prefix, and provider usage. The client preserves native usage and does not infer cache behavior or cost from this option.

Read through the frame whose `output.choices[0].finish_reason` is `stop` and then through clean SSE EOF to confirm completion. The final frame is held until EOF and then returned with `is_complete() == true`. A clean EOF without `stop`, a late provider error, a size-limit violation, or a stream interruption returns an error. Intermediate frames remain unmodified provider data. The 8 MiB request-body limit, 8 MiB per-SSE-event limit, 64 MiB response-stream limit, and 120-second deadline are local safeguards, not provider-published service limits. The module never retries automatically because a request may already have reached the hosted service.

The contract and tool-history format follow Alibaba Cloud's first-party documentation: [Knowledge Chat](https://help.aliyun.com/en/model-studio/knowledgechat), [Multi-turn Conversations: Correctly Passing Tool Call History](https://help.aliyun.com/en/model-studio/rag/multi-turn-chat), and [addFile](https://help.aliyun.com/en/model-studio/api-bailian-2023-12-29-addfile). Tests use a mock transport and make no live requests.
