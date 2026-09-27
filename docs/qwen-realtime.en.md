# Qwen-Omni Realtime (WebSocket)

This module implements Alibaba Cloud Model Studio's Qwen-Omni Realtime WebSocket session protocol. The current route supports the two regions and three realtime models listed by the first-party connection guide: Beijing, Singapore, Qwen3.8 Omni Flash, Qwen3.5 Omni Plus, and Qwen3.5 Omni Flash.

Sources: Alibaba's [Realtime guide](https://help.aliyun.com/zh/model-studio/realtime), [client event reference](https://help.aliyun.com/zh/model-studio/client-events), and [server event reference](https://help.aliyun.com/zh/model-studio/server-events).

## Region, workspace, and credentials

`QwenRealtimeRoute::new` requires a region and workspace ID and constructs only the corresponding full WSS route:

- Beijing: `wss://{WorkspaceId}.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime?model=...`
- Singapore: `wss://{WorkspaceId}.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/realtime?model=...`

The current API key is sent in the handshake as `Authorization: Bearer …`. The host owns and supplies a key that matches the selected region and workspace on each connection. `QwenRealtimeScope` binds the profile, account, region, workspace, and endpoint fingerprint; it never stores credentials.

```rust,ignore
let route = QwenRealtimeRoute::new(QwenRealtimeRegion::Beijing, workspace_id)?;
let scope = QwenRealtimeScope::new("qwen-mainland", "billing-account-1", &route)?;
let config = QwenRealtimeConfig::new(QwenRealtimeModel::Qwen38OmniFlashRealtime);
let (session, driver) = QwenRealtimeSession::connect(
    transport,
    route,
    scope,
    api_key, // a per-connection Secret<String>
    config,
    RealtimeLimits::default(),
).await?;
```

After connecting, the driver waits for `session.created`, sends `session.update`, and returns only after `session.updated`. The model is included in both the URL `model` query parameter and the session config, as shown by Alibaba. Output can be text-only or text plus audio; audio-only is not documented as valid. The configured audio path uses the documented 16 kHz mono PCM16 input and 24 kHz PCM16 output defaults. `semantic_vad` is supported for the listed model family; manual mode sets `turn_detection` to `null`.

## Input and output

- Audio input uses `RealtimeInput::Audio` with `Pcm16 { sample_rate_hz: 16000 }` and is sent as Base64 in `input_audio_buffer.append`.
- An image/video frame uses `RealtimeInput::Image`, accepts only `image/jpeg`, and is limited to 256 KiB after Base64 encoding. Qwen receives JPEG frames one at a time through `input_image_buffer.append`; at least one audio append must be queued successfully before the first image. Alibaba recommends roughly one frame per second. Capture, frame extraction, and pacing belong to the host.
- Text input creates a user `input_text` item with `conversation.item.create`, then explicitly sends `response.create`.
- Server VAD commits audio and creates a response automatically. Manual mode requires the host to call `CommitAudio` after audio input and then `ContinueResponse`. These explicit operations are rejected locally in VAD mode to prevent duplicate commits or responses. `Interrupt` maps to Qwen's `response.cancel`.
- WebSocket audio output is delivered as PCM16, 24 kHz byte deltas. `response.audio_transcript.*` and text-mode `response.text.*` remain distinct events. `response.done` preserves the native response and usage.
- `error` is surfaced as `ProviderError` and does not automatically end the session; unknown events retain native JSON. A remote disconnect emits `ConnectionInterrupted` and stops the driver, without reconnecting or replaying input.

This adapter does not capture or play device audio, extract video frames, reconnect automatically, or execute Function Calling/MCP tools and return their results. Image/audio frames are bounded by the session frame limit and sender queue; on a full queue the caller gets an error and chooses how to proceed.

`RealtimeInput::ClearAudio` sends `input_audio_buffer.clear` to discard uncommitted input audio. Its acknowledgement remains a native provider event. It does not cancel output or close the session; `FinishSession` remains unsupported.

Contract: [client events](https://help.aliyun.com/en/model-studio/client-events).
