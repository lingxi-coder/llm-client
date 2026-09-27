# Preserving Anthropic-native content

Known Anthropic Messages blocks such as `text`, `thinking`, and `tool_use` continue to decode into their corresponding common types. Other complete blocks are retained as `ContentBlock::ProviderContent` with the `AnthropicMessages` protocol identity. Text blocks with native citations are preserved the same way. This allows a later Anthropic Messages request to replay `server_tool_use`, `web_search_tool_result`, `web_fetch_tool_result`, code execution results, tool-search results, and content block types added in the future.

`ProviderContent` means that Anthropic-native data must be preserved; it does not mean a client tool call. The codec does not execute Anthropic-hosted web search, web fetch, code execution, or tool search, and it does not convert these operations into `ToolUse`. Replay on another protocol is rejected. Applications that need a local operation should use a separately declared client tool and its host permission flow.

For streams, unknown top-level events and the start, delta, and stop frames of opaque native or unknown content blocks are emitted as `StreamEvent::ProviderEvent { protocol, payload }`. Each frame remains separate, including repeated identical frames. These events are observational only: they cannot be converted into message content or sent to a client tool for execution. If a native block reaches `content_block_stop` and all its deltas are understood, the codec also emits `ProviderContent`. Server-tool `input_json_delta` fragments are assembled into the final `input`, while other fields from the opening block are preserved. An unknown delta or invalid fragmented JSON keeps the event frames and suppresses `ProviderContent` for that block so incomplete data is not replayed. Pending native blocks and server-tool input share an 8 MiB aggregate buffer limit.

Non-empty initial text and subsequent fragments from a streamed text block are emitted as `TextDelta`. When a `citations_delta` arrives, the codec combines the text and citations received so far and emits complete native `ProviderContent` at `content_block_stop`, preserving the other fields from the opening block. Anthropic defines each `citations_delta` as one citation to add to the current block; citation objects such as character, page, and content-block locations are retained unchanged. Plain text blocks do not produce an extra native content block.

Observable native frames remain separate `ProviderEvent`s. If citations first appear mid-stream, the codec emits that block’s buffered start/text/citation frames in their original per-block order when it detects the citation. Their observation events can therefore appear later than other stream events and do not imply global response-frame timing. An unknown or malformed delta keeps its frames and suppresses replayable `ProviderContent` for that block. Web-search attribution continues through `WebSearch`. The codec does not promise to adapt every Anthropic tool or citation format into a common execution interface.

Consumers of `ModelStream` can inspect each stream event. The structured stream collector retains events in `StructuredStreamResult.events`; a plain `ChatResponse` has no raw event transcript, so callers that save only the final response do not retain `ProviderEvent` data.

## Official documentation

- [Messages API](https://platform.claude.com/docs/en/api/http/messages)
- [Streaming Messages](https://platform.claude.com/docs/en/build-with-claude/streaming)
- [Citations, including citation streaming](https://platform.claude.com/docs/en/build-with-claude/citations)
