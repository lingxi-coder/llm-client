# OpenAI Responses computer decisions

The current [OpenAI computer tool](https://developers.openai.com/api/docs/guides/tools-computer-use#use-the-computer-tool)
returns an ordered `computer_call.actions` array. The caller provides the
environment, authorizes and executes permitted actions, then returns a screenshot
with the matching `call_id`. A call's `completed` status describes model generation.
It does not report local input completion or a verified business result.

Enable the caller-executed tool with
`ChatRequest::set_openai_computer_tool(Some(OpenAiComputerToolConfig::default()))`.
The SDK stores this declaration in `native_options` and encodes
`tools: [{"type":"computer"}]` on official OpenAI Responses routes.
It is separate from provider-executed `HostedTool` declarations and ordinary
function `ToolSpec`s.

Provider types live in `providers::openai::computer`:

- `OpenAiComputerCall` retains the call ID, generation status, safety checks and
  ordered `ComputerAction`s.
- A typed call requires the API's item `id` and an explicit
  `pending_safety_checks` array, including `[]` when no checks are pending.
  Missing fields are rejected before a typed call can be published.
- `OpenAiComputerCallOutput` retains the paired call ID and `ComputerScreenshot`.
- `ContentBlock::Native` and `StreamEvent::Native` carry a `NativeExtension`
  tagged with the corresponding provider format. Use the types' `from_extension`
  or `from_content_block` helpers; do not parse opaque `ProviderContent` into input.

The Responses stream publishes complete typed computer calls after validating
the terminal response's entire set of calls and matching its response ID to
`response.created`. When an `output_item.done` computer call was observed, its
actions and safety checks must also match the terminal item. Nonstream
responses require a nonempty response ID before publishing completed calls.
When an `output_item.added` computer call was observed, its item ID and call ID
must match the completed terminal output at the same index. A completed
response missing that output is rejected.
Partial action events and
`response.output_item.done` are observations rather than dispatch authority.
An early `[DONE]` after a computer item is a stream interruption; it cannot
replace the terminal response.
Consumers must also require the successful terminal event, current caller scope
and local permission checks before handling actions. Errors, cancellation and
incomplete responses cannot authorize a batch prefix.

After handling a call, build an output for that exact call with
`OpenAiComputerCallOutput::into_content_block_for`. Keep the SDK's scoped
`ContinuationRef` on the next request to continue the same response and account.
`OpenAiComputerCallOutput::from_response_item` can inspect returned output items,
including a failed status and `created_by`; those readback-only fields cannot be
submitted as caller input. A returned item can omit its screenshot URL unless
the API read includes `computer_call_output.output.image_url`, and can include
both a file ID and URL. An output built for submission must still provide exactly
one screenshot source.
When the first call requests only `screenshot`, return a fresh screenshot without
emitting input. The SDK does not automatically acknowledge safety checks or
execute actions. The caller owns acknowledgements, observation freshness,
unknown execution outcomes and retry decisions.

The scoped continuation identifies the response, route, model and account. It
does not contain the server's pending call IDs. The caller must retain the
original call alongside that continuation and require `validate_for_call` before
returning an output. Constructing an output's native block alone checks its
shape and bounds; the encoder cannot infer that it belongs to an unseen previous
response. Public DTOs and stream observations are not authorization tokens.
Block-level replay compatibility for an output describes its wire family.
It does not establish the request's role, computer-tool declaration or scoped
continuation; the Responses request validator checks those separately. Generated
calls and raw computer provider items are incompatible with replay as input.

These types target the current `computer` / `actions[]` protocol. Legacy
`computer_use_preview` single-action calls are outside this interface. Other
protocol codecs reject typed computer content rather than converting it into
function calls or silently omitting it.

Validation tests use recorded protocol fixtures and injected transports. They
do not establish live model availability, API-key access, desktop execution or
task success.
