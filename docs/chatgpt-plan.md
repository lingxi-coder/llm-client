# Sign in with ChatGPT plan usage

`llm-client` supports the public Responses API route for an OpenAI OAuth access
token that has the separately granted `chatgpt.tokens.use.direct` scope. The
host application owns registration, browser sign-in, PKCE, callback state,
ID-token verification, account selection, token refresh, and protected storage.
An identity-only sign-in does not authorize inference.

Use a distinct OpenAI Responses profile with `auth: "chat_gpt_plan"`,
`base_url: "https://api.openai.com/v1"`, `model_list: "none"`, and
`supports_websockets: false`. Keep it separate from Codex's
`chatgpt.com/backend-api/codex` connection. The builder rejects a plan profile
that targets another endpoint or enables WebSocket.
The profile and every model use subscription billing. Independent provider
services (including background jobs, audio, images, the Files API, Containers,
Realtime, and Live) are unavailable through this grant. Plan connections are
never automatic failover sources or targets; switching accounts or billing
paths is a host decision.

```json
{
  "provider_id": "openai",
  "profile_name": "chatgpt-plan",
  "base_url": "https://api.openai.com/v1",
  "protocol": "open_ai_responses",
  "auth": "chat_gpt_plan",
  "model_list": "none",
  "pricing": { "billingMode": "subscription" },
  "models": [
    {
      "display_model": "gpt-6.1-sol",
      "request_model": "gpt-6.1-sol",
      "billing_model": "gpt-6.1-sol"
    }
  ]
}
```

Replace the example model with an account-visible slug from model discovery.

The host supplies the selected account's current access token in
`RequestOptions::credential`. Call `chat().stream_in(...)` and set
`ChatRequest::controls.responses.store = Some(false)`. The SDK checks the final
request after authentication, including when the host supplies a custom
authenticator: `stream: true`, `store: false`, an
`input` array, and no unsupported Responses fields, HTTP continuation, or
hosted tools other than eligible web search. It rejects automatic Files API
uploads before they can start. The host must supply full conversation history
for each HTTP request and consume the stream through `response.completed`.
The final model must match the selected model row, including after draft edits
or a request finalizer. The SDK also checks nested input content and tools for
unsupported audio, video, file IDs, and hosted capabilities.
Unknown input item types, including MCP approval items, are rejected until this
restricted route supports them explicitly. A sealed plan call cannot be sent
over an externally supplied WebSocket connection.
The plan stream reports `[DONE]` without `response.completed` as interrupted,
and reports `response.incomplete` as a failure with its native response body.
Conflicting account-routing or credential headers are rejected before dispatch.

For a current account-specific model picker, bind `OpenAiClient` to this exact
profile and call `chatgpt_plan_models(&access_token)`. It requests the public
`GET /v1/models`, filters rows with `visibility: "list"`, and preserves server
order. Refresh the list after switching accounts. The returned `slug` is the
model ID to use in the request; the host must configure a matching model in its
profile before routing it through the shared chat service.
Model-list failures return `ChatGptPlanModelError::Provider` with the HTTP status,
request ID, and original JSON error or direct-admission detail so the host can
apply the documented recovery for that response.
Inference failures return `LlmError::ProviderResponse` with the status, request
ID when available, original body, classified `LlmErrorKind`, and `retry_after`.
For `response.failed` after an HTTP stream opens, the HTTP status remains 200;
the same holds for an `error` event in that stream. Use `body.error.code` for
the provider's specific recovery action.

The current shared `ToolSpec` encoder emits bare function tools. The plan route
rejects those because OpenAI currently requires function/custom tools in a
namespace or in `additional_tools` input items. That wire form needs a separate
typed SDK addition before host tools can be used through this route.

Official contract and current limitations:

- [Registration and sign-in](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
- [Models and inference](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference)
- [Preview limitations](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations)
