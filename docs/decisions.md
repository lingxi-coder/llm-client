# Provider-native Decisions

Decision resources execute the provider's native protocol. The SDK no longer exposes a cross-provider `client.decisions()` API based on Chat Structured Outputs. Normal JSON Schema output remains a Chat capability.

## OpenAI

`providers::openai::decisions::{CreateRequest, Decisions}` sends exactly one request to an explicit complete endpoint, normally `https://api.openai.com/v1/decisions`:

```rust,no_run
use lingxi_llm_client::{HttpTransport, RequestOptions};
use lingxi_llm_client::providers::openai::decisions::{CreateRequest, Decisions};
use serde_json::json;
async fn example() -> Result<(), Box<dyn std::error::Error>> {
    let transport = HttpTransport::new()?;
    let service = Decisions::new(&transport, "https://api.openai.com/v1/decisions");
    let request = CreateRequest {
        model: "gpt-6-luna".into(), input: json!("Please refund the duplicate charge."),
        questions: vec![json!({"type":"predicate", "name":"refund", "instructions":"Is a refund requested?"})],
    };
    let options = RequestOptions::default(); // Supply options.credential from the host.
    let _response = service.create(&request, &options).await?;
    Ok(())
}
```

An exact-profile `OpenAiClient` also provides `decisions(endpoint)`. Bound ChatGPT plan credentials are rejected. Native request content/questions and successful response fields are retained as JSON; semantic normalization and answer validation belong to the independent Decision runtime.

## TypeSafe

`providers::typesafe::{SystemOne, SystemOneRequest}` sends `model`, `state`, named `questions`, and optional image `assets` to an explicit `/v1/systemone` endpoint. `TypeSafeClient::system_one(endpoint)` supplies the exact-profile resource binding. This service does not require Chat models or a Chat-compatible endpoint.

Image assets trigger TypeSafe's required `TypeSafe-Danger-Zone-Images: true` header. Each asset reference must appear in the supplied evidence. Model selection and any feature-specific limits remain explicit.

## Execution and upgrade

The host supplies credentials through `RequestOptions`, never endpoint URLs or stored SDK configuration. Requests have a default 30-second deadline, bounded 16 MiB request/response bodies, no automatic retry or provider fallback, and no redirect-based credential forwarding. Transport cancellation occurs by dropping the operation. Provider HTTP errors retain status, request ID and reported usage without exposing the raw error body.

SDK configuration loading strips only the recognized obsolete `connection.decisions` and `fallback.decisions` metadata with `implementation: structured_output`. Other fields remain subject to normal strict validation. Ordinary provider configuration and account credentials are preserved.

Official contracts: [OpenAI](https://developers.openai.com/api/reference/resources/decisions/methods/create), [TypeSafe](https://docs.typesafe.ai/api/system-one).
