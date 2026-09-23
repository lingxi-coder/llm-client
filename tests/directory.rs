//! Reading what a provider says it serves right now.
//!
//! These name providers on purpose — they assert what the shipped data says.
//! Gate 30 scans `src/` only.

use lingxi_agent_api::protocol::{AuthStrategy, LlmError, ProtocolFamily, ProviderProfile};
use lingxi_llm_client::{
    builtin_providers, AnthropicMessagesDirectory, GeminiDirectory, HttpResponse, LlmClient,
    LlmClientBuilder, ModelDirectory, ModelPage, OpenAiChatDirectory,
};
use serde_json::{json, Value};
use std::sync::Arc;

mod support;

fn client_of(profiles: &[ProviderProfile]) -> LlmClient {
    let http = Arc::new(support::NoHttp);
    let mut b = LlmClientBuilder::with_transport(http, profiles);
    for strategy in AuthStrategy::ALL {
        b.register_authenticator(strategy, Arc::new(support::NoAuth));
    }
    b.build().expect("every protocol has a codec")
}

fn profile(protocol: &str, base_url: &str, model_list: Option<&str>) -> ProviderProfile {
    let mut spec = json!({
        "provider_id": "acme",
        "profile_name": "acme",
        "base_url": base_url,
        "protocol": protocol,
        "auth": "api_key",
        "models": [{"display_model": "m", "request_model": "m", "billing_model": "m"}],
    });
    if let Some(route) = model_list {
        spec["model_list"] = json!(route);
    }
    serde_json::from_value(spec).expect("profile fixture parses")
}

fn ok(body: Value) -> HttpResponse {
    HttpResponse {
        status: 200,
        headers: vec![("content-type".to_owned(), "application/json".to_owned())],
        body: serde_json::to_vec(&body).unwrap().into(),
    }
}

/// One page of each wire, as the vendor actually sends it.
fn chat_page() -> Value {
    json!({"object": "list", "data": [
        {"id": "gpt-4.1", "object": "model", "created": 1_744_070_400, "owned_by": "system"},
        {"id": "o4-mini", "object": "model", "created": 1_744_070_400, "owned_by": "system"},
    ]})
}

/// The same shape as above, from an aggregator that publishes far more of it.
fn aggregator_page() -> Value {
    json!({"data": [{
        "id": "deepseek/deepseek-chat-v3.1:free",
        "name": "DeepSeek: DeepSeek V3.1 (free)",
        "description": "A large hybrid reasoning model.",
        "context_length": 163_840,
        "top_provider": {"max_completion_tokens": 32_768},
        "pricing": {"prompt": "0", "completion": "0"},
    }]})
}

fn messages_page(has_more: bool) -> Value {
    json!({
        "data": [
            {"type": "model", "id": "claude-opus-4-5-20260101", "display_name": "Claude Opus 4.5"},
            {"type": "model", "id": "claude-sonnet-4-5-20250929", "display_name": "Claude Sonnet 4.5"},
        ],
        "has_more": has_more,
        "first_id": "claude-opus-4-5-20260101",
        "last_id": "claude-sonnet-4-5-20250929",
    })
}

fn generate_content_page(next: Option<&str>) -> Value {
    let mut body = json!({"models": [{
        "name": "models/gemini-2.5-pro",
        "version": "2.5",
        "displayName": "Gemini 2.5 Pro",
        "description": "Stable release of Gemini 2.5 Pro.",
        "inputTokenLimit": 1_048_576,
        "outputTokenLimit": 65_536,
        "supportedGenerationMethods": ["generateContent", "countTokens"],
    }]});
    if let Some(token) = next {
        body["nextPageToken"] = json!(token);
    }
    body
}

fn page_of(d: &dyn ModelDirectory, body: Value) -> Result<ModelPage, LlmError> {
    d.decode_page(&ok(body))
}

/// Each shape reads the ids its own wire publishes, and the detail it
/// publishes beside them.
#[test]
fn each_shape_reads_the_page_its_own_wire_sends() {
    let page = page_of(&OpenAiChatDirectory, chat_page()).expect("a list page decodes");
    let ids: Vec<&str> = page
        .models
        .iter()
        .map(|m| m.request_model.as_str())
        .collect();
    assert_eq!(ids, ["gpt-4.1", "o4-mini"]);
    assert_eq!(page.next_cursor, None);

    let page = page_of(&AnthropicMessagesDirectory, messages_page(false)).unwrap();
    assert_eq!(page.models[0].request_model, "claude-opus-4-5-20260101");
    assert_eq!(
        page.models[0].display_name.as_deref(),
        Some("Claude Opus 4.5")
    );

    let page = page_of(&GeminiDirectory, generate_content_page(None)).unwrap();
    assert_eq!(
        page.models[0].request_model, "gemini-2.5-pro",
        "a request carries the bare id, not the resource path it is listed under"
    );
    assert_eq!(page.models[0].context_window, Some(1_048_576));
    assert_eq!(page.models[0].max_output_tokens, Some(65_536));
}

/// Where a shape publishes more than the bare id, it is read; a vendor that
/// publishes only ids still lists.
#[test]
fn detail_beside_the_id_is_read_where_a_vendor_publishes_it() {
    let rich = page_of(&OpenAiChatDirectory, aggregator_page()).unwrap();
    let m = &rich.models[0];
    assert_eq!(m.request_model, "deepseek/deepseek-chat-v3.1:free");
    assert_eq!(
        m.display_name.as_deref(),
        Some("DeepSeek: DeepSeek V3.1 (free)")
    );
    assert_eq!(m.context_window, Some(163_840));
    assert_eq!(m.max_output_tokens, Some(32_768));

    let bare = page_of(&OpenAiChatDirectory, chat_page()).unwrap();
    assert_eq!(bare.models[0].display_name, None);
    assert_eq!(bare.models[0].context_window, None);
}

/// The failure this whole route key exists to prevent. A page read by the
/// wrong shape has none of the keys that shape looks for — and "no models" is
/// a claim a merge would act on by withdrawing the entire catalog.
#[test]
fn a_page_of_another_shape_is_refused_rather_than_read_as_an_empty_catalog() {
    let wrong: Vec<(&str, Result<ModelPage, LlmError>)> = vec![
        (
            "list read as the resource-path shape",
            page_of(&GeminiDirectory, chat_page()),
        ),
        (
            "resource-path page read as the list shape",
            page_of(&OpenAiChatDirectory, generate_content_page(None)),
        ),
        (
            "resource-path page read as the id-cursor shape",
            page_of(&AnthropicMessagesDirectory, generate_content_page(None)),
        ),
    ];
    for (what, got) in wrong {
        let err = got
            .err()
            .unwrap_or_else(|| panic!("{what} must not decode"));
        assert!(
            matches!(err, LlmError::ProviderInternal { .. }),
            "{what}: {err}"
        );
    }
}

/// And the distinction that makes the refusal above meaningful: a vendor that
/// really serves nothing says so with an empty array, which is not an error.
#[test]
fn an_empty_list_is_an_answer_and_a_missing_list_is_not() {
    let empty = page_of(&OpenAiChatDirectory, json!({"object": "list", "data": []}))
        .expect("an empty catalog is a thing a provider may say");
    assert!(empty.models.is_empty());

    assert!(
        page_of(&OpenAiChatDirectory, json!({"object": "list"})).is_err(),
        "a page with no list at all is not a provider serving nothing"
    );
}

/// A row with no id cannot be requested, so it is not a model. Skipping it
/// would turn a shape mismatch into a quietly shorter catalog.
#[test]
fn a_row_that_names_no_model_is_refused() {
    assert!(page_of(
        &OpenAiChatDirectory,
        json!({"data": [{"id": "fine"}, {"object": "model", "created": 1}]})
    )
    .is_err());
}

/// The cursor goes back out on the next request, and only while the provider
/// says there is more. Both wires here keep sending the last id on the final
/// page, so following that alone would walk the same page forever.
#[test]
fn a_cursor_is_produced_only_while_the_provider_says_there_is_more() {
    let more = page_of(&AnthropicMessagesDirectory, messages_page(true)).unwrap();
    assert_eq!(
        more.next_cursor.as_deref(),
        Some("claude-sonnet-4-5-20250929")
    );
    let done = page_of(&AnthropicMessagesDirectory, messages_page(false)).unwrap();
    assert_eq!(
        done.next_cursor, None,
        "the last page still names last_id; has_more is what ends the walk"
    );

    let more = page_of(&GeminiDirectory, generate_content_page(Some("tok-2"))).unwrap();
    assert_eq!(more.next_cursor.as_deref(), Some("tok-2"));
    let done = page_of(&GeminiDirectory, generate_content_page(None)).unwrap();
    assert_eq!(done.next_cursor, None);
    let blank = page_of(&GeminiDirectory, {
        let mut b = generate_content_page(None);
        b["nextPageToken"] = json!("");
        b
    })
    .unwrap();
    assert_eq!(
        blank.next_cursor, None,
        "an empty token is how this wire spells 'no more', not a page to ask for"
    );
}

/// A cursor handed back reaches the next request, in the parameter its own
/// wire reads it from.
#[test]
fn the_cursor_is_carried_into_the_next_request() {
    let p = profile("anthropic_messages", "https://x.test", None);
    let first = AnthropicMessagesDirectory.list_request(&p, None);
    assert_eq!(first.method, "GET");
    assert!(
        first.url.starts_with("https://x.test/v1/models?"),
        "{}",
        first.url
    );
    assert!(!first.url.contains("after_id"), "{}", first.url);
    let next = AnthropicMessagesDirectory.list_request(&p, Some("mid-7"));
    assert!(next.url.contains("after_id=mid-7"), "{}", next.url);

    let g = profile("gemini_generate_content", "https://y.test/v1beta", None);
    let next = GeminiDirectory.list_request(&g, Some("tok-2"));
    assert!(
        next.url.starts_with("https://y.test/v1beta/models?"),
        "{}",
        next.url
    );
    assert!(next.url.contains("pageToken=tok-2"), "{}", next.url);
}

/// A cursor is the provider's own opaque string — one wire's is base64 with
/// its padding intact. Pasted in raw it would end the query value early.
#[test]
fn an_opaque_cursor_survives_being_put_in_a_url() {
    let g = profile("gemini_generate_content", "https://y.test/v1beta", None);
    let req = GeminiDirectory.list_request(&g, Some("a+b/c=&x=1"));
    assert!(
        req.url.contains("pageToken=a%2Bb%2Fc%3D%26x%3D1"),
        "{}",
        req.url
    );
    assert_eq!(
        req.url.matches("pageToken").count(),
        1,
        "a cursor must not be able to add parameters of its own"
    );
}

/// Listing carries no credential: this crate holds none, and the caller
/// attaches one after encoding exactly as it does for a completion (gate 64).
#[test]
fn a_list_request_leaves_the_credential_to_whoever_owns_it() {
    let p = profile("open_ai_chat", "https://x.test/v1", None);
    let req = OpenAiChatDirectory.list_request(&p, None);
    assert!(req.body.is_empty());
    assert!(req.timeout.is_some(), "a refresh must not hang forever");
    for (name, _) in &req.headers {
        assert!(
            !["authorization", "x-api-key", "x-goog-api-key"]
                .contains(&name.to_ascii_lowercase().as_str()),
            "{name} is the authenticator's to write"
        );
    }
}

/// A refused directory is the same error a refused completion is. A caller
/// deciding whether to prompt for a key should not have to know which request
/// got the 401.
#[test]
fn a_refusal_reads_as_the_error_a_completion_would_have_given() {
    let resp = HttpResponse {
        status: 401,
        headers: vec![],
        body: serde_json::to_vec(&json!({"error": {"message": "no key"}}))
            .unwrap()
            .into(),
    };
    for d in [
        &OpenAiChatDirectory as &dyn ModelDirectory,
        &AnthropicMessagesDirectory,
        &GeminiDirectory,
    ] {
        assert!(
            matches!(d.decode_page(&resp), Err(LlmError::Authentication { .. })),
            "a 401 from {:?} must not read as an empty catalog",
            d.shape()
        );
    }
}

/// Every shipped preset either resolves to a reader or says it publishes no
/// directory. Silence would be a third state that looks like the first and
/// behaves like the second.
#[test]
fn every_preset_either_publishes_a_directory_or_says_it_does_not() {
    let profiles = builtin_providers().unwrap();
    let c = client_of(&profiles);
    let mut declared_none = vec![];
    for p in &profiles {
        if p.model_list == lingxi_agent_api::protocol::DirectoryRoute::NotPublished {
            declared_none.push(p.profile_name.as_str());
            assert!(c.directory_for(p).is_none());
            continue;
        }
        assert!(
            c.directory_for(p).is_some(),
            "{} would look for a directory in a shape nothing here reads, which \
             answers every refresh with an error instead of a list",
            p.profile_name
        );
    }
    assert_eq!(
        declared_none,
        ["deepseek-search", "glm-coding"],
        "compatibility endpoints without a configured model-directory route"
    );
}

/// The wire an endpoint speaks is not evidence about its model list. One
/// preset here proves it: it speaks one wire for completions and publishes its
/// list in the other's shape, at the other's path.
#[test]
fn a_directory_shape_is_not_assumed_from_the_wire_the_endpoint_speaks() {
    let profiles = builtin_providers().unwrap();
    let c = client_of(&profiles);
    let by = |name: &str| {
        profiles
            .iter()
            .find(|p| p.profile_name == name)
            .unwrap_or_else(|| panic!("{name} is shipped"))
    };

    let openai = by("openai");
    assert_eq!(openai.protocol, ProtocolFamily::OpenAiResponses);
    assert_eq!(
        c.directory_for(openai).map(|d| d.shape()),
        Some(ProtocolFamily::OpenAiChat),
        "this vendor serves one shared list for both its wires, in the other one's shape"
    );

    // And the default still holds everywhere it is right.
    let anthropic = by("anthropic");
    assert_eq!(
        c.directory_for(anthropic).map(|d| d.shape()),
        Some(ProtocolFamily::AnthropicMessages)
    );
}

/// Every preset's first page goes to a real URL under its own base.
#[test]
fn every_preset_lists_from_its_own_endpoint() {
    let profiles = builtin_providers().unwrap();
    let c = client_of(&profiles);
    for p in &profiles {
        let Some(d) = c.directory_for(p) else {
            continue;
        };
        let req = d.list_request(p, None);
        assert_eq!(req.method, "GET", "{}", p.profile_name);
        assert!(
            req.url.starts_with(p.base_url.trim_end_matches('/')),
            "{} lists from {} which is not under its own base {}",
            p.profile_name,
            req.url,
            p.base_url
        );
        let path = req.url.split('?').next().unwrap();
        assert!(
            path.ends_with("/models"),
            "{} lists from {path}",
            p.profile_name
        );
    }
}

#[test]
fn anthropic_directory_sends_the_selected_api_version() {
    for version in [None, Some("2024-01-01")] {
        let mut p = profile("anthropic_messages", "https://x.test", None);
        if let Some(version) = version {
            p.extra = json!({"api_version": version});
        }
        let req = AnthropicMessagesDirectory.list_request(&p, None);
        let versions: Vec<_> = req
            .headers
            .iter()
            .filter(|(key, _)| key.eq_ignore_ascii_case("anthropic-version"))
            .collect();
        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0].1, version.unwrap_or("2023-06-01"));
    }
}
