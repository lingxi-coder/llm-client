//! Where a credential ends up on a request.
//!
//! The credential is passed in, never looked up (gate 64). What these pin is
//! that it arrives in the header the wire actually reads, and that a missing one
//! fails loudly instead of going out unauthenticated.

use lingxi_llm_client::protocol::{LlmError, ProviderProfile, Secret};
use lingxi_llm_client::{ApiKeyAuthenticator, Authenticator, BearerAuthenticator, HttpRequest};
use serde_json::json;

fn request() -> HttpRequest {
    HttpRequest {
        method: "POST".to_owned(),
        url: "https://x.test/v1/messages".to_owned(),
        headers: vec![("content-type".to_owned(), "application/json".to_owned())],
        body: bytes::Bytes::new(),
        timeout: None,
    }
}

fn profile(protocol: &str, extra: serde_json::Value) -> ProviderProfile {
    serde_json::from_value(json!({
        "provider_id": "acme",
        "profile_name": "acme",
        "base_url": "https://x.test",
        "protocol": protocol,
        "auth": "api_key",
        "extra": extra,
    }))
    .expect("profile fixture parses")
}

fn header(req: &HttpRequest, name: &str) -> Option<String> {
    req.headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.clone())
}

fn apply(auth: &dyn Authenticator, profile: &ProviderProfile, key: Option<&str>) -> HttpRequest {
    let secret = key.map(|k| Secret::new(k.to_owned()));
    let mut req = request();
    futures::executor::block_on(auth.apply(&mut req, profile, secret.as_ref()))
        .expect("the credential is supplied");
    req
}

/// The header is a property of the wire, not the vendor — every endpoint
/// speaking a given protocol reads the key from the same place. That is what
/// lets this stay free of provider names (gate 30).
#[test]
fn a_key_lands_in_the_header_its_wire_reads() {
    for (protocol, name, expected) in [
        ("anthropic_messages", "x-api-key", "sk-test"),
        ("gemini_generate_content", "x-goog-api-key", "sk-test"),
        ("open_ai_chat", "authorization", "Bearer sk-test"),
        ("open_ai_responses", "authorization", "Bearer sk-test"),
    ] {
        let req = apply(
            &ApiKeyAuthenticator,
            &profile(protocol, json!({})),
            Some("sk-test"),
        );
        assert_eq!(
            header(&req, name).as_deref(),
            Some(expected),
            "{protocol} reads its key from {name}"
        );
    }
}

/// A profile whose endpoint disagrees with its wire says so in data rather than
/// being special-cased in code.
#[test]
fn a_profile_may_name_the_header_itself() {
    let req = apply(
        &ApiKeyAuthenticator,
        &profile("open_ai_chat", json!({"credential_header": "x-house-key"})),
        Some("sk-test"),
    );
    assert_eq!(header(&req, "x-house-key").as_deref(), Some("sk-test"));
    assert_eq!(
        header(&req, "authorization"),
        None,
        "and it does not also go to the default"
    );
}

/// Sending the request anyway would reach the provider as an anonymous call and
/// come back a 401, which reads as "your key is wrong" rather than "you have
/// not set one".
#[test]
fn a_missing_credential_is_refused_by_name_not_sent_anonymously() {
    for auth in [
        &ApiKeyAuthenticator as &dyn Authenticator,
        &BearerAuthenticator,
    ] {
        let p = profile("open_ai_chat", json!({}));
        let mut req = request();
        let err = futures::executor::block_on(auth.apply(&mut req, &p, None))
            .expect_err("no credential was supplied");
        assert!(
            matches!(&err, LlmError::Authentication { message } if message.contains("acme")),
            "the error names the profile a user has to fix: {err:?}"
        );
        assert_eq!(header(&req, "authorization"), None);
    }
}

/// Two authorization headers are rejected by some endpoints and silently
/// resolved to the first by others. Neither is worth shipping.
#[test]
fn applying_twice_replaces_rather_than_appends() {
    let p = profile("open_ai_chat", json!({}));
    let mut req = request();
    for key in ["stale", "fresh"] {
        let secret = Secret::new(key.to_owned());
        futures::executor::block_on(BearerAuthenticator.apply(&mut req, &p, Some(&secret)))
            .unwrap();
    }
    let all: Vec<_> = req
        .headers
        .iter()
        .filter(|(k, _)| k.eq_ignore_ascii_case("authorization"))
        .collect();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].1, "Bearer fresh");
}

#[test]
fn request_debug_redacts_credentials_urls_and_prompts() {
    for extra in [json!({}), json!({"credential_header": "x-house-key"})] {
        let mut req = apply(
            &ApiKeyAuthenticator,
            &profile("open_ai_chat", extra),
            Some("private-credential"),
        );
        req.url = "https://user:url-password@example.test/messages?key=query-secret".into();
        req.body = bytes::Bytes::from_static(b"private-prompt");
        let debug = format!("{req:?}");
        for secret in [
            "private-credential",
            "url-password",
            "query-secret",
            "private-prompt",
        ] {
            assert!(!debug.contains(secret), "Debug leaked {secret}");
        }
        assert!(debug.contains("POST"));
    }
}
