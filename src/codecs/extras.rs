//! Per-profile additions to a request: body fields and headers a wire does not
//! define but a particular endpoint understands.
//!
//! An OpenAI-compatible endpoint is rarely only OpenAI-compatible. Aggregators
//! take a `provider` object choosing which upstream serves the call, a `models`
//! array for their own fallback, `transforms`, an end-user id; several want
//! attribution headers. None of that belongs in `CompletionRequest`, which is
//! the neutral shape, and none of it can be special-cased by provider name —
//! gate 30 keeps provider names out of this crate entirely. So it arrives as
//! data on the profile and is merged here.
//!
//! Two rules, both of which exist to stop configuration doing damage quietly:
//!
//! **Additive only.** A key the codec already wrote wins. Configuration fills
//! gaps; it cannot redirect a request. Letting `extra.body.model` through would
//! send a different model than the one that was resolved, priced and recorded,
//! and nothing downstream would notice.
//!
//! **No credentials.** Headers that carry authorization are refused outright.
//! The authenticator owns those, it runs after encoding, and a profile that
//! could set them would be a way to smuggle a key past it — in a crate whose
//! whole point is that it does not hold credentials (gate 64).

use lingxi_agent_api::protocol::ProviderProfile;
use serde_json::{Map, Value};

/// Headers a profile may never set. The authenticator attaches these, after
/// encoding, from a credential the caller passed in.
const RESERVED_HEADERS: &[&str] = &[
    "authorization",
    "proxy-authorization",
    "x-api-key",
    "api-key",
    "cookie",
];

/// Body fields that must come from typed request data rather than profile
/// configuration. Unlike ordinary additive extras, these change request
/// identity or bind it to provider-side state.
const RESERVED_BODY_KEYS: &[&str] = &["model", "previous_response_id"];

/// Merge `extra.body` into a request body.
///
/// Returns the keys that were refused because the codec had already set them,
/// so a caller can say so rather than silently doing something else. Nothing is
/// overwritten either way.
pub(crate) fn merge_body(profile: &ProviderProfile, body: &mut Map<String, Value>) -> Vec<String> {
    let Some(extra) = profile.extra.get("body").and_then(Value::as_object) else {
        return vec![];
    };
    let mut refused = Vec::new();
    for (key, value) in extra {
        if RESERVED_BODY_KEYS.contains(&key.as_str()) || body.contains_key(key) {
            refused.push(key.clone());
            continue;
        }
        body.insert(key.clone(), value.clone());
    }
    refused
}

/// Append `extra.headers` to a request's headers.
///
/// Returns the names that were refused: a reserved one, or a duplicate of a
/// header the codec already wrote.
pub(crate) fn merge_headers(
    profile: &ProviderProfile,
    headers: &mut Vec<(String, String)>,
) -> Vec<String> {
    let Some(extra) = profile.extra.get("headers").and_then(Value::as_object) else {
        return vec![];
    };
    let mut refused = Vec::new();
    for (name, value) in extra {
        let Some(value) = value.as_str() else {
            refused.push(name.clone());
            continue;
        };
        let lower = name.to_ascii_lowercase();
        let reserved = RESERVED_HEADERS.contains(&lower.as_str());
        let already = headers.iter().any(|(k, _)| k.eq_ignore_ascii_case(name));
        if reserved || already {
            refused.push(name.clone());
            continue;
        }
        headers.push((name.clone(), value.to_owned()));
    }
    refused
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn profile(extra: Value) -> ProviderProfile {
        serde_json::from_value(json!({
            "provider_id": "acme",
            "profile_name": "acme",
            "base_url": "https://x.test",
            "protocol": "open_ai_chat",
            "auth": "api_key",
            "extra": extra,
        }))
        .expect("profile fixture parses")
    }

    #[test]
    fn a_nested_object_the_wire_does_not_define_is_passed_through_whole() {
        // The point of the mechanism: an aggregator's routing preferences are a
        // structure, not a flag, and this crate must not know its name.
        let p = profile(json!({
            "body": {
                "provider": {"sort": "throughput", "data_collection": "deny"},
                "user": "stable-id",
            }
        }));
        let mut body = Map::new();
        assert!(merge_body(&p, &mut body).is_empty());
        assert_eq!(
            body.get("provider"),
            Some(&json!({"sort": "throughput", "data_collection": "deny"}))
        );
        assert_eq!(body.get("user"), Some(&json!("stable-id")));
    }

    #[test]
    fn configuration_cannot_redirect_a_request_to_another_model() {
        // The model was resolved, priced and recorded. Silently sending a
        // different one would make every one of those records wrong.
        let p = profile(json!({"body": {"model": "someone-elses-model"}}));
        let mut body = Map::new();
        body.insert("model".into(), json!("the-resolved-one"));
        assert_eq!(merge_body(&p, &mut body), vec!["model".to_owned()]);
        assert_eq!(body.get("model"), Some(&json!("the-resolved-one")));
    }

    #[test]
    fn configuration_cannot_inject_a_responses_continuation() {
        let p = profile(json!({"body": {"previous_response_id": "resp_untyped"}}));
        let mut body = Map::new();
        assert_eq!(
            merge_body(&p, &mut body),
            vec!["previous_response_id".to_owned()]
        );
        assert!(!body.contains_key("previous_response_id"));
    }

    #[test]
    fn an_authorization_header_cannot_be_smuggled_in_through_config() {
        for name in ["Authorization", "x-api-key", "COOKIE"] {
            let p = profile(json!({"headers": {name: "sk-not-yours"}}));
            let mut headers = vec![];
            assert_eq!(merge_headers(&p, &mut headers), vec![name.to_owned()]);
            assert!(
                headers.is_empty(),
                "{name} belongs to the authenticator, which runs after this"
            );
        }
    }

    #[test]
    fn a_header_the_codec_already_wrote_is_not_duplicated() {
        let p = profile(json!({"headers": {"Content-Type": "text/plain"}}));
        let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
        assert_eq!(
            merge_headers(&p, &mut headers),
            vec!["Content-Type".to_owned()]
        );
        assert_eq!(headers.len(), 1);
        assert_eq!(headers[0].1, "application/json");
    }

    #[test]
    fn attribution_headers_go_through() {
        let p = profile(json!({"headers": {"HTTP-Referer": "https://app.test", "X-Title": "App"}}));
        let mut headers = vec![];
        assert!(merge_headers(&p, &mut headers).is_empty());
        assert_eq!(headers.len(), 2);
    }

    #[test]
    fn a_header_whose_value_is_not_a_string_is_refused() {
        let p = profile(json!({"headers": {"X-Count": 3}}));
        let mut headers = vec![];
        assert_eq!(merge_headers(&p, &mut headers), vec!["X-Count".to_owned()]);
        assert!(headers.is_empty());
    }

    #[test]
    fn a_profile_that_declares_nothing_changes_nothing() {
        let p = profile(json!({"stream_usage_opt_in": true}));
        let mut body = Map::new();
        let mut headers = vec![];
        assert!(merge_body(&p, &mut body).is_empty());
        assert!(merge_headers(&p, &mut headers).is_empty());
        assert!(body.is_empty() && headers.is_empty());
    }
}
