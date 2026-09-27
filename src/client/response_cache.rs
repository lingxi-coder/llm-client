//! Provider-neutral projection of explicitly reported gateway cache metadata.
use crate::protocol::{ProviderProfile, ResponseCacheObservation, ResponseCacheStatus};

/// Read only OpenRouter's documented response-cache headers from an official
/// OpenRouter profile. Unknown or absent status values are not observations.
pub(crate) fn openrouter_observation(
    profile: &ProviderProfile,
    request_url: &str,
    headers: &[(String, String)],
) -> Option<ResponseCacheObservation> {
    if profile.provider_id.as_str() != "openrouter"
        || !official_profile_base(&profile.base_url)
        || !official_request_url(request_url)
    {
        return None;
    }

    let status = match header(headers, "x-openrouter-cache-status")?.trim() {
        value if value.eq_ignore_ascii_case("HIT") => ResponseCacheStatus::Hit,
        value if value.eq_ignore_ascii_case("MISS") => ResponseCacheStatus::Miss,
        _ => return None,
    };
    let age_seconds = (status == ResponseCacheStatus::Hit)
        .then(|| header(headers, "x-openrouter-cache-age").and_then(parse_seconds))
        .flatten();
    let ttl_seconds = header(headers, "x-openrouter-cache-ttl").and_then(parse_seconds);
    let source_generation_id = (status == ResponseCacheStatus::Hit)
        .then(|| {
            header(headers, "x-openrouter-cache-source-id")
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        })
        .flatten();

    Some(ResponseCacheObservation {
        status,
        age_seconds,
        ttl_seconds,
        source_generation_id,
    })
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

fn parse_seconds(value: &str) -> Option<u64> {
    value.trim().parse().ok()
}

pub(crate) fn official_profile_base(endpoint: &str) -> bool {
    let Ok(url) = url::Url::parse(endpoint) else {
        return false;
    };
    url.scheme() == "https"
        && url.host_str() == Some("openrouter.ai")
        && url.port_or_known_default() == Some(443)
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && matches!(url.path(), "/api/v1" | "/api/v1/")
}

pub(crate) fn official_request_url(endpoint: &str) -> bool {
    let Ok(url) = url::Url::parse(endpoint) else {
        return false;
    };
    url.scheme() == "https"
        && url.host_str() == Some("openrouter.ai")
        && url.port_or_known_default() == Some(443)
        && url.username().is_empty()
        && url.password().is_none()
        && url.query().is_none()
        && url.fragment().is_none()
        && matches!(
            url.path(),
            "/api/v1/chat/completions"
                | "/api/v1/responses"
                | "/api/v1/messages"
                | "/api/v1/embeddings"
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn profile(provider: &str, endpoint: &str) -> ProviderProfile {
        serde_json::from_value(json!({
            "provider_id":provider,
            "profile_name":"test",
            "base_url":endpoint,
            "protocol":"open_ai_chat",
            "auth":"none",
            "models":[{"request_model":"m","display_model":"m","billing_model":"m"}]
        }))
        .unwrap()
    }

    #[test]
    fn projects_only_documented_status_and_status_specific_headers() {
        let profile = profile("openrouter", "https://openrouter.ai/api/v1");
        let hit = openrouter_observation(
            &profile,
            "https://openrouter.ai/api/v1/chat/completions",
            &[
                ("x-openrouter-cache-status".into(), "HIT".into()),
                ("X-OpenRouter-Cache-Age".into(), "12".into()),
                ("X-OpenRouter-Cache-TTL".into(), "288".into()),
                ("X-OpenRouter-Cache-Source-Id".into(), "gen-source".into()),
            ],
        )
        .unwrap();
        assert_eq!(hit.status, ResponseCacheStatus::Hit);
        assert_eq!(hit.age_seconds, Some(12));
        assert_eq!(hit.ttl_seconds, Some(288));
        assert_eq!(hit.source_generation_id.as_deref(), Some("gen-source"));

        let miss = openrouter_observation(
            &profile,
            "https://openrouter.ai/api/v1/chat/completions",
            &[
                ("X-OpenRouter-Cache-Status".into(), "MISS".into()),
                ("X-OpenRouter-Cache-Age".into(), "7".into()),
                ("X-OpenRouter-Cache-TTL".into(), "300".into()),
                (
                    "X-OpenRouter-Cache-Source-Id".into(),
                    "ignored-on-miss".into(),
                ),
            ],
        )
        .unwrap();
        assert_eq!(miss.status, ResponseCacheStatus::Miss);
        assert_eq!(miss.age_seconds, None);
        assert_eq!(miss.ttl_seconds, Some(300));
        assert_eq!(miss.source_generation_id, None);
    }

    #[test]
    fn unknown_status_and_non_official_routes_are_not_projected() {
        let openrouter_profile = profile("openrouter", "https://openrouter.ai/api/v1");
        assert!(openrouter_observation(
            &openrouter_profile,
            "https://openrouter.ai/api/v1/chat/completions",
            &[("X-OpenRouter-Cache-Status".into(), "WARMING".into())]
        )
        .is_none());
        assert!(openrouter_observation(
            &openrouter_profile,
            "https://openrouter.ai/api/v1/chat/completions",
            &[]
        )
        .is_none());

        for (provider, endpoint) in [
            ("openai", "https://api.openai.com/v1"),
            ("openrouter", "https://gateway.example.test/api/v1"),
            ("openrouter", "http://openrouter.ai/api/v1"),
        ] {
            assert!(openrouter_observation(
                &profile(provider, endpoint),
                "https://openrouter.ai/api/v1/chat/completions",
                &[("X-OpenRouter-Cache-Status".into(), "HIT".into())]
            )
            .is_none());
        }
        assert!(openrouter_observation(
            &openrouter_profile,
            "https://openrouter.ai/api/v1/custom/chat/completions",
            &[("X-OpenRouter-Cache-Status".into(), "HIT".into())]
        )
        .is_none());
    }
}
