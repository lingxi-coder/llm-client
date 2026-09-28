//! Provider-neutral projection of explicitly reported gateway cache metadata.
use crate::protocol::{LlmError, ProviderProfile, ResponseCacheObservation, ResponseCacheStatus};
use crate::transport::HttpRequest;

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

/// OpenRouter gateway response caching, independent of provider prompt caching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenRouterResponseCache {
    /// Send `X-OpenRouter-Cache: false`, including when a remote preset enables it.
    Disabled,
    /// Cache this exact request body. Refresh replaces only its matching entry.
    Enabled {
        ttl_seconds: Option<u32>,
        refresh: bool,
    },
}

pub(crate) fn validate_openrouter_response_cache(
    policy: Option<OpenRouterResponseCache>,
    profile: &ProviderProfile,
) -> Result<(), LlmError> {
    let Some(policy) = policy else {
        return Ok(());
    };
    let base = url::Url::parse(&profile.base_url).map_err(|_| LlmError::InvalidRequest {
        message: "OpenRouter response-cache profile has an invalid endpoint".into(),
    })?;
    if profile.provider_id.as_str() != "openrouter" || !official_profile_base(base.as_str()) {
        return Err(LlmError::UnsupportedCapability {
            message: "gateway response caching requires an official OpenRouter HTTPS route".into(),
        });
    }
    if matches!(policy, OpenRouterResponseCache::Enabled { ttl_seconds: Some(ttl), .. } if !(1..=86_400).contains(&ttl))
    {
        return Err(LlmError::InvalidRequest {
            message: "OpenRouter response-cache TTL must be 1–86400 seconds".into(),
        });
    }
    Ok(())
}

pub(crate) fn apply_openrouter_response_cache(
    policy: Option<OpenRouterResponseCache>,
    profile: &ProviderProfile,
    request: &mut HttpRequest,
) -> Result<(), LlmError> {
    validate_openrouter_response_cache(policy, profile)?;
    let Some(policy) = policy else {
        return Ok(());
    };
    let url = url::Url::parse(&request.url).map_err(|_| LlmError::InvalidRequest {
        message: "OpenRouter response-cache request has an invalid endpoint".into(),
    })?;
    if !official_request_url(url.as_str()) {
        return Err(LlmError::UnsupportedCapability {
            message: "gateway response caching requires a documented OpenRouter API endpoint"
                .into(),
        });
    }
    match policy {
        OpenRouterResponseCache::Disabled => request
            .headers
            .push(("X-OpenRouter-Cache".into(), "false".into())),
        OpenRouterResponseCache::Enabled {
            ttl_seconds,
            refresh,
        } => {
            request
                .headers
                .push(("X-OpenRouter-Cache".into(), "true".into()));
            if let Some(ttl) = ttl_seconds {
                request
                    .headers
                    .push(("X-OpenRouter-Cache-TTL".into(), ttl.to_string()));
            }
            if refresh {
                request
                    .headers
                    .push(("X-OpenRouter-Cache-Clear".into(), "true".into()));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod policy_tests {
    use super::*;
    use bytes::Bytes;
    use serde_json::json;
    fn profile(provider: &str, base_url: &str) -> ProviderProfile {
        serde_json::from_value(json!({
            "provider_id":provider, "profile_name":"route", "base_url":base_url,
            "protocol":"open_ai_chat", "auth":"none"
        }))
        .unwrap()
    }

    fn request() -> HttpRequest {
        HttpRequest {
            method: "POST".into(),
            url: "https://openrouter.ai/api/v1/chat/completions".into(),
            headers: vec![],
            body: Bytes::new(),
            timeout: None,
        }
    }

    #[test]
    fn response_cache_headers_are_separate_from_prompt_cache() {
        let profile = profile("openrouter", "https://openrouter.ai/api/v1");
        let mut outgoing = request();
        apply_openrouter_response_cache(
            Some(OpenRouterResponseCache::Enabled {
                ttl_seconds: Some(600),
                refresh: true,
            }),
            &profile,
            &mut outgoing,
        )
        .unwrap();
        assert_eq!(
            outgoing.headers,
            [
                ("X-OpenRouter-Cache".into(), "true".into()),
                ("X-OpenRouter-Cache-TTL".into(), "600".into()),
                ("X-OpenRouter-Cache-Clear".into(), "true".into()),
            ]
        );
        let mut disabled = request();
        apply_openrouter_response_cache(
            Some(OpenRouterResponseCache::Disabled),
            &profile,
            &mut disabled,
        )
        .unwrap();
        assert_eq!(
            disabled.headers,
            [("X-OpenRouter-Cache".into(), "false".into())]
        );
    }

    #[test]
    fn response_cache_rejects_non_openrouter_and_bad_ttl_before_http() {
        let mut request = request();
        assert!(apply_openrouter_response_cache(
            Some(OpenRouterResponseCache::Disabled),
            &profile("openai", "https://api.openai.com/v1"),
            &mut request,
        )
        .is_err());
        assert!(request.headers.is_empty());
        assert!(apply_openrouter_response_cache(
            Some(OpenRouterResponseCache::Enabled {
                ttl_seconds: Some(86_401),
                refresh: false,
            }),
            &profile("openrouter", "https://openrouter.ai/api/v1"),
            &mut request,
        )
        .is_err());
        assert!(request.headers.is_empty());
    }

    #[test]
    fn response_cache_accepts_only_documented_openrouter_endpoint_urls() {
        let canonical_profile = profile("openrouter", "https://openrouter.ai/api/v1");
        for path in [
            "/api/v1/chat/completions",
            "/api/v1/responses",
            "/api/v1/messages",
            "/api/v1/embeddings",
        ] {
            let mut outgoing = request();
            outgoing.url = format!("https://openrouter.ai{path}");
            apply_openrouter_response_cache(
                Some(OpenRouterResponseCache::Disabled),
                &canonical_profile,
                &mut outgoing,
            )
            .unwrap_or_else(|error| panic!("documented route {path} rejected: {error}"));
        }

        for endpoint in [
            "https://openrouter.ai/api/v1/custom/responses",
            "https://openrouter.ai/api/v1/responses?redirect=https://evil.test",
            "https://openrouter.ai:444/api/v1/responses",
            "https://user@openrouter.ai/api/v1/responses",
        ] {
            let mut outgoing = request();
            outgoing.url = endpoint.into();
            assert!(
                apply_openrouter_response_cache(
                    Some(OpenRouterResponseCache::Disabled),
                    &canonical_profile,
                    &mut outgoing,
                )
                .is_err(),
                "unexpectedly accepted {endpoint}"
            );
            assert!(outgoing.headers.is_empty());
        }

        for base in [
            "https://openrouter.ai/api/v1/custom",
            "https://openrouter.ai/api/v1?proxy=1",
            "https://openrouter.ai:444/api/v1",
        ] {
            let mut outgoing = request();
            assert!(
                apply_openrouter_response_cache(
                    Some(OpenRouterResponseCache::Disabled),
                    &profile("openrouter", base),
                    &mut outgoing,
                )
                .is_err(),
                "unexpectedly accepted base URL {base}"
            );
        }
    }
}
