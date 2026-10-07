//! Typed provider response metadata decoded from HTTP headers.
//!
//! Header names, canonical request-id headers, and provider reset formats are
//! wire protocol details. Hosts consume these values while retaining their
//! account state, clocks, retry policy, and UI rendering.

use crate::protocol::ProtocolFamily;
use serde_json::{Map, Value};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// Native `getRateLimitResetDelayMs` cap for unified reset waits.
pub const ANTHROPIC_PERSISTENT_RESET_CAP_MILLIS: u64 = 6 * 60 * 60 * 1000;

/// A pair of utilization and reset values for one Anthropic quota window.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AnthropicQuotaWindowHeaders {
    pub utilization: Option<f64>,
    pub reset_epoch_seconds: Option<u64>,
    pub surpassed_threshold: Option<f64>,
}

/// Anthropic quota headers after provider-specific parsing.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AnthropicRateLimitHeaders {
    pub representative_claim: Option<String>,
    pub overage_status: Option<String>,
    pub overage_disabled_reason: Option<String>,
    pub status: Option<String>,
    pub reset_epoch_seconds: Option<u64>,
    pub overage_reset_epoch_seconds: Option<u64>,
    pub fallback_available: Option<bool>,
    pub overage_in_use: bool,
    pub upgrade_paths: Option<Vec<String>>,
    pub overage_period_monthly_utilization: Option<f64>,
    pub overage_period_channel_utilization: Option<f64>,
    pub five_hour: AnthropicQuotaWindowHeaders,
    pub seven_day: AnthropicQuotaWindowHeaders,
    pub overage: AnthropicQuotaWindowHeaders,
}

impl AnthropicRateLimitHeaders {
    /// Decode the Anthropic unified and request-window rate-limit headers.
    #[must_use]
    pub fn decode(headers: &[(String, String)]) -> Self {
        let window = |abbrev: &str| {
            let threshold = format!("anthropic-ratelimit-unified-{abbrev}-surpassed-threshold");
            AnthropicQuotaWindowHeaders {
                utilization: finite_number(
                    headers,
                    &format!("anthropic-ratelimit-unified-{abbrev}-utilization"),
                ),
                reset_epoch_seconds: epoch_seconds(
                    headers,
                    &format!("anthropic-ratelimit-unified-{abbrev}-reset"),
                ),
                surpassed_threshold: finite_number(headers, &threshold),
            }
        };

        Self {
            representative_claim: header_value(
                headers,
                "anthropic-ratelimit-unified-representative-claim",
            )
            .map(str::to_owned),
            overage_status: header_value(headers, "anthropic-ratelimit-unified-overage-status")
                .map(str::to_owned),
            overage_disabled_reason: header_value(
                headers,
                "anthropic-ratelimit-unified-overage-disabled-reason",
            )
            .map(str::to_owned),
            status: header_value(headers, "anthropic-ratelimit-unified-status")
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
            reset_epoch_seconds: epoch_seconds(headers, "anthropic-ratelimit-unified-reset"),
            overage_reset_epoch_seconds: epoch_seconds(
                headers,
                "anthropic-ratelimit-unified-overage-reset",
            ),
            fallback_available: header_value(headers, "anthropic-ratelimit-unified-fallback")
                .map(|value| value == "available"),
            overage_in_use: header_value(headers, "anthropic-ratelimit-unified-overage-in-use")
                == Some("true"),
            upgrade_paths: header_value(headers, "anthropic-ratelimit-unified-upgrade-paths")
                .filter(|value| !value.is_empty())
                .map(|value| value.split(',').map(str::trim).map(str::to_owned).collect()),
            overage_period_monthly_utilization: finite_number(
                headers,
                "anthropic-ratelimit-unified-overage-period-monthly-utilization",
            ),
            overage_period_channel_utilization: finite_number(
                headers,
                "anthropic-ratelimit-unified-overage-period-channel-utilization",
            ),
            five_hour: window("5h"),
            seven_day: window("7d"),
            overage: window("overage"),
        }
    }

    /// Native error-path view for a 429 response, including its nested quota
    /// error body. A body-only `credits_required` error is retained.
    #[must_use]
    pub fn decode_429_error(
        headers: &[(String, String)],
        body: Option<&Value>,
    ) -> Option<AnthropicRateLimitError> {
        let representative_claim =
            nonempty_header(headers, "anthropic-ratelimit-unified-representative-claim");
        let overage_status = nonempty_header(headers, "anthropic-ratelimit-unified-overage-status");
        let (credits_required, body_disabled_reason) = credits_required_from_body(body);
        if representative_claim.is_none() && overage_status.is_none() && !credits_required {
            return None;
        }
        Some(AnthropicRateLimitError {
            representative_claim,
            overage_status,
            reset_epoch_seconds: nonempty_header(headers, "anthropic-ratelimit-unified-reset")
                .and_then(|_| epoch_seconds(headers, "anthropic-ratelimit-unified-reset")),
            overage_reset_epoch_seconds: nonempty_header(
                headers,
                "anthropic-ratelimit-unified-overage-reset",
            )
            .and_then(|_| epoch_seconds(headers, "anthropic-ratelimit-unified-overage-reset")),
            overage_disabled_reason: nonempty_header(
                headers,
                "anthropic-ratelimit-unified-overage-disabled-reason",
            )
            .or(body_disabled_reason),
            credits_required,
        })
    }

    /// Resolve an Anthropic request-window reset using the SDK's current wire
    /// formats. Host policy decides whether and when to use the result.
    #[must_use]
    pub fn request_reset_delay(headers: &[(String, String)], now: SystemTime) -> Option<Duration> {
        let raw = header_value(headers, "anthropic-ratelimit-requests-reset")?.trim();
        let target = parse_anthropic_utc_seconds(raw)?;
        let now_seconds = now.duration_since(UNIX_EPOCH).ok()?.as_secs();
        Some(Duration::from_secs(target.saturating_sub(now_seconds)))
    }

    /// Decode Native's unified reset delay, including its positive-delay gate
    /// and six-hour cap. Callers retain ownership of retry admission.
    #[must_use]
    pub fn unified_reset_delay(headers: &[(String, String)], now: SystemTime) -> Option<Duration> {
        let reset_seconds =
            javascript_number(header_value(headers, "anthropic-ratelimit-unified-reset")?)?;
        if !reset_seconds.is_finite() {
            return None;
        }
        let now_millis = now.duration_since(UNIX_EPOCH).ok()?.as_millis() as f64;
        let remaining_millis = (reset_seconds * 1000.0 - now_millis).round();
        if remaining_millis.is_nan() || remaining_millis <= 0.0 {
            return None;
        }
        let cap_millis = ANTHROPIC_PERSISTENT_RESET_CAP_MILLIS as f64;
        Some(Duration::from_millis(
            remaining_millis.min(cap_millis) as u64
        ))
    }
}

/// Provider response fields decoded once for Host retry, history, and account
/// state consumers.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProviderResponseHeaders {
    pub request_id: Option<String>,
    /// Nonnegative retry hint consumed as `Option<Duration>` by Host retry policy;
    /// HTTP dates at or before `now` resolve to zero. Native 429/529 signed
    /// suppression and marker state are separate and are not encoded here.
    pub retry_after: Option<Duration>,
    pub openai_reset: Option<Duration>,
    pub anthropic_request_reset: Option<Duration>,
    pub anthropic_unified_reset: Option<Duration>,
    pub stream_metadata: Value,
    pub anthropic_rate_limits: Option<AnthropicRateLimitHeaders>,
}

/// Native 2.1.291 provider identity for its client-request-id middleware.
/// Host route resolution must map to this enum from trusted provider facts;
/// custom Anthropic-compatible routes stay `Other`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeAnthropicProvider {
    FirstParty,
    AnthropicAws,
    Bedrock,
    Other,
}

/// Resolved first-party base-url fact. `ExplicitlyAssumed` corresponds to
/// Native `_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL`; the SDK does not read
/// process environment or infer trust from a caller-controlled URL.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstPartyBaseUrlFact {
    Default,
    AnthropicApiHost,
    ExplicitlyAssumed,
    Custom,
}

/// Three-way environment state is required because Native's per-attempt gate
/// tests `=== undefined`, while its fetch fallback uses JavaScript truthiness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicAwsBaseUrlFact {
    Undefined,
    DefinedEmpty,
    DefinedNonEmpty,
}

/// Trusted, resolved facts for the current Anthropic request route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NativeAnthropicRequestHeaderFacts {
    pub protocol: ProtocolFamily,
    pub provider: NativeAnthropicProvider,
    /// Source-level process base URL observed by Native's `vg()` gate.
    pub first_party_base_url: FirstPartyBaseUrlFact,
    /// Resolved selected-route URL; custom profiles must not inherit process
    /// first-party eligibility unless the explicit Native override is set.
    pub selected_base_url: FirstPartyBaseUrlFact,
    pub anthropic_aws_base_url: AnthropicAwsBaseUrlFact,
}

/// Distinct Native request-id insertion gates. An empty but defined
/// `ANTHROPIC_AWS_BASE_URL` disables per-attempt insertion while allowing the
/// fetch fallback, so these facts must not be collapsed into one boolean.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct NativeAnthropicRequestHeaderPlan {
    pub per_attempt_client_request_id: bool,
    pub fetch_fallback_client_request_id: bool,
}

impl NativeAnthropicRequestHeaderFacts {
    /// Build facts from a current selected provider profile. Only Native's
    /// explicit Anthropic profile identities participate; arbitrary compatible
    /// providers and Bedrock remain distinct.
    #[must_use]
    pub fn from_provider_profile(
        profile: &crate::protocol::ProviderProfile,
        selected_url: &str,
    ) -> Option<Self> {
        let provider = match (profile.protocol, profile.provider_id.as_str()) {
            (ProtocolFamily::AnthropicMessages, "anthropic") => NativeAnthropicProvider::FirstParty,
            (ProtocolFamily::AnthropicMessages, "anthropicAws") => {
                NativeAnthropicProvider::AnthropicAws
            }
            (ProtocolFamily::BedrockClaude, _) => NativeAnthropicProvider::Bedrock,
            _ => return None,
        };
        Some(Self::from_process_environment(
            profile.protocol,
            provider,
            selected_url,
        ))
    }

    /// Resolve explicit Native environment gates after the caller has selected
    /// a trusted provider kind. The process base URL and resolved route URL are
    /// kept separate because Native reads both at different boundaries.
    #[must_use]
    pub fn from_process_environment(
        protocol: ProtocolFamily,
        provider: NativeAnthropicProvider,
        selected_url: &str,
    ) -> Self {
        let assume_first_party = std::env::var("_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL")
            .ok()
            .is_some_and(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            });
        let first_party_base_url = if assume_first_party {
            FirstPartyBaseUrlFact::ExplicitlyAssumed
        } else {
            match std::env::var("ANTHROPIC_BASE_URL") {
                Ok(url) if anthropic_api_host(&url) => FirstPartyBaseUrlFact::AnthropicApiHost,
                Ok(_) | Err(std::env::VarError::NotUnicode(_)) => FirstPartyBaseUrlFact::Custom,
                Err(std::env::VarError::NotPresent) => FirstPartyBaseUrlFact::Default,
            }
        };
        let selected_base_url = if anthropic_api_host(selected_url) {
            FirstPartyBaseUrlFact::AnthropicApiHost
        } else {
            FirstPartyBaseUrlFact::Custom
        };
        let anthropic_aws_base_url = match std::env::var_os("ANTHROPIC_AWS_BASE_URL") {
            None => AnthropicAwsBaseUrlFact::Undefined,
            Some(value) if value.is_empty() => AnthropicAwsBaseUrlFact::DefinedEmpty,
            Some(_) => AnthropicAwsBaseUrlFact::DefinedNonEmpty,
        };
        Self {
            protocol,
            provider,
            first_party_base_url,
            selected_base_url,
            anthropic_aws_base_url,
        }
    }

    #[must_use]
    pub fn plan(self) -> NativeAnthropicRequestHeaderPlan {
        let first_party_base_allows = match self.first_party_base_url {
            FirstPartyBaseUrlFact::Default | FirstPartyBaseUrlFact::AnthropicApiHost => {
                self.selected_base_url != FirstPartyBaseUrlFact::Custom
            }
            FirstPartyBaseUrlFact::ExplicitlyAssumed => true,
            FirstPartyBaseUrlFact::Custom => false,
        };
        let first_party = self.protocol == ProtocolFamily::AnthropicMessages
            && self.provider == NativeAnthropicProvider::FirstParty
            && first_party_base_allows;
        let anthropic_aws = self.protocol == ProtocolFamily::AnthropicMessages
            && self.provider == NativeAnthropicProvider::AnthropicAws;
        let aws_unset = self.anthropic_aws_base_url == AnthropicAwsBaseUrlFact::Undefined;
        let aws_fetch_default =
            self.anthropic_aws_base_url != AnthropicAwsBaseUrlFact::DefinedNonEmpty;
        NativeAnthropicRequestHeaderPlan {
            per_attempt_client_request_id: first_party || (anthropic_aws && aws_unset),
            fetch_fallback_client_request_id: first_party || (anthropic_aws && aws_fetch_default),
        }
    }
}

fn anthropic_api_host(url: &str) -> bool {
    url::Url::parse(url)
        .is_ok_and(|url| url.host_str() == Some("api.anthropic.com") && url.port().is_none())
}

/// The two Native request stages that can supply `x-client-request-id`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeRequestIdStage {
    PerAttempt,
    FetchFallback,
}

/// Apply Native's UUIDv4 `x-client-request-id` rule at one explicit SDK stage.
/// Any caller-supplied spelling/value is preserved, including an empty value.
pub fn ensure_native_client_request_id(
    headers: &mut Vec<(String, String)>,
    facts: NativeAnthropicRequestHeaderFacts,
    stage: NativeRequestIdStage,
) -> Result<Option<String>, crate::protocol::LlmError> {
    if let Some((_, value)) = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("x-client-request-id"))
    {
        return Ok(Some(value.clone()));
    }
    let plan = facts.plan();
    let eligible = match stage {
        NativeRequestIdStage::PerAttempt => plan.per_attempt_client_request_id,
        NativeRequestIdStage::FetchFallback => plan.fetch_fallback_client_request_id,
    };
    if !eligible {
        return Ok(None);
    }
    let request_id = uuid_v4()?;
    headers.push(("x-client-request-id".into(), request_id.clone()));
    Ok(Some(request_id))
}

/// Native `sZt` carries traceparent only when a trace context exists and the
/// route is first-party/Anthropic-AWS eligible or explicit propagation is on.
#[must_use]
pub fn should_propagate_native_traceparent(
    has_trace_context: bool,
    plan: NativeAnthropicRequestHeaderPlan,
    propagate_traceparent: bool,
) -> bool {
    has_trace_context && (plan.per_attempt_client_request_id || propagate_traceparent)
}

/// Read the explicit Native trace-propagation opt-in independently from the
/// per-attempt request-id gate.
#[must_use]
pub fn native_traceparent_opt_in_from_environment() -> bool {
    std::env::var("CLAUDE_CODE_PROPAGATE_TRACEPARENT")
        .ok()
        .is_some_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
}

/// Add the current trace context only when the Native `sZt` trace gate allows
/// it. Caller-supplied traceparent headers keep their existing value.
#[must_use]
pub fn ensure_native_traceparent(
    headers: &mut Vec<(String, String)>,
    traceparent: &str,
    facts: NativeAnthropicRequestHeaderFacts,
    propagate_traceparent: bool,
) -> bool {
    if !should_propagate_native_traceparent(
        !traceparent.is_empty(),
        facts.plan(),
        propagate_traceparent,
    ) || headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("traceparent"))
    {
        return false;
    }
    headers.push(("traceparent".into(), traceparent.to_owned()));
    true
}

fn uuid_v4() -> Result<String, crate::protocol::LlmError> {
    use ring::rand::{SecureRandom, SystemRandom};
    let mut bytes = [0_u8; 16];
    SystemRandom::new().fill(&mut bytes).map_err(|_| {
        crate::protocol::LlmError::ProviderInternal {
            message: "secure request-id generation failed".into(),
        }
    })?;
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Ok(format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0],
        bytes[1],
        bytes[2],
        bytes[3],
        bytes[4],
        bytes[5],
        bytes[6],
        bytes[7],
        bytes[8],
        bytes[9],
        bytes[10],
        bytes[11],
        bytes[12],
        bytes[13],
        bytes[14],
        bytes[15]
    ))
}

impl ProviderResponseHeaders {
    /// Decode response metadata for the selected protocol. `now` makes the
    /// HTTP-date retry delay deterministic for callers and tests.
    #[must_use]
    pub fn decode(
        protocol: ProtocolFamily,
        provider_id: &str,
        headers: &[(String, String)],
        now: SystemTime,
    ) -> Self {
        let request_id = response_request_id(protocol, provider_id, headers);
        let retry_after = retry_after(headers, now);
        let openai_reset = openai_reset(headers);
        let stream_metadata = stream_metadata(headers);
        let is_anthropic = protocol == ProtocolFamily::AnthropicMessages;
        let anthropic_rate_limits =
            is_anthropic.then(|| AnthropicRateLimitHeaders::decode(headers));
        Self {
            request_id,
            retry_after,
            openai_reset,
            anthropic_request_reset: is_anthropic
                .then(|| AnthropicRateLimitHeaders::request_reset_delay(headers, now))
                .flatten(),
            anthropic_unified_reset: is_anthropic
                .then(|| AnthropicRateLimitHeaders::unified_reset_delay(headers, now))
                .flatten(),
            stream_metadata,
            anthropic_rate_limits,
        }
    }
}

/// Nested error fields consumed by Anthropic's rejected-quota state path.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AnthropicRateLimitError {
    pub representative_claim: Option<String>,
    pub overage_status: Option<String>,
    pub reset_epoch_seconds: Option<u64>,
    pub overage_reset_epoch_seconds: Option<u64>,
    pub overage_disabled_reason: Option<String>,
    pub credits_required: bool,
}

fn credits_required_from_body(body: Option<&Value>) -> (bool, Option<String>) {
    let Some(details) = body
        .and_then(|body| body.get("error"))
        .and_then(|error| error.get("error"))
        .and_then(|error| error.get("details"))
    else {
        return (false, None);
    };
    if details.get("error_code").and_then(Value::as_str) != Some("credits_required") {
        return (false, None);
    }
    (
        true,
        details
            .get("disabled_reason")
            .and_then(Value::as_str)
            .map(str::to_owned),
    )
}

fn response_request_id(
    protocol: ProtocolFamily,
    provider_id: &str,
    headers: &[(String, String)],
) -> Option<String> {
    let canonical: &[&str] = match protocol {
        ProtocolFamily::AnthropicMessages => &["request-id", "x-request-id"],
        ProtocolFamily::AzureOpenAi => &["apim-request-id", "x-ms-request-id", "x-request-id"],
        ProtocolFamily::BedrockClaude => &["x-amzn-requestid", "x-amzn-request-id"],
        ProtocolFamily::GeminiGenerateContent
        | ProtocolFamily::VertexGemini
        | ProtocolFamily::VertexClaude => &["x-goog-request-id", "x-request-id"],
        _ if provider_id.eq_ignore_ascii_case("openai") => &["x-request-id"],
        _ => &["x-request-id", "request-id"],
    };
    canonical
        .iter()
        .find_map(|name| header_value(headers, name))
        .or_else(|| {
            [
                "apim-request-id",
                "x-ms-request-id",
                "x-amzn-requestid",
                "x-amzn-request-id",
                "x-goog-request-id",
            ]
            .iter()
            .find_map(|name| header_value(headers, name))
        })
        .map(str::to_owned)
}

/// Decode the SDK's cross-provider request-id fallback list when the Host has
/// no resolved profile available at the response boundary.
#[must_use]
pub fn generic_request_id(headers: &[(String, String)]) -> Option<String> {
    [
        "request-id",
        "x-request-id",
        "apim-request-id",
        "x-ms-request-id",
        "x-amzn-requestid",
        "x-amzn-request-id",
        "x-goog-request-id",
    ]
    .iter()
    .find_map(|name| header_value(headers, name))
    .map(str::to_owned)
}

fn retry_after(headers: &[(String, String)], now: SystemTime) -> Option<Duration> {
    if let Some(milliseconds) =
        header_value(headers, "retry-after-ms").and_then(javascript_parse_float)
    {
        if milliseconds != 0.0 {
            return duration_from_seconds(milliseconds / 1000.0);
        }
    }

    let value = header_value(headers, "retry-after")?;
    if let Some(seconds) = javascript_parse_float(value) {
        return duration_from_seconds(seconds);
    }

    retry_after_http_date(value, now)
}

fn duration_from_seconds(seconds: f64) -> Option<Duration> {
    if !seconds.is_finite() || seconds < 0.0 {
        return None;
    }
    Duration::try_from_secs_f64(seconds).ok()
}

fn retry_after_http_date(value: &str, now: SystemTime) -> Option<Duration> {
    let date = chrono::DateTime::parse_from_rfc2822(value.trim()).ok()?;
    let remaining_millis = i128::from(date.timestamp_millis()) - unix_millis(now);
    if remaining_millis <= 0 {
        return Some(Duration::ZERO);
    }
    u64::try_from(remaining_millis)
        .ok()
        .map(Duration::from_millis)
}

fn unix_millis(time: SystemTime) -> i128 {
    match time.duration_since(UNIX_EPOCH) {
        Ok(duration) => duration.as_millis() as i128,
        Err(error) => -(error.duration().as_millis() as i128),
    }
}

/// Match JavaScript `parseFloat`'s leading numeric prefix for the Native header
/// precedence. Invalid, negative, or non-finite selected values are left for
/// the caller's retry policy; this shared `Duration` cannot represent the
/// signed delay used by Native's separate 429/529 suppression path.
fn javascript_parse_float(value: &str) -> Option<f64> {
    let value = value.trim_start_matches(is_javascript_whitespace);
    let bytes = value.as_bytes();
    let mut index = 0;
    let negative = match bytes.first() {
        Some(b'-') => {
            index = 1;
            true
        }
        Some(b'+') => {
            index = 1;
            false
        }
        _ => false,
    };

    if value[index..].starts_with("Infinity") {
        return Some(if negative {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        });
    }

    let integer_start = index;
    while bytes.get(index).is_some_and(u8::is_ascii_digit) {
        index += 1;
    }
    let mut saw_digit = index > integer_start;
    if bytes.get(index) == Some(&b'.') {
        index += 1;
        let fractional_start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        saw_digit |= index > fractional_start;
    }
    if !saw_digit {
        return None;
    }

    if matches!(bytes.get(index), Some(b'e' | b'E')) {
        let exponent_marker = index;
        index += 1;
        if matches!(bytes.get(index), Some(b'+' | b'-')) {
            index += 1;
        }
        let exponent_start = index;
        while bytes.get(index).is_some_and(u8::is_ascii_digit) {
            index += 1;
        }
        if index == exponent_start {
            index = exponent_marker;
        }
    }

    value[..index].parse().ok()
}

/// Parse the `Retry-After` delta-seconds wire form without selecting provider
/// policy or consulting the clock.
#[must_use]
pub fn retry_after_delta_seconds(headers: &[(String, String)]) -> Option<Duration> {
    header_value(headers, "retry-after")?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

fn openai_reset(headers: &[(String, String)]) -> Option<Duration> {
    let requests = header_value(headers, "x-ratelimit-reset-requests").and_then(parse_go_duration);
    let tokens = header_value(headers, "x-ratelimit-reset-tokens").and_then(parse_go_duration);
    match (requests, tokens) {
        (Some(requests), Some(tokens)) => Some(requests.max(tokens)),
        (Some(value), None) | (None, Some(value)) => Some(value),
        (None, None) => None,
    }
}

/// Parse a Go-style duration used by OpenAI reset headers.
#[must_use]
pub fn parse_go_duration(raw: &str) -> Option<Duration> {
    let value = raw.trim();
    if value.is_empty() {
        return None;
    }
    let bytes = value.as_bytes();
    let mut index = 0;
    let mut total_seconds = 0.0f64;
    let mut saw_component = false;
    while index < bytes.len() {
        let number_start = index;
        while index < bytes.len() && (bytes[index].is_ascii_digit() || bytes[index] == b'.') {
            index += 1;
        }
        if number_start == index {
            return None;
        }
        let number = value[number_start..index].parse::<f64>().ok()?;
        let unit_start = index;
        while index < bytes.len() && !bytes[index].is_ascii_digit() && bytes[index] != b'.' {
            index += 1;
        }
        let multiplier = match &value[unit_start..index] {
            "h" => 3600.0,
            "m" => 60.0,
            "s" | "" => 1.0,
            "ms" => 0.001,
            "us" | "µs" => 0.000_001,
            "ns" => 0.000_000_001,
            _ => return None,
        };
        total_seconds += number * multiplier;
        saw_component = true;
    }
    if !saw_component || !total_seconds.is_finite() {
        return None;
    }
    Duration::try_from_secs_f64(total_seconds.max(0.0)).ok()
}

fn parse_anthropic_utc_seconds(raw: &str) -> Option<u64> {
    let bytes = raw.as_bytes();
    if bytes.len() != 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || bytes[19] != b'Z'
    {
        return None;
    }
    let year = raw[0..4].parse::<i32>().ok()?;
    let month = raw[5..7].parse::<u32>().ok()?;
    let day = raw[8..10].parse::<u32>().ok()?;
    let hour = raw[11..13].parse::<u32>().ok()?;
    let minute = raw[14..16].parse::<u32>().ok()?;
    let second = raw[17..19].parse::<u32>().ok()?;
    chrono::NaiveDate::from_ymd_opt(year, month, day)?
        .and_hms_opt(hour, minute, second)?
        .and_utc()
        .timestamp()
        .try_into()
        .ok()
}

/// Filter provider control headers that are retained with a streamed result.
#[must_use]
pub fn stream_metadata(headers: &[(String, String)]) -> Value {
    let mut metadata = Map::new();
    for (name, value) in headers {
        let name = name.to_ascii_lowercase();
        if name == "openai-model"
            || name == "x-models-etag"
            || name == "x-reasoning-included"
            || name == "x-request-id"
            || name == "x-codex-turn-state"
            || name == "retry-after"
            || name.starts_with("x-ratelimit-")
        {
            metadata.insert(name, Value::String(value.clone()));
        }
    }
    if metadata.is_empty() {
        Value::Null
    } else {
        Value::Object(metadata)
    }
}

fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(key, _)| key.eq_ignore_ascii_case(name))
        .map(|(_, value)| value.as_str())
}

fn nonempty_header(headers: &[(String, String)], name: &str) -> Option<String> {
    header_value(headers, name)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn epoch_seconds(headers: &[(String, String)], name: &str) -> Option<u64> {
    let raw = header_value(headers, name).filter(|value| !value.is_empty())?;
    let number = javascript_number(raw)?;
    (number.is_finite() && number >= 0.0 && number.fract() == 0.0 && number < u64::MAX as f64)
        .then_some(number as u64)
}

fn finite_number(headers: &[(String, String)], name: &str) -> Option<f64> {
    let raw = header_value(headers, name).filter(|value| !value.is_empty())?;
    javascript_number(raw).filter(|value| value.is_finite())
}

fn javascript_number(value: &str) -> Option<f64> {
    let value = value.trim_matches(is_javascript_whitespace);
    if value.is_empty() {
        return Some(0.0);
    }
    if value == "Infinity" || value == "+Infinity" {
        return Some(f64::INFINITY);
    }
    if value == "-Infinity" {
        return Some(f64::NEG_INFINITY);
    }
    let radix = if let Some(digits) = value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
    {
        Some((digits, 16))
    } else if let Some(digits) = value
        .strip_prefix("0b")
        .or_else(|| value.strip_prefix("0B"))
    {
        Some((digits, 2))
    } else {
        value
            .strip_prefix("0o")
            .or_else(|| value.strip_prefix("0O"))
            .map(|digits| (digits, 8))
    };
    if let Some((digits, radix)) = radix {
        if digits.is_empty() {
            return None;
        }
        let mut number = 0.0;
        for character in digits.chars() {
            number = number * f64::from(radix) + f64::from(character.to_digit(radix)?);
        }
        return Some(number);
    }
    value.parse::<f64>().ok()
}

fn is_javascript_whitespace(character: char) -> bool {
    matches!(
        character,
        '\u{0009}'
            | '\u{000A}'
            | '\u{000B}'
            | '\u{000C}'
            | '\u{000D}'
            | '\u{0020}'
            | '\u{00A0}'
            | '\u{1680}'
            | '\u{2000}'
            ..='\u{200A}'
                | '\u{2028}'
                | '\u{2029}'
                | '\u{202F}'
                | '\u{205F}'
                | '\u{3000}'
                | '\u{FEFF}'
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(entries: &[(&str, &str)]) -> Vec<(String, String)> {
        entries
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }

    #[test]
    fn decodes_protocol_request_ids_and_retry_fields_case_insensitively() {
        let headers = headers(&[
            ("X-Request-Id", "generic"),
            ("Request-Id", "anthropic"),
            ("Retry-After-Ms", "250"),
        ]);
        let parsed = ProviderResponseHeaders::decode(
            ProtocolFamily::AnthropicMessages,
            "anthropic",
            &headers,
            UNIX_EPOCH,
        );
        assert_eq!(parsed.request_id.as_deref(), Some("anthropic"));
        assert_eq!(parsed.retry_after, Some(Duration::from_millis(250)));
    }

    #[test]
    fn request_id_header_selection_prefers_provider_canonical_fields() {
        let bedrock = headers(&[
            ("x-request-id", "fallback"),
            ("X-Amzn-RequestId", "bedrock"),
        ]);
        assert_eq!(
            ProviderResponseHeaders::decode(
                ProtocolFamily::BedrockClaude,
                "bedrock",
                &bedrock,
                UNIX_EPOCH,
            )
            .request_id
            .as_deref(),
            Some("bedrock")
        );

        let openai = headers(&[("request-id", "fallback"), ("X-Request-ID", "openai")]);
        assert_eq!(
            ProviderResponseHeaders::decode(
                ProtocolFamily::OpenAiResponses,
                "openai",
                &openai,
                UNIX_EPOCH,
            )
            .request_id
            .as_deref(),
            Some("openai")
        );
    }

    #[test]
    fn generic_request_id_fallback_preserves_existing_priority() {
        let headers = headers(&[
            ("x-request-id", "openai"),
            ("request-id", "anthropic"),
            ("x-amzn-requestid", "bedrock"),
        ]);
        assert_eq!(generic_request_id(&headers).as_deref(), Some("anthropic"));
    }

    #[test]
    fn retry_after_ms_precedes_retry_after_and_uses_javascript_float_prefixes() {
        let headers = headers(&[("retry-after-ms", "250.5ms"), ("retry-after", "9")]);
        let decoded = ProviderResponseHeaders::decode(
            ProtocolFamily::AnthropicMessages,
            "anthropic",
            &headers,
            UNIX_EPOCH,
        );
        assert_eq!(decoded.retry_after, Some(Duration::from_micros(250_500)));
    }

    #[test]
    fn zero_or_invalid_retry_after_ms_falls_back_to_retry_after() {
        for milliseconds in ["0", "invalid"] {
            let headers = headers(&[
                ("retry-after-ms", milliseconds),
                ("retry-after", "0.25seconds"),
            ]);
            let decoded = ProviderResponseHeaders::decode(
                ProtocolFamily::AnthropicMessages,
                "anthropic",
                &headers,
                UNIX_EPOCH,
            );
            assert_eq!(decoded.retry_after, Some(Duration::from_millis(250)));
        }
    }

    #[test]
    fn nonzero_invalid_retry_after_ms_does_not_fall_back() {
        for milliseconds in ["-1", "Infinity", "1e309"] {
            let headers = headers(&[("retry-after-ms", milliseconds), ("retry-after", "7")]);
            let decoded = ProviderResponseHeaders::decode(
                ProtocolFamily::AnthropicMessages,
                "anthropic",
                &headers,
                UNIX_EPOCH,
            );
            assert_eq!(decoded.retry_after, None, "retry-after-ms={milliseconds}");
        }
    }

    #[test]
    fn retry_after_seconds_preserves_zero_fraction_and_values_above_native_timer_cap() {
        let zero = ProviderResponseHeaders::decode(
            ProtocolFamily::AnthropicMessages,
            "anthropic",
            &headers(&[("retry-after", "0")]),
            UNIX_EPOCH,
        );
        assert_eq!(zero.retry_after, Some(Duration::ZERO));

        let fractional = ProviderResponseHeaders::decode(
            ProtocolFamily::AnthropicMessages,
            "anthropic",
            &headers(&[("retry-after", "0.25")]),
            UNIX_EPOCH,
        );
        assert_eq!(fractional.retry_after, Some(Duration::from_millis(250)));

        for raw in ["-1", "Infinity", "1e309"] {
            let invalid = ProviderResponseHeaders::decode(
                ProtocolFamily::AnthropicMessages,
                "anthropic",
                &headers(&[("retry-after", raw)]),
                UNIX_EPOCH,
            );
            assert_eq!(invalid.retry_after, None, "retry-after={raw}");
        }

        let above_native_timer_cap = ProviderResponseHeaders::decode(
            ProtocolFamily::AnthropicMessages,
            "anthropic",
            &headers(&[("retry-after-ms", "2147483648")]),
            UNIX_EPOCH,
        );
        assert_eq!(
            above_native_timer_cap.retry_after,
            Some(Duration::from_millis(2_147_483_648))
        );
    }

    #[test]
    fn retry_after_http_date_resolves_against_now_and_clamps_past_dates_to_zero() {
        let date_headers = headers(&[("retry-after", "Tue, 13 Jan 1970 12:00:00 GMT")]);
        let decoded = ProviderResponseHeaders::decode(
            ProtocolFamily::AnthropicMessages,
            "anthropic",
            &date_headers,
            UNIX_EPOCH + Duration::from_secs(1_070_000),
        );
        assert_eq!(decoded.retry_after, Some(Duration::from_secs(10_000)));

        let at_now = ProviderResponseHeaders::decode(
            ProtocolFamily::AnthropicMessages,
            "anthropic",
            &date_headers,
            UNIX_EPOCH + Duration::from_secs(1_080_000),
        );
        assert_eq!(at_now.retry_after, Some(Duration::ZERO));

        let past = ProviderResponseHeaders::decode(
            ProtocolFamily::AnthropicMessages,
            "anthropic",
            &date_headers,
            UNIX_EPOCH + Duration::from_secs(1_080_001),
        );
        assert_eq!(past.retry_after, Some(Duration::ZERO));

        let invalid = ProviderResponseHeaders::decode(
            ProtocolFamily::AnthropicMessages,
            "anthropic",
            &headers(&[("retry-after", "not a date")]),
            UNIX_EPOCH,
        );
        assert_eq!(invalid.retry_after, None);
    }

    #[test]
    fn retry_after_delta_seconds_helper_keeps_its_explicit_wire_contract() {
        assert_eq!(
            retry_after_delta_seconds(&headers(&[("retry-after", "5")])),
            Some(Duration::from_secs(5))
        );
        assert_eq!(
            retry_after_delta_seconds(&headers(&[(
                "retry-after",
                "Tue, 13 Jan 1970 12:00:00 GMT",
            )])),
            None
        );
    }

    #[test]
    fn stream_metadata_keeps_only_the_sdk_control_header_set() {
        let headers = headers(&[
            ("OpenAI-Model", "gpt-current"),
            ("X-RateLimit-Remaining", "3"),
            ("Retry-After", "2"),
            ("Set-Cookie", "private=value"),
        ]);
        let metadata = stream_metadata(&headers);
        assert_eq!(metadata["openai-model"], "gpt-current");
        assert_eq!(metadata["x-ratelimit-remaining"], "3");
        assert_eq!(metadata["retry-after"], "2");
        assert!(metadata.get("set-cookie").is_none());
    }

    #[test]
    fn decodes_quota_fields_without_an_empty_threshold() {
        let headers = headers(&[
            ("Anthropic-RateLimit-Unified-Status", "allowed"),
            ("anthropic-ratelimit-unified-5h-utilization", "0.95"),
            ("anthropic-ratelimit-unified-5h-reset", "1700000000"),
            ("anthropic-ratelimit-unified-5h-surpassed-threshold", ""),
            ("anthropic-ratelimit-unified-overage-in-use", "true"),
            ("anthropic-ratelimit-unified-upgrade-paths", "team, max"),
        ]);
        let decoded = AnthropicRateLimitHeaders::decode(&headers);
        assert_eq!(decoded.status.as_deref(), Some("allowed"));
        assert_eq!(decoded.five_hour.surpassed_threshold, None);
        assert!(decoded.overage_in_use);
        assert_eq!(
            decoded
                .upgrade_paths
                .as_deref()
                .map(|values| values.iter().map(String::as_str).collect::<Vec<_>>()),
            Some(vec!["team", "max"])
        );
    }

    #[test]
    fn quota_numbers_drop_empty_values_and_parse_radix_forms() {
        let headers = headers(&[
            ("anthropic-ratelimit-unified-5h-utilization", ""),
            ("anthropic-ratelimit-unified-5h-reset", "0x3e9"),
            ("anthropic-ratelimit-unified-5h-surpassed-threshold", ""),
        ]);
        let decoded = AnthropicRateLimitHeaders::decode(&headers);
        assert_eq!(decoded.five_hour.utilization, None);
        assert_eq!(decoded.five_hour.reset_epoch_seconds, Some(1001));
        assert_eq!(decoded.five_hour.surpassed_threshold, None);
    }

    #[test]
    fn quota_finite_readings_distinguish_empty_from_whitespace_and_zero() {
        for (raw, expected) in [
            ("", None),
            (" ", Some(0.0)),
            ("0", Some(0.0)),
            ("-0", Some(-0.0)),
            ("0x10", Some(16.0)),
            ("garbage", None),
            ("NaN", None),
            ("Infinity", None),
            ("-Infinity", None),
        ] {
            let decoded = AnthropicRateLimitHeaders::decode(&headers(&[
                ("anthropic-ratelimit-unified-5h-utilization", raw),
                ("anthropic-ratelimit-unified-5h-surpassed-threshold", raw),
                (
                    "anthropic-ratelimit-unified-overage-period-monthly-utilization",
                    raw,
                ),
                (
                    "anthropic-ratelimit-unified-overage-period-channel-utilization",
                    raw,
                ),
            ]));
            let bits = expected.map(f64::to_bits);
            assert_eq!(
                decoded.five_hour.utilization.map(f64::to_bits),
                bits,
                "{raw:?}"
            );
            assert_eq!(
                decoded.five_hour.surpassed_threshold.map(f64::to_bits),
                bits,
                "{raw:?}"
            );
            assert_eq!(
                decoded.overage_period_monthly_utilization.map(f64::to_bits),
                bits,
                "{raw:?}"
            );
            assert_eq!(
                decoded.overage_period_channel_utilization.map(f64::to_bits),
                bits,
                "{raw:?}"
            );
        }
    }

    #[test]
    fn decodes_429_nested_error_without_treating_unrelated_body_as_quota() {
        let headers = headers(&[]);
        let body = serde_json::json!({
            "error": {"error": {"details": {
                "error_code": "credits_required",
                "disabled_reason": "out_of_credits"
            }}}
        });
        let decoded = AnthropicRateLimitHeaders::decode_429_error(&headers, Some(&body)).unwrap();
        assert!(decoded.credits_required);
        assert_eq!(
            decoded.overage_disabled_reason.as_deref(),
            Some("out_of_credits")
        );
        assert!(AnthropicRateLimitHeaders::decode_429_error(&headers, None).is_none());
        let empty_reset = vec![
            (
                "anthropic-ratelimit-unified-overage-status".to_owned(),
                "rejected".to_owned(),
            ),
            (
                "anthropic-ratelimit-unified-reset".to_owned(),
                String::new(),
            ),
        ];
        assert_eq!(
            AnthropicRateLimitHeaders::decode_429_error(&empty_reset, None)
                .unwrap()
                .reset_epoch_seconds,
            None
        );
    }

    #[test]
    fn native_request_id_gates_keep_first_party_aws_and_bedrock_distinct() {
        let first_party = |base| NativeAnthropicRequestHeaderFacts {
            protocol: ProtocolFamily::AnthropicMessages,
            provider: NativeAnthropicProvider::FirstParty,
            first_party_base_url: base,
            selected_base_url: match base {
                FirstPartyBaseUrlFact::Default | FirstPartyBaseUrlFact::AnthropicApiHost => {
                    FirstPartyBaseUrlFact::AnthropicApiHost
                }
                other => other,
            },
            anthropic_aws_base_url: AnthropicAwsBaseUrlFact::Undefined,
        };
        for base in [
            FirstPartyBaseUrlFact::Default,
            FirstPartyBaseUrlFact::AnthropicApiHost,
            FirstPartyBaseUrlFact::ExplicitlyAssumed,
        ] {
            let plan = first_party(base).plan();
            assert!(plan.per_attempt_client_request_id);
            assert!(plan.fetch_fallback_client_request_id);
        }
        let assumed_loopback = NativeAnthropicRequestHeaderFacts {
            protocol: ProtocolFamily::AnthropicMessages,
            provider: NativeAnthropicProvider::FirstParty,
            first_party_base_url: FirstPartyBaseUrlFact::ExplicitlyAssumed,
            selected_base_url: FirstPartyBaseUrlFact::Custom,
            anthropic_aws_base_url: AnthropicAwsBaseUrlFact::Undefined,
        };
        assert!(assumed_loopback.plan().per_attempt_client_request_id);
        assert!(assumed_loopback.plan().fetch_fallback_client_request_id);
        assert_eq!(
            first_party(FirstPartyBaseUrlFact::Custom).plan(),
            NativeAnthropicRequestHeaderPlan::default()
        );
        let wrong_protocol = NativeAnthropicRequestHeaderFacts {
            protocol: ProtocolFamily::OpenAiResponses,
            provider: NativeAnthropicProvider::FirstParty,
            first_party_base_url: FirstPartyBaseUrlFact::Default,
            selected_base_url: FirstPartyBaseUrlFact::Custom,
            anthropic_aws_base_url: AnthropicAwsBaseUrlFact::Undefined,
        };
        assert_eq!(
            wrong_protocol.plan(),
            NativeAnthropicRequestHeaderPlan::default()
        );

        let aws = |base_url| NativeAnthropicRequestHeaderFacts {
            protocol: ProtocolFamily::AnthropicMessages,
            provider: NativeAnthropicProvider::AnthropicAws,
            first_party_base_url: FirstPartyBaseUrlFact::Custom,
            selected_base_url: FirstPartyBaseUrlFact::Custom,
            anthropic_aws_base_url: base_url,
        };
        assert_eq!(
            aws(AnthropicAwsBaseUrlFact::Undefined).plan(),
            NativeAnthropicRequestHeaderPlan {
                per_attempt_client_request_id: true,
                fetch_fallback_client_request_id: true,
            }
        );
        assert_eq!(
            aws(AnthropicAwsBaseUrlFact::DefinedEmpty).plan(),
            NativeAnthropicRequestHeaderPlan {
                per_attempt_client_request_id: false,
                fetch_fallback_client_request_id: true,
            }
        );
        assert_eq!(
            aws(AnthropicAwsBaseUrlFact::DefinedNonEmpty).plan(),
            NativeAnthropicRequestHeaderPlan::default()
        );

        let bedrock = NativeAnthropicRequestHeaderFacts {
            protocol: ProtocolFamily::AnthropicMessages,
            provider: NativeAnthropicProvider::Bedrock,
            first_party_base_url: FirstPartyBaseUrlFact::Default,
            selected_base_url: FirstPartyBaseUrlFact::Custom,
            anthropic_aws_base_url: AnthropicAwsBaseUrlFact::Undefined,
        };
        assert_eq!(bedrock.plan(), NativeAnthropicRequestHeaderPlan::default());
    }

    #[test]
    fn native_route_facts_use_selected_profile_identity_not_endpoint_guessing() {
        let profile = |provider_id: &str, protocol: &str| {
            serde_json::from_value::<crate::protocol::ProviderProfile>(serde_json::json!({
                "provider_id": provider_id,
                "profile_name": "selected",
                "base_url": "https://api.anthropic.com",
                "protocol": protocol,
                "auth": "none",
                "models": []
            }))
            .unwrap()
        };
        assert_eq!(
            NativeAnthropicRequestHeaderFacts::from_provider_profile(
                &profile("anthropic", "anthropic_messages"),
                "https://api.anthropic.com",
            )
            .unwrap()
            .provider,
            NativeAnthropicProvider::FirstParty
        );
        assert_eq!(
            NativeAnthropicRequestHeaderFacts::from_provider_profile(
                &profile("anthropicAws", "anthropic_messages"),
                "https://gateway.example",
            )
            .unwrap()
            .provider,
            NativeAnthropicProvider::AnthropicAws
        );
        assert_eq!(
            NativeAnthropicRequestHeaderFacts::from_provider_profile(
                &profile("bedrock", "bedrock_claude"),
                "https://bedrock.example",
            )
            .unwrap()
            .provider,
            NativeAnthropicProvider::Bedrock
        );
        assert!(NativeAnthropicRequestHeaderFacts::from_provider_profile(
            &profile("custom-gateway", "anthropic_messages"),
            "https://gateway.example",
        )
        .is_none());
    }

    #[test]
    fn native_request_id_preserves_caller_value_and_generates_uuid_v4_when_missing() {
        let ineligible = NativeAnthropicRequestHeaderFacts {
            protocol: ProtocolFamily::AnthropicMessages,
            provider: NativeAnthropicProvider::Other,
            first_party_base_url: FirstPartyBaseUrlFact::Custom,
            selected_base_url: FirstPartyBaseUrlFact::Custom,
            anthropic_aws_base_url: AnthropicAwsBaseUrlFact::DefinedNonEmpty,
        };
        let mut supplied = vec![("X-Client-Request-Id".into(), "caller-value".into())];
        assert_eq!(
            ensure_native_client_request_id(
                &mut supplied,
                ineligible,
                NativeRequestIdStage::PerAttempt,
            )
            .unwrap()
            .as_deref(),
            Some("caller-value")
        );
        assert_eq!(supplied.len(), 1);

        let mut empty_caller = vec![("x-client-request-id".into(), String::new())];
        assert_eq!(
            ensure_native_client_request_id(
                &mut empty_caller,
                NativeAnthropicRequestHeaderFacts {
                    protocol: ProtocolFamily::AnthropicMessages,
                    provider: NativeAnthropicProvider::FirstParty,
                    first_party_base_url: FirstPartyBaseUrlFact::Default,
                    selected_base_url: FirstPartyBaseUrlFact::AnthropicApiHost,
                    anthropic_aws_base_url: AnthropicAwsBaseUrlFact::Undefined,
                },
                NativeRequestIdStage::PerAttempt,
            )
            .unwrap(),
            Some(String::new())
        );
        assert_eq!(empty_caller.len(), 1);

        let eligible = NativeAnthropicRequestHeaderFacts {
            protocol: ProtocolFamily::AnthropicMessages,
            provider: NativeAnthropicProvider::FirstParty,
            first_party_base_url: FirstPartyBaseUrlFact::Default,
            selected_base_url: FirstPartyBaseUrlFact::AnthropicApiHost,
            anthropic_aws_base_url: AnthropicAwsBaseUrlFact::Undefined,
        };
        let mut generated = Vec::new();
        let request_id = ensure_native_client_request_id(
            &mut generated,
            eligible,
            NativeRequestIdStage::PerAttempt,
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            generated,
            vec![("x-client-request-id".into(), request_id.clone())]
        );
        assert_eq!(request_id.len(), 36);
        assert_eq!(&request_id[14..15], "4");
        assert!(matches!(&request_id[19..20], "8" | "9" | "a" | "b"));
    }

    #[test]
    fn aws_defined_empty_defers_uuid_to_fetch_stage() {
        let facts = NativeAnthropicRequestHeaderFacts {
            protocol: ProtocolFamily::AnthropicMessages,
            provider: NativeAnthropicProvider::AnthropicAws,
            first_party_base_url: FirstPartyBaseUrlFact::Custom,
            selected_base_url: FirstPartyBaseUrlFact::Custom,
            anthropic_aws_base_url: AnthropicAwsBaseUrlFact::DefinedEmpty,
        };
        let mut headers = Vec::new();
        assert_eq!(
            ensure_native_client_request_id(&mut headers, facts, NativeRequestIdStage::PerAttempt,)
                .unwrap(),
            None
        );
        let id = ensure_native_client_request_id(
            &mut headers,
            facts,
            NativeRequestIdStage::FetchFallback,
        )
        .unwrap();
        assert!(id.is_some());
        assert_eq!(headers.len(), 1);
    }

    #[test]
    fn native_traceparent_gate_is_separate_from_request_id_fetch_fallback() {
        let fetch_only = NativeAnthropicRequestHeaderPlan {
            per_attempt_client_request_id: false,
            fetch_fallback_client_request_id: true,
        };
        assert!(!should_propagate_native_traceparent(
            true, fetch_only, false
        ));
        assert!(should_propagate_native_traceparent(true, fetch_only, true));
        assert!(!should_propagate_native_traceparent(
            false, fetch_only, true
        ));
    }

    #[test]
    fn native_traceparent_requires_context_and_preserves_caller_header() {
        let facts = NativeAnthropicRequestHeaderFacts {
            protocol: ProtocolFamily::AnthropicMessages,
            provider: NativeAnthropicProvider::Other,
            first_party_base_url: FirstPartyBaseUrlFact::Custom,
            selected_base_url: FirstPartyBaseUrlFact::Custom,
            anthropic_aws_base_url: AnthropicAwsBaseUrlFact::DefinedNonEmpty,
        };
        let mut headers = Vec::new();
        assert!(!ensure_native_traceparent(
            &mut headers,
            "00-trace",
            facts,
            false,
        ));
        assert!(ensure_native_traceparent(
            &mut headers,
            "00-trace",
            facts,
            true,
        ));
        assert_eq!(headers, vec![("traceparent".into(), "00-trace".into())]);
        assert!(!ensure_native_traceparent(
            &mut headers,
            "00-replacement",
            facts,
            true,
        ));
        assert_eq!(headers[0].1, "00-trace");
    }
}
