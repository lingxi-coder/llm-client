//! Explicit price selection. Unknown rates never imply free usage.
use super::{ServiceTier, Submission};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TokenRates {
    pub input_per_million: Option<f64>,
    pub output_per_million: Option<f64>,
    pub cache_read_per_million: Option<f64>,
    pub cache_write_per_million: Option<f64>,
    pub cache_write_1h_per_million: Option<f64>,
    /// Absent means reasoning is included in output at the output rate.
    pub reasoning_per_million: Option<f64>,
    /// Price in the stated currency for one provider-hosted web search.
    /// Absent means unknown, not free.
    pub web_search_per_request: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PricingContext {
    pub service_tier: Option<ServiceTier>,
    pub submission: Submission,
    /// Full prompt length, including cached input, for context price bands.
    pub input_tokens: Option<u64>,
    pub unix_seconds: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PriceRule {
    pub multiplier: Option<PriceMultiplier>,
    pub service_tier: ServiceTier,
    pub submission: Submission,
    pub min_input_tokens: Option<u64>,
    pub max_input_tokens: Option<u64>,
    /// Inclusive start in the official billing time zone.
    pub valid_from: Option<PriceBoundary>,
    /// Exclusive end in the official billing time zone.
    pub valid_until: Option<PriceBoundary>,
    pub rates: TokenRates,
    pub source: Option<String>,
    pub verified_at: Option<String>,
    /// A provider-documented modifier; never applied to other rules implicitly.
    pub apply_peak_schedule: bool,
}

/// A published local date/time and its billing time zone. Never uses the machine's zone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PriceBoundary {
    /// ISO local date/time (YYYY-MM-DDTHH:MM:SS), or RFC 3339 with explicit offset.
    pub local: String,
    /// IANA zone, such as Asia/Shanghai or America/Los_Angeles. None means unpublished.
    pub time_zone: Option<String>,
}

impl PriceBoundary {
    /// Earliest/latest possible UTC Unix second. Equal when the zone or offset is known.
    /// An unpublished zone covers UTC offsets -12 through +14 hours; queries inside
    /// that transition interval cannot safely select either price.
    pub fn utc_bounds(&self) -> Result<(u64, u64), String> {
        use jiff::civil::DateTime;
        use jiff::tz::TimeZone;
        let zone = self
            .time_zone
            .as_deref()
            .map(|name| {
                TimeZone::get(name).map_err(|_| format!("unknown billing time zone {name:?}"))
            })
            .transpose()?;
        let seconds = |timestamp: i64| {
            u64::try_from(timestamp).map_err(|_| "price boundary precedes Unix epoch".to_owned())
        };
        if let (Some(instant), Some(written)) = (
            crate::rfc3339::timestamp(&self.local),
            crate::rfc3339::pieces(&self.local),
        ) {
            let written = written
                .date()
                .to_datetime(written.time().unwrap_or_default());
            if zone
                .as_ref()
                .is_some_and(|zone| zone.to_datetime(instant) != written)
            {
                return Err("price boundary offset disagrees with its IANA time zone".into());
            }
            let timestamp = seconds(instant.as_second())?;
            return Ok((timestamp, timestamp));
        }
        let local = DateTime::strptime("%Y-%m-%dT%H:%M:%S", &self.local).map_err(|_| {
            "price boundary requires an ISO local datetime or explicit RFC 3339 offset".to_owned()
        })?;
        if let Some(zone) = zone {
            let instant = zone
                .to_ambiguous_timestamp(local)
                .unambiguous()
                .map_err(|_| {
                    "ambiguous or nonexistent local price boundary; supply an explicit valid offset"
                        .to_owned()
                })?;
            let timestamp = seconds(instant.as_second())?;
            Ok((timestamp, timestamp))
        } else {
            let nominal = TimeZone::UTC
                .to_timestamp(local)
                .map_err(|_| "price boundary is out of range".to_owned())?
                .as_second();
            Ok((seconds(nominal - 14 * 3600)?, seconds(nominal + 12 * 3600)?))
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QuotaPrice {
    pub service_tier: ServiceTier,
    pub multiplier: f64,
    pub unit: String,
    pub source: String,
    pub verified_at: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriceStatus {
    Priced,
    Unknown,
    Unsupported,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriceQuote {
    /// Selected rule, or an unresolved candidate when the time zone is unknown.
    pub rule: Option<PriceRule>,
    pub multiplier: Option<PriceMultiplier>,
    pub status: PriceStatus,
    pub context: PricingContext,
    pub rates: Option<TokenRates>,
    pub quota: Option<QuotaPrice>,
    pub currency: String,
    pub unit: String,
    pub source: Option<String>,
    pub verified_at: Option<String>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PriceBucket {
    Input,
    Output,
    CacheRead,
    CacheWrite,
    #[serde(rename = "cache_write_1h")]
    CacheWrite1h,
    Reasoning,
    WebSearchRequest,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PriceMultiplier {
    pub factor: f64,
    pub buckets: Vec<PriceBucket>,
}

#[cfg(test)]
mod time_tests {
    use super::PriceBoundary;
    fn boundary(local: &str, zone: Option<&str>) -> PriceBoundary {
        PriceBoundary {
            local: local.into(),
            time_zone: zone.map(str::to_owned),
        }
    }
    #[test]
    fn named_zones_normalize_to_the_same_utc_instant() {
        let utc = boundary("2027-01-01T00:00:00Z", None).utc_bounds().unwrap();
        assert_eq!(
            boundary("2027-01-01T08:00:00", Some("Asia/Shanghai"))
                .utc_bounds()
                .unwrap(),
            utc
        );
        assert_eq!(
            boundary("2026-12-31T16:00:00", Some("America/Los_Angeles"))
                .utc_bounds()
                .unwrap(),
            utc
        );
        assert_eq!(
            boundary("2027-01-01T00:00:00", Some("UTC"))
                .utc_bounds()
                .unwrap(),
            utc
        );
    }
    #[test]
    fn summer_and_winter_offsets_follow_iana_dst_rules() {
        for (local, expected) in [
            ("2026-07-01T00:00:00", "2026-07-01T07:00:00Z"),
            ("2026-12-01T00:00:00", "2026-12-01T08:00:00Z"),
        ] {
            assert_eq!(
                boundary(local, Some("America/Los_Angeles"))
                    .utc_bounds()
                    .unwrap(),
                boundary(expected, None).utc_bounds().unwrap()
            );
        }
    }
    #[test]
    fn dst_boundaries_require_valid_unambiguous_local_times() {
        for local in [
            "2026-03-08T02:30:00",
            "2026-11-01T01:30:00",
            "2026-07-01T00:00:00-08:00",
        ] {
            assert!(boundary(local, Some("America/Los_Angeles"))
                .utc_bounds()
                .is_err());
        }
        assert!(
            boundary("2026-11-01T01:30:00-07:00", Some("America/Los_Angeles"))
                .utc_bounds()
                .is_ok()
        );
        assert!(
            boundary("2026-11-01T01:30:00-08:00", Some("America/Los_Angeles"))
                .utc_bounds()
                .is_ok()
        );
        assert!(boundary("2026-12-01T00:00:00", Some("invalid/time-zone"))
            .utc_bounds()
            .is_err());
    }
    #[test]
    fn unpublished_zones_are_uncertain_instead_of_assumed_utc() {
        let (from, to) = boundary("2027-01-01T00:00:00", None).utc_bounds().unwrap();
        let utc = boundary("2027-01-01T00:00:00Z", None)
            .utc_bounds()
            .unwrap()
            .0;
        assert_eq!(from, utc - 14 * 3600);
        assert_eq!(to, utc + 12 * 3600);
    }
}
