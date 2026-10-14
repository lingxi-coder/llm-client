//! Current Native 2.1.295 model-catalog facts used by model completion and validation.
//!
//! The alias table maps family defaults to catalog IDs; those IDs are then
//! resolved through the catalog's provider-specific `provider_ids`. Alias
//! keys such as `sonnet55` are never selectors or wire model IDs.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeModelProvider {
    FirstParty,
    Bedrock,
    Vertex,
    Foundry,
    AnthropicAws,
    AnthropicGoogleCloud,
    Mantle,
    Gateway,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeModelFamily {
    Haiku,
    Sonnet,
    Opus,
    Fable,
    Mythos,
}

#[derive(Debug, Clone, Copy)]
struct CatalogEntry {
    id: &'static str,
    fallback_3p: Option<&'static str>,
    key: Option<&'static str>,
    family: NativeModelFamily,
    first_party: &'static str,
    bedrock: Option<&'static str>,
    vertex: Option<&'static str>,
    foundry: Option<&'static str>,
    anthropic_aws: Option<&'static str>,
    anthropic_google_cloud: Option<&'static str>,
    mantle: Option<&'static str>,
    gateway: Option<&'static str>,
}

const CATALOG: &[CatalogEntry] = &[
    CatalogEntry {
        id: "claude-3-5-haiku",
        fallback_3p: None,
        key: Some("haiku35"),
        family: NativeModelFamily::Haiku,
        first_party: "claude-3-5-haiku-20241022",
        bedrock: Some("us.anthropic.claude-3-5-haiku-20241022-v1:0"),
        vertex: Some("claude-3-5-haiku@20241022"),
        foundry: Some("claude-3-5-haiku"),
        anthropic_aws: Some("claude-3-5-haiku-20241022"),
        anthropic_google_cloud: None,
        mantle: None,
        gateway: Some("claude-3-5-haiku-20241022"),
    },
    CatalogEntry {
        id: "claude-haiku-4-5",
        fallback_3p: None,
        key: Some("haiku45"),
        family: NativeModelFamily::Haiku,
        first_party: "claude-haiku-4-5-20251001",
        bedrock: Some("us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        vertex: Some("claude-haiku-4-5@20251001"),
        foundry: Some("claude-haiku-4-5"),
        anthropic_aws: Some("claude-haiku-4-5-20251001"),
        anthropic_google_cloud: Some("claude-haiku-4-5-20251001"),
        mantle: Some("anthropic.claude-haiku-4-5"),
        gateway: Some("claude-haiku-4-5-20251001"),
    },
    CatalogEntry {
        id: "claude-haiku-5-5",
        fallback_3p: Some("claude-haiku-4-5"),
        key: Some("haiku55"),
        family: NativeModelFamily::Haiku,
        first_party: "claude-haiku-5-5",
        bedrock: Some("us.anthropic.claude-haiku-5-5"),
        vertex: Some("claude-haiku-5-5"),
        foundry: Some("claude-haiku-5-5"),
        anthropic_aws: Some("claude-haiku-5-5"),
        anthropic_google_cloud: Some("claude-haiku-5-5"),
        mantle: Some("anthropic.claude-haiku-5-5"),
        gateway: None,
    },
    CatalogEntry {
        id: "claude-3-5-sonnet",
        fallback_3p: None,
        key: Some("sonnet35"),
        family: NativeModelFamily::Sonnet,
        first_party: "claude-3-5-sonnet-20241022",
        bedrock: Some("us.anthropic.claude-3-5-sonnet-20241022-v2:0"),
        vertex: Some("claude-3-5-sonnet-v2@20241022"),
        foundry: Some("claude-3-5-sonnet"),
        anthropic_aws: Some("claude-3-5-sonnet-20241022"),
        anthropic_google_cloud: None,
        mantle: None,
        gateway: Some("claude-3-5-sonnet-20241022"),
    },
    CatalogEntry {
        id: "claude-3-7-sonnet",
        fallback_3p: None,
        key: Some("sonnet37"),
        family: NativeModelFamily::Sonnet,
        first_party: "claude-3-7-sonnet-20250219",
        bedrock: Some("us.anthropic.claude-3-7-sonnet-20250219-v1:0"),
        vertex: Some("claude-3-7-sonnet@20250219"),
        foundry: Some("claude-3-7-sonnet"),
        anthropic_aws: Some("claude-3-7-sonnet-20250219"),
        anthropic_google_cloud: None,
        mantle: None,
        gateway: Some("claude-3-7-sonnet-20250219"),
    },
    CatalogEntry {
        id: "claude-sonnet-4-0",
        fallback_3p: None,
        key: Some("sonnet40"),
        family: NativeModelFamily::Sonnet,
        first_party: "claude-sonnet-4-20250514",
        bedrock: Some("us.anthropic.claude-sonnet-4-20250514-v1:0"),
        vertex: Some("claude-sonnet-4@20250514"),
        foundry: Some("claude-sonnet-4"),
        anthropic_aws: Some("claude-sonnet-4-20250514"),
        anthropic_google_cloud: None,
        mantle: None,
        gateway: Some("claude-sonnet-4-20250514"),
    },
    CatalogEntry {
        id: "claude-sonnet-4-5",
        fallback_3p: Some("claude-sonnet-4-0"),
        key: Some("sonnet45"),
        family: NativeModelFamily::Sonnet,
        first_party: "claude-sonnet-4-5-20250929",
        bedrock: Some("us.anthropic.claude-sonnet-4-5-20250929-v1:0"),
        vertex: Some("claude-sonnet-4-5@20250929"),
        foundry: Some("claude-sonnet-4-5"),
        anthropic_aws: Some("claude-sonnet-4-5-20250929"),
        anthropic_google_cloud: Some("claude-sonnet-4-5-20250929"),
        mantle: None,
        gateway: Some("claude-sonnet-4-5-20250929"),
    },
    CatalogEntry {
        id: "claude-sonnet-4-6",
        fallback_3p: Some("claude-sonnet-4-5"),
        key: Some("sonnet46"),
        family: NativeModelFamily::Sonnet,
        first_party: "claude-sonnet-4-6",
        bedrock: Some("us.anthropic.claude-sonnet-4-6"),
        vertex: Some("claude-sonnet-4-6"),
        foundry: Some("claude-sonnet-4-6"),
        anthropic_aws: Some("claude-sonnet-4-6"),
        anthropic_google_cloud: Some("claude-sonnet-4-6"),
        mantle: None,
        gateway: Some("claude-sonnet-4-6"),
    },
    CatalogEntry {
        id: "claude-sonnet-5",
        fallback_3p: Some("claude-sonnet-4-6"),
        key: Some("sonnet5"),
        family: NativeModelFamily::Sonnet,
        first_party: "claude-sonnet-5",
        bedrock: Some("us.anthropic.claude-sonnet-5"),
        vertex: Some("claude-sonnet-5"),
        foundry: Some("claude-sonnet-5"),
        anthropic_aws: Some("claude-sonnet-5"),
        anthropic_google_cloud: Some("claude-sonnet-5"),
        mantle: Some("anthropic.claude-sonnet-5"),
        gateway: Some("claude-sonnet-5"),
    },
    CatalogEntry {
        id: "claude-sonnet-5-5",
        fallback_3p: Some("claude-sonnet-5"),
        key: Some("sonnet55"),
        family: NativeModelFamily::Sonnet,
        first_party: "claude-sonnet-5-5",
        bedrock: Some("us.anthropic.claude-sonnet-5-5"),
        vertex: Some("claude-sonnet-5-5"),
        foundry: Some("claude-sonnet-5-5"),
        anthropic_aws: Some("claude-sonnet-5-5"),
        anthropic_google_cloud: Some("claude-sonnet-5-5"),
        mantle: Some("anthropic.claude-sonnet-5-5"),
        gateway: Some("claude-sonnet-5-5"),
    },
    CatalogEntry {
        id: "claude-opus-4-0",
        fallback_3p: None,
        key: Some("opus40"),
        family: NativeModelFamily::Opus,
        first_party: "claude-opus-4-20250514",
        bedrock: Some("us.anthropic.claude-opus-4-20250514-v1:0"),
        vertex: Some("claude-opus-4@20250514"),
        foundry: Some("claude-opus-4"),
        anthropic_aws: Some("claude-opus-4-20250514"),
        anthropic_google_cloud: None,
        mantle: None,
        gateway: Some("claude-opus-4-20250514"),
    },
    CatalogEntry {
        id: "claude-opus-4-1",
        fallback_3p: None,
        key: Some("opus41"),
        family: NativeModelFamily::Opus,
        first_party: "claude-opus-4-1-20250805",
        bedrock: Some("us.anthropic.claude-opus-4-1-20250805-v1:0"),
        vertex: Some("claude-opus-4-1@20250805"),
        foundry: Some("claude-opus-4-1"),
        anthropic_aws: Some("claude-opus-4-1-20250805"),
        anthropic_google_cloud: None,
        mantle: None,
        gateway: Some("claude-opus-4-1-20250805"),
    },
    CatalogEntry {
        id: "claude-opus-4-5",
        fallback_3p: Some("claude-opus-4-1"),
        key: Some("opus45"),
        family: NativeModelFamily::Opus,
        first_party: "claude-opus-4-5-20251101",
        bedrock: Some("us.anthropic.claude-opus-4-5-20251101-v1:0"),
        vertex: Some("claude-opus-4-5@20251101"),
        foundry: Some("claude-opus-4-5"),
        anthropic_aws: Some("claude-opus-4-5-20251101"),
        anthropic_google_cloud: Some("claude-opus-4-5-20251101"),
        mantle: None,
        gateway: Some("claude-opus-4-5-20251101"),
    },
    CatalogEntry {
        id: "claude-opus-4-6",
        fallback_3p: Some("claude-opus-4-5"),
        key: Some("opus46"),
        family: NativeModelFamily::Opus,
        first_party: "claude-opus-4-6",
        bedrock: Some("us.anthropic.claude-opus-4-6-v1"),
        vertex: Some("claude-opus-4-6"),
        foundry: Some("claude-opus-4-6"),
        anthropic_aws: Some("claude-opus-4-6"),
        anthropic_google_cloud: Some("claude-opus-4-6"),
        mantle: None,
        gateway: Some("claude-opus-4-6"),
    },
    CatalogEntry {
        id: "claude-opus-4-7",
        fallback_3p: Some("claude-opus-4-6"),
        key: Some("opus47"),
        family: NativeModelFamily::Opus,
        first_party: "claude-opus-4-7",
        bedrock: Some("us.anthropic.claude-opus-4-7"),
        vertex: Some("claude-opus-4-7"),
        foundry: Some("claude-opus-4-7"),
        anthropic_aws: Some("claude-opus-4-7"),
        anthropic_google_cloud: Some("claude-opus-4-7"),
        mantle: Some("anthropic.claude-opus-4-7"),
        gateway: Some("claude-opus-4-7"),
    },
    CatalogEntry {
        id: "claude-opus-4-8",
        fallback_3p: Some("claude-opus-4-7"),
        key: Some("opus48"),
        family: NativeModelFamily::Opus,
        first_party: "claude-opus-4-8",
        bedrock: Some("us.anthropic.claude-opus-4-8"),
        vertex: Some("claude-opus-4-8"),
        foundry: Some("claude-opus-4-8"),
        anthropic_aws: Some("claude-opus-4-8"),
        anthropic_google_cloud: Some("claude-opus-4-8"),
        mantle: Some("anthropic.claude-opus-4-8"),
        gateway: Some("claude-opus-4-8"),
    },
    CatalogEntry {
        id: "claude-opus-5",
        fallback_3p: Some("claude-opus-4-8"),
        key: Some("opus5"),
        family: NativeModelFamily::Opus,
        first_party: "claude-opus-5",
        bedrock: Some("us.anthropic.claude-opus-5"),
        vertex: Some("claude-opus-5"),
        foundry: Some("claude-opus-5"),
        anthropic_aws: Some("claude-opus-5"),
        anthropic_google_cloud: Some("claude-opus-5"),
        mantle: Some("anthropic.claude-opus-5"),
        gateway: Some("claude-opus-5"),
    },
    CatalogEntry {
        id: "claude-opus-5-5",
        fallback_3p: Some("claude-opus-5"),
        key: Some("opus55"),
        family: NativeModelFamily::Opus,
        first_party: "claude-opus-5-5",
        bedrock: Some("us.anthropic.claude-opus-5-5"),
        vertex: Some("claude-opus-5-5"),
        foundry: Some("claude-opus-5-5"),
        anthropic_aws: Some("claude-opus-5-5"),
        anthropic_google_cloud: Some("claude-opus-5-5"),
        mantle: Some("anthropic.claude-opus-5-5"),
        gateway: Some("claude-opus-5-5"),
    },
    CatalogEntry {
        id: "claude-fable-5",
        fallback_3p: Some("claude-opus-5-5"),
        key: Some("fable5"),
        family: NativeModelFamily::Fable,
        first_party: "claude-fable-5",
        bedrock: Some("us.anthropic.claude-fable-5"),
        vertex: Some("claude-fable-5"),
        foundry: Some("claude-fable-5"),
        anthropic_aws: Some("claude-fable-5"),
        anthropic_google_cloud: Some("claude-fable-5"),
        mantle: Some("anthropic.claude-fable-5"),
        gateway: Some("claude-fable-5"),
    },
    CatalogEntry {
        id: "claude-fable-5-1",
        fallback_3p: Some("claude-fable-5"),
        key: Some("fable51"),
        family: NativeModelFamily::Fable,
        first_party: "claude-fable-5-1",
        bedrock: Some("us.anthropic.claude-fable-5-1"),
        vertex: Some("claude-fable-5-1"),
        foundry: Some("claude-fable-5-1"),
        anthropic_aws: Some("claude-fable-5-1"),
        anthropic_google_cloud: Some("claude-fable-5-1"),
        mantle: Some("anthropic.claude-fable-5-1"),
        gateway: Some("claude-fable-5-1"),
    },
    CatalogEntry {
        id: "claude-mythos-5",
        fallback_3p: None,
        key: None,
        family: NativeModelFamily::Mythos,
        first_party: "claude-mythos-5",
        bedrock: None,
        vertex: None,
        foundry: None,
        anthropic_aws: None,
        anthropic_google_cloud: None,
        mantle: None,
        gateway: None,
    },
    CatalogEntry {
        id: "claude-mythos-5-1",
        fallback_3p: None,
        key: None,
        family: NativeModelFamily::Mythos,
        first_party: "claude-mythos-5-1",
        bedrock: None,
        vertex: None,
        foundry: None,
        anthropic_aws: None,
        anthropic_google_cloud: None,
        mantle: None,
        gateway: None,
    },
];

/// Resolve one Native alias-table default to the catalog ID before selecting
/// that model's provider-specific ID.
pub(crate) fn family_default_catalog_id(
    family: NativeModelFamily,
    provider: NativeModelProvider,
) -> &'static str {
    match family {
        NativeModelFamily::Haiku => match provider {
            NativeModelProvider::FirstParty => "claude-haiku-5-5",
            NativeModelProvider::Bedrock
            | NativeModelProvider::Vertex
            | NativeModelProvider::Foundry
            | NativeModelProvider::AnthropicAws
            | NativeModelProvider::AnthropicGoogleCloud
            | NativeModelProvider::Mantle
            | NativeModelProvider::Gateway => "claude-haiku-4-5",
        },
        NativeModelFamily::Sonnet => match provider {
            NativeModelProvider::Bedrock
            | NativeModelProvider::Vertex
            | NativeModelProvider::Foundry
            | NativeModelProvider::Mantle => "claude-sonnet-4-5",
            NativeModelProvider::AnthropicAws | NativeModelProvider::Gateway => "claude-sonnet-4-6",
            NativeModelProvider::FirstParty | NativeModelProvider::AnthropicGoogleCloud => {
                "claude-sonnet-5-5"
            }
        },
        NativeModelFamily::Opus => match provider {
            NativeModelProvider::Foundry => "claude-opus-4-6",
            NativeModelProvider::Gateway => "claude-opus-4-7",
            NativeModelProvider::FirstParty
            | NativeModelProvider::Bedrock
            | NativeModelProvider::Vertex
            | NativeModelProvider::AnthropicAws
            | NativeModelProvider::AnthropicGoogleCloud
            | NativeModelProvider::Mantle => "claude-opus-5-5",
        },
        NativeModelFamily::Fable => match provider {
            NativeModelProvider::Gateway => "claude-fable-5",
            NativeModelProvider::FirstParty
            | NativeModelProvider::Bedrock
            | NativeModelProvider::Vertex
            | NativeModelProvider::Foundry
            | NativeModelProvider::AnthropicAws
            | NativeModelProvider::AnthropicGoogleCloud
            | NativeModelProvider::Mantle => "claude-fable-5-1",
        },
        NativeModelFamily::Mythos => "claude-mythos-5",
    }
}

/// Resolve a static catalog ID through its exact provider-ID field. Native
/// uses the first-party ID for its gateway when a model row has no gateway ID.
pub(crate) fn provider_model_id(
    catalog_id: &str,
    provider: NativeModelProvider,
) -> Option<&'static str> {
    let entry = CATALOG.iter().find(|entry| entry.id == catalog_id)?;
    provider_id_for_entry(entry, provider)
}

fn provider_id_for_entry(
    entry: &CatalogEntry,
    provider: NativeModelProvider,
) -> Option<&'static str> {
    match provider {
        NativeModelProvider::FirstParty => Some(entry.first_party),
        NativeModelProvider::Bedrock => entry.bedrock,
        NativeModelProvider::Vertex => entry.vertex,
        NativeModelProvider::Foundry => entry.foundry,
        NativeModelProvider::AnthropicAws => entry.anthropic_aws,
        NativeModelProvider::AnthropicGoogleCloud => entry.anthropic_google_cloud,
        NativeModelProvider::Mantle => entry.mantle,
        NativeModelProvider::Gateway => Some(entry.gateway.unwrap_or(entry.first_party)),
    }
}

pub(crate) fn static_family_model_id(
    family: NativeModelFamily,
    provider: NativeModelProvider,
) -> Option<&'static str> {
    let catalog_id = family_default_catalog_id(family, provider);
    let entry = CATALOG
        .iter()
        .find(|entry| entry.id == catalog_id && entry.family == family)?;
    provider_id_for_entry(entry, provider)
}

#[cfg(test)]
pub(crate) fn static_family_key(family: NativeModelFamily) -> &'static str {
    match family {
        NativeModelFamily::Haiku => "haiku55",
        NativeModelFamily::Sonnet => "sonnet55",
        NativeModelFamily::Opus => "opus55",
        NativeModelFamily::Fable => "fable51",
        NativeModelFamily::Mythos => "mythos5",
    }
}

pub(crate) fn native_model_key(catalog_id: &str) -> Option<&'static str> {
    CATALOG
        .iter()
        .find(|entry| entry.id == catalog_id)
        .and_then(|entry| entry.key)
}

pub(crate) fn canonical_model_id_for_provider_id(provider_id: &str) -> Option<&'static str> {
    CATALOG.iter().find_map(|entry| {
        let ids = [
            Some(entry.first_party),
            entry.bedrock,
            entry.vertex,
            entry.foundry,
            entry.anthropic_aws,
            entry.anthropic_google_cloud,
            entry.mantle,
            entry.gateway,
        ];
        ids.into_iter()
            .flatten()
            .any(|id| id.eq_ignore_ascii_case(provider_id))
            .then_some(entry.id)
    })
}

/// A source-selected fallback row from Native `ikn`'s sorted catalog scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NativeModelSuggestionTarget {
    pub(crate) trigger_catalog_id: &'static str,
    pub(crate) fallback_catalog_id: &'static str,
    pub(crate) trigger_family: NativeModelFamily,
    pub(crate) fallback_family: NativeModelFamily,
}

/// Return the first `fallback_3p` trigger matching Native `ikn`'s lowercased
/// substring check. Native sorts candidates by `BAt`, family, then `Fto` before
/// testing `needle` and `needleUnderscore`; input is not trimmed.
pub(crate) fn native_model_suggestion_target(
    input_units: &[u16],
) -> Option<NativeModelSuggestionTarget> {
    let lowered = lowercase_utf16(input_units);
    let mut candidates: Vec<&CatalogEntry> = CATALOG
        .iter()
        .filter(|entry| entry.fallback_3p.is_some())
        .collect();
    candidates.sort_by(|left, right| {
        family_rank(left.family)
            .cmp(&family_rank(right.family))
            .then_with(|| family_name(left.family).cmp(family_name(right.family)))
            .then_with(|| compare_native_version_desc(left, right))
    });
    candidates.into_iter().find_map(|entry| {
        let needle = entry.id.strip_prefix("claude-").unwrap_or(entry.id);
        let underscore = needle.replace('-', "_");
        if !contains_ascii_units(&lowered, needle.as_bytes())
            && !contains_ascii_units(&lowered, underscore.as_bytes())
        {
            return None;
        }
        let fallback_catalog_id = entry.fallback_3p?;
        let fallback = CATALOG
            .iter()
            .find(|candidate| candidate.id == fallback_catalog_id)?;
        Some(NativeModelSuggestionTarget {
            trigger_catalog_id: entry.id,
            fallback_catalog_id,
            trigger_family: entry.family,
            fallback_family: fallback.family,
        })
    })
}

/// Return Native `go(provider)`'s static per-catalog-ID provider defaults.
/// Missing provider IDs use the first non-null row in Native `Yl`/`Ho` order;
/// if a provider has no configured row, each key uses that model's first-party
/// ID. This is only the static seed: the host resolves model overrides and
/// dynamic provider values before passing the effective map to the facade.
pub(crate) fn native_static_provider_defaults(
    provider: NativeModelProvider,
) -> Vec<(&'static str, &'static str)> {
    let config_rows = CATALOG.iter().filter(|entry| entry.key.is_some());
    let first_provider_id = config_rows
        .clone()
        .find_map(|entry| provider_id_for_entry(entry, provider));
    config_rows
        .map(|entry| {
            (
                entry.id,
                provider_id_for_entry(entry, provider)
                    .or(first_provider_id)
                    .unwrap_or(entry.first_party),
            )
        })
        .collect()
}

fn family_rank(family: NativeModelFamily) -> u8 {
    match family {
        NativeModelFamily::Fable => 0,
        NativeModelFamily::Opus => 1,
        NativeModelFamily::Sonnet => 2,
        NativeModelFamily::Haiku => 3,
        NativeModelFamily::Mythos => 4,
    }
}

fn family_name(family: NativeModelFamily) -> &'static str {
    match family {
        NativeModelFamily::Haiku => "haiku",
        NativeModelFamily::Sonnet => "sonnet",
        NativeModelFamily::Opus => "opus",
        NativeModelFamily::Fable => "fable",
        NativeModelFamily::Mythos => "mythos",
    }
}

fn compare_native_version_desc(left: &CatalogEntry, right: &CatalogEntry) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    let left = version_segments(left);
    let right = version_segments(right);
    for index in 0..left.len().max(right.len()) {
        let left = left.get(index).copied().unwrap_or(-1);
        let right = right.get(index).copied().unwrap_or(-1);
        match right.cmp(&left) {
            Ordering::Equal => {}
            ordering => return ordering,
        }
    }
    Ordering::Equal
}

fn version_segments(entry: &CatalogEntry) -> Vec<i64> {
    let name = entry.id.strip_prefix("claude-").unwrap_or(entry.id);
    let family = family_name(entry.family);
    name.replacen(family, "", 1)
        .trim_start_matches('-')
        .split('-')
        .map(|segment| segment.parse::<i64>().unwrap_or(-1))
        .collect()
}

fn lowercase_utf16(units: &[u16]) -> Vec<u16> {
    let mut lowered = Vec::with_capacity(units.len());
    for decoded in char::decode_utf16(units.iter().copied()) {
        match decoded {
            Ok(character) => {
                for lowered_character in character.to_lowercase() {
                    let mut encoded = [0; 2];
                    lowered.extend_from_slice(lowered_character.encode_utf16(&mut encoded));
                }
            }
            Err(error) => lowered.push(error.unpaired_surrogate()),
        }
    }
    lowered
}

fn contains_ascii_units(haystack: &[u16], needle: &[u8]) -> bool {
    if needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|window| {
        window
            .iter()
            .zip(needle)
            .all(|(unit, expected)| *unit == u16::from(*expected))
    })
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_model_id_for_provider_id, family_default_catalog_id, native_model_key,
        provider_model_id, static_family_model_id, NativeModelFamily as Family,
        NativeModelProvider as Provider,
    };

    #[test]
    fn family_keys_resolve_through_catalog_ids_to_provider_ids() {
        assert_eq!(
            family_default_catalog_id(Family::Haiku, Provider::Bedrock),
            "claude-haiku-4-5"
        );
        assert_eq!(native_model_key("claude-haiku-4-5"), Some("haiku45"));
        assert_eq!(native_model_key("haiku45"), None);
        assert_eq!(
            static_family_model_id(Family::Haiku, Provider::Bedrock),
            Some("us.anthropic.claude-haiku-4-5-20251001-v1:0")
        );
    }

    #[test]
    fn haiku_55_is_catalogued_and_is_the_first_party_family_default() {
        assert_eq!(native_model_key("claude-haiku-5-5"), Some("haiku55"));
        assert_eq!(
            provider_model_id("claude-haiku-5-5", Provider::FirstParty),
            Some("claude-haiku-5-5")
        );
        assert_eq!(
            provider_model_id("claude-haiku-5-5", Provider::Bedrock),
            Some("us.anthropic.claude-haiku-5-5")
        );
        assert_eq!(
            provider_model_id("claude-haiku-5-5", Provider::Mantle),
            Some("anthropic.claude-haiku-5-5")
        );
        assert_eq!(
            family_default_catalog_id(Family::Haiku, Provider::FirstParty),
            "claude-haiku-5-5"
        );
        assert_eq!(super::static_family_key(Family::Haiku), "haiku55");
        assert_eq!(
            static_family_model_id(Family::Haiku, Provider::FirstParty),
            Some("claude-haiku-5-5")
        );
        for provider in [
            Provider::Bedrock,
            Provider::Vertex,
            Provider::Foundry,
            Provider::AnthropicAws,
            Provider::AnthropicGoogleCloud,
            Provider::Mantle,
            Provider::Gateway,
        ] {
            assert_eq!(
                family_default_catalog_id(Family::Haiku, provider),
                "claude-haiku-4-5"
            );
        }
    }

    #[test]
    fn provider_alias_defaults_are_the_current_native_alias_table() {
        assert_eq!(
            family_default_catalog_id(Family::Sonnet, Provider::FirstParty),
            "claude-sonnet-5-5"
        );
        assert_eq!(
            family_default_catalog_id(Family::Sonnet, Provider::Bedrock),
            "claude-sonnet-4-5"
        );
        assert_eq!(
            family_default_catalog_id(Family::Sonnet, Provider::AnthropicAws),
            "claude-sonnet-4-6"
        );
        assert_eq!(
            family_default_catalog_id(Family::Opus, Provider::Foundry),
            "claude-opus-4-6"
        );
        assert_eq!(
            family_default_catalog_id(Family::Opus, Provider::Gateway),
            "claude-opus-4-7"
        );
    }

    #[test]
    fn provider_ids_reverse_to_canonical_catalog_ids() {
        assert_eq!(
            canonical_model_id_for_provider_id("claude-haiku-4-5@20251001"),
            Some("claude-haiku-4-5")
        );
        assert_eq!(
            provider_model_id("claude-opus-4-7", Provider::Gateway),
            Some("claude-opus-4-7")
        );
    }
}
