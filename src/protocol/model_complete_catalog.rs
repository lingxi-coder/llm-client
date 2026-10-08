//! Current Native 2.1.293 model-catalog facts used by `model.complete`.
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

/// Native's `CATALOG_ID_TO_KEY` entries for the current static catalog.
const CATALOG_ID_TO_KEY: &[(&str, &str)] = &[
    ("claude-3-5-haiku", "haiku35"),
    ("claude-haiku-4-5", "haiku45"),
    ("claude-haiku-5-5", "haiku55"),
    ("claude-3-5-sonnet", "sonnet35"),
    ("claude-3-7-sonnet", "sonnet37"),
    ("claude-sonnet-4-0", "sonnet40"),
    ("claude-sonnet-4-5", "sonnet45"),
    ("claude-sonnet-4-6", "sonnet46"),
    ("claude-sonnet-5", "sonnet5"),
    ("claude-sonnet-5-5", "sonnet55"),
    ("claude-opus-4-0", "opus40"),
    ("claude-opus-4-1", "opus41"),
    ("claude-opus-4-5", "opus45"),
    ("claude-opus-4-6", "opus46"),
    ("claude-opus-4-7", "opus47"),
    ("claude-opus-4-8", "opus48"),
    ("claude-opus-5", "opus5"),
    ("claude-opus-5-5", "opus55"),
    ("claude-fable-5", "fable5"),
    ("claude-fable-5-1", "fable51"),
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
    provider_model_id(catalog_id, provider)
}

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
    CATALOG_ID_TO_KEY
        .iter()
        .find_map(|(id, key)| (*id == catalog_id).then_some(*key))
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
