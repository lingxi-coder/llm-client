//! One immutable routing, listing and pricing view, installed only after commit.
use super::attachments::{first_party_endpoint, FirstPartyEndpoint};
use crate::protocol::{ProviderProfile, Region};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) struct RuntimeSnapshot {
    pub region: Region,
    pub profiles: Vec<ProviderProfile>,
    by_name: BTreeMap<String, usize>,
    first_party_endpoints: Vec<FirstPartyEndpoint>,
    /// Each name's rows are grouped by profile, in their original catalog order.
    pub model_index: BTreeMap<String, Vec<(usize, usize)>>,
    tracked: Option<BTreeMap<String, BTreeSet<String>>>,
}
impl RuntimeSnapshot {
    pub fn new(
        region: Region,
        mut profiles: Vec<ProviderProfile>,
        tracked: Option<BTreeMap<String, BTreeSet<String>>>,
    ) -> Self {
        for profile in &mut profiles {
            profile.info.features = profile.inference_features();
            for model in &mut profile.models {
                model.info.pricing = model.pricing.clone();
                model.info.features = model.info.features.on_connection(&profile.info.features);
            }
        }
        let first_party_endpoints = profiles.iter().map(first_party_endpoint).collect();
        let mut by_name = BTreeMap::new();
        let mut model_index: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
        for (pi, p) in profiles.iter().enumerate() {
            by_name.insert(p.profile_name.clone(), pi);
            if !p.chat_enabled {
                continue;
            }
            for (mi, m) in p.models.iter().enumerate() {
                let mut names: BTreeSet<_> = m.aliases.iter().map(String::as_str).collect();
                names.extend([m.request_model.as_str(), m.display_model.as_str()]);
                for name in names {
                    model_index.entry(name.into()).or_default().push((pi, mi));
                }
            }
        }
        Self {
            region,
            profiles,
            by_name,
            first_party_endpoints,
            model_index,
            tracked,
        }
    }
    pub fn profile(&self, name: &str) -> Option<&ProviderProfile> {
        self.by_name.get(name).map(|i| &self.profiles[*i])
    }
    pub(crate) fn first_party_endpoint(&self, name: &str) -> FirstPartyEndpoint {
        self.first_party_endpoints[*self
            .by_name
            .get(name)
            .expect("resolved profile belongs to its configuration snapshot")]
    }
    pub fn tracks(&self, profile: &ProviderProfile, model: &crate::protocol::ModelProfile) -> bool {
        self.tracked.as_ref().is_none_or(|tracked| {
            tracked
                .get(profile.provider_id.as_str())
                .is_some_and(|ids| ids.contains(&model.request_model))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn endpoint_classification_follows_each_configuration_snapshot() {
        let mut profile: ProviderProfile = serde_json::from_value(serde_json::json!({
            "provider_id": "openai",
            "profile_name": "openai",
            "base_url": "https://api.openai.com/v1",
            "protocol": "open_ai_chat",
            "auth": "none",
            "models": []
        }))
        .unwrap();
        let first = RuntimeSnapshot::new(Region::International, vec![profile.clone()], None);
        assert!(matches!(
            first.first_party_endpoint("openai"),
            FirstPartyEndpoint::OpenAi
        ));

        profile.base_url = "https://api.openai.com.example.test/v1".into();
        let second = RuntimeSnapshot::new(Region::International, vec![profile], None);
        assert!(matches!(
            second.first_party_endpoint("openai"),
            FirstPartyEndpoint::Other
        ));
        assert!(matches!(
            first.first_party_endpoint("openai"),
            FirstPartyEndpoint::OpenAi
        ));
    }
}
