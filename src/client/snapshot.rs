//! One immutable routing, listing and pricing view, installed only after commit.
use crate::protocol::{ProviderProfile, Region};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) struct RuntimeSnapshot {
    pub region: Region,
    pub profiles: Vec<ProviderProfile>,
    by_name: BTreeMap<String, usize>,
    pub model_index: BTreeMap<String, Vec<(usize, usize)>>,
    pub groups: BTreeMap<String, Vec<usize>>,
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
        let mut by_name = BTreeMap::new();
        let mut model_index: BTreeMap<String, Vec<(usize, usize)>> = BTreeMap::new();
        let mut groups: BTreeMap<String, Vec<usize>> = BTreeMap::new();
        for (pi, p) in profiles.iter().enumerate() {
            by_name.insert(p.profile_name.clone(), pi);
            groups.entry(p.group().into()).or_default().push(pi);
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
            model_index,
            groups,
            tracked,
        }
    }
    pub fn profile(&self, name: &str) -> Option<&ProviderProfile> {
        self.by_name.get(name).map(|i| &self.profiles[*i])
    }
    pub fn tracks(&self, profile: &ProviderProfile, model: &crate::protocol::ModelProfile) -> bool {
        self.tracked.as_ref().is_none_or(|tracked| {
            tracked
                .get(profile.provider_id.as_str())
                .is_some_and(|ids| ids.contains(&model.request_model))
        })
    }
}
