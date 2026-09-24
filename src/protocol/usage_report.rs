//! Provider measurements and their validity, independent of display totals.
use super::Usage;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UsageState {
    #[default]
    Missing,
    Partial,
    Complete,
    Invalid,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageReport {
    pub usage: Option<Usage>,
    pub state: UsageState,
}
impl UsageReport {
    pub fn measured(usage: Usage, state: UsageState) -> Self {
        Self {
            usage: Some(usage),
            state,
        }
    }
    pub fn complete(&self) -> Option<&Usage> {
        (self.state == UsageState::Complete)
            .then_some(self.usage.as_ref())
            .flatten()
    }
}
impl From<Usage> for UsageReport {
    fn from(usage: Usage) -> Self {
        Self::measured(usage, UsageState::Partial)
    }
}
