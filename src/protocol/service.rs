//! Independent service settings and credential placement.
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", content = "value", rename_all = "snake_case")]
pub enum ServiceSetting<T> {
    #[default]
    Inherit,
    Enabled(T),
    Disabled,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ServiceAuth {
    None,
    Bearer,
    ApiKey { header: String },
}
