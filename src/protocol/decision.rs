//! Finite, single-choice decisions over shared text or image context.

use super::{ImageSource, ModelListing, UsageReport};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// The first implementation uses a provider-enforced JSON Schema response.
/// A native Decisions wire can be added without changing the public request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DecisionImplementation {
    StructuredOutput,
}

/// An explicit opt-in on a provider connection. Model and region checks apply
/// independently; a compatible chat wire alone does not enable decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSupport {
    pub implementation: DecisionImplementation,
    pub text: bool,
    pub image: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DecisionContextPart {
    Text { text: String },
    Image { source: Box<ImageSource> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionOption {
    pub id: String,
    pub label: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionQuestion {
    pub id: String,
    pub prompt: String,
    pub options: Vec<DecisionOption>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionRequest {
    pub model: String,
    pub context: Vec<DecisionContextPart>,
    pub questions: Vec<DecisionQuestion>,
}

/// One earlier connection that reported usage before another was tried.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionAttemptReport {
    pub model: String,
    pub executed_profile: String,
    pub usage: UsageReport,
}

/// Retained on both success and completed-provider failures. `usage` describes
/// the final attempt; earlier reported usage stays separate in `prior_attempts`.
/// Missing usage must not be interpreted as zero cost.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionCallReport {
    pub model: String,
    pub executed_profile: Option<String>,
    pub implementation: DecisionImplementation,
    pub usage: UsageReport,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prior_attempts: Vec<DecisionAttemptReport>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionResult {
    /// One offered option ID for each original question ID.
    pub answers: BTreeMap<String, String>,
    pub report: DecisionCallReport,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionModelListing {
    pub model: ModelListing,
    pub text: bool,
    pub image: bool,
    pub implementation: DecisionImplementation,
}
