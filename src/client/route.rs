//! The result of `LlmClient::resolve(model)`: which connection to start on,
//! which wire model to send, and which sibling connections may take over.
//!
//! Ported from the previous project's `llm-client/src/registry.rs`. The shape
//! that matters is `connection_chain`: a hop is already resolved to its profile
//! and wire model id, so nothing has to be looked up again at failover time.

use crate::protocol::{FailoverTriggers, ModelCapabilitySupport, ProviderId};
use serde::{Deserialize, Serialize};

/// One sibling connection to fall over to, already resolved.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConnectionHop {
    /// Profile name of the connection to try.
    pub profile_name: String,
    /// Provider-local model value to send on that connection.
    pub request_model: String,
}

/// What a finished request is billed under. Kept separate from the wire model
/// because a provider may serve several wire ids under one billing id.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PricingModelRef {
    pub pricing_provider_id: ProviderId,
    pub billing_model: String,
    pub request_model: String,
    pub display_model: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResolvedRoute {
    pub provider_id: ProviderId,
    pub profile_name: String,
    /// The model id as the provider wants it on the wire.
    pub request_model: String,
    pub display_model: String,
    pub pricing_model: PricingModelRef,
    pub capability_support: ModelCapabilitySupport,
    /// Sibling connections of the head's group, in order. Empty when the
    /// profile stands alone. The head itself is not in this list.
    pub connection_chain: Vec<ConnectionHop>,
    pub failover: FailoverTriggers,
}
