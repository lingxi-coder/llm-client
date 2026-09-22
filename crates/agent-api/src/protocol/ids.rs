//! Identifiers. String ids hold a provider's or a peer's own string verbatim;
//! `u64` ids are minted by the kernel, monotonic, never reused (§13).

use serde::{Deserialize, Serialize};
use std::fmt;

macro_rules! string_id {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub String);

        impl $name {
            pub fn new(id: impl Into<String>) -> Self {
                Self(id.into())
            }
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<&str> for $name {
            fn from(s: &str) -> Self {
                Self(s.to_owned())
            }
        }

        impl From<String> for $name {
            fn from(s: String) -> Self {
                Self(s)
            }
        }
    };
}

macro_rules! u64_id {
    ($(#[$m:meta])* $name:ident) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
        #[serde(transparent)]
        pub struct $name(pub u64);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}#{}", stringify!($name), self.0)
            }
        }
    };
}

string_id! {
    /// Pairs a `ContentBlock::ToolUse` with its `ContentBlock::ToolResult`.
    ///
    /// One id space: the provider's own string (`toolu_01…`, `call_…`) when the
    /// call came from a provider, a locally minted one otherwise. The previous
    /// project used a UUID newtype here and had to carry a second `provider_id`
    /// field on both sides and keep them in sync on every egress.
    ToolUseId
}
string_id! {
    /// A provider-minted response identifier, kept opaque for continuation.
    ResponseId
}
string_id! {
    /// A (sub)agent. `None` in an event means the main agent.
    AgentId
}
string_id! { SessionId }
string_id! { PluginId }
string_id! {
    /// Open string: any `ProviderProfile` may declare a new one (§7.2).
    ProviderId
}
string_id! {
    /// A work mode name (`chat`, `code`, `work`, `plan`, or user-defined).
    ModeName
}

u64_id! {
    /// One turn of the loop. Completion events carry the turn they answer;
    /// a mismatch is a late result and is dropped (§13 rule 1).
    TurnId
}
u64_id! {
    /// One issued effect. Every completion event names the effect it answers.
    EffectId
}
u64_id! {
    /// One tool batch inside the executor.
    BatchId
}
u64_id! {
    /// One permission question put to the frontend. At most one is in flight.
    AskId
}
u64_id! {
    /// One compaction.
    CompId
}
