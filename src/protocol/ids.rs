//! Opaque identifiers for providers, responses, and tool calls.

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

string_id! {
    /// Pairs a `ContentBlock::ToolUse` with its `ContentBlock::ToolResult`.
    ///
    /// One id space: the provider's own string (`toolu_01…`, `call_…`) when the
    /// call came from a provider, a locally generated one otherwise.
    ToolUseId
}
string_id! {
    /// A provider-minted response identifier, kept opaque for continuation.
    ResponseId
}
string_id! {
    /// Open string: any `ProviderProfile` may declare a new one.
    ProviderId
}
