//! Provenance of a registry entry (§7.0, borrowed from the harness named in §9,
//! dumpable registry). `--dump-registry` prints one of these per entry, so an
//! empty origin is a gate failure (gate 39).

use crate::protocol::ids::PluginId;
use crate::protocol::scope::{ManifestKind, Scope};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum OriginSource {
    /// A capability crate compiled into the binary.
    Crate {
        name: String,
    },
    Plugin {
        id: PluginId,
    },
    McpServer {
        name: String,
    },
    /// A settings file, with the JSON path of the entry inside it.
    Settings {
        file: PathBuf,
        path: String,
    },
    /// A `profiles/*.toml` entry.
    Profile {
        name: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Origin {
    pub scope: Scope,
    pub source: OriginSource,
    /// Registered by an `optional = […]` profile entry: failure degrades, not aborts.
    pub optional: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub manifest_kind: Option<ManifestKind>,
}

impl Origin {
    pub fn krate(name: &'static str) -> Self {
        Self {
            scope: Scope::Builtin,
            source: OriginSource::Crate {
                name: name.to_owned(),
            },
            optional: false,
            manifest_kind: None,
        }
    }

    pub fn profile(name: impl Into<String>, optional: bool) -> Self {
        Self {
            scope: Scope::Builtin,
            source: OriginSource::Profile { name: name.into() },
            optional,
            manifest_kind: None,
        }
    }

    pub fn plugin(id: PluginId, manifest_kind: ManifestKind) -> Self {
        Self {
            scope: Scope::Plugin(id.clone()),
            source: OriginSource::Plugin { id },
            optional: false,
            manifest_kind: Some(manifest_kind),
        }
    }

    pub fn mcp_server(name: impl Into<String>, scope: Scope) -> Self {
        Self {
            scope,
            source: OriginSource::McpServer { name: name.into() },
            optional: false,
            manifest_kind: None,
        }
    }

    pub fn settings(file: PathBuf, path: impl Into<String>, scope: Scope) -> Self {
        Self {
            scope,
            source: OriginSource::Settings {
                file,
                path: path.into(),
            },
            optional: false,
            manifest_kind: None,
        }
    }
}
