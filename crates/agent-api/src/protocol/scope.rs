//! The one `Scope` enum (gate 4). Settings layers, permission-rule sources and
//! registry entries all name their layer with this type.

use crate::protocol::ids::PluginId;
use serde::{Deserialize, Serialize};

/// Where something came from, by layer.
///
/// Precedence (higher wins on conflict; deny still wins across layers, gate 44):
/// `Managed > Session > Local > Project > User > Plugin > Builtin`.
/// §7.0 lists the order without `Session`; it is placed below `Managed` and
/// above `Local`, where the previous project's command-line layer sits. M1 may
/// move it — but by editing [`Scope::rank`], not by adding a second enum.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    /// Administrator-managed settings.
    Managed,
    /// `~/.agents/…`.
    User,
    /// `<project>/.agents/…` (shared, checked in).
    Project,
    /// `<project>/.agents/*.local.*` (not checked in).
    Local,
    /// This session only: mode rules, `/permissions` additions, CLI flags.
    Session,
    /// Materialized from a plugin.
    Plugin(PluginId),
    /// Compiled into the binary.
    Builtin,
}

impl Scope {
    /// Lower rank wins. See the type docs for the order.
    pub fn rank(&self) -> u8 {
        match self {
            Scope::Managed => 0,
            Scope::Session => 1,
            Scope::Local => 2,
            Scope::Project => 3,
            Scope::User => 4,
            Scope::Plugin(_) => 5,
            Scope::Builtin => 6,
        }
    }

    /// `true` when `self` overrides `other` on conflict.
    pub fn overrides(&self, other: &Scope) -> bool {
        self.rank() < other.rank()
    }
}

/// Which manifest a plugin was loaded from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManifestKind {
    /// `.agent-plugin/plugin.json` — this project's manifest.
    AgentPlugin,
    /// The ecosystem manifest read only behind the compat switch (§11, M3).
    CompatClaudePlugin,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_overrides_everything_and_builtin_nothing() {
        let all = [
            Scope::Managed,
            Scope::Session,
            Scope::Local,
            Scope::Project,
            Scope::User,
            Scope::Plugin(PluginId::new("p")),
            Scope::Builtin,
        ];
        for s in &all[1..] {
            assert!(Scope::Managed.overrides(s));
            assert!(!Scope::Builtin.overrides(s) || matches!(s, Scope::Builtin));
        }
        // Strictly ordered: no two distinct scopes share a rank.
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert!(a.overrides(b), "{a:?} should override {b:?}");
            }
        }
    }
}
