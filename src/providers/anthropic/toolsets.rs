//! Anthropic-defined client-side Browser and Computer toolsets.
//!
//! These declarations describe tools that the host executes. They are kept
//! separate from [`crate::protocol::HostedTool`], which describes provider-side
//! execution.

use super::types::{AnthropicMcpCacheControl, AnthropicToolCaller};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

macro_rules! client_tool_members {
    ($name:ident { $($variant:ident => $wire:literal $([default = $default:literal])?),+ $(,)? }) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
        pub enum $name {
            $(#[serde(rename = $wire)] $variant),+
        }

        impl $name {
            pub const ALL: &'static [Self] = &[$(Self::$variant),+];
            pub const NAMES: &'static [&'static str] = &[$($wire),+];

            pub const fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $wire),+
                }
            }

            pub const fn default_enabled(self) -> bool {
                match self {
                    $(Self::$variant => client_tool_members!(@default $($default)?)),+
                }
            }

            pub fn parse(name: &str) -> Option<Self> {
                match name {
                    $($wire => Some(Self::$variant),)+
                    _ => None,
                }
            }
        }
    };
    (@default $value:literal) => { $value };
    (@default) => { true };
}

client_tool_members!(AnthropicBrowserMember {
    CloseTab => "close_tab",
    DoubleClick => "double_click",
    FileUpload => "file_upload" [default = false],
    Find => "find",
    FormInput => "form_input",
    GetPageText => "get_page_text",
    HoldKey => "hold_key",
    Hover => "hover",
    JavascriptExec => "javascript_exec" [default = false],
    Key => "key",
    LeftClick => "left_click",
    LeftClickDrag => "left_click_drag",
    LeftMouseDown => "left_mouse_down",
    LeftMouseUp => "left_mouse_up",
    ListTabs => "list_tabs",
    MiddleClick => "middle_click",
    MouseMove => "mouse_move",
    Navigate => "navigate",
    NewTab => "new_tab",
    ReadConsole => "read_console" [default = false],
    ReadNetwork => "read_network" [default = false],
    ReadPage => "read_page",
    RightClick => "right_click",
    Screenshot => "screenshot",
    Scroll => "scroll",
    ScrollTo => "scroll_to",
    SwitchTab => "switch_tab",
    TripleClick => "triple_click",
    Type => "type",
    Wait => "wait",
    Zoom => "zoom"
});

client_tool_members!(AnthropicComputerMember {
    Screenshot => "screenshot",
    Zoom => "zoom",
    LeftClick => "left_click",
    RightClick => "right_click",
    MiddleClick => "middle_click",
    DoubleClick => "double_click",
    TripleClick => "triple_click",
    LeftClickDrag => "left_click_drag",
    MouseMove => "mouse_move",
    LeftMouseDown => "left_mouse_down",
    LeftMouseUp => "left_mouse_up",
    CursorPosition => "cursor_position",
    Scroll => "scroll",
    Type => "type",
    Key => "key",
    HoldKey => "hold_key",
    Wait => "wait"
});

/// Sparse per-member controls shared by the stable Browser and Computer
/// toolsets. `None` preserves the provider default.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicClientToolConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub defer_loading: Option<bool>,
}

fn deserialize_configs<'de, D, M>(
    deserializer: D,
) -> Result<BTreeMap<M, AnthropicClientToolConfig>, D::Error>
where
    D: serde::Deserializer<'de>,
    M: Ord + Deserialize<'de>,
{
    let values =
        Option::<BTreeMap<M, Option<AnthropicClientToolConfig>>>::deserialize(deserializer)?;
    Ok(values
        .unwrap_or_default()
        .into_iter()
        .map(|(member, config)| (member, config.unwrap_or_default()))
        .collect())
}

/// Settings for the stable `browser_toolset_20260801` declaration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicBrowserToolsetConfig {
    #[serde(
        default,
        deserialize_with = "deserialize_configs",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub configs: BTreeMap<AnthropicBrowserMember, AnthropicClientToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicMcpCacheControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_callers: Option<Vec<AnthropicToolCaller>>,
}

/// Settings for the stable `computer_toolset_20260801` declaration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicComputerToolsetConfig {
    #[serde(
        default,
        deserialize_with = "deserialize_configs",
        skip_serializing_if = "BTreeMap::is_empty"
    )]
    pub configs: BTreeMap<AnthropicComputerMember, AnthropicClientToolConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_control: Option<AnthropicMcpCacheControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_callers: Option<Vec<AnthropicToolCaller>>,
}

/// One Anthropic-defined client toolset declaration. The provider executes no
/// Browser or Computer member calls; the host handles the returned `tool_use`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AnthropicClientToolset {
    #[serde(rename = "browser_toolset_20260801")]
    Browser(AnthropicBrowserToolsetConfig),
    #[serde(rename = "computer_toolset_20260801")]
    Computer(AnthropicComputerToolsetConfig),
}

impl AnthropicClientToolset {
    /// The `toolset_name` value attached to member `tool_use` and `tool_result`
    /// blocks.
    pub const fn kind_name(&self) -> &'static str {
        match self {
            Self::Browser(_) => "browser",
            Self::Computer(_) => "computer",
        }
    }

    /// The dated stable tool type emitted in the Messages `tools` array.
    pub const fn wire_type(&self) -> &'static str {
        match self {
            Self::Browser(_) => "browser_toolset_20260801",
            Self::Computer(_) => "computer_toolset_20260801",
        }
    }

    /// All fixed member names from the stable schema, including members that
    /// are disabled by default or explicitly disabled in this request.
    pub const fn members(&self) -> &'static [&'static str] {
        match self {
            Self::Browser(_) => AnthropicBrowserMember::NAMES,
            Self::Computer(_) => AnthropicComputerMember::NAMES,
        }
    }

    pub fn member_config(&self, name: &str) -> Option<&AnthropicClientToolConfig> {
        match self {
            Self::Browser(config) => {
                AnthropicBrowserMember::parse(name).and_then(|member| config.configs.get(&member))
            }
            Self::Computer(config) => {
                AnthropicComputerMember::parse(name).and_then(|member| config.configs.get(&member))
            }
        }
    }

    pub fn member_enabled(&self, name: &str) -> Option<bool> {
        match self {
            Self::Browser(config) => AnthropicBrowserMember::parse(name).map(|member| {
                config
                    .configs
                    .get(&member)
                    .and_then(|value| value.enabled)
                    .unwrap_or_else(|| member.default_enabled())
            }),
            Self::Computer(config) => AnthropicComputerMember::parse(name).map(|member| {
                config
                    .configs
                    .get(&member)
                    .and_then(|value| value.enabled)
                    .unwrap_or_else(|| member.default_enabled())
            }),
        }
    }

    pub fn member_deferred(&self, name: &str) -> Option<bool> {
        self.member_config(name)
            .map(|config| config.defer_loading.unwrap_or(false))
    }

    pub fn cache_control(&self) -> Option<AnthropicMcpCacheControl> {
        match self {
            Self::Browser(config) => config.cache_control,
            Self::Computer(config) => config.cache_control,
        }
    }

    pub fn allowed_callers(&self) -> Option<&[AnthropicToolCaller]> {
        match self {
            Self::Browser(config) => config.allowed_callers.as_deref(),
            Self::Computer(config) => config.allowed_callers.as_deref(),
        }
    }

    pub(crate) fn to_wire_value(&self) -> serde_json::Value {
        let mut value = serde_json::json!({"type": self.wire_type()});
        let (configs, cache_control, allowed_callers) = match self {
            Self::Browser(config) => (
                serde_json::to_value(&config.configs).ok(),
                config.cache_control,
                config.allowed_callers.as_ref(),
            ),
            Self::Computer(config) => (
                serde_json::to_value(&config.configs).ok(),
                config.cache_control,
                config.allowed_callers.as_ref(),
            ),
        };
        if let Some(configs) =
            configs.filter(|configs| !configs.as_object().is_none_or(|map| map.is_empty()))
        {
            value["configs"] = configs;
        }
        if let Some(cache_control) = cache_control {
            let mut marker = serde_json::json!({"type": "ephemeral"});
            if let Some(ttl) = cache_control.ttl {
                marker["ttl"] = serde_json::Value::String(
                    match ttl {
                        super::types::AnthropicMcpCacheTtl::FiveMinutes => "5m",
                        super::types::AnthropicMcpCacheTtl::OneHour => "1h",
                    }
                    .into(),
                );
            }
            value["cache_control"] = marker;
        }
        if let Some(allowed_callers) = allowed_callers {
            value["allowed_callers"] = serde_json::to_value(allowed_callers).unwrap_or_default();
        }
        value
    }
}
