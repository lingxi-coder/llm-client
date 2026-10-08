//! Typed native request options owned by anthropic.
use super::types::*;
use crate::protocol::{ChatRequest, HostedTool, JsonPayload, NativeExtension, NativeType};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::value::RawValue;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", content = "config", rename_all = "snake_case")]
pub enum AnthropicHostedTool {
    ToolSearch(AnthropicToolSearchConfig),
    WebFetch(AnthropicWebFetchConfig),
    CodeExecution(AnthropicCodeExecutionConfig),
    Mcp(AnthropicMcpConfig),
}

impl<'de> Deserialize<'de> for AnthropicHostedTool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        // Read the internally tagged wire envelope as raw source. Serde's
        // internally tagged enum visitor buffers the payload into Value-like
        // content before visiting `config`, which cannot retain arbitrary MCP
        // schema depth or escaped lone UTF-16 units.
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        let payload = JsonPayload::parse(raw.get()).map_err(serde::de::Error::custom)?;
        let kind = payload
            .string_at("/type")
            .ok_or_else(|| serde::de::Error::custom("Anthropic hosted tool type is missing"))?;
        let config = payload
            .payload_at("/config")
            .ok_or_else(|| serde::de::Error::custom("Anthropic hosted tool config is missing"))?;
        match kind.as_str() {
            "tool_search" => {
                serde_json::from_str::<AnthropicToolSearchConfig>(config.as_json_source())
                    .map(Self::ToolSearch)
                    .map_err(serde::de::Error::custom)
            }
            "web_fetch" => serde_json::from_str::<AnthropicWebFetchConfig>(config.as_json_source())
                .map(Self::WebFetch)
                .map_err(serde::de::Error::custom),
            "code_execution" => {
                serde_json::from_str::<AnthropicCodeExecutionConfig>(config.as_json_source())
                    .map(Self::CodeExecution)
                    .map_err(serde::de::Error::custom)
            }
            "mcp" => serde_json::from_str::<AnthropicMcpConfig>(config.as_json_source())
                .map(Self::Mcp)
                .map_err(serde::de::Error::custom),
            _ => Err(serde::de::Error::custom(
                "unknown Anthropic hosted tool type",
            )),
        }
    }
}
impl NativeType for AnthropicHostedTool {
    const FORMAT: &'static str = "anthropic.hosted_tool.v1";
}
impl From<AnthropicHostedTool> for HostedTool {
    fn from(value: AnthropicHostedTool) -> Self {
        Self::Native(
            NativeExtension::from_typed(value)
                .expect("native hosted tools contain serializable JSON data"),
        )
    }
}
impl ChatRequest {
    pub fn hosted_anthropic_tool_search(&self) -> Option<&AnthropicToolSearchConfig> {
        self.hosted_tools
            .iter()
            .find_map(|tool| match tool.native::<AnthropicHostedTool>() {
                Some(AnthropicHostedTool::ToolSearch(config)) => Some(config),
                _ => None,
            })
    }
    pub fn set_hosted_anthropic_tool_search(&mut self, config: Option<AnthropicToolSearchConfig>) {
        self.hosted_tools.retain(|tool| {
            !matches!(
                tool.native::<AnthropicHostedTool>(),
                Some(AnthropicHostedTool::ToolSearch(_))
            )
        });
        if let Some(config) = config {
            self.hosted_tools
                .push(AnthropicHostedTool::ToolSearch(config).into());
        }
    }
    pub fn hosted_anthropic_web_fetch(&self) -> Option<&AnthropicWebFetchConfig> {
        self.hosted_tools
            .iter()
            .find_map(|tool| match tool.native::<AnthropicHostedTool>() {
                Some(AnthropicHostedTool::WebFetch(config)) => Some(config),
                _ => None,
            })
    }
    pub fn set_hosted_anthropic_web_fetch(&mut self, config: Option<AnthropicWebFetchConfig>) {
        self.hosted_tools.retain(|tool| {
            !matches!(
                tool.native::<AnthropicHostedTool>(),
                Some(AnthropicHostedTool::WebFetch(_))
            )
        });
        if let Some(config) = config {
            self.hosted_tools
                .push(AnthropicHostedTool::WebFetch(config).into());
        }
    }
    pub fn hosted_anthropic_code_execution(&self) -> Option<&AnthropicCodeExecutionConfig> {
        self.hosted_tools
            .iter()
            .find_map(|tool| match tool.native::<AnthropicHostedTool>() {
                Some(AnthropicHostedTool::CodeExecution(config)) => Some(config),
                _ => None,
            })
    }
    pub fn anthropic_mcp_servers(&self) -> impl Iterator<Item = &AnthropicMcpConfig> {
        self.hosted_tools
            .iter()
            .filter_map(|tool| match tool.native::<AnthropicHostedTool>() {
                Some(AnthropicHostedTool::Mcp(config)) => Some(config),
                _ => None,
            })
    }
}

/// Anthropic message-level policy, stored in a format-tagged extension.
impl NativeType for AnthropicMessageOptions {
    const FORMAT: &'static str = "anthropic.message_options.v1";
}

/// Anthropic request options beyond provider-executed tools.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicRequestOptions {
    pub client_toolsets: Vec<AnthropicClientToolset>,
}
impl NativeType for AnthropicRequestOptions {
    const FORMAT: &'static str = "anthropic.request_options.v1";
}
impl ChatRequest {
    pub fn anthropic_client_toolsets(&self) -> &[AnthropicClientToolset] {
        self.native_options
            .iter()
            .find(|extension| extension.is::<AnthropicRequestOptions>())
            .and_then(|extension| extension.decode::<AnthropicRequestOptions>().ok())
            .map_or(&[], |options| options.client_toolsets.as_slice())
    }
    pub fn set_anthropic_client_toolsets(&mut self, client_toolsets: Vec<AnthropicClientToolset>) {
        self.native_options
            .retain(|extension| !extension.is::<AnthropicRequestOptions>());
        if !client_toolsets.is_empty() {
            self.native_options.push(
                NativeExtension::from_typed(AnthropicRequestOptions { client_toolsets })
                    .expect("toolset JSON is serializable"),
            );
        }
    }
}
impl crate::protocol::ConversationMessage {
    pub fn anthropic_options(&self) -> Option<&AnthropicMessageOptions> {
        self.native_options
            .iter()
            .find(|extension| extension.is::<AnthropicMessageOptions>())
            .and_then(|extension| extension.decode::<AnthropicMessageOptions>().ok())
    }
    pub fn with_anthropic_options(mut self, options: AnthropicMessageOptions) -> Self {
        self.native_options
            .retain(|extension| !extension.is::<AnthropicMessageOptions>());
        self.native_options.push(
            NativeExtension::from_typed(options).expect("message options JSON is serializable"),
        );
        self
    }
}

/// Provider-owned policy for a caller-executed function declaration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AnthropicToolOptions {
    pub allowed_callers: Vec<AnthropicToolCaller>,
}
impl NativeType for AnthropicToolOptions {
    const FORMAT: &'static str = "anthropic.tool_options.v1";
}

fn deserialize_present_json_payload<'de, D>(
    deserializer: D,
) -> Result<Option<JsonPayload>, D::Error>
where
    D: Deserializer<'de>,
{
    JsonPayload::deserialize(deserializer).map(Some)
}

impl crate::protocol::ToolSpec {
    pub fn anthropic_allowed_callers(&self) -> &[AnthropicToolCaller] {
        self.native_options
            .iter()
            .find(|extension| extension.is::<AnthropicToolOptions>())
            .and_then(|extension| extension.decode::<AnthropicToolOptions>().ok())
            .map_or(&[], |options| options.allowed_callers.as_slice())
    }
    pub fn set_anthropic_allowed_callers(&mut self, allowed_callers: Vec<AnthropicToolCaller>) {
        self.native_options
            .retain(|extension| !extension.is::<AnthropicToolOptions>());
        if !allowed_callers.is_empty() {
            self.native_options.push(
                NativeExtension::from_typed(AnthropicToolOptions { allowed_callers })
                    .expect("caller enum is serializable"),
            );
        }
    }
    pub fn with_anthropic_allowed_callers(
        mut self,
        allowed_callers: Vec<AnthropicToolCaller>,
    ) -> Self {
        self.set_anthropic_allowed_callers(allowed_callers);
        self
    }
}

/// Unmodified native container and usage observations from Messages.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AnthropicResponseMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<AnthropicContainerMetadata>,
    #[serde(
        default,
        deserialize_with = "deserialize_present_json_payload",
        skip_serializing_if = "Option::is_none"
    )]
    pub usage: Option<crate::protocol::JsonPayload>,
    /// Provider stop diagnostics, preserved without interpreting their schema.
    #[serde(
        default,
        deserialize_with = "deserialize_present_json_payload",
        skip_serializing_if = "Option::is_none"
    )]
    pub stop_details: Option<crate::protocol::JsonPayload>,
    /// Provider response resumability, retained with exact source and presence.
    #[serde(
        default,
        deserialize_with = "deserialize_present_json_payload",
        skip_serializing_if = "Option::is_none"
    )]
    pub resumable: Option<crate::protocol::JsonPayload>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fallback: Option<super::fallback_response::FallbackResponse>,
}
impl NativeType for AnthropicResponseMetadata {
    const FORMAT: &'static str = "anthropic.response_metadata.v1";
}
impl crate::protocol::ChatResponse {
    pub fn anthropic_fallback(&self) -> Option<&super::fallback_response::FallbackResponse> {
        self.native_metadata
            .iter()
            .find(|extension| extension.is::<AnthropicResponseMetadata>())
            .and_then(|extension| extension.decode::<AnthropicResponseMetadata>().ok())
            .and_then(|metadata| metadata.fallback.as_ref())
    }
    pub fn set_anthropic_fallback(
        &mut self,
        fallback: Option<super::fallback_response::FallbackResponse>,
    ) {
        if let Some(extension) = self
            .native_metadata
            .iter_mut()
            .find(|extension| extension.is::<AnthropicResponseMetadata>())
        {
            extension
                .edit::<AnthropicResponseMetadata, _>(|metadata| metadata.fallback = fallback)
                .expect("native response metadata is JSON");
        } else if fallback.is_some() {
            self.native_metadata.push(
                NativeExtension::from_typed(AnthropicResponseMetadata {
                    fallback,
                    ..Default::default()
                })
                .expect("native response metadata is JSON"),
            );
        }
    }
    pub fn anthropic_container(&self) -> Option<&AnthropicContainerMetadata> {
        self.native_metadata
            .iter()
            .find(|extension| extension.is::<AnthropicResponseMetadata>())
            .and_then(|extension| extension.decode::<AnthropicResponseMetadata>().ok())
            .and_then(|metadata| metadata.container.as_ref())
    }
    pub fn anthropic_usage(&self) -> Option<&crate::protocol::JsonPayload> {
        self.native_metadata
            .iter()
            .find(|extension| extension.is::<AnthropicResponseMetadata>())
            .and_then(|extension| extension.decode::<AnthropicResponseMetadata>().ok())
            .and_then(|metadata| metadata.usage.as_ref())
    }
    pub fn anthropic_stop_details(&self) -> Option<&crate::protocol::JsonPayload> {
        self.native_metadata
            .iter()
            .find(|extension| extension.is::<AnthropicResponseMetadata>())
            .and_then(|extension| extension.decode::<AnthropicResponseMetadata>().ok())
            .and_then(|metadata| metadata.stop_details.as_ref())
    }
    pub fn anthropic_resumable(&self) -> Option<&crate::protocol::JsonPayload> {
        self.native_metadata
            .iter()
            .find(|extension| extension.is::<AnthropicResponseMetadata>())
            .and_then(|extension| extension.decode::<AnthropicResponseMetadata>().ok())
            .and_then(|metadata| metadata.resumable.as_ref())
    }
    pub fn set_anthropic_resumable(&mut self, resumable: Option<crate::protocol::JsonPayload>) {
        if let Some(extension) = self
            .native_metadata
            .iter_mut()
            .find(|extension| extension.is::<AnthropicResponseMetadata>())
        {
            extension
                .edit::<AnthropicResponseMetadata, _>(|metadata| metadata.resumable = resumable)
                .expect("native response metadata is JSON");
        } else if resumable.is_some() {
            self.native_metadata.push(
                NativeExtension::from_typed(AnthropicResponseMetadata {
                    resumable,
                    ..Default::default()
                })
                .expect("native response metadata is JSON"),
            );
        }
    }
    pub fn set_anthropic_stop_details(
        &mut self,
        stop_details: Option<crate::protocol::JsonPayload>,
    ) {
        if let Some(extension) = self
            .native_metadata
            .iter_mut()
            .find(|extension| extension.is::<AnthropicResponseMetadata>())
        {
            extension
                .edit::<AnthropicResponseMetadata, _>(|metadata| {
                    metadata.stop_details = stop_details
                })
                .expect("native response metadata is JSON");
        } else if stop_details.is_some() {
            self.native_metadata.push(
                NativeExtension::from_typed(AnthropicResponseMetadata {
                    stop_details,
                    ..Default::default()
                })
                .expect("native response metadata is JSON"),
            );
        }
    }
    pub fn set_anthropic_metadata(
        &mut self,
        container: Option<AnthropicContainerMetadata>,
        usage: Option<crate::protocol::JsonPayload>,
    ) {
        let stop_details = self.anthropic_stop_details().cloned();
        let resumable = self.anthropic_resumable().cloned();
        let fallback = self.anthropic_fallback().cloned();
        self.native_metadata
            .retain(|extension| !extension.is::<AnthropicResponseMetadata>());
        if container.is_some()
            || usage.is_some()
            || stop_details.is_some()
            || resumable.is_some()
            || fallback.is_some()
        {
            self.native_metadata.push(
                NativeExtension::from_typed(AnthropicResponseMetadata {
                    container,
                    usage,
                    stop_details,
                    resumable,
                    fallback,
                })
                .expect("native response metadata is JSON"),
            );
        }
    }
}

#[cfg(test)]
mod response_metadata_tests {
    use super::*;

    fn cold_extension(data_source: &str) -> NativeExtension {
        let source = format!(
            r#"{{"format":"{}","data":{data_source}}}"#,
            AnthropicResponseMetadata::FORMAT
        );
        serde_json::from_str(&source).expect("native extension fixture is valid")
    }

    fn cold_roundtrip(data_source: &str) -> NativeExtension {
        let extension = cold_extension(data_source);
        let encoded = serde_json::to_string(&extension).expect("extension serializes");
        serde_json::from_str(&encoded).expect("extension cold-deserializes")
    }

    #[test]
    fn response_metadata_preserves_present_null_and_resumable_json_values() {
        for (resumable_member, expected) in [
            ("", None),
            (r#""resumable":null,"#, Some("null")),
            (r#""resumable":true,"#, Some("true")),
            (r#""resumable":false,"#, Some("false")),
        ] {
            let data = format!(r#"{{{resumable_member}"usage":null,"stop_details":null}}"#);
            let extension = cold_roundtrip(&data);
            let metadata = extension
                .decode::<AnthropicResponseMetadata>()
                .expect("metadata cold-decodes");
            assert_eq!(
                metadata.usage.as_ref().map(JsonPayload::as_json_source),
                Some("null"),
                "usage source survives: {data}"
            );
            assert_eq!(
                metadata
                    .stop_details
                    .as_ref()
                    .map(JsonPayload::as_json_source),
                Some("null"),
                "stop_details source survives: {data}"
            );
            assert_eq!(
                metadata.resumable.as_ref().map(JsonPayload::as_json_source),
                expected,
                "resumable presence/source survives: {data}"
            );
        }
    }

    #[test]
    fn response_metadata_edit_keeps_null_and_deep_lone_resumable_source() {
        let depth = 16_384;
        let deep = format!("{}0{}", "[".repeat(depth), "]".repeat(depth));
        let resumable_source = format!(r#"{{"deep":{deep},"lone":"\ud800"}}"#);
        let data = format!(
            r#"{{"usage":null,"stop_details":null,"resumable":{resumable_source},"future":{{"kept":true}}}}"#
        );
        let mut extension = cold_roundtrip(&data);
        assert_eq!(
            extension
                .decode::<AnthropicResponseMetadata>()
                .unwrap()
                .resumable
                .as_ref()
                .unwrap()
                .as_json_source(),
            resumable_source
        );

        extension
            .edit::<AnthropicResponseMetadata, _>(|metadata| {
                metadata.fallback =
                    Some(super::super::fallback_response::FallbackResponse::default())
            })
            .expect("editing another typed field succeeds");

        let metadata = extension
            .decode::<AnthropicResponseMetadata>()
            .expect("edited metadata decodes");
        assert_eq!(metadata.usage.as_ref().unwrap().as_json_source(), "null");
        assert_eq!(
            metadata.stop_details.as_ref().unwrap().as_json_source(),
            "null"
        );
        assert_eq!(
            metadata.resumable.as_ref().unwrap().as_json_source(),
            resumable_source
        );
        assert_eq!(
            extension.data().raw_source_at("/future").as_deref(),
            Some(r#"{"kept":true}"#)
        );
    }
}
