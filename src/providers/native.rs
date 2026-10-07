//! Validation shared by provider-native request extensions.
use super::{
    anthropic::native::AnthropicHostedTool,
    google::native::GoogleHostedTool,
    openai::{computer::OpenAiComputerToolConfig, native::OpenAiHostedTool},
    openrouter::native::OpenRouterHostedTool,
    qwen::native::QwenHostedTool,
    xai::native::XaiHostedTool,
};
use crate::protocol::{ChatRequest, HostedTool, LlmError, NativeExtension, NativeType};

impl ChatRequest {
    /// Reject malformed, unknown, and duplicate native tools before provider
    /// helpers read optional typed views. No unknown format is silently dropped.
    pub fn validate_hosted_tools(&self) -> Result<(), LlmError> {
        validate_request_options(&self.native_options)?;
        for tool in &self.tools {
            validate_options(
                &tool.native_options,
                super::anthropic::native::AnthropicToolOptions::FORMAT,
                |value| {
                    value
                        .decode::<super::anthropic::native::AnthropicToolOptions>()
                        .map(|_| ())
                },
            )?;
        }
        for message in &self.messages {
            validate_options(
                &message.native_options,
                super::anthropic::types::AnthropicMessageOptions::FORMAT,
                |value| {
                    value
                        .decode::<super::anthropic::types::AnthropicMessageOptions>()
                        .map(|_| ())
                },
            )?;
        }
        let mut unique = std::collections::BTreeSet::new();
        let mut anthropic_mcp_count = 0;
        for tool in &self.hosted_tools {
            let key = match tool {
                HostedTool::WebSearch(_) => ("common".to_string(), "web_search".to_string()),
                HostedTool::Native(extension) => {
                    validate_native_tool(extension, &mut anthropic_mcp_count)?
                }
            };
            if !unique.insert(key) {
                return Err(LlmError::InvalidRequest {
                    message:
                        "the same hosted tool or remote MCP server may appear only once per request"
                            .into(),
                });
            }
        }
        if anthropic_mcp_count > 20 {
            return Err(LlmError::InvalidRequest {
                message: "Anthropic MCP supports at most 20 servers per request".into(),
            });
        }
        Ok(())
    }
}

fn validate_native_tool(
    extension: &NativeExtension,
    anthropic_mcp_count: &mut usize,
) -> Result<(String, String), LlmError> {
    match extension.format() {
        AnthropicHostedTool::FORMAT => match extension.decode::<AnthropicHostedTool>()? {
            AnthropicHostedTool::WebFetch(config) => config.validate()?,
            AnthropicHostedTool::Mcp(config) => {
                config.validate()?;
                *anthropic_mcp_count += 1;
                return Ok(("anthropic_mcp".into(), config.name().into()));
            }
            _ => {}
        },
        OpenAiHostedTool::FORMAT => match extension.decode::<OpenAiHostedTool>()? {
            OpenAiHostedTool::ToolSearch(config) => config.validate()?,
            OpenAiHostedTool::RemoteMcp(config) => {
                config.validate()?;
                return Ok(("remote_mcp".into(), config.server_label().into()));
            }
            _ => {}
        },
        XaiHostedTool::FORMAT => {
            let XaiHostedTool::RemoteMcp(config) = extension.decode::<XaiHostedTool>()?;
            config.validate()?;
            return Ok(("remote_mcp".into(), config.server_label().into()));
        }
        GoogleHostedTool::FORMAT => {
            extension.decode::<GoogleHostedTool>()?;
        }
        OpenRouterHostedTool::FORMAT => {
            extension.decode::<OpenRouterHostedTool>()?;
        }
        QwenHostedTool::FORMAT => {
            extension.decode::<QwenHostedTool>()?;
        }
        format => {
            return Err(LlmError::UnsupportedCapability {
                message: format!("unsupported native hosted-tool format {format}"),
            });
        }
    }
    let kind = extension
        .data()
        .get("type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| LlmError::InvalidRequest {
            message: "native hosted tool has no type".into(),
        })?;
    Ok((extension.format().to_string(), kind.to_string()))
}

fn validate_options(
    options: &[NativeExtension],
    expected: &str,
    decode: impl Fn(&NativeExtension) -> Result<(), LlmError>,
) -> Result<(), LlmError> {
    let mut seen = false;
    for extension in options {
        if extension.format() != expected {
            return Err(LlmError::UnsupportedCapability {
                message: format!("unsupported native options format {}", extension.format()),
            });
        }
        if seen {
            return Err(LlmError::InvalidRequest {
                message: format!("duplicate native options format {expected}"),
            });
        }
        decode(extension)?;
        seen = true;
    }
    Ok(())
}

fn validate_request_options(options: &[NativeExtension]) -> Result<(), LlmError> {
    let mut seen = std::collections::BTreeSet::new();
    for extension in options {
        let format = extension.format();
        if !seen.insert(format) {
            return Err(LlmError::InvalidRequest {
                message: format!("duplicate native options format {format}"),
            });
        }
        match format {
            super::anthropic::native::AnthropicRequestOptions::FORMAT => {
                extension.decode::<super::anthropic::native::AnthropicRequestOptions>()?;
            }
            OpenAiComputerToolConfig::FORMAT => {
                extension.decode::<OpenAiComputerToolConfig>()?;
            }
            super::google::computer::GeminiComputerToolConfig::FORMAT => {
                extension
                    .decode::<super::google::computer::GeminiComputerToolConfig>()?
                    .validate()?;
            }
            format => {
                return Err(LlmError::UnsupportedCapability {
                    message: format!("unsupported native options format {format}"),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request() -> ChatRequest {
        serde_json::from_value(json!({"model":"test", "messages":[]})).unwrap()
    }
    #[test]
    fn malformed_or_unknown_native_tools_are_rejected() {
        for (format, data) in [
            ("unknown.tools.v1", json!({})),
            (
                AnthropicHostedTool::FORMAT,
                json!({"type":"mcp", "config":null}),
            ),
        ] {
            let mut request = request();
            request.hosted_tools.push(HostedTool::Native(
                NativeExtension::new(format, data).unwrap(),
            ));
            assert!(request.validate_hosted_tools().is_err());
        }
    }
    #[test]
    fn malformed_duplicate_and_wrong_location_options_are_rejected() {
        let mut request = request();
        let extension = NativeExtension::from_typed(
            super::super::anthropic::native::AnthropicRequestOptions::default(),
        )
        .unwrap();
        request.native_options = vec![extension.clone(), extension];
        assert!(request.validate_hosted_tools().is_err());
        request.native_options = vec![NativeExtension::new(
            super::super::anthropic::native::AnthropicRequestOptions::FORMAT,
            json!({"client_toolsets":false}),
        )
        .unwrap()];
        assert!(request.validate_hosted_tools().is_err());
        request.native_options = vec![
            NativeExtension::from_typed(OpenAiComputerToolConfig::default()).unwrap(),
            NativeExtension::from_typed(OpenAiComputerToolConfig::default()).unwrap(),
        ];
        assert!(request.validate_hosted_tools().is_err());
        request.native_options = vec![NativeExtension::new(
            OpenAiComputerToolConfig::FORMAT,
            json!({"unexpected":true}),
        )
        .unwrap()];
        assert!(request.validate_hosted_tools().is_err());
        request.native_options =
            vec![NativeExtension::new("unknown.request.option.v1", json!({})).unwrap()];
        assert!(request.validate_hosted_tools().is_err());
        request.native_options.clear();
        let mut message = crate::protocol::ConversationMessage::user_text("hello");
        message.native_options.push(
            NativeExtension::from_typed(
                super::super::anthropic::native::AnthropicRequestOptions::default(),
            )
            .unwrap(),
        );
        request.messages.push(message);
        assert!(request.validate_hosted_tools().is_err());
    }
}
