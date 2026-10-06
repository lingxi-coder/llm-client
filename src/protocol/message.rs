//! Messages and content blocks shared by every codec.

use crate::protocol::ids::{ProviderId, ToolUseId};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// One block of structured content within a message.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Provider-owned typed content, distinct from ordinary function calls.
    /// The provider codec validates the native format; callers own execution.
    Native {
        value: crate::protocol::NativeExtension,
    },
    /// Native content required to replay reasoning or a hosted tool's turn unchanged.
    /// Kept separate from client tools, which the caller must execute.
    /// Replaying onto a different protocol is rejected by the built-in codecs.
    /// Chat reasoning uses a `chat_reasoning` envelope whose fields belong to
    /// the enclosing assistant message, rather than to its wire content array.
    ProviderContent {
        protocol: crate::protocol::provider::ProtocolFamily,
        value: Value,
    },
    /// Plain UTF-8 text.
    Text {
        text: String,
        /// Gemini's opaque signature, attached to this exact text part.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },

    /// A tool invocation requested by the model.
    ToolUse {
        id: ToolUseId,
        name: String,
        input: Value,
        /// Provider-issued call ID, if one exists. The local `id` also pairs
        /// calls with results when an older provider omits this field.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        provider_id: Option<String>,
        /// Native provider metadata describing how the tool was invoked.
        /// Anthropic uses this to identify calls made from Code Execution and
        /// to associate them with the paused server-tool call.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        caller: Option<Value>,
        /// Anthropic client-toolset family for Browser/Computer members.
        /// Dispatch member calls by this field together with `name` because
        /// toolset members may share names with other tools.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        toolset_name: Option<String>,
        /// Gemini's opaque signature, attached to this exact call part.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        thought_signature: Option<String>,
    },

    /// The result of a previously requested tool call.
    ToolResult {
        tool_use_id: ToolUseId,
        /// Model-facing text. Always populated, even when `blocks` is set.
        content: String,
        is_error: bool,
        /// Structured blocks when the result is an array (an MCP result with
        /// images or resources). Sent verbatim when present.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        blocks: Option<Vec<Value>>,
        /// Anthropic client-toolset family of the paired call, echoed on
        /// Browser/Computer member results.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        toolset_name: Option<String>,
    },

    /// Extended-thinking trace. `signature` must round-trip: providers that sign
    /// thinking reject a replayed block whose signature is missing.
    Thinking {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },

    /// Provider-opaque redacted reasoning. Round-trips unmodified.
    RedactedThinking {
        data: String,
    },

    Image {
        source: ImageSource,
    },

    /// A document (PDF, text) attached by the user.
    Document {
        source: DocumentSource,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
    },

    /// A video attached by the user. Providers may support only a subset of
    /// these source forms and codecs reject sources they cannot represent.
    Video {
        source: VideoSource,
    },

    /// Base64-encoded audio input for OpenRouter Chat or Gemini GenerateContent.
    /// `format` is a provider-supported format such as `wav` or `mp3`;
    /// codecs validate their own formats and encode the appropriate wire part.
    /// This content part does not accept URLs.
    Audio {
        format: String,
        data: String,
    },
}

/// Audio formats documented by OpenRouter for Chat Completions audio output.
/// A selected model may support only a subset of these formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OpenRouterChatAudioFormat {
    Wav,
    Mp3,
    Flac,
    Opus,
    Pcm16,
}

impl OpenRouterChatAudioFormat {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Wav => "wav",
            Self::Mp3 => "mp3",
            Self::Flac => "flac",
            Self::Opus => "opus",
            Self::Pcm16 => "pcm16",
        }
    }
}

/// Per-request voice and format for OpenRouter Chat Completions audio output.
///
/// `write_metadata` stores this under `openrouter_chat_audio` in
/// [`crate::protocol::ChatRequest::metadata`]. The Chat codec turns it into
/// OpenRouter's top-level `modalities` and `audio` fields. OpenRouter requires
/// a streaming request for audio output.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenRouterChatAudioOutput {
    pub voice: String,
    pub format: OpenRouterChatAudioFormat,
}

impl OpenRouterChatAudioOutput {
    pub fn new(voice: impl Into<String>, format: OpenRouterChatAudioFormat) -> Self {
        Self {
            voice: voice.into(),
            format,
        }
    }

    /// Add the provider-specific configuration while retaining unrelated
    /// object metadata. Returns an error instead of discarding non-object
    /// metadata that the caller may already rely on.
    pub fn write_metadata(&self, metadata: &mut Value) -> Result<(), &'static str> {
        if metadata.is_null() {
            *metadata = Value::Object(Default::default());
        }
        let Some(object) = metadata.as_object_mut() else {
            return Err("OpenRouter Chat audio configuration requires object request metadata");
        };
        object.insert(
            "openrouter_chat_audio".to_owned(),
            serde_json::json!({
                "voice": self.voice,
                "format": self.format.as_str(),
            }),
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 {
        media_type: String,
        data: String,
    },
    Url {
        url: String,
    },
    /// An application-owned attachment. The identifier is stable across
    /// devices; the client resolves it through its configured resolver before
    /// sending the request to a provider.
    Attachment {
        attachment: AttachmentRef,
    },
    /// A provider-owned file prepared for one concrete connection. This is a
    /// model input reference only; it is not an application display URL.
    ProviderFile {
        file: ProviderFileSource,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    Base64 {
        media_type: String,
        data: String,
    },
    Text {
        media_type: String,
        data: String,
    },
    Url {
        url: String,
    },
    /// An application-owned attachment. The identifier is stable across
    /// devices; the client resolves it through its configured resolver before
    /// sending the request to a provider.
    Attachment {
        attachment: AttachmentRef,
    },
    /// A provider-owned file prepared for one concrete connection. This is a
    /// model input reference only; it is not an application display URL.
    ProviderFile {
        file: ProviderFileSource,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VideoSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
    Attachment { attachment: AttachmentRef },
    ProviderFile { file: ProviderFileSource },
}

/// Stable identity and display metadata for bytes owned by the host
/// application. The same id and revision must always resolve to identical
/// bytes, so conversation history can safely refer to the attachment from
/// another device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttachmentRef {
    pub attachment_id: String,
    pub revision: String,
    pub filename: String,
    pub media_type: String,
    pub size_bytes: u64,
}

/// A file reference owned by one provider connection, used only while
/// preparing a request for that provider. It must never replace an
/// application-owned [`AttachmentRef`] in durable conversation history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProviderFileSource {
    pub protocol: crate::protocol::provider::ProtocolFamily,
    pub provider_id: ProviderId,
    /// Connection which created the provider file. The client checks it
    /// against the current route before sending the reference.
    pub profile_name: String,
    /// Deterministic, non-secret fingerprint of the configured file endpoint.
    /// Routes with a documented canonical resource identity (Foundry Files)
    /// use that canonical base. Missing fingerprints from older serialized
    /// references are rejected.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub endpoint_fingerprint: String,
    /// Optional stable, non-secret account identity supplied by the host.
    /// Direct model-input references must carry a non-empty scope matching
    /// the request's `file_account_scope`. The automatic `AttachmentRef` path
    /// may create an internal per-attempt scope when none is configured; that
    /// scope cannot be reused across requests.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub account_scope: Option<String>,
    pub file_id: String,
    /// Some APIs require a URI for model input while exposing a separate file
    /// id for management. This is not a public or cross-device display URL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// Provider-reported expiry for this file. The original timestamp string
    /// is retained so references round-trip without normalizing provider data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    /// Provider-reported processing status, retained verbatim for local
    /// readiness checks. It is not sent as model-input data.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub processing_status: Option<String>,
    /// Required by providers whose model-input URI does not carry the
    /// uploaded file's media type.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// Purpose used to upload this file when the provider requires it again
    /// for list, delete, or model-input operations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,
}

/// One message of the transcript.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConversationMessage {
    pub role: MessageRole,
    pub content: Vec<ContentBlock>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub native_options: Vec<super::NativeExtension>,
}

impl ConversationMessage {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: text.into(),
                thought_signature: None,
            }],
            native_options: Vec::new(),
        }
    }

    pub fn system_text(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::System,
            content: vec![ContentBlock::Text {
                text: text.into(),
                thought_signature: None,
            }],
            native_options: Vec::new(),
        }
    }

    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content,
            native_options: Vec::new(),
        }
    }

    /// Every tool call this message requests, in order, as
    /// `(id, toolset_name, member_name, input)`. Dispatch namespaced calls by
    /// the `(toolset_name, member_name)` pair.
    pub fn tool_uses(&self) -> impl Iterator<Item = (&ToolUseId, Option<&str>, &str, &Value)> {
        self.content.iter().filter_map(|b| match b {
            ContentBlock::ToolUse {
                id,
                name,
                input,
                toolset_name,
                ..
            } => Some((id, toolset_name.as_deref(), name.as_str(), input)),
            _ => None,
        })
    }

    /// Concatenated text of every text block. Thinking is excluded.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text, .. } => Some(text.as_str()),
                ContentBlock::ProviderContent { value, .. } => {
                    value.get("text").and_then(Value::as_str)
                }
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::anthropic::types::{
        AnthropicClearAt, AnthropicMessageEffort, AnthropicMessageOptions,
    };

    #[test]
    fn tool_result_round_trips_with_and_without_blocks() {
        let b = ContentBlock::ToolResult {
            tool_use_id: ToolUseId::new("toolu_1"),
            content: "ok".into(),
            is_error: false,
            blocks: None,
            toolset_name: None,
        };
        let s = serde_json::to_string(&b).unwrap();
        assert!(!s.contains("blocks"));
        assert!(!s.contains("toolset_name"));
        assert_eq!(serde_json::from_str::<ContentBlock>(&s).unwrap(), b);
    }

    #[test]
    fn client_toolset_names_round_trip_and_null_decodes_as_absent() {
        let use_block = ContentBlock::ToolUse {
            id: ToolUseId::new("toolu_browser"),
            name: "screenshot".into(),
            input: serde_json::json!({"tab_id": "1"}),
            provider_id: None,
            caller: None,
            toolset_name: Some("browser".into()),
            thought_signature: None,
        };
        let use_value = serde_json::to_value(&use_block).unwrap();
        assert_eq!(use_value["toolset_name"], "browser");
        assert_eq!(
            serde_json::from_value::<ContentBlock>(use_value).unwrap(),
            use_block
        );

        let result_block = ContentBlock::ToolResult {
            tool_use_id: ToolUseId::new("toolu_browser"),
            content: "done".into(),
            is_error: false,
            blocks: None,
            toolset_name: Some("browser".into()),
        };
        let result_value = serde_json::to_value(&result_block).unwrap();
        assert_eq!(result_value["toolset_name"], "browser");
        assert_eq!(
            serde_json::from_value::<ContentBlock>(result_value).unwrap(),
            result_block
        );

        let null_use = serde_json::json!({
            "type": "tool_use",
            "id": "toolu_legacy",
            "name": "lookup",
            "input": {},
            "toolset_name": null
        });
        assert!(matches!(
            serde_json::from_value::<ContentBlock>(null_use).unwrap(),
            ContentBlock::ToolUse {
                toolset_name: None,
                ..
            }
        ));
        let omitted_result = serde_json::json!({
            "type": "tool_result",
            "tool_use_id": "toolu_legacy",
            "content": "done",
            "is_error": false
        });
        assert!(matches!(
            serde_json::from_value::<ContentBlock>(omitted_result).unwrap(),
            ContentBlock::ToolResult {
                toolset_name: None,
                ..
            }
        ));
    }

    #[test]
    fn tool_uses_exposes_toolset_name_with_member_name() {
        let message = ConversationMessage::assistant(vec![ContentBlock::ToolUse {
            id: ToolUseId::new("toolu_browser"),
            name: "screenshot".into(),
            input: serde_json::json!({"tab_id": "1"}),
            provider_id: None,
            caller: None,
            toolset_name: Some("browser".into()),
            thought_signature: None,
        }]);

        let (id, toolset_name, name, input) = message.tool_uses().next().unwrap();
        assert_eq!(id.as_str(), "toolu_browser");
        assert_eq!(toolset_name, Some("browser"));
        assert_eq!(name, "screenshot");
        assert_eq!(input["tab_id"], "1");
    }

    #[test]
    fn thinking_signature_round_trips() {
        let b = ContentBlock::Thinking {
            text: "t".into(),
            signature: Some("sig".into()),
        };
        let s = serde_json::to_string(&b).unwrap();
        assert_eq!(serde_json::from_str::<ContentBlock>(&s).unwrap(), b);
    }

    #[test]
    fn anthropic_message_options_are_typed_optional_metadata() {
        let message = ConversationMessage::system_text("Reminder").with_anthropic_options(
            AnthropicMessageOptions {
                clear_at: Some(AnthropicClearAt::NextUserMessage),
                effort: Some(AnthropicMessageEffort::XHigh),
            },
        );
        let value = serde_json::to_value(&message).unwrap();
        assert_eq!(
            value["native_options"][0]["data"],
            serde_json::json!({
                "clear_at":"next_user_message",
                "effort":"xhigh"
            })
        );

        let legacy: ConversationMessage = serde_json::from_value(serde_json::json!({
            "role":"user",
            "content":[{"type":"text","text":"hello"}]
        }))
        .unwrap();
        assert!(legacy.native_options.is_empty());
        assert!(
            serde_json::to_value(ConversationMessage::user_text("hello"))
                .unwrap()
                .get("native_options")
                .is_none()
        );
        assert!(
            serde_json::from_value::<AnthropicMessageOptions>(serde_json::json!({
                "unknown":true
            }))
            .is_err()
        );
    }
}
