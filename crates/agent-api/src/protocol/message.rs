//! Messages and content blocks. `ContentBlock` is defined exactly once in the
//! workspace (gate 29); every codec converts to and from this shape.

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
    /// Native content required to replay a hosted tool's turn unchanged.
    /// Kept separate from client tools, which the caller must execute.
    /// Replaying onto a different protocol is rejected by the built-in codecs.
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
    },

    /// Extended-thinking trace. `signature` must round-trip: providers that sign
    /// thinking reject a replayed block whose signature is missing (gate 18).
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
    /// Deterministic, non-secret fingerprint of the full configured base URL.
    /// Missing fingerprints from older serialized references are rejected.
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
}

impl ConversationMessage {
    pub fn user_text(text: impl Into<String>) -> Self {
        Self {
            role: MessageRole::User,
            content: vec![ContentBlock::Text {
                text: text.into(),
                thought_signature: None,
            }],
        }
    }

    pub fn assistant(content: Vec<ContentBlock>) -> Self {
        Self {
            role: MessageRole::Assistant,
            content,
        }
    }

    /// Every tool call this message requests, in order.
    pub fn tool_uses(&self) -> impl Iterator<Item = (&ToolUseId, &str, &Value)> {
        self.content.iter().filter_map(|b| match b {
            ContentBlock::ToolUse {
                id, name, input, ..
            } => Some((id, name.as_str(), input)),
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

    #[test]
    fn tool_result_round_trips_with_and_without_blocks() {
        let b = ContentBlock::ToolResult {
            tool_use_id: ToolUseId::new("toolu_1"),
            content: "ok".into(),
            is_error: false,
            blocks: None,
        };
        let s = serde_json::to_string(&b).unwrap();
        assert!(!s.contains("blocks"));
        assert_eq!(serde_json::from_str::<ContentBlock>(&s).unwrap(), b);
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
}
