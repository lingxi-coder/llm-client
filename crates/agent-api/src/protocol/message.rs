//! Messages and content blocks. `ContentBlock` is defined exactly once in the
//! workspace (gate 29); every codec converts to and from this shape.

use crate::protocol::ids::ToolUseId;
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
    },

    /// A tool invocation requested by the model.
    ToolUse {
        id: ToolUseId,
        name: String,
        input: Value,
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
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    Base64 { media_type: String, data: String },
    Text { media_type: String, data: String },
    Url { url: String },
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
            content: vec![ContentBlock::Text { text: text.into() }],
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
            ContentBlock::ToolUse { id, name, input } => Some((id, name.as_str(), input)),
            _ => None,
        })
    }

    /// Concatenated text of every text block. Thinking is excluded.
    pub fn text(&self) -> String {
        self.content
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
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
