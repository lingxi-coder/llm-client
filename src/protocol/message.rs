//! Messages and content blocks shared by every codec.

use crate::protocol::ids::{ProviderId, ToolUseId};
use crate::protocol::JsonPayload;
use serde::de::DeserializeOwned;
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::value::RawValue;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    Assistant,
    System,
}

/// One block of structured content within a message.
#[derive(Debug, Clone, PartialEq)]
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
        value: JsonPayload,
    },
    /// Plain UTF-8 text.
    Text {
        text: String,
        /// Gemini's opaque signature, attached to this exact text part.
        thought_signature: Option<String>,
        /// Exact provider field presence; the inner option preserves JSON null.
        citations: Option<Option<JsonPayload>>,
    },
    /// Text decoded from JSON with one or more unpaired JavaScript UTF-16
    /// units. `text` is the lossy display string; codecs use the exact units
    /// when this block is replayed into a later request.
    TextJsUtf16 {
        text: String,
        /// Runtime-only exact source units. Provider encoders move these into
        /// the selected codec's JSON-string override map; they are never
        /// serialized as a provider field.
        utf16_code_units: Vec<u16>,
        thought_signature: Option<String>,
        citations: Option<Option<JsonPayload>>,
    },

    /// A tool invocation requested by the model.
    ToolUse {
        id: ToolUseId,
        name: String,
        input: JsonPayload,
        /// Provider-issued call ID, if one exists. The local `id` also pairs
        /// calls with results when an older provider omits this field.
        provider_id: Option<String>,
        /// Native provider metadata describing how the tool was invoked.
        /// Anthropic uses this to identify calls made from Code Execution and
        /// to associate them with the paused server-tool call.
        caller: Option<JsonPayload>,
        /// Anthropic client-toolset family for Browser/Computer members.
        /// Dispatch member calls by this field together with `name` because
        /// toolset members may share names with other tools.
        toolset_name: Option<String>,
        /// Gemini's opaque signature, attached to this exact call part.
        thought_signature: Option<String>,
    },

    /// The result of a previously requested tool call.
    ToolResult {
        tool_use_id: ToolUseId,
        /// Model-facing text. Always populated, even when `blocks` is set.
        content: JsonPayload,
        /// `None` preserves an omitted provider key; explicit false remains present.
        is_error: Option<bool>,
        /// Structured blocks when the result is an array (an MCP result with
        /// images or resources). Sent verbatim when present.
        blocks: Option<JsonPayload>,
        /// Anthropic cache reference for this exact tool-result receipt.
        cache_reference: Option<String>,
        /// Anthropic client-toolset family of the paired call, echoed on
        /// Browser/Computer member results.
        toolset_name: Option<String>,
    },

    /// Extended-thinking trace. `signature` must round-trip: providers that sign
    /// thinking reject a replayed block whose signature is missing.
    Thinking {
        text: String,
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

impl Serialize for ContentBlock {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(None)?;
        match self {
            Self::Native { value } => {
                map.serialize_entry("type", "native")?;
                map.serialize_entry("value", value)?;
            }
            Self::ProviderContent { protocol, value } => {
                map.serialize_entry("type", "provider_content")?;
                map.serialize_entry("protocol", protocol)?;
                map.serialize_entry("value", value)?;
            }
            Self::Text {
                text,
                thought_signature,
                citations,
            } => {
                map.serialize_entry("type", "text")?;
                map.serialize_entry("text", text)?;
                serialize_some(&mut map, "thought_signature", thought_signature)?;
                serialize_some(&mut map, "citations", citations)?;
            }
            Self::TextJsUtf16 {
                text,
                thought_signature,
                citations,
                ..
            } => {
                map.serialize_entry("type", "text_js_utf16")?;
                map.serialize_entry("text", text)?;
                serialize_some(&mut map, "thought_signature", thought_signature)?;
                serialize_some(&mut map, "citations", citations)?;
            }
            Self::ToolUse {
                id,
                name,
                input,
                provider_id,
                caller,
                toolset_name,
                thought_signature,
            } => {
                map.serialize_entry("type", "tool_use")?;
                map.serialize_entry("id", id)?;
                map.serialize_entry("name", name)?;
                map.serialize_entry("input", input)?;
                serialize_some(&mut map, "provider_id", provider_id)?;
                serialize_some(&mut map, "caller", caller)?;
                serialize_some(&mut map, "toolset_name", toolset_name)?;
                serialize_some(&mut map, "thought_signature", thought_signature)?;
            }
            Self::ToolResult {
                tool_use_id,
                content,
                is_error,
                blocks,
                cache_reference,
                toolset_name,
            } => {
                content
                    .validate_root_string("tool result content")
                    .map_err(serde::ser::Error::custom)?;
                if let Some(blocks) = blocks {
                    blocks
                        .validate_root_array("tool result blocks")
                        .map_err(serde::ser::Error::custom)?;
                }
                map.serialize_entry("type", "tool_result")?;
                map.serialize_entry("tool_use_id", tool_use_id)?;
                map.serialize_entry("content", content)?;
                serialize_some(&mut map, "is_error", is_error)?;
                serialize_some(&mut map, "blocks", blocks)?;
                serialize_some(&mut map, "cache_reference", cache_reference)?;
                serialize_some(&mut map, "toolset_name", toolset_name)?;
            }
            Self::Thinking { text, signature } => {
                map.serialize_entry("type", "thinking")?;
                map.serialize_entry("text", text)?;
                serialize_some(&mut map, "signature", signature)?;
            }
            Self::RedactedThinking { data } => {
                map.serialize_entry("type", "redacted_thinking")?;
                map.serialize_entry("data", data)?;
            }
            Self::Image { source } => {
                map.serialize_entry("type", "image")?;
                map.serialize_entry("source", source)?;
            }
            Self::Document { source, title } => {
                map.serialize_entry("type", "document")?;
                map.serialize_entry("source", source)?;
                serialize_some(&mut map, "title", title)?;
            }
            Self::Video { source } => {
                map.serialize_entry("type", "video")?;
                map.serialize_entry("source", source)?;
            }
            Self::Audio { format, data } => {
                map.serialize_entry("type", "audio")?;
                map.serialize_entry("format", format)?;
                map.serialize_entry("data", data)?;
            }
        }
        map.end()
    }
}

fn serialize_some<M, T>(map: &mut M, key: &'static str, value: &Option<T>) -> Result<(), M::Error>
where
    M: SerializeMap,
    T: Serialize,
{
    if let Some(value) = value {
        map.serialize_entry(key, value)?;
    }
    Ok(())
}

impl<'de> Deserialize<'de> for ContentBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = Box::<RawValue>::deserialize(deserializer)?;
        decode_content_block(raw.get()).map_err(serde::de::Error::custom)
    }
}

fn decode_content_block(source: &str) -> Result<ContentBlock, String> {
    let scanner = crate::response_json::ResponseJsonScanner::parse(
        source.as_bytes(),
        "content block is not valid JSON",
    )
    .map_err(|error| error.to_string())?;
    let kind = scanner
        .string_at("/type")
        .ok_or_else(|| "content block is missing a string type".to_owned())?;
    let required_string = |pointer: &str| scalar_string_at(&scanner, pointer);
    let optional_string = |pointer: &str| optional_scalar_string_at(&scanner, pointer);
    let optional_payload = |pointer: &str| optional_json_payload_at(&scanner, pointer);
    let input_payload = || payload_at(&scanner, "/input");

    match kind {
        "provider_content" => Ok(ContentBlock::ProviderContent {
            protocol: required_value_at(&scanner, "/protocol")?,
            value: payload_at(&scanner, "/value")?,
        }),
        "text" | "text_js_utf16" => {
            let text_pointer = "/text";
            let text = scanner
                .string_at(text_pointer)
                .ok_or_else(|| "text content block is missing a string text".to_owned())?
                .to_owned();
            let utf16_code_units = scanner
                .string_units_at(text_pointer)
                .ok_or_else(|| "text content block has no UTF-16 source units".to_owned())?;
            let thought_signature = optional_string("/thought_signature")?;
            let citations = optional_nullable_json_payload_at(&scanner, "/citations")?;
            if kind == "text_js_utf16" || String::from_utf16(&utf16_code_units).is_err() {
                Ok(ContentBlock::TextJsUtf16 {
                    text,
                    utf16_code_units,
                    thought_signature,
                    citations,
                })
            } else {
                Ok(ContentBlock::Text {
                    text,
                    thought_signature,
                    citations,
                })
            }
        }
        "tool_use" => Ok(ContentBlock::ToolUse {
            id: required_value_at(&scanner, "/id")?,
            name: required_string("/name")?,
            input: input_payload()?,
            provider_id: optional_string("/provider_id")?,
            caller: optional_payload("/caller")?,
            toolset_name: optional_string("/toolset_name")?,
            thought_signature: optional_string("/thought_signature")?,
        }),
        "tool_result" => Ok(ContentBlock::ToolResult {
            tool_use_id: required_value_at(&scanner, "/tool_use_id")?,
            content: {
                let content = payload_at(&scanner, "/content")?;
                content
                    .validate_root_string("tool result content")
                    .map_err(|error| error.to_string())?;
                content
            },
            is_error: optional_required_bool_at(&scanner, "/is_error")?,
            blocks: optional_json_payload_at(&scanner, "/blocks")?
                .map(|blocks| {
                    blocks
                        .validate_root_array("tool result blocks")
                        .map_err(|error| error.to_string())?;
                    Ok::<_, String>(blocks)
                })
                .transpose()?,
            cache_reference: optional_string("/cache_reference")?,
            toolset_name: optional_string("/toolset_name")?,
        }),
        "native" => Ok(ContentBlock::Native {
            value: serde_json::from_str(raw_field_at(&scanner, "/value")?)
                .map_err(|error| error.to_string())?,
        }),
        "thinking" => Ok(ContentBlock::Thinking {
            text: required_string("/text")?,
            signature: optional_string("/signature")?,
        }),
        "redacted_thinking" => Ok(ContentBlock::RedactedThinking {
            data: required_string("/data")?,
        }),
        "image" => Ok(ContentBlock::Image {
            source: required_value_at(&scanner, "/source")?,
        }),
        "document" => Ok(ContentBlock::Document {
            source: required_value_at(&scanner, "/source")?,
            title: optional_string("/title")?,
        }),
        "video" => Ok(ContentBlock::Video {
            source: required_value_at(&scanner, "/source")?,
        }),
        "audio" => Ok(ContentBlock::Audio {
            format: required_string("/format")?,
            data: required_string("/data")?,
        }),
        other => Err(format!("unknown content block type {other:?}")),
    }
}

fn raw_field_at<'a>(
    scanner: &'a crate::response_json::ResponseJsonScanner,
    pointer: &str,
) -> Result<&'a str, String> {
    scanner
        .raw_source_at(pointer)
        .ok_or_else(|| format!("required JSON field {pointer} is missing"))
}

fn required_value_at<T: DeserializeOwned>(
    scanner: &crate::response_json::ResponseJsonScanner,
    pointer: &str,
) -> Result<T, String> {
    serde_json::from_str(raw_field_at(scanner, pointer)?).map_err(|error| error.to_string())
}

fn optional_json_payload_at(
    scanner: &crate::response_json::ResponseJsonScanner,
    pointer: &str,
) -> Result<Option<JsonPayload>, String> {
    scanner
        .raw_source_at(pointer)
        .map(|source| JsonPayload::parse(source).map_err(|error| error.to_string()))
        .transpose()
}

fn optional_nullable_json_payload_at(
    scanner: &crate::response_json::ResponseJsonScanner,
    pointer: &str,
) -> Result<Option<Option<JsonPayload>>, String> {
    let Some(source) = scanner.raw_source_at(pointer) else {
        return Ok(None);
    };
    if scanner.kind_at(pointer) == Some(crate::response_json::FlatJsonKind::Null) {
        Ok(Some(None))
    } else {
        JsonPayload::parse(source)
            .map(|value| Some(Some(value)))
            .map_err(|error| error.to_string())
    }
}

fn optional_required_bool_at(
    scanner: &crate::response_json::ResponseJsonScanner,
    pointer: &str,
) -> Result<Option<bool>, String> {
    scanner
        .raw_source_at(pointer)
        .map(|source| serde_json::from_str(source).map_err(|error| error.to_string()))
        .transpose()
}

fn scalar_string_at(
    scanner: &crate::response_json::ResponseJsonScanner,
    pointer: &str,
) -> Result<String, String> {
    let units = scanner
        .string_units_at(pointer)
        .ok_or_else(|| format!("required string field {pointer} is missing"))?;
    String::from_utf16(&units)
        .map_err(|_| format!("field {pointer} contains unpaired UTF-16 units"))
}

fn optional_scalar_string_at(
    scanner: &crate::response_json::ResponseJsonScanner,
    pointer: &str,
) -> Result<Option<String>, String> {
    if scanner.raw_source_at(pointer).is_none()
        || scanner.kind_at(pointer) == Some(crate::response_json::FlatJsonKind::Null)
    {
        Ok(None)
    } else {
        scalar_string_at(scanner, pointer).map(Some)
    }
}

fn payload_at(
    scanner: &crate::response_json::ResponseJsonScanner,
    pointer: &str,
) -> Result<JsonPayload, String> {
    let source = raw_field_at(scanner, pointer)?;
    JsonPayload::parse(source).map_err(|error| error.to_string())
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
    pub fn write_metadata(&self, metadata: &mut JsonPayload) -> Result<(), &'static str> {
        if metadata.is_root_null() {
            *metadata = JsonPayload::parse("{}").expect("empty object is valid JSON");
        }
        if !metadata.is_root_object() {
            return Err("OpenRouter Chat audio configuration requires object request metadata");
        }
        #[derive(Serialize)]
        struct AudioMetadata<'a> {
            voice: &'a str,
            format: &'a str,
        }
        let audio = serde_json::to_string(&AudioMetadata {
            voice: &self.voice,
            format: self.format.as_str(),
        })
        .map_err(|_| "OpenRouter Chat audio configuration could not be serialized")?;
        *metadata = metadata
            .with_canonical_fields(
                crate::exact_json::JsonEncoding::Serde,
                &[("openrouter_chat_audio", &audio)],
            )
            .map_err(|_| "OpenRouter Chat audio configuration could not be added to metadata")?;
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
                citations: None,
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
                citations: None,
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
    pub fn tool_uses(
        &self,
    ) -> impl Iterator<Item = (&ToolUseId, Option<&str>, &str, &JsonPayload)> {
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
                ContentBlock::Text { text, .. } | ContentBlock::TextJsUtf16 { text, .. } => {
                    Some(text.clone())
                }
                ContentBlock::ProviderContent { value, .. } => {
                    let scanner = value.scanner().ok()?;
                    scanner.string_at("/text").map(str::to_owned)
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
            cache_reference: None,
            tool_use_id: ToolUseId::new("toolu_1"),
            content: "ok".into(),
            is_error: Some(false),
            blocks: None,
            toolset_name: None,
        };
        let s = serde_json::to_string(&b).unwrap();
        assert!(!s.contains("blocks"));
        assert!(!s.contains("toolset_name"));
        assert_eq!(serde_json::from_str::<ContentBlock>(&s).unwrap(), b);

        let content = JsonPayload::string_from_utf16_code_units(&[0xD800]);
        let blocks = JsonPayload::parse(r#"[{"type":"text","text":"\udfff"}]"#).unwrap();
        let exact = ContentBlock::ToolResult {
            tool_use_id: ToolUseId::new("toolu_2"),
            content,
            is_error: None,
            blocks: Some(blocks),
            cache_reference: None,
            toolset_name: None,
        };
        let encoded = serde_json::to_string(&exact).unwrap();
        assert!(encoded.contains(r#""content":"\ud800""#));
        assert!(encoded.contains(r#""blocks":[{"type":"text","text":"\udfff"}]"#));
        let decoded: ContentBlock = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, exact);

        let invalid_content = ContentBlock::ToolResult {
            tool_use_id: ToolUseId::new("toolu_invalid"),
            content: JsonPayload::parse("{}").unwrap(),
            is_error: None,
            blocks: None,
            cache_reference: None,
            toolset_name: None,
        };
        assert!(serde_json::to_string(&invalid_content).is_err());
        let invalid_blocks = ContentBlock::ToolResult {
            tool_use_id: ToolUseId::new("toolu_invalid"),
            content: JsonPayload::string("fallback"),
            is_error: None,
            blocks: Some(JsonPayload::parse("{}").unwrap()),
            cache_reference: None,
            toolset_name: None,
        };
        assert!(serde_json::to_string(&invalid_blocks).is_err());
    }

    #[test]
    fn text_citations_preserve_absence_null_and_value() {
        for (input, expected) in [
            (r#"{"type":"text","text":"x"}"#, None),
            (
                r#"{"type":"text","text":"x","citations":null}"#,
                Some("null"),
            ),
            (r#"{"type":"text","text":"x","citations":[]}"#, Some("[]")),
        ] {
            let block: ContentBlock = serde_json::from_str(input).unwrap();
            let output = serde_json::to_string(&block).unwrap();
            match expected {
                Some(expected) => assert!(output.ends_with(&format!("\"citations\":{expected}}}"))),
                None => assert!(!output.contains("citations")),
            }
        }

        let exact_source = r#"{"type":"text","text":"x","citations":{"future":"\ud800"}}"#;
        let exact: ContentBlock = serde_json::from_str(exact_source).unwrap();
        let ContentBlock::Text {
            citations: Some(Some(citations)),
            ..
        } = exact
        else {
            panic!("citation object must remain present")
        };
        assert_eq!(citations.as_json_source(), r#"{"future":"\ud800"}"#);
        assert_eq!(
            serde_json::to_string(&ContentBlock::Text {
                text: "x".into(),
                thought_signature: None,
                citations: Some(Some(citations)),
            })
            .unwrap(),
            exact_source
        );
    }

    #[test]
    fn tagged_tool_input_and_caller_keep_deep_and_lone_source_without_value_trees() {
        let depth = 16_384;
        let mut input_source = String::with_capacity(depth * 2 + 1);
        input_source.extend(std::iter::repeat_n('[', depth));
        input_source.push('0');
        input_source.extend(std::iter::repeat_n(']', depth));
        let source = format!(
            r#"{{"type":"tool_use","id":"toolu_deep","name":"read","input":{input_source},"caller":{{"future":"\ud800"}}}}"#
        );
        let block: ContentBlock = serde_json::from_str(&source).unwrap();
        let ContentBlock::ToolUse {
            input,
            caller: Some(caller),
            ..
        } = &block
        else {
            panic!("expected tool use")
        };
        assert_eq!(input.as_json_source(), input_source);
        assert_eq!(caller.as_json_source(), r#"{"future":"\ud800"}"#);
        assert_eq!(serde_json::to_string(&block).unwrap(), source);
    }

    #[test]
    fn tool_result_is_error_preserves_omission_and_explicit_false() {
        let absent: ContentBlock = serde_json::from_str(
            r#"{"type":"tool_result","tool_use_id":"toolu_1","content":"ok"}"#,
        )
        .unwrap();
        assert!(matches!(
            absent,
            ContentBlock::ToolResult { is_error: None, .. }
        ));
        assert!(!serde_json::to_string(&absent).unwrap().contains("is_error"));

        let explicit_false: ContentBlock = serde_json::from_str(
            r#"{"type":"tool_result","tool_use_id":"toolu_1","content":"ok","is_error":false}"#,
        )
        .unwrap();
        assert!(matches!(
            explicit_false,
            ContentBlock::ToolResult {
                is_error: Some(false),
                ..
            }
        ));
        assert!(serde_json::to_string(&explicit_false)
            .unwrap()
            .ends_with("\"is_error\":false}"));

        let explicit_true: ContentBlock = serde_json::from_str(
            r#"{"type":"tool_result","tool_use_id":"toolu_1","content":"failed","is_error":true}"#,
        )
        .unwrap();
        assert!(matches!(
            explicit_true,
            ContentBlock::ToolResult {
                is_error: Some(true),
                ..
            }
        ));
        assert!(serde_json::to_string(&explicit_true)
            .unwrap()
            .ends_with("\"is_error\":true}"));

        assert!(serde_json::from_str::<ContentBlock>(
            r#"{"type":"tool_result","tool_use_id":"toolu_1","content":"invalid","is_error":null}"#
        )
        .is_err());
    }

    #[test]
    fn client_toolset_names_round_trip_and_null_decodes_as_absent() {
        let use_block = ContentBlock::ToolUse {
            id: ToolUseId::new("toolu_browser"),
            name: "screenshot".into(),
            input: JsonPayload::parse(r#"{"tab_id":"1"}"#).unwrap(),
            provider_id: None,
            caller: None,
            toolset_name: Some("browser".into()),
            thought_signature: None,
        };
        let use_json = serde_json::to_string(&use_block).unwrap();
        assert!(use_json.contains("\"toolset_name\":\"browser\""));
        assert_eq!(
            serde_json::from_str::<ContentBlock>(&use_json).unwrap(),
            use_block
        );

        let result_block = ContentBlock::ToolResult {
            cache_reference: None,
            tool_use_id: ToolUseId::new("toolu_browser"),
            content: "done".into(),
            is_error: Some(false),
            blocks: None,
            toolset_name: Some("browser".into()),
        };
        let result_json = serde_json::to_string(&result_block).unwrap();
        assert!(result_json.contains("\"toolset_name\":\"browser\""));
        assert_eq!(
            serde_json::from_str::<ContentBlock>(&result_json).unwrap(),
            result_block
        );

        let cached_result = ContentBlock::ToolResult {
            tool_use_id: ToolUseId::new("toolu_cached"),
            content: JsonPayload::string("done"),
            is_error: None,
            blocks: None,
            cache_reference: Some("cache_ref_1".into()),
            toolset_name: None,
        };
        let cached_json = serde_json::to_string(&cached_result).unwrap();
        assert!(cached_json.contains("\"cache_reference\":\"cache_ref_1\""));
        assert_eq!(
            serde_json::from_str::<ContentBlock>(&cached_json).unwrap(),
            cached_result
        );

        let null_use = r#"{"type":"tool_use","id":"toolu_legacy","name":"lookup","input":{},"toolset_name":null}"#;
        assert!(matches!(
            serde_json::from_str::<ContentBlock>(null_use).unwrap(),
            ContentBlock::ToolUse {
                toolset_name: None,
                ..
            }
        ));
        let omitted_result = r#"{"type":"tool_result","tool_use_id":"toolu_legacy","content":"done","is_error":false}"#;
        assert!(matches!(
            serde_json::from_str::<ContentBlock>(omitted_result).unwrap(),
            ContentBlock::ToolResult {
                toolset_name: None,
                ..
            }
        ));
        let null_result = r#"{"type":"tool_result","tool_use_id":"toolu_legacy","content":"done","is_error":false,"toolset_name":null}"#;
        assert!(matches!(
            serde_json::from_str::<ContentBlock>(null_result).unwrap(),
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
            input: JsonPayload::parse(r#"{"tab_id":"1"}"#).unwrap(),
            provider_id: None,
            caller: None,
            toolset_name: Some("browser".into()),
            thought_signature: None,
        }]);

        let (id, toolset_name, name, input) = message.tool_uses().next().unwrap();
        assert_eq!(id.as_str(), "toolu_browser");
        assert_eq!(toolset_name, Some("browser"));
        assert_eq!(name, "screenshot");
        assert_eq!(input.as_json_source(), r#"{"tab_id":"1"}"#);
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

        let legacy: ConversationMessage =
            serde_json::from_str(r#"{"role":"user","content":[{"type":"text","text":"hello"}]}"#)
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

    #[test]
    fn tagged_tool_input_serde_preserves_deep_json_and_lone_utf16() {
        let depth = 16_384;
        let mut deep = String::with_capacity(depth * 2 + 1);
        deep.extend(std::iter::repeat_n('[', depth));
        deep.push('0');
        deep.extend(std::iter::repeat_n(']', depth));
        let input_source = format!(r#"{{"\ud800":"\udfff","deep":{deep}}}"#);
        let wire =
            format!(r#"{{"type":"tool_use","id":"call","name":"read","input":{input_source}}}"#);

        let message_source =
            format!(r#"{{"role":"assistant","content":[{wire}],"native_options":[]}}"#);
        let message: ConversationMessage = serde_json::from_str(&message_source).unwrap();
        let ContentBlock::ToolUse { input, .. } = &message.content[0] else {
            panic!("expected a tool call")
        };
        assert_eq!(input.as_json_source(), input_source);

        let serialized = serde_json::to_string(&message).unwrap();
        assert!(serialized.contains(r#""input":{"\ud800":"\udfff","deep":["#));
        let round_trip: ConversationMessage = serde_json::from_str(&serialized).unwrap();
        let ContentBlock::ToolUse { input, .. } = &round_trip.content[0] else {
            panic!("expected a tool call after round trip")
        };
        assert_eq!(input.as_json_source(), input_source);
    }

    #[test]
    fn tagged_provider_content_serde_keeps_actual_json_value_shape() {
        let wire = r#"{"type":"provider_content","protocol":"anthropic_messages","value":{"type":"server_tool_use","name":"web_search","input":{"query":"x","lone":"\ud800"}}}"#;
        let block: ContentBlock = serde_json::from_str(wire).unwrap();
        let ContentBlock::ProviderContent { value, .. } = &block else {
            panic!("expected provider content")
        };
        assert_eq!(
            value.as_json_source(),
            r#"{"type":"server_tool_use","name":"web_search","input":{"query":"x","lone":"\ud800"}}"#
        );
        assert_eq!(serde_json::to_string(&block).unwrap(), wire);
    }
}
