//! Image generation contracts, independent of conversation messages.

use super::{AttachmentRef, Secret};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageApi {
    OpenAi,
    Gemini,
    Qwen,
    Wan,
    Xai,
    Minimax,
    Zai,
    OpenRouter,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageRouteConfig {
    pub api: ImageApi,
    /// Full API root; no path is derived from the chat connection URL.
    pub base_url: String,
    /// An optional separate native task API root, used by Qwen.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task_base_url: Option<String>,
    /// The explicit API-key header, if it is not Authorization: Bearer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_header: Option<String>,
    /// Registered custom endpoint authenticator, when built-in key placement is insufficient.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub authenticator: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageCapabilities {
    pub text_to_image: bool,
    pub reference_generation: bool,
    pub editing: bool,
    pub mask_editing: bool,
    pub async_generate: bool,
    pub async_edit: bool,
    pub max_inputs: Option<u32>,
    pub max_outputs: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageModelProfile {
    pub display_model: String,
    pub request_model: String,
    pub route: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub hidden: bool,
    #[serde(default)]
    pub capabilities: ImageCapabilities,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageServiceConfig {
    pub routes: BTreeMap<String, ImageRouteConfig>,
    pub models: Vec<ImageModelProfile>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageInput {
    Url { url: String },
    Base64 { media_type: String, data: String },
    Attachment { attachment: AttachmentRef },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageReferenceKind {
    General,
    Character,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageReference {
    pub kind: ImageReferenceKind,
    pub image: ImageInput,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSize {
    Auto,
    Pixels { width: u32, height: u32 },
    Tier { name: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    Png,
    Jpeg,
    Webp,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageQuality {
    Auto,
    Low,
    Medium,
    High,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ImageOutputOptions {
    pub count: Option<u32>,
    pub size: Option<ImageSize>,
    pub aspect_ratio: Option<(u32, u32)>,
    pub format: Option<ImageFormat>,
    pub quality: Option<ImageQuality>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub references: Vec<ImageReference>,
    #[serde(default)]
    pub output: ImageOutputOptions,
    #[serde(default)]
    pub provider_options: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageMask {
    /// Index of the masked input. OpenAI supports only index zero.
    pub image_index: usize,
    pub source: ImageInput,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageEditRequest {
    pub model: String,
    pub prompt: String,
    pub images: Vec<ImageInput>,
    #[serde(default)]
    pub mask: Option<ImageMask>,
    #[serde(default)]
    pub output: ImageOutputOptions,
    #[serde(default)]
    pub provider_options: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageRequest {
    Generate(ImageGenerationRequest),
    Edit(ImageEditRequest),
}

impl ImageRequest {
    pub fn model(&self) -> &str {
        match self {
            Self::Generate(request) => &request.model,
            Self::Edit(request) => &request.model,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageData {
    Url { url: String },
    Base64 { data: String },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageArtifact {
    pub data: ImageData,
    #[serde(default)]
    pub media_type: Option<String>,
    #[serde(default)]
    pub revised_prompt: Option<String>,
    #[serde(default)]
    pub expires_at: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageOutcome {
    Complete,
    Partial,
    Blocked,
    NoImage,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageResponse {
    pub outcome: ImageOutcome,
    pub images: Vec<ImageArtifact>,
    pub text: Option<String>,
    pub requested_model: String,
    pub reported_model: Option<String>,
    pub executed_profile: String,
    pub provider_id: String,
    pub request_id: Option<String>,
    pub usage: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageTaskRef {
    pub version: u8,
    pub provider_id: String,
    pub profile_name: String,
    pub route: String,
    pub endpoint_fingerprint: String,
    pub account_scope: String,
    pub request_model: String,
    pub provider_task_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum ImageTaskSnapshot {
    Pending,
    Running,
    Succeeded { response: Box<ImageResponse> },
    Failed { message: String },
    Cancelled,
    Expired,
    Unknown { raw_status: String },
}

#[derive(Debug, Clone, Default)]
pub struct ImageRequestOptions {
    pub credential: Option<Secret<String>>,
    pub account_scope: Option<String>,
    pub total_timeout: Option<Duration>,
    pub max_response_bytes: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageModelListing {
    pub profile_name: String,
    pub provider_id: String,
    pub display_model: String,
    pub request_model: String,
    pub capabilities: ImageCapabilities,
}
