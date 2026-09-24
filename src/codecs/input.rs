//! Pure request data passed to wire codecs. No credentials or execution state.
use crate::protocol::{
    CompletionRequest, CredentialConfig, ModelProfile, ProviderInfo, ProviderProfile,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestMode {
    Complete,
    Stream,
}

/// A connection snapshot containing only the data used to encode a request.
/// Credential configuration and prices are removed when the snapshot is made.
#[derive(Debug, Clone)]
pub struct CodecContext {
    pub(crate) profile: ProviderProfile,
    pub(crate) request_model: String,
    pub(crate) stream: bool,
    pub(crate) file_account_scope: Option<String>,
}
impl CodecContext {
    /// Build a context from a wire ID. Use [`Self::for_model`] when selecting
    /// one of several catalog rows that share that ID.
    pub fn new(
        profile: &ProviderProfile,
        request_model: impl Into<String>,
        mode: RequestMode,
    ) -> Self {
        let request_model = request_model.into();
        Self::from_models(
            profile,
            &request_model,
            profile
                .models
                .iter()
                .filter(|model| model.request_model == request_model),
            mode,
        )
    }
    /// Encode one selected catalog row, including when multiple rows share a wire ID.
    pub fn for_model(profile: &ProviderProfile, model: &ModelProfile, mode: RequestMode) -> Self {
        Self::from_models(profile, &model.request_model, std::iter::once(model), mode)
    }
    fn from_models<'a>(
        profile: &ProviderProfile,
        request_model: &str,
        models: impl Iterator<Item = &'a ModelProfile>,
        mode: RequestMode,
    ) -> Self {
        let connection = ProviderProfile {
            inference: profile.inference.clone(),
            regions: Vec::new(),
            provider_id: profile.provider_id.clone(),
            profile_name: profile.profile_name.clone(),
            base_url: profile.base_url.clone(),
            protocol: profile.protocol,
            model_list: Default::default(),
            auth: profile.auth,
            credential: CredentialConfig::None,
            models: models
                .map(|model| {
                    let mut model = model.clone();
                    model.pricing = None;
                    model.info.pricing = None;
                    model.billing_mode = None;
                    model
                })
                .collect(),
            pricing: Default::default(),
            signing: profile.signing.clone(),
            azure: profile.azure.clone(),
            supports_websockets: false,
            supports_websocket_compression: false,
            websocket_connect_timeout_ms: None,
            connection: Default::default(),
            info: ProviderInfo {
                features: profile.info.features.clone(),
                ..Default::default()
            },
            extra: profile.extra.clone(),
        };
        Self {
            profile: connection,
            request_model: request_model.to_owned(),
            stream: mode == RequestMode::Stream,
            file_account_scope: None,
        }
    }
    pub fn with_file_scope(mut self, scope: Option<&str>) -> Self {
        self.file_account_scope = scope.map(str::to_owned);
        self
    }
    pub fn profile(&self) -> &ProviderProfile {
        &self.profile
    }
    pub fn request_model(&self) -> &str {
        &self.request_model
    }
    pub fn mode(&self) -> RequestMode {
        if self.stream {
            RequestMode::Stream
        } else {
            RequestMode::Complete
        }
    }
    pub fn file_scope(&self) -> Option<&str> {
        self.file_account_scope.as_deref()
    }
}

/// Borrowed model input. Media bindings are prepared separately from the
/// caller-owned conversation and never carry upload/cache or credential state.
#[derive(Clone, Copy)]
pub struct EncodeRequest<'a> {
    request: &'a CompletionRequest,
    media: &'a [PreparedMedia<'a>],
    bindings: &'a [ContentBinding<'a>],
}
impl<'a> EncodeRequest<'a> {
    pub fn new(request: &'a CompletionRequest) -> Self {
        Self {
            request,
            media: &[],
            bindings: &[],
        }
    }
    pub fn request(self) -> &'a CompletionRequest {
        self.request
    }
    pub fn blocks(self) -> impl Iterator<Item = &'a crate::protocol::ContentBlock> {
        self.request
            .messages
            .iter()
            .flat_map(|m| &m.content)
            .map(move |block| self.block(block))
    }
    pub fn media(self) -> &'a [PreparedMedia<'a>] {
        self.media
    }
}

/// Immutable attachment data resolved by the caller or attachment service.
#[derive(Clone, Copy)]
pub struct PreparedMedia<'a> {
    pub attachment: &'a crate::protocol::AttachmentRef,
    pub bytes: &'a [u8],
}

/// An attempt-local replacement for a single caller-owned content block.
/// Pointer identity associates a binding with its original block without cloning the conversation.
pub struct ContentBinding<'a> {
    pub original: &'a crate::protocol::ContentBlock,
    pub replacement: crate::protocol::ContentBlock,
}
impl<'a> EncodeRequest<'a> {
    pub fn with_media(mut self, media: &'a [PreparedMedia<'a>]) -> Self {
        self.media = media;
        self
    }
    pub fn with_bindings(mut self, bindings: &'a [ContentBinding<'a>]) -> Self {
        self.bindings = bindings;
        self
    }
    pub fn block<'b>(
        self,
        original: &'b crate::protocol::ContentBlock,
    ) -> &'b crate::protocol::ContentBlock
    where
        'a: 'b,
    {
        self.bindings
            .iter()
            .find(|binding| std::ptr::eq(binding.original, original))
            .map_or(original, |binding| &binding.replacement)
    }
    pub fn inline_media(
        self,
        block: &crate::protocol::ContentBlock,
    ) -> Result<Option<PreparedMedia<'a>>, crate::protocol::LlmError> {
        use crate::protocol::{ContentBlock, DocumentSource, ImageSource, LlmError, VideoSource};
        let attachment = match block {
            ContentBlock::Image {
                source: ImageSource::Attachment { attachment },
            }
            | ContentBlock::Document {
                source: DocumentSource::Attachment { attachment },
                ..
            }
            | ContentBlock::Video {
                source: VideoSource::Attachment { attachment },
            } => attachment,
            _ => return Ok(None),
        };
        let media = self
            .media
            .iter()
            .find(|media| media.attachment == attachment)
            .copied()
            .ok_or_else(super::unresolved_attachment_error)?;
        if media.bytes.len() as u64 != attachment.size_bytes {
            return Err(LlmError::InvalidRequest {
                message: "prepared media length does not match attachment metadata".into(),
            });
        }
        Ok(Some(media))
    }
}
