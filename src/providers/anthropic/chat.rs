//! Anthropic chat policy, body budgeting and native stream observations.
use crate::client::{
    AttachmentKind, FirstPartyEndpoint, RequestOptions, ResolvedRequest,
    INLINE_IMAGE_PREFERENCE_LIMIT,
};
use crate::codecs::{CodecContext, EncodeRequest, PreparedMedia, WireCodec};
use crate::files;
use crate::protocol::JsonPayload;
use crate::protocol::{
    ChatRequest, LlmError, ProtocolFamily, ProviderFileSource, ProviderProfile, StreamEvent,
};

pub(crate) struct Backend;
pub(crate) static BACKEND: Backend = Backend;
impl super::super::dispatch::ChatBackend for Backend {
    fn attachment_budget(
        &self,
        req: &ResolvedRequest<'_>,
        context: &CodecContext,
        opts: &RequestOptions,
        codec: &dyn WireCodec,
        endpoint: FirstPartyEndpoint,
    ) -> Result<Option<usize>, LlmError> {
        attachment_budget(req, context, opts, codec, endpoint)
    }
}

pub(crate) const MAX_REQUEST_BODY_BYTES: usize = 32_000_000;

pub(crate) fn validate_host(req: &ChatRequest, context: &CodecContext) -> Result<(), LlmError> {
    if req
        .messages
        .iter()
        .any(|message| message.anthropic_options().is_some())
        || crate::providers::anthropic::conversation::has_tool_changes(req)
        || !req.anthropic_client_toolsets().is_empty()
        || (crate::providers::anthropic::conversation::supports_profile(context.profile())
            && req
                .messages
                .iter()
                .any(|message| message.role == crate::protocol::MessageRole::System))
    {
        crate::providers::anthropic::conversation::validate(req, context)?;
        crate::codecs::cache::validate(req, context)?;
    }

    crate::providers::anthropic::web_fetch::validate(req, context)?;
    crate::providers::anthropic::tool_search::validate(req, context)?;
    crate::providers::anthropic::client_toolsets::validate(req, context)?;
    if crate::providers::anthropic::web_fetch::has_fetch(req) {
        crate::codecs::structured::validate(req, context)?;
    }
    crate::providers::anthropic::mcp::validate(req, context)?;
    if crate::providers::anthropic::web_fetch::has_fetch(req)
        || crate::providers::anthropic::mcp::has_mcp(req)
    {
        crate::codecs::cache::validate(req, context)?;
    }
    crate::providers::anthropic::code_execution::validate(req, context)?;
    Ok(())
}

pub(crate) fn extra_file_expirations(req: &ChatRequest) -> impl Iterator<Item = &str> {
    req.hosted_anthropic_code_execution()
        .into_iter()
        .flat_map(|config| &config.files)
        .filter_map(|file| file.expires_at.as_deref())
}

pub(crate) fn validate_body_size(body_len: usize) -> Result<(), LlmError> {
    if body_len > MAX_REQUEST_BODY_BYTES {
        return Err(LlmError::RequestTooLarge {
            message: format!("final Anthropic body exceeds {MAX_REQUEST_BODY_BYTES} bytes"),
        });
    }
    Ok(())
}

pub(crate) fn supports_exact_count(profile: &ProviderProfile) -> bool {
    profile.protocol == ProtocolFamily::AnthropicMessages
}

pub(crate) fn attachment_budget(
    req: &ResolvedRequest<'_>,
    context: &CodecContext,
    opts: &RequestOptions,
    codec: &dyn WireCodec,
    endpoint: FirstPartyEndpoint,
) -> Result<Option<usize>, LlmError> {
    if endpoint != FirstPartyEndpoint::Anthropic || req.attachments.is_empty() {
        return Ok(None);
    }
    let media = req
        .attachments
        .iter()
        .map(|payload| PreparedMedia {
            attachment: &payload.attachment,
            bytes: &payload.bytes,
        })
        .collect::<Vec<_>>();
    let overhead = super::mcp_authorization::anthropic_mcp_authorization_body_overhead(
        req.request,
        &opts.mcp_authorizations,
    )?;
    let inline =
        codec.encoded_body_len(EncodeRequest::new(req.request).with_media(&media), context)?;
    let promote_small_images = inline.saturating_add(overhead) > MAX_REQUEST_BODY_BYTES;
    let projected = projected_anthropic_file_request(
        req,
        context.profile(),
        context.request_model(),
        opts,
        promote_small_images,
    )?;
    let projected = codec
        .encoded_body_len(
            EncodeRequest::new(req.request)
                .with_media(&media)
                .with_bindings(&projected),
            context,
        )?
        .saturating_add(overhead);
    if projected > MAX_REQUEST_BODY_BYTES {
        return Err(LlmError::RequestTooLarge {
            message: format!(
                "Anthropic request body projects to {projected} bytes with bounded file IDs; the limit is {MAX_REQUEST_BODY_BYTES} bytes"
            ),
        });
    }
    Ok(Some(if promote_small_images { 0 } else { usize::MAX }))
}

fn projected_anthropic_file_request<'a>(
    req: &ResolvedRequest<'a>,
    profile: &ProviderProfile,
    model: &str,
    opts: &RequestOptions,
    promote_small_images: bool,
) -> Result<Vec<crate::codecs::ContentBinding<'a>>, LlmError> {
    let mut projected = Vec::new();
    for payload in &req.attachments {
        let capabilities = files::capabilities(profile, model, &payload.attachment.media_type);
        if !capabilities.upload
            || capabilities.model_input == files::ModelFileReference::Unsupported
        {
            continue;
        }
        if payload.kind == AttachmentKind::Image
            && !promote_small_images
            && payload.bytes.len() <= INLINE_IMAGE_PREFERENCE_LIMIT
        {
            continue;
        }
        if payload.kind == AttachmentKind::Video {
            continue;
        }
        let file = ProviderFileSource {
            protocol: profile.protocol,
            provider_id: profile.provider_id.clone(),
            profile_name: profile.profile_name.clone(),
            endpoint_fingerprint: files::provider_file_endpoint_fingerprint(&profile.base_url),
            account_scope: opts.file_account_scope.clone(),
            // The two JSON quote bytes are included in the preflight bound.
            expires_at: None,
            processing_status: None,
            file_id: "x".repeat(files::MAX_AUTOMATIC_ANTHROPIC_FILE_ID_JSON_BYTES - 2),
            uri: None,
            media_type: Some(payload.attachment.media_type.clone()),
            purpose: None,
        };
        projected.push(crate::client::provider_file_binding(
            req.request,
            payload,
            file,
        )?);
    }
    Ok(projected)
}

#[derive(Default)]
pub(crate) struct StreamObservation {
    enabled: bool,
    container: Option<crate::providers::anthropic::types::AnthropicContainerMetadata>,
    usage: Option<JsonPayload>,
    stop_details: Option<JsonPayload>,
    resumable: Option<JsonPayload>,
    fallback: super::fallback_response::FallbackResponse,
}
impl StreamObservation {
    pub(crate) fn new(context: &CodecContext) -> Self {
        Self {
            enabled: super::code_execution::is_official_profile(context.profile())
                || super::code_execution::supports_execution(context),
            ..Self::default()
        }
    }
    pub(crate) fn observe(&mut self, event: &StreamEvent) {
        if let StreamEvent::NativeControl {
            protocol: ProtocolFamily::AnthropicMessages,
            control,
        } = event
        {
            if let Ok(super::fallback_response::FallbackControl::Boundary {
                stop_reason,
                iterations,
                iterations_present,
            }) = control.decode::<super::fallback_response::FallbackControl>()
            {
                self.fallback.observe_boundary(
                    stop_reason.as_deref(),
                    iterations,
                    *iterations_present,
                );
            }
            return;
        }
        let StreamEvent::ProviderEvent {
            protocol: ProtocolFamily::AnthropicMessages,
            payload,
        } = event
        else {
            return;
        };
        if payload.string_is("/type", "message_delta") {
            if let Some(resumable) = super::code_execution::stream_resumable(payload) {
                self.resumable = Some(resumable);
            }
            let stop_reason = payload.string_at("/delta/stop_reason");
            self.fallback.observe_stop_reason(stop_reason.as_deref());
            if let Some(usage) = payload.payload_at("/usage") {
                self.fallback.observe_usage(&usage);
            }
            if let Some(details) = payload.payload_at("/delta/stop_details") {
                self.stop_details = (!details.is_root_null()).then_some(details);
            }
        }
        if super::fallback_response::control_index(payload).is_some() {
            if let Some(block) = payload.payload_at("/content_block") {
                self.fallback.observe_block(&block);
            }
        }
        if !self.enabled {
            return;
        }
        if let Some(usage) = super::code_execution::stream_usage(payload) {
            if let Some(current) = &mut self.usage {
                if let Ok(merged) = current.merge_overwriting_objects(&usage) {
                    *current = merged;
                }
            } else {
                self.usage = Some(usage);
            }
        }
        if let Some(container) = super::code_execution::stream_container(payload) {
            if !container.is_root_null() {
                self.container = Some(
                    crate::providers::anthropic::types::AnthropicContainerMetadata {
                        envelope: container,
                    },
                );
            } else {
                self.container = None;
            }
        }
    }
    pub(crate) fn stop_details(&self) -> Option<&JsonPayload> {
        self.stop_details.as_ref()
    }
    pub(crate) fn fallback(&self) -> Option<&super::fallback_response::FallbackResponse> {
        self.fallback.observed().then_some(&self.fallback)
    }
    pub(crate) fn container(
        &self,
    ) -> Option<&crate::providers::anthropic::types::AnthropicContainerMetadata> {
        self.container.as_ref()
    }
    pub(crate) fn usage(&self) -> Option<&JsonPayload> {
        self.usage.as_ref()
    }
    pub(crate) fn resumable(&self) -> Option<&JsonPayload> {
        self.resumable.as_ref()
    }
}

#[cfg(test)]
mod resumable_observation_tests {
    use super::StreamObservation;
    use crate::protocol::{JsonPayload, ProtocolFamily, StreamEvent};

    #[test]
    fn stream_observation_uses_the_current_boolean_delta_merge() {
        let mut observation = StreamObservation::default();
        let event = |source: &str| StreamEvent::ProviderEvent {
            protocol: ProtocolFamily::AnthropicMessages,
            payload: JsonPayload::parse(source).unwrap(),
        };
        observation.observe(&event(
            r#"{"type":"message_delta","delta":{"resumable":false},"resumable":true}"#,
        ));
        assert_eq!(
            observation.resumable().map(JsonPayload::as_json_source),
            Some("false")
        );
        observation.observe(&event(
            r#"{"type":"message_delta","delta":{"resumable":null},"resumable":true}"#,
        ));
        assert_eq!(
            observation.resumable().map(JsonPayload::as_json_source),
            Some("true")
        );
        observation.observe(&event(
            r#"{"type":"message_delta","delta":{"resumable":"true"},"resumable":2}"#,
        ));
        assert_eq!(
            observation.resumable().map(JsonPayload::as_json_source),
            Some("true")
        );
        observation.observe(&event(
            r#"{"type":"message_start","message":{"resumable":false}}"#,
        ));
        assert_eq!(
            observation.resumable().map(JsonPayload::as_json_source),
            Some("true")
        );
    }
}
