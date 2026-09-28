//! Provider dispatch policy. Runtime owns I/O; providers own request semantics.
use crate::client::{FirstPartyEndpoint, RequestOptions, ResolvedRequest};
use crate::codecs::{CodecContext, PreparedMedia, RequestMode, WireCodec};
use crate::protocol::{
    ChatRequest, ContentBlock, LlmError, ProtocolFamily, ProviderProfile, ResponseCacheObservation,
    Secret, StreamEvent,
};
use crate::transport::HttpRequest;
use std::collections::BTreeMap;
use std::sync::Arc;

/// A small chat capability contract. Implementations use the caller's registry,
/// never instantiate a codec, transport, authenticator, or a second client.
pub(crate) trait ChatBackend: Send + Sync {
    fn codec(
        &self,
        profile: &ProviderProfile,
        codecs: &BTreeMap<ProtocolFamily, Arc<dyn WireCodec>>,
    ) -> Result<Arc<dyn WireCodec>, LlmError> {
        codecs
            .get(&profile.protocol)
            .cloned()
            .ok_or_else(|| LlmError::UnsupportedCapability {
                message: format!("no codec for {:?}", profile.protocol),
            })
    }

    fn validate_host(
        &self,
        req: &ChatRequest,
        context: &CodecContext,
        opts: &RequestOptions,
        codec: &dyn WireCodec,
    ) -> Result<(), LlmError> {
        // Typed native options must be validated even on another provider: a
        // compatible codec cannot silently drop options it does not support.
        super::google::chat::validate_host(req, context)?;
        super::anthropic::chat::validate_host(req, context)?;
        crate::codecs::inference::validate(req, context.profile(), context.request_model())?;
        super::qwen::chat::validate_host(req, context)?;
        super::openrouter::chat::validate_host(req, context, opts)?;
        validate_mcp_authorizations(&opts.mcp_authorizations, req, context.profile())?;
        let has_audio = req
            .messages
            .iter()
            .flat_map(|message| &message.content)
            .any(|block| matches!(block, ContentBlock::Audio { .. }))
            || req.metadata.get("openrouter_chat_audio").is_some();
        if has_audio {
            if !super::openrouter::chat::accepts_audio(context)
                && (!super::google::chat::accepts_audio(context)
                    || req.metadata.get("openrouter_chat_audio").is_some())
            {
                return Err(LlmError::UnsupportedCapability { message: "Chat audio requires OpenRouter Chat or Gemini GenerateContent; OpenRouter output settings require OpenRouter".into() });
            }
            codec.validate_request(req, context)?;
        }
        Ok(())
    }

    fn validate_preparation(
        &self,
        req: &ChatRequest,
        context: &CodecContext,
        opts: &RequestOptions,
    ) -> Result<(), LlmError> {
        if context.mode() == RequestMode::CountTokens
            && !self.supports_exact_count(context.profile())
        {
            return Err(LlmError::UnsupportedCapability {
                message: "exact token counting is unavailable for this protocol".into(),
            });
        }
        if req.continuation.is_some()
            && context.profile().protocol != ProtocolFamily::OpenAiResponses
        {
            return Err(LlmError::UnsupportedCapability {
                message: format!(
                    "profile {:?} uses {:?}, which cannot continue a Responses id",
                    context.profile().profile_name,
                    context.profile().protocol
                ),
            });
        }
        super::google::chat::validate_host(req, context)?;
        crate::providers::dispatch::validate_output_contract(
            req,
            context.profile(),
            context.request_model(),
        )?;
        super::anthropic::code_execution::validate(req, context)?;
        super::openrouter::chat::validate_host(req, context, opts)?;
        validate_mcp_authorizations(&opts.mcp_authorizations, req, context.profile())
    }

    fn preflight_media(
        &self,
        req: &ChatRequest,
        media: &[PreparedMedia<'_>],
        context: &CodecContext,
        codec: &dyn WireCodec,
    ) -> Result<(), LlmError> {
        super::google::chat::preflight_media(req, media, context, codec)
    }

    fn attachment_budget(
        &self,
        _req: &ResolvedRequest<'_>,
        _context: &CodecContext,
        _opts: &RequestOptions,
        _codec: &dyn WireCodec,
        _endpoint: FirstPartyEndpoint,
    ) -> Result<Option<usize>, LlmError> {
        Ok(None)
    }

    fn apply_request_options(
        &self,
        opts: &RequestOptions,
        profile: &ProviderProfile,
        request: &mut HttpRequest,
    ) -> Result<(), LlmError> {
        super::openrouter::response_cache::apply_openrouter_response_cache(
            opts.openrouter_response_cache,
            profile,
            request,
        )?;
        apply_mcp_authorizations(&opts.mcp_authorizations, profile, request)
    }

    fn validate_prepared_body(
        &self,
        size: usize,
        endpoint: FirstPartyEndpoint,
    ) -> Result<(), LlmError> {
        if endpoint == FirstPartyEndpoint::Anthropic {
            super::anthropic::chat::validate_body_size(size)?;
        }
        Ok(())
    }

    fn validate_sealed_body(&self, profile: &ProviderProfile, size: usize) -> Result<(), LlmError> {
        if profile.protocol == ProtocolFamily::AnthropicMessages {
            super::anthropic::chat::validate_body_size(size)?;
        }
        Ok(())
    }

    fn supports_exact_count(&self, profile: &ProviderProfile) -> bool {
        super::anthropic::chat::supports_exact_count(profile)
    }

    fn response_cache(
        &self,
        _profile: &ProviderProfile,
        _request_url: &str,
        _headers: &[(String, String)],
    ) -> Option<ResponseCacheObservation> {
        None
    }

    fn stream_observation(&self, context: &CodecContext) -> StreamObservation {
        StreamObservation {
            anthropic: super::anthropic::chat::StreamObservation::new(context),
        }
    }

    fn replay_policy(&self, req: &ChatRequest, opts: &RequestOptions) -> ReplayPolicy {
        let remote_mcp = req.remote_mcp_servers().next().is_some()
            || req.xai_remote_mcp_servers().next().is_some();
        let stateful = super::anthropic::mcp::has_mcp(req)
            || super::anthropic::web_fetch::has_fetch(req)
            || super::openrouter::server_tools::has_tools(req)
            || req.hosted_openai_tool_search().is_some()
            || req.hosted_code_interpreter().is_some()
            || super::anthropic::code_execution::has_execution(req);
        let hosted = stateful || remote_mcp || super::google::chat::has_hosted_tools(req);
        ReplayPolicy {
            pin_to_connection: req.continuation.is_some()
                || opts.openrouter_response_cache.is_some()
                || hosted,
            repair_missing_files: req.continuation.is_none() && !hosted,
            allow_failover: !stateful,
        }
    }
}

struct CompatibleBackend;
impl ChatBackend for CompatibleBackend {}
static COMPATIBLE: CompatibleBackend = CompatibleBackend;

/// Select only by stable provider identity. An error from a known provider is
/// returned to the caller, never retried through the compatible implementation.
pub(crate) fn chat(profile: &ProviderProfile) -> &'static dyn ChatBackend {
    match profile.provider_id.as_str() {
        "openai" => &super::openai::chat::BACKEND,
        "anthropic" => &super::anthropic::chat::BACKEND,
        "google" => &super::google::chat::BACKEND,
        "qwen" => &super::qwen::chat::BACKEND,
        "minimax" => &super::minimax::chat::BACKEND,
        "zhipu" => &super::zhipu::chat::BACKEND,
        "kimi" => &super::kimi::chat::BACKEND,
        "xai" => &super::xai::chat::BACKEND,
        "openrouter" => &super::openrouter::chat::BACKEND,
        "deepseek" => &super::deepseek::chat::BACKEND,
        "github-copilot" => &super::github_copilot::chat::BACKEND,
        _ => &COMPATIBLE,
    }
}

pub(crate) struct ReplayPolicy {
    pub(crate) pin_to_connection: bool,
    pub(crate) repair_missing_files: bool,
    pub(crate) allow_failover: bool,
}

#[derive(Default)]
pub(crate) struct StreamObservation {
    anthropic: super::anthropic::chat::StreamObservation,
}
impl StreamObservation {
    pub(crate) fn observe(&mut self, event: &StreamEvent) {
        self.anthropic.observe(event);
    }
    pub(crate) fn anthropic_container(
        &self,
    ) -> Option<&crate::providers::anthropic::types::AnthropicContainerMetadata> {
        self.anthropic.container()
    }
    pub(crate) fn anthropic_usage(&self) -> Option<&serde_json::Value> {
        self.anthropic.usage()
    }
}

pub(crate) fn extra_file_expirations(req: &ChatRequest) -> impl Iterator<Item = &str> {
    super::anthropic::chat::extra_file_expirations(req)
}

pub(crate) fn apply_mcp_authorizations(
    credentials: &BTreeMap<String, Secret<String>>,
    profile: &ProviderProfile,
    request: &mut HttpRequest,
) -> Result<(), LlmError> {
    if (profile.provider_id.as_str() == "anthropic"
        && profile.protocol == ProtocolFamily::AnthropicMessages)
        || profile.protocol == ProtocolFamily::FoundryClaude
    {
        super::anthropic::mcp_authorization::apply_mcp_authorizations(credentials, profile, request)
    } else {
        super::openai::mcp_authorization::apply_mcp_authorizations(credentials, profile, request)
    }
}
pub(crate) fn validate_mcp_authorizations(
    credentials: &BTreeMap<String, Secret<String>>,
    request: &ChatRequest,
    profile: &ProviderProfile,
) -> Result<(), LlmError> {
    crate::providers::anthropic::mcp::validate_profile_extra(profile)?;
    let openai_configs = request.remote_mcp_servers().collect::<Vec<_>>();
    let xai_configs = request.xai_remote_mcp_servers().collect::<Vec<_>>();
    let anthropic_configs = request.anthropic_mcp_servers().collect::<Vec<_>>();
    if !openai_configs.is_empty() || !xai_configs.is_empty() || !anthropic_configs.is_empty() {
        request.validate_hosted_tools()?;
    }
    let configured_families = usize::from(!openai_configs.is_empty())
        + usize::from(!xai_configs.is_empty())
        + usize::from(!anthropic_configs.is_empty());
    if configured_families > 1 {
        return Err(LlmError::UnsupportedCapability {
            message: "Anthropic, OpenAI, and xAI remote MCP configurations cannot be mixed".into(),
        });
    }
    if !openai_configs.is_empty() {
        super::openai::mcp_authorization::validate_mcp_profile(
            profile,
            "openai",
            "remote_mcp",
            "openai_responses",
            "api.openai.com",
        )?;
    }
    if !xai_configs.is_empty() {
        super::openai::mcp_authorization::validate_mcp_profile(
            profile,
            "xai",
            "xai_remote_mcp",
            "xai_responses",
            "api.x.ai",
        )?;
    }
    if !anthropic_configs.is_empty()
        && !crate::providers::anthropic::code_execution::is_official_profile(profile)
        && profile.protocol != ProtocolFamily::FoundryClaude
    {
        return Err(LlmError::UnsupportedCapability {
            message: "typed Anthropic MCP requires Anthropic Messages or Foundry Claude".into(),
        });
    }
    if credentials.is_empty() {
        return Ok(());
    }
    if !anthropic_configs.is_empty() {
        if profile.protocol == ProtocolFamily::FoundryClaude {
            super::anthropic::mcp_authorization::foundry_mcp_endpoint(profile)?;
        }
        let names = anthropic_configs
            .iter()
            .map(|config| config.name())
            .collect::<std::collections::BTreeSet<_>>();
        for (name, credential) in credentials {
            if !names.contains(name.as_str())
                || credential.expose_secret().trim().is_empty()
                || credential.expose_secret().len() > 16 * 1024
                || credential.expose_secret().chars().any(char::is_control)
            {
                return Err(LlmError::InvalidRequest {
                    message:
                        "Anthropic MCP authorization has an unknown server name or invalid secret"
                            .into(),
                });
            }
        }
        return Ok(());
    }
    let expected_provider = if !xai_configs.is_empty() {
        "xai"
    } else {
        "openai"
    };
    let expected_marker = if expected_provider == "xai" {
        ("xai_remote_mcp", "xai_responses", "api.x.ai")
    } else {
        ("remote_mcp", "openai_responses", "api.openai.com")
    };
    if openai_configs.is_empty() && xai_configs.is_empty()
        || super::openai::mcp_authorization::validate_mcp_profile(
            profile,
            expected_provider,
            expected_marker.0,
            expected_marker.1,
            expected_marker.2,
        )
        .is_err()
    {
        return Err(LlmError::UnsupportedCapability {
            message: "remote MCP credentials do not match the active provider profile".into(),
        });
    }
    let labels = request
        .remote_mcp_servers()
        .map(|config| config.server_label())
        .chain(
            request
                .xai_remote_mcp_servers()
                .map(|config| config.server_label()),
        )
        .collect::<std::collections::BTreeSet<_>>();
    for (label, credential) in credentials {
        if !labels.contains(label.as_str())
            || credential.expose_secret().trim().is_empty()
            || credential.expose_secret().len() > 16 * 1024
        {
            return Err(LlmError::InvalidRequest {
                message: "remote MCP authorization has an unknown label or invalid secret".into(),
            });
        }
    }
    Ok(())
}

/// First-party prompt/output contracts also run before attachment resolution.
pub(crate) fn validate_output_contract(
    req: &ChatRequest,
    profile: &ProviderProfile,
    request_model: &str,
) -> Result<(), LlmError> {
    super::qwen::structured::validate_output_contract(req, profile, request_model)?;
    super::deepseek::structured::validate_output_contract(req, profile)
}
