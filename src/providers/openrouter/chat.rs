//! OpenRouter server tools and gateway response-cache policy.
use crate::client::RequestOptions;
use crate::codecs::CodecContext;
use crate::protocol::{ChatRequest, LlmError, ProtocolFamily};
pub(crate) struct Backend;
pub(crate) static BACKEND: Backend = Backend;
impl super::super::dispatch::ChatBackend for Backend {
    fn response_cache(
        &self,
        profile: &crate::protocol::ProviderProfile,
        request_url: &str,
        headers: &[(String, String)],
    ) -> Option<crate::protocol::ResponseCacheObservation> {
        super::response_cache::openrouter_observation(profile, request_url, headers)
    }
}
pub(crate) fn validate_host(
    req: &ChatRequest,
    context: &CodecContext,
    opts: &RequestOptions,
) -> Result<bool, LlmError> {
    super::response_cache::validate_openrouter_response_cache(
        opts.openrouter_response_cache,
        context.profile(),
    )?;
    if super::server_tools::has_tools(req) {
        super::server_tools::validate(req, context.profile(), opts.account_scope.as_deref(), true)
    } else {
        Ok(false)
    }
}
pub(crate) fn accepts_audio(context: &CodecContext) -> bool {
    context.profile().provider_id.as_str() == "openrouter"
        && context.profile().protocol == ProtocolFamily::OpenAiChat
}

pub(crate) fn messages_endpoint(profile: &crate::protocol::ProviderProfile) -> Option<String> {
    (profile.provider_id.as_str() == "openrouter"
        && profile.base_url.trim_end_matches('/') == "https://openrouter.ai/api/v1")
        .then(|| "https://openrouter.ai/api/v1/messages".to_owned())
}
