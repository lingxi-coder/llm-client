use super::*;

/// The shape that paginates with an id cursor: `data`, `has_more`, `last_id`,
/// and `after_id` to continue.
#[derive(Debug, Default, Clone, Copy)]
pub struct AnthropicMessagesDirectory;

impl ModelDirectory for AnthropicMessagesDirectory {
    fn shape(&self) -> ProtocolFamily {
        ProtocolFamily::AnthropicMessages
    }

    fn list_request(&self, profile: &ProviderProfile, cursor: Option<&str>) -> HttpRequest {
        let url = endpoint(profile, "/v1/models");
        let url = match cursor {
            Some(after) => with_query(&url, &[("limit", PAGE_SIZE), ("after_id", after)]),
            None => with_query(&url, &[("limit", PAGE_SIZE)]),
        };
        let mut req = get(url, profile);
        // The Models API requires the same version header as Messages.
        req.headers
            .retain(|(name, _)| !name.eq_ignore_ascii_case("anthropic-version"));
        req.headers.push((
            "anthropic-version".to_owned(),
            profile
                .extra
                .get("api_version")
                .and_then(Value::as_str)
                .unwrap_or(crate::codecs::anthropic::DEFAULT_API_VERSION)
                .to_owned(),
        ));
        req
    }

    fn decode_page(&self, resp: &HttpResponse) -> Result<ModelPage, LlmError> {
        let body = ok_or_classified(resp, crate::codecs::anthropic::classify_error)?;
        let models = rows(&body, "data")?
            .iter()
            .map(|row| {
                Ok(LiveModel {
                    request_model: required_id(row, "id")?,
                    display_name: text(row, "display_name"),
                    description: None,
                    context_window: None,
                    max_output_tokens: None,
                })
            })
            .collect::<Result<Vec<_>, LlmError>>()?;
        // `has_more` is what ends the walk; `last_id` alone is sent on the
        // final page too, so following it would loop forever.
        let next_cursor = match body.get("has_more").and_then(Value::as_bool) {
            Some(true) => cursor(&body, "last_id"),
            _ => None,
        };
        Ok(ModelPage {
            models,
            next_cursor,
        })
    }
}
