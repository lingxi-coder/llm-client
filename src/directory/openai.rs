use super::*;

/// The shape whose list is a flat `data` array of `{id}` objects, with no
/// pagination: the whole catalog arrives in one response.
///
/// Several endpoints speaking this shape publish more than the id — a name, a
/// blurb, a context length — and those are read when present. The bare shape
/// carries only the id, which is still enough to say what exists.
#[derive(Debug, Default, Clone, Copy)]
pub struct OpenAiChatDirectory;

impl ModelDirectory for OpenAiChatDirectory {
    fn shape(&self) -> ProtocolFamily {
        ProtocolFamily::OpenAiChat
    }

    fn list_request(&self, profile: &ProviderProfile, _cursor: Option<&str>) -> HttpRequest {
        // This shape has no cursor, so there is nothing to carry forward and a
        // caller can never have one to hand back.
        get(endpoint(profile, "/models"), profile)
    }

    fn decode_page(&self, resp: &HttpResponse) -> Result<ModelPage, LlmError> {
        let body = ok_or_classified(resp, crate::codecs::openai::chat::classify_error)?;
        let models = rows(&body, "data")?
            .iter()
            .map(|row| {
                Ok(LiveModel {
                    request_model: required_id(row, "id")?,
                    display_name: text(row, "name"),
                    description: text(row, "description"),
                    context_window: number(row, "context_length"),
                    max_output_tokens: number(row, "max_output_tokens").or_else(|| {
                        row.get("top_provider")
                            .and_then(|t| number(t, "max_completion_tokens"))
                    }),
                })
            })
            .collect::<Result<Vec<_>, LlmError>>()?;
        Ok(ModelPage {
            models,
            next_cursor: None,
        })
    }
}
