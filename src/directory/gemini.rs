use super::*;

/// The shape whose list is `models`, whose id is a resource path, and which
/// paginates with an opaque page token.
#[derive(Debug, Default, Clone, Copy)]
pub struct GeminiDirectory;

impl ModelDirectory for GeminiDirectory {
    fn shape(&self) -> ProtocolFamily {
        ProtocolFamily::GeminiGenerateContent
    }

    fn list_request(&self, profile: &ProviderProfile, cursor: Option<&str>) -> HttpRequest {
        let url = endpoint(profile, "/models");
        let url = match cursor {
            Some(token) => with_query(&url, &[("pageSize", PAGE_SIZE), ("pageToken", token)]),
            None => with_query(&url, &[("pageSize", PAGE_SIZE)]),
        };
        get(url, profile)
    }

    fn decode_page(&self, resp: &HttpResponse) -> Result<ModelPage, LlmError> {
        let body = ok_or_classified(resp, crate::codecs::gemini::classify_error)?;
        let models = rows(&body, "models")?
            .iter()
            .map(|row| {
                // The id here is a resource path. A request carries the bare
                // id, so the collection prefix is stripped once, here, rather
                // than by every caller that has to match one against a catalog.
                let name = required_id(row, "name")?;
                let id = name.strip_prefix("models/").unwrap_or(&name).to_owned();
                Ok(LiveModel {
                    request_model: id,
                    display_name: text(row, "displayName"),
                    description: text(row, "description"),
                    context_window: number(row, "inputTokenLimit"),
                    max_output_tokens: number(row, "outputTokenLimit"),
                })
            })
            .collect::<Result<Vec<_>, LlmError>>()?;
        Ok(ModelPage {
            models,
            next_cursor: cursor(&body, "nextPageToken"),
        })
    }
}
