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
        decode_page(resp).map(|decoded| decoded.page)
    }

    fn decode_page_with_exclusions(
        &self,
        resp: &HttpResponse,
    ) -> Result<DecodedModelPage, LlmError> {
        decode_page(resp)
    }
}

/// Google's list shape includes models for different API operations. This
/// client only calls `generateContent`, so a complete methods list that omits
/// it is explicit evidence that the model cannot serve this directory's use.
/// Older-compatible endpoints sometimes omit the field; those rows remain
/// discoverable because absence is unknown, not a negative capability.
fn decode_page(resp: &HttpResponse) -> Result<DecodedModelPage, LlmError> {
    let body = ok_or_classified(resp, crate::codecs::gemini::classify_error)?;
    let mut models = Vec::new();
    let mut incompatible_model_ids = Vec::new();
    let mut explicitly_compatible_model_ids = Vec::new();
    for row in rows(&body, "models")? {
        // The id here is a resource path. A request carries the bare id, so the
        // collection prefix is stripped once here rather than by every caller.
        let name = required_id(row, "name")?;
        let id = name.strip_prefix("models/").unwrap_or(&name).to_owned();
        match supports_generate_content(row)? {
            Some(false) => {
                incompatible_model_ids.push(id);
                continue;
            }
            Some(true) => explicitly_compatible_model_ids.push(id.clone()),
            None => {}
        }
        models.push(LiveModel {
            inference_features: None,
            request_model: id,
            display_name: text(row, "displayName"),
            description: text(row, "description"),
            context_window: number(row, "inputTokenLimit"),
            max_output_tokens: number(row, "outputTokenLimit"),
        });
    }
    Ok(DecodedModelPage {
        page: ModelPage {
            models,
            next_cursor: next_page_cursor(&body)?,
        },
        incompatible_model_ids,
        explicitly_compatible_model_ids,
    })
}

fn supports_generate_content(row: &Value) -> Result<Option<bool>, LlmError> {
    let Some(methods) = row.get("supportedGenerationMethods") else {
        return Ok(None);
    };
    let methods = methods
        .as_array()
        .ok_or_else(|| LlmError::ProviderInternal {
            message: "a Gemini model's supportedGenerationMethods must be an array of strings"
                .into(),
        })?;
    let mut supports = false;
    for method in methods {
        let method = method.as_str().ok_or_else(|| LlmError::ProviderInternal {
            message: "a Gemini model's supportedGenerationMethods must contain only strings".into(),
        })?;
        supports |= method == "generateContent";
    }
    Ok(Some(supports))
}

fn next_page_cursor(body: &Value) -> Result<Option<String>, LlmError> {
    match body.get("nextPageToken") {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(token)) if token.is_empty() => Ok(None),
        Some(Value::String(token)) => Ok(Some(token.clone())),
        Some(_) => Err(LlmError::ProviderInternal {
            message: "the Gemini model directory's nextPageToken must be a string".into(),
        }),
    }
}
