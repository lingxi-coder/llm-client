//! Shared image request validation and transport-neutral wire primitives.
use super::ImageError;
use crate::protocol::*;
use crate::transport::{HttpRequest, HttpResponse};
use bytes::Bytes;
use serde_json::{json, Map, Value};

pub(crate) fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
pub(crate) fn unsupported(message: impl Into<String>) -> LlmError {
    LlmError::UnsupportedCapability {
        message: message.into(),
    }
}
pub(crate) struct ImageRules {
    pub options: &'static [&'static str],
    pub ratio: bool,
    pub size: bool,
    pub count: bool,
    pub format: bool,
    pub quality: bool,
}
pub(crate) fn validate(
    req: &ImageRequest,
    model: &ImageModelProfile,
    rules: &ImageRules,
    asynchronous: bool,
) -> Result<(), ImageError> {
    let (prompt, input_count, output, options, editing, mask) = match req {
        ImageRequest::Generate(r) => (
            &r.prompt,
            r.references.len(),
            &r.output,
            &r.provider_options,
            false,
            false,
        ),
        ImageRequest::Edit(r) => (
            &r.prompt,
            r.images.len(),
            &r.output,
            &r.provider_options,
            true,
            r.mask.is_some(),
        ),
    };
    if prompt.trim().is_empty() {
        return Err(invalid("image prompt is empty").into());
    }
    let caps = &model.capabilities;
    if editing {
        if input_count == 0 || !caps.editing {
            return Err(unsupported("model does not support image editing").into());
        }
    } else if input_count == 0 {
        if !caps.text_to_image {
            return Err(unsupported("model does not support text-to-image").into());
        }
    } else if !caps.reference_generation {
        return Err(unsupported("model does not support reference generation").into());
    }
    if mask && !caps.mask_editing {
        return Err(unsupported("model does not support mask editing").into());
    }
    if asynchronous
        && if editing {
            !caps.async_edit
        } else {
            !caps.async_generate
        }
    {
        return Err(unsupported("model has no native asynchronous image task endpoint").into());
    }
    if caps
        .max_inputs
        .is_some_and(|max| input_count > max as usize)
    {
        return Err(invalid("too many input images").into());
    }
    if output
        .count
        .is_some_and(|count| count == 0 || caps.max_outputs.is_some_and(|max| count > max))
    {
        return Err(invalid("image count is outside the model limit").into());
    }
    if output.aspect_ratio.is_some_and(|(w, h)| w == 0 || h == 0) {
        return Err(invalid("image aspect ratio must be positive").into());
    }
    if output.aspect_ratio.is_some() && matches!(output.size, Some(ImageSize::Pixels { .. })) {
        return Err(
            invalid("explicit pixel dimensions cannot be combined with an aspect ratio").into(),
        );
    }
    if matches!(
        output.size,
        Some(ImageSize::Pixels { width: 0, .. } | ImageSize::Pixels { height: 0, .. })
    ) {
        return Err(invalid("image size must be positive").into());
    }
    if let ImageRequest::Edit(edit) = req {
        if edit
            .mask
            .as_ref()
            .is_some_and(|mask| mask.image_index >= edit.images.len())
        {
            return Err(invalid("image mask index is outside the input list").into());
        }
    }
    let allowed = rules.options;
    for key in options.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(unsupported(format!(
                "image option {key:?} is not supported on this route"
            ))
            .into());
        }
    }
    if output.aspect_ratio.is_some() && !rules.ratio {
        return Err(unsupported("aspect ratio is unsupported on this image route").into());
    }
    if output.size.is_some() && !rules.size {
        return Err(unsupported("size is unsupported on this image route").into());
    }
    if output.count.is_some() && !rules.count {
        return Err(unsupported("image count is unsupported on this image route").into());
    }
    if output.format.is_some() && !rules.format {
        return Err(unsupported("output format is unsupported on this image route").into());
    }
    if output.quality.is_some() && !rules.quality {
        return Err(unsupported("quality is unsupported on this image route").into());
    }
    for image in request_images(req) {
        validate_input(image)?;
    }
    if let ImageRequest::Edit(edit) = req {
        if let Some(mask) = &edit.mask {
            if !matches!(
                mask.source,
                ImageInput::Base64 { .. } | ImageInput::Attachment { .. }
            ) {
                return Err(unsupported("mask input requires Base64 or an attachment").into());
            }
            validate_input(&mask.source)?;
        }
    }
    Ok(())
}
fn validate_input(input: &ImageInput) -> Result<(), ImageError> {
    match input {
        ImageInput::Url { url } => {
            let parsed = url::Url::parse(url).map_err(|_| invalid("invalid image URL"))?;
            if !matches!(parsed.scheme(), "http" | "https") || parsed.host_str().is_none() {
                return Err(invalid("image URL must be HTTP or HTTPS").into());
            }
        }
        ImageInput::Base64 { media_type, data } => {
            if !media_type.starts_with("image/")
                || !media_type
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'+' | b'-' | b'.'))
            {
                return Err(invalid("invalid image media type").into());
            }
            if data.len() > 64 * 1024 * 1024 * 4 / 3 + 4 {
                return Err(LlmError::RequestTooLarge {
                    message: "inline image exceeds 64 MiB".into(),
                }
                .into());
            }
        }
        ImageInput::Attachment { attachment } => {
            if !attachment.media_type.starts_with("image/")
                || !attachment
                    .media_type
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'+' | b'-' | b'.'))
            {
                return Err(invalid("invalid image attachment media type").into());
            }
        }
    }
    Ok(())
}
pub(crate) fn common_body(req: &ImageRequest, model: &ImageModelProfile) -> Map<String, Value> {
    let (prompt, output, options) = match req {
        ImageRequest::Generate(r) => (&r.prompt, &r.output, &r.provider_options),
        ImageRequest::Edit(r) => (&r.prompt, &r.output, &r.provider_options),
    };
    let mut body = Map::new();
    body.insert("model".into(), json!(model.request_model));
    body.insert("prompt".into(), json!(prompt));
    if let Some(n) = output.count {
        body.insert("n".into(), json!(n));
    }
    if let Some(size) = &output.size {
        let rendered = match size {
            ImageSize::Auto => "auto".into(),
            ImageSize::Pixels { width, height } => format!("{width}x{height}"),
            ImageSize::Tier { name } => name.clone(),
        };
        body.insert("size".into(), json!(rendered));
    }
    if let Some((width, height)) = output.aspect_ratio {
        body.insert("aspect_ratio".into(), json!(format!("{width}:{height}")));
    }
    if let Some(format) = output.format {
        body.insert("output_format".into(), json!(format_name(format)));
    }
    if let Some(quality) = output.quality {
        body.insert("quality".into(), json!(quality_name(quality)));
    }
    for (key, value) in options {
        body.insert(key.clone(), value.clone());
    }
    body
}
pub(crate) fn post(
    endpoint: String,
    body: Map<String, Value>,
    headers: Vec<(String, String)>,
) -> Result<HttpRequest, ImageError> {
    Ok(HttpRequest {
        method: "POST".into(),
        url: endpoint,
        headers,
        body: Bytes::from(serde_json::to_vec(&body).map_err(|e| invalid(e.to_string()))?),
        timeout: None,
    })
}
pub(crate) fn json_headers() -> Vec<(String, String)> {
    vec![("content-type".into(), "application/json".into())]
}
pub(crate) fn input_output(req: &ImageRequest) -> (&str, &ImageOutputOptions) {
    match req {
        ImageRequest::Generate(r) => (&r.prompt, &r.output),
        ImageRequest::Edit(r) => (&r.prompt, &r.output),
    }
}
pub(crate) fn request_images(req: &ImageRequest) -> Vec<&ImageInput> {
    match req {
        ImageRequest::Generate(r) => r.references.iter().map(|r| &r.image).collect(),
        ImageRequest::Edit(r) => r.images.iter().collect(),
    }
}
pub(crate) fn image_data_uri(image: &ImageInput) -> Result<String, ImageError> {
    match image {
        ImageInput::Url { url } if url.starts_with("https://") || url.starts_with("http://") => {
            Ok(url.clone())
        }
        ImageInput::Url { .. } => Err(invalid("image URL must be HTTP or HTTPS").into()),
        ImageInput::Base64 { media_type, data } if media_type.starts_with("image/") => {
            Ok(format!("data:{media_type};base64,{data}"))
        }
        ImageInput::Base64 { .. } => Err(invalid("image data has a non-image media type").into()),
        ImageInput::Attachment { .. } => Err(invalid("unresolved image attachment").into()),
    }
}
pub(crate) fn format_name(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::Webp => "webp",
    }
}
pub(crate) fn quality_name(quality: ImageQuality) -> &'static str {
    match quality {
        ImageQuality::Auto => "auto",
        ImageQuality::Low => "low",
        ImageQuality::Medium => "medium",
        ImageQuality::High => "high",
    }
}
pub(crate) fn image_artifact(item: &Value) -> Option<ImageArtifact> {
    let data = if let Some(url) = item
        .get("url")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        ImageData::Url { url: url.into() }
    } else if let Some(data) = item
        .get("b64_json")
        .or_else(|| item.get("base64"))
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        ImageData::Base64 { data: data.into() }
    } else {
        return None;
    };
    Some(ImageArtifact {
        data,
        media_type: item
            .get("media_type")
            .or_else(|| item.get("mime_type"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        revised_prompt: item
            .get("revised_prompt")
            .and_then(Value::as_str)
            .map(str::to_owned),
        expires_at: None,
    })
}
pub(crate) fn parse_json(resp: &HttpResponse) -> Result<Value, ImageError> {
    let body: Value = match serde_json::from_slice(&resp.body) {
        Ok(value) => value,
        Err(error) if (200..300).contains(&resp.status) => {
            return Err(ImageError::InvalidResponse(error.to_string()));
        }
        Err(_) => Value::Null,
    };
    if (200..300).contains(&resp.status) {
        return Ok(body);
    }
    let message = body
        .pointer("/error/message")
        .or_else(|| body.get("message"))
        .and_then(Value::as_str)
        .unwrap_or("image provider request failed");
    let message = format!("HTTP {}: {message}", resp.status);
    Err(match resp.status {
        400 | 422 => LlmError::InvalidRequest { message },
        401 => LlmError::Authentication { message },
        403 => LlmError::PermissionDenied { message },
        404 => LlmError::ModelUnavailable { message },
        429 => LlmError::RateLimited {
            message,
            retry_after: None,
        },
        500..=599 => LlmError::ProviderInternal { message },
        _ => LlmError::ProviderInternal { message },
    }
    .into())
}
pub(crate) fn response(
    body: &Value,
    model: &ImageModelProfile,
    profile: &ProviderProfile,
    images: Vec<ImageArtifact>,
    text: Option<String>,
    blocked: bool,
    partial: bool,
) -> Result<ImageResponse, ImageError> {
    if images.is_empty() && !blocked && text.is_none() {
        return Err(ImageError::InvalidResponse(
            "provider returned no image, text, or refusal".into(),
        ));
    }
    Ok(ImageResponse {
        outcome: if blocked {
            ImageOutcome::Blocked
        } else if partial {
            ImageOutcome::Partial
        } else if images.is_empty() {
            ImageOutcome::NoImage
        } else {
            ImageOutcome::Complete
        },
        images,
        text,
        requested_model: model.request_model.clone(),
        reported_model: body
            .get("model")
            .or_else(|| body.get("modelVersion"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        executed_profile: profile.profile_name.clone(),
        provider_id: profile.provider_id.as_str().to_owned(),
        request_id: body
            .get("request_id")
            .or_else(|| body.get("id"))
            .and_then(Value::as_str)
            .map(str::to_owned),
        usage: body
            .get("usage")
            .or_else(|| body.get("usageMetadata"))
            .cloned(),
    })
}
pub(crate) fn openai_images(body: &Value) -> Vec<ImageArtifact> {
    body.get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(image_artifact)
        .collect()
}
pub(crate) fn require_size(
    req: &ImageRequest,
    allowed: impl FnOnce(&ImageSize) -> bool,
    message: &str,
) -> Result<(), ImageError> {
    if input_output(req)
        .1
        .size
        .as_ref()
        .is_some_and(|size| !allowed(size))
    {
        return Err(unsupported(message).into());
    }
    Ok(())
}
pub(crate) fn task_model(task: &ImageTaskRef) -> ImageModelProfile {
    ImageModelProfile {
        display_model: task.request_model.clone(),
        request_model: task.request_model.clone(),
        route: task.route.clone(),
        aliases: vec![],
        hidden: false,
        capabilities: Default::default(),
    }
}
pub(crate) fn task_id_value(id: Option<&Value>) -> Result<String, ImageError> {
    id.and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| ImageError::InvalidResponse("provider accepted no task ID".into()))
}
pub(crate) fn task_get(url: String, task_id: &str) -> Result<HttpRequest, ImageError> {
    if !task_id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(invalid("invalid provider task ID").into());
    }
    Ok(HttpRequest {
        method: "GET".into(),
        url,
        headers: vec![],
        body: Bytes::new(),
        timeout: None,
    })
}
pub(crate) fn task_status(
    body: &Value,
    status: Option<&Value>,
    success: impl FnOnce() -> Result<ImageResponse, ImageError>,
) -> Result<ImageTaskSnapshot, ImageError> {
    let status = status
        .and_then(Value::as_str)
        .ok_or_else(|| ImageError::InvalidResponse("task status missing".into()))?;
    Ok(match status.to_ascii_uppercase().as_str() {
        "PENDING" => ImageTaskSnapshot::Pending,
        "RUNNING" | "PROCESSING" => ImageTaskSnapshot::Running,
        "SUCCEEDED" | "SUCCESS" => ImageTaskSnapshot::Succeeded {
            response: Box::new(success()?),
        },
        "FAILED" | "FAIL" => ImageTaskSnapshot::Failed {
            message: body
                .get("message")
                .or_else(|| body.pointer("/output/message"))
                .and_then(Value::as_str)
                .unwrap_or("provider task failed")
                .into(),
        },
        "CANCELED" | "CANCELLED" => ImageTaskSnapshot::Cancelled,
        "EXPIRED" => ImageTaskSnapshot::Expired,
        _ => ImageTaskSnapshot::Unknown {
            raw_status: status.into(),
        },
    })
}
