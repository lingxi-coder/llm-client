use super::ImageError;
use crate::protocol::{
    ImageApi, ImageArtifact, ImageData, ImageFormat, ImageInput, ImageModelProfile, ImageOutcome,
    ImageQuality, ImageReferenceKind, ImageRequest, ImageResponse, ImageRouteConfig, ImageSize,
    ImageTaskRef, ImageTaskSnapshot, LlmError, ProviderProfile,
};
use crate::transport::{HttpRequest, HttpResponse};
use base64::Engine;
use bytes::Bytes;
use serde_json::{json, Map, Value};

fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
fn unsupported(message: impl Into<String>) -> LlmError {
    LlmError::UnsupportedCapability {
        message: message.into(),
    }
}

pub(super) fn validate(
    req: &ImageRequest,
    model: &ImageModelProfile,
    route: &ImageRouteConfig,
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
        if route.api == ImageApi::OpenAi
            && edit.mask.as_ref().is_some_and(|mask| mask.image_index != 0)
        {
            return Err(unsupported("OpenAI masks apply only to the first input image").into());
        }
    }
    let allowed: &[&str] = match route.api {
        ImageApi::OpenAi => &["background", "moderation", "output_compression"],
        ImageApi::Gemini => &[],
        ImageApi::Qwen => &[
            "negative_prompt",
            "seed",
            "prompt_extend",
            "prompt_extend_mode",
            "enable_thinking",
            "watermark",
        ],
        ImageApi::Wan => &["watermark", "thinking_mode"],
        ImageApi::Xai => &["resolution"],
        ImageApi::Minimax => &["seed", "prompt_optimizer", "response_format"],
        ImageApi::Zai => &["user_id"],
        ImageApi::OpenRouter => &["seed", "background", "output_compression", "provider"],
    };
    for key in options.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(unsupported(format!(
                "image option {key:?} is not supported on this route"
            ))
            .into());
        }
    }
    let supports_ratio = matches!(
        route.api,
        ImageApi::Gemini | ImageApi::Xai | ImageApi::Minimax | ImageApi::OpenRouter
    );
    let supports_size = !matches!(route.api, ImageApi::Xai);
    let supports_count = !matches!(route.api, ImageApi::Gemini | ImageApi::Zai);
    let supports_format = matches!(route.api, ImageApi::OpenAi | ImageApi::OpenRouter);
    let supports_quality = matches!(
        route.api,
        ImageApi::OpenAi | ImageApi::Xai | ImageApi::Zai | ImageApi::OpenRouter
    );
    if output.aspect_ratio.is_some() && !supports_ratio {
        return Err(unsupported("aspect ratio is unsupported on this image route").into());
    }
    if output.size.is_some() && !supports_size {
        return Err(unsupported("size is unsupported on this image route").into());
    }
    if output.count.is_some() && !supports_count {
        return Err(unsupported("image count is unsupported on this image route").into());
    }
    if output.format.is_some() && !supports_format {
        return Err(unsupported("output format is unsupported on this image route").into());
    }
    if output.quality.is_some() && !supports_quality {
        return Err(unsupported("quality is unsupported on this image route").into());
    }
    if matches!(route.api, ImageApi::Minimax) && editing {
        return Err(unsupported("MiniMax character references are generation, not editing").into());
    }
    if route.api == ImageApi::Minimax {
        if let ImageRequest::Generate(request) = req {
            if request
                .references
                .iter()
                .any(|r| r.kind != ImageReferenceKind::Character)
            {
                return Err(unsupported("MiniMax supports character references only").into());
            }
        }
    }
    if matches!(route.api, ImageApi::Zai) && (editing || input_count > 0) {
        return Err(unsupported("Z.AI image route accepts a text prompt only").into());
    }
    if matches!(route.api, ImageApi::OpenAi) && !editing && input_count > 0 {
        return Err(unsupported("OpenAI reference images require edit()").into());
    }
    if route.api == ImageApi::Zai
        && output
            .quality
            .is_some_and(|quality| !matches!(quality, ImageQuality::High))
    {
        return Err(unsupported("GLM-Image accepts only HD quality").into());
    }
    if route.api == ImageApi::Xai && output.quality == Some(ImageQuality::High) {
        return Err(unsupported("xAI image quality accepts low, medium, or auto").into());
    }
    if let Some(size) = &output.size {
        match (route.api, size) {
            (ImageApi::Minimax, ImageSize::Tier { .. })
            | (ImageApi::Qwen | ImageApi::Zai | ImageApi::OpenAi, ImageSize::Tier { .. })
            | (ImageApi::Minimax | ImageApi::Zai | ImageApi::Wan, ImageSize::Auto) => {
                return Err(
                    unsupported("this image route cannot represent the selected size").into(),
                )
            }
            (ImageApi::Gemini, ImageSize::Pixels { .. }) => {
                return Err(unsupported("Gemini image size uses a resolution tier").into())
            }
            (ImageApi::Gemini, ImageSize::Tier { name })
                if !matches!(name.as_str(), "1K" | "2K" | "4K") =>
            {
                return Err(unsupported("Gemini image size must be 1K, 2K, or 4K").into())
            }
            (ImageApi::Wan, ImageSize::Tier { name })
                if !matches!(name.as_str(), "1K" | "2K" | "4K") =>
            {
                return Err(unsupported("Wan image size must be 1K, 2K, or 4K").into())
            }
            _ => {}
        }
    }
    for image in request_images(req) {
        if route.api == ImageApi::Gemini && matches!(image, ImageInput::Url { .. }) {
            return Err(unsupported("Gemini image inputs require Base64 or an attachment").into());
        }
        if route.api == ImageApi::Minimax && !matches!(image, ImageInput::Url { .. }) {
            return Err(unsupported("MiniMax character reference requires a public URL").into());
        }
        if route.api == ImageApi::OpenAi && editing && matches!(image, ImageInput::Url { .. }) {
            return Err(unsupported("OpenAI edit requires Base64 or an attachment").into());
        }
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

pub(super) fn encode(
    req: &ImageRequest,
    model: &ImageModelProfile,
    route: &ImageRouteConfig,
    asynchronous: bool,
) -> Result<HttpRequest, ImageError> {
    let (prompt, output, options) = match req {
        ImageRequest::Generate(r) => (&r.prompt, &r.output, &r.provider_options),
        ImageRequest::Edit(r) => (&r.prompt, &r.output, &r.provider_options),
    };
    if route.api == ImageApi::OpenAi && matches!(req, ImageRequest::Edit(_)) {
        return openai_edit(req, model, route);
    }
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
        if !(route.api == ImageApi::Gemini && matches!(size, ImageSize::Auto)) {
            body.insert("size".into(), json!(rendered));
        }
    }
    if let Some((width, height)) = output.aspect_ratio {
        body.insert("aspect_ratio".into(), json!(format!("{width}:{height}")));
    }
    if let Some(format) = output.format {
        body.insert("output_format".into(), json!(format_name(format)));
    }
    if let Some(quality) = output.quality {
        body.insert(
            "quality".into(),
            json!(if route.api == ImageApi::Zai {
                "hd"
            } else {
                quality_name(quality)
            }),
        );
    }
    for (key, value) in options {
        body.insert(key.clone(), value.clone());
    }
    let mut headers = vec![("content-type".into(), "application/json".into())];
    let endpoint = match route.api {
        ImageApi::OpenAi | ImageApi::Xai | ImageApi::Qwen => {
            if asynchronous {
                let base = route
                    .task_base_url
                    .as_ref()
                    .ok_or_else(|| unsupported("native task endpoint not configured"))?;
                headers.push(("X-DashScope-Async".into(), "enable".into()));
                body = qwen_task_body(req, model, &body)?;
                format!(
                    "{}/services/aigc/image-generation/generation",
                    base.trim_end_matches('/')
                )
            } else {
                match req {
                    ImageRequest::Edit(_) if route.api == ImageApi::Xai => {
                        format!("{}/images/edits", route.base_url.trim_end_matches('/'))
                    }
                    _ => format!(
                        "{}/images/generations",
                        route.base_url.trim_end_matches('/')
                    ),
                }
            }
        }
        ImageApi::OpenRouter => format!("{}/images", route.base_url.trim_end_matches('/')),
        ImageApi::Minimax => format!("{}/image_generation", route.base_url.trim_end_matches('/')),
        ImageApi::Zai => format!(
            "{}/{}images/generations",
            route.base_url.trim_end_matches('/'),
            if asynchronous { "async/" } else { "" }
        ),
        ImageApi::Gemini => format!(
            "{}/models/{}:generateContent",
            route.base_url.trim_end_matches('/'),
            model.request_model
        ),
        ImageApi::Wan => {
            let root = if asynchronous {
                route.task_base_url.as_ref().unwrap_or(&route.base_url)
            } else {
                &route.base_url
            };
            if asynchronous {
                headers.push(("X-DashScope-Async".into(), "enable".into()));
            }
            format!(
                "{}/services/aigc/{}/generation",
                root.trim_end_matches('/'),
                if asynchronous {
                    "image-generation"
                } else {
                    "multimodal-generation"
                }
            )
        }
    };
    match route.api {
        ImageApi::Wan => {
            let mut content = Vec::new();
            for image in request_images(req) {
                content.push(json!({"image": image_data_uri(image)?}));
            }
            content.push(json!({"text": prompt}));
            let mut parameters = body.clone();
            parameters.remove("model");
            parameters.remove("prompt");
            if let Some(size) = parameters.get_mut("size") {
                if let Some(s) = size.as_str() {
                    *size = json!(s.replace('x', "*"));
                }
            }
            body = serde_json::from_value(json!({"model": model.request_model, "input": {"messages": [{"role":"user", "content":content}]}, "parameters":parameters})).expect("object");
        }
        ImageApi::Gemini => {
            let mut parts = vec![json!({"text": prompt})];
            for image in request_images(req) {
                parts.push(gemini_image(image)?);
            }
            let mut config = Map::new();
            config.insert("responseModalities".into(), json!(["IMAGE"]));
            if let Some(ratio) = body.remove("aspect_ratio") {
                config.insert("imageConfig".into(), json!({"aspectRatio": ratio}));
            }
            if let Some(size) = body.remove("size") {
                config.entry("imageConfig").or_insert_with(|| json!({}))["imageSize"] = size;
            }
            body = serde_json::from_value(
                json!({"contents": [{"role":"user", "parts": parts}], "generationConfig": config}),
            )
            .expect("object");
        }
        ImageApi::Minimax => {
            body.remove("output_format");
            if let Some(ImageSize::Pixels { width, height }) = &output.size {
                body.remove("size");
                body.insert("width".into(), json!(width));
                body.insert("height".into(), json!(height));
            }
            let references = request_images(req);
            if !references.is_empty() {
                let mut array = Vec::new();
                for image in references {
                    array.push(json!({"type":"character", "image_file": image_data_uri(image)?}));
                }
                body.insert("subject_reference".into(), json!(array));
            }
        }
        ImageApi::Xai => {
            if let ImageRequest::Edit(edit) = req {
                let image = edit
                    .images
                    .first()
                    .ok_or_else(|| invalid("edit requires an image"))?;
                body.insert(
                    "image".into(),
                    json!({"type":"image_url", "url": image_data_uri(image)?}),
                );
            }
        }
        ImageApi::Qwen if !asynchronous => {
            let images = request_images(req)
                .iter()
                .map(|image| image_data_uri(image))
                .collect::<Result<Vec<_>, _>>()?;
            if !images.is_empty() {
                body.insert(
                    "image".into(),
                    if images.len() == 1 {
                        json!(images[0])
                    } else {
                        json!(images)
                    },
                );
            }
        }
        ImageApi::OpenRouter => {
            let references = request_images(req)
                .iter()
                .map(|image| {
                    image_data_uri(image)
                        .map(|url| json!({"type":"image_url", "image_url":{"url":url}}))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if !references.is_empty() {
                body.insert("input_references".into(), json!(references));
            }
        }
        _ => {}
    }
    Ok(HttpRequest {
        method: "POST".into(),
        url: endpoint,
        headers,
        body: Bytes::from(serde_json::to_vec(&body).map_err(|e| invalid(e.to_string()))?),
        timeout: None,
    })
}

fn request_images(req: &ImageRequest) -> Vec<&ImageInput> {
    match req {
        ImageRequest::Generate(r) => r.references.iter().map(|r| &r.image).collect(),
        ImageRequest::Edit(r) => r.images.iter().collect(),
    }
}
fn image_data_uri(image: &ImageInput) -> Result<String, ImageError> {
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
fn gemini_image(image: &ImageInput) -> Result<Value, ImageError> {
    match image {
        ImageInput::Base64 { media_type, data } => {
            Ok(json!({"inlineData": {"mimeType": media_type, "data": data}}))
        }
        ImageInput::Url { url } => Ok(json!({"fileData": {"fileUri": url}})),
        ImageInput::Attachment { .. } => Err(invalid("unresolved image attachment").into()),
    }
}
fn format_name(format: ImageFormat) -> &'static str {
    match format {
        ImageFormat::Png => "png",
        ImageFormat::Jpeg => "jpeg",
        ImageFormat::Webp => "webp",
    }
}
fn quality_name(quality: ImageQuality) -> &'static str {
    match quality {
        ImageQuality::Auto => "auto",
        ImageQuality::Low => "low",
        ImageQuality::Medium => "medium",
        ImageQuality::High => "high",
    }
}

fn qwen_task_body(
    req: &ImageRequest,
    model: &ImageModelProfile,
    common: &Map<String, Value>,
) -> Result<Map<String, Value>, ImageError> {
    let (prompt, images) = match req {
        ImageRequest::Generate(r) => (&r.prompt, request_images(req)),
        ImageRequest::Edit(r) => (&r.prompt, r.images.iter().collect()),
    };
    let mut content = vec![json!({"text": prompt})];
    for image in images {
        content.push(json!({"image": image_data_uri(image)?}));
    }
    let mut parameters = common.clone();
    parameters.remove("model");
    parameters.remove("prompt");
    if let Some(size) = parameters.get_mut("size") {
        if let Some(s) = size.as_str() {
            *size = json!(s.replace('x', "*"));
        }
    }
    let body = json!({"model": model.request_model, "input": {"messages": [{"role":"user", "content":content}]}, "parameters": parameters});
    Ok(body.as_object().expect("object").clone())
}

fn openai_edit(
    req: &ImageRequest,
    model: &ImageModelProfile,
    route: &ImageRouteConfig,
) -> Result<HttpRequest, ImageError> {
    let ImageRequest::Edit(edit) = req else {
        unreachable!()
    };
    let boundary = "lingxi-image-edit-0f731eab";
    let mut body = Vec::new();
    {
        let mut field = |name: &str, value: &str| {
            body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes());
        };
        field("model", &model.request_model);
        field("prompt", &edit.prompt);
        if let Some(n) = edit.output.count {
            field("n", &n.to_string());
        }
        if let Some(size) = &edit.output.size {
            field(
                "size",
                &match size {
                    ImageSize::Auto => "auto".into(),
                    ImageSize::Pixels { width, height } => format!("{width}x{height}"),
                    ImageSize::Tier { name } => name.clone(),
                },
            );
        }
        if let Some(format) = edit.output.format {
            field("output_format", format_name(format));
        }
        if let Some(quality) = edit.output.quality {
            field("quality", quality_name(quality));
        }
        for (key, value) in &edit.provider_options {
            let rendered = value
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| value.to_string());
            field(key, &rendered);
        }
    }
    for (index, image) in edit.images.iter().enumerate() {
        append_file(
            &mut body,
            boundary,
            if edit.images.len() == 1 {
                "image"
            } else {
                "image[]"
            },
            &format!("image-{index}.png"),
            image,
        )?;
    }
    if let Some(mask) = &edit.mask {
        append_file(&mut body, boundary, "mask", "mask.png", &mask.source)?;
    }
    body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
    Ok(HttpRequest {
        method: "POST".into(),
        url: format!("{}/images/edits", route.base_url.trim_end_matches('/')),
        headers: vec![(
            "content-type".into(),
            format!("multipart/form-data; boundary={boundary}"),
        )],
        body: Bytes::from(body),
        timeout: None,
    })
}
fn append_file(
    body: &mut Vec<u8>,
    boundary: &str,
    name: &str,
    filename: &str,
    image: &ImageInput,
) -> Result<(), ImageError> {
    let ImageInput::Base64 { media_type, data } = image else {
        return Err(unsupported("OpenAI image edit requires Base64 or an attachment").into());
    };
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|_| invalid("invalid Base64 image input"))?;
    body.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {media_type}\r\n\r\n").as_bytes());
    body.extend_from_slice(&decoded);
    body.extend_from_slice(b"\r\n");
    Ok(())
}

pub(super) fn parse_json(resp: &HttpResponse) -> Result<Value, ImageError> {
    let body: Value = match serde_json::from_slice(&resp.body) {
        Ok(value) => value,
        Err(error) if (200..300).contains(&resp.status) => {
            return Err(ImageError::InvalidResponse(error.to_string()));
        }
        Err(_) => Value::Null,
    };
    if (200..300).contains(&resp.status) {
        if let Some(code) = body
            .pointer("/base_resp/status_code")
            .and_then(Value::as_i64)
        {
            if code != 0 {
                return Err(LlmError::ProviderInternal {
                    message: body
                        .pointer("/base_resp/status_msg")
                        .and_then(Value::as_str)
                        .unwrap_or("image provider error")
                        .into(),
                }
                .into());
            }
        }
        if body.get("code").is_some() && body.get("data").is_none() && body.get("output").is_none()
        {
            return Err(LlmError::ProviderInternal {
                message: body
                    .get("message")
                    .and_then(Value::as_str)
                    .unwrap_or("image provider error")
                    .into(),
            }
            .into());
        }
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

fn image_artifact(item: &Value) -> Option<ImageArtifact> {
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

pub(super) fn decode(
    api: ImageApi,
    body: &Value,
    model: &ImageModelProfile,
    profile: &ProviderProfile,
) -> Result<ImageResponse, ImageError> {
    let mut images = Vec::new();
    let mut text = None;
    let mut blocked = false;
    let mut partial = false;
    match api {
        ImageApi::Gemini => {
            blocked = body
                .get("promptFeedback")
                .and_then(|v| v.get("blockReason"))
                .is_some();
            if let Some(candidate) = body
                .get("candidates")
                .and_then(Value::as_array)
                .and_then(|v| v.first())
            {
                let finish = candidate
                    .get("finishReason")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                blocked |= matches!(finish, "SAFETY" | "IMAGE_SAFETY" | "PROHIBITED_CONTENT");
                if let Some(parts) = candidate
                    .pointer("/content/parts")
                    .and_then(Value::as_array)
                {
                    for part in parts {
                        if part.get("thought").and_then(Value::as_bool) == Some(true) {
                            continue;
                        }
                        if let Some(data) = part.pointer("/inlineData/data").and_then(Value::as_str)
                        {
                            images.push(ImageArtifact {
                                data: ImageData::Base64 { data: data.into() },
                                media_type: part
                                    .pointer("/inlineData/mimeType")
                                    .and_then(Value::as_str)
                                    .map(str::to_owned),
                                revised_prompt: None,
                                expires_at: None,
                            });
                        }
                        if let Some(value) = part.get("text").and_then(Value::as_str) {
                            text.get_or_insert_with(String::new).push_str(value);
                        }
                    }
                }
            }
        }
        ImageApi::Minimax => {
            if let Some(urls) = body.pointer("/data/image_urls").and_then(Value::as_array) {
                images.extend(
                    urls.iter()
                        .filter_map(Value::as_str)
                        .map(|url| ImageArtifact {
                            data: ImageData::Url { url: url.into() },
                            media_type: None,
                            revised_prompt: None,
                            expires_at: None,
                        }),
                );
            }
            if let Some(encoded) = body.pointer("/data/image_base64").and_then(Value::as_array) {
                images.extend(
                    encoded
                        .iter()
                        .filter_map(Value::as_str)
                        .map(|data| ImageArtifact {
                            data: ImageData::Base64 { data: data.into() },
                            media_type: Some("image/jpeg".into()),
                            revised_prompt: None,
                            expires_at: None,
                        }),
                );
            }
            partial = body
                .pointer("/metadata/failed_count")
                .and_then(Value::as_str)
                .is_some_and(|v| v != "0");
        }
        ImageApi::Qwen => {
            if let Some(data) = body.get("data").and_then(Value::as_array) {
                images.extend(data.iter().filter_map(image_artifact));
            }
            images.extend(choices_images(body));
        }
        ImageApi::Wan => {
            images.extend(choices_images(body));
        }
        _ => {
            if let Some(data) = body.get("data").and_then(Value::as_array) {
                images.extend(data.iter().filter_map(image_artifact));
            }
        }
    }
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

fn choices_images(body: &Value) -> Vec<ImageArtifact> {
    body.pointer("/output/choices")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|choice| choice.pointer("/message/content").and_then(Value::as_array))
        .flatten()
        .filter(|item| item.get("type").and_then(Value::as_str) == Some("image"))
        .filter_map(|item| item.get("image").and_then(Value::as_str))
        .map(|url| ImageArtifact {
            data: ImageData::Url { url: url.into() },
            media_type: Some("image/png".into()),
            revised_prompt: None,
            expires_at: None,
        })
        .collect()
}

pub(super) fn task_id(api: ImageApi, body: &Value) -> Result<String, ImageError> {
    let id = match api {
        ImageApi::Qwen | ImageApi::Wan => body.pointer("/output/task_id"),
        ImageApi::Zai => body.get("id"),
        _ => return Err(unsupported("image route has no task response").into()),
    };
    id.and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| ImageError::InvalidResponse("provider accepted no task ID".into()))
}
pub(super) fn task_query(
    route: &ImageRouteConfig,
    task_id: &str,
) -> Result<HttpRequest, ImageError> {
    if !task_id
        .bytes()
        .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    {
        return Err(LlmError::InvalidRequest {
            message: "invalid provider task ID".into(),
        }
        .into());
    }
    let url = match route.api {
        ImageApi::Qwen | ImageApi::Wan => format!(
            "{}/tasks/{task_id}",
            route
                .task_base_url
                .as_ref()
                .ok_or_else(|| unsupported("task API root missing"))?
                .trim_end_matches('/')
        ),
        ImageApi::Zai => format!(
            "{}/async-result/{task_id}",
            route.base_url.trim_end_matches('/')
        ),
        _ => return Err(unsupported("image route has no task query endpoint").into()),
    };
    Ok(HttpRequest {
        method: "GET".into(),
        url,
        headers: vec![],
        body: Bytes::new(),
        timeout: None,
    })
}
pub(super) fn decode_task(
    api: ImageApi,
    body: &Value,
    task: &ImageTaskRef,
    profile: &ProviderProfile,
) -> Result<ImageTaskSnapshot, ImageError> {
    let (status, result) = match api {
        ImageApi::Qwen | ImageApi::Wan => (body.pointer("/output/task_status"), body.get("output")),
        ImageApi::Zai => (
            body.get("task_status"),
            body.get("image_result").or_else(|| body.get("data")),
        ),
        _ => return Err(unsupported("image route has no task decoder").into()),
    };
    let status = status
        .and_then(Value::as_str)
        .ok_or_else(|| ImageError::InvalidResponse("task status missing".into()))?;
    Ok(match status.to_ascii_uppercase().as_str() {
        "PENDING" => ImageTaskSnapshot::Pending,
        "RUNNING" | "PROCESSING" => ImageTaskSnapshot::Running,
        "SUCCEEDED" | "SUCCESS" => {
            if matches!(api, ImageApi::Qwen | ImageApi::Wan) {
                let model = ImageModelProfile {
                    display_model: task.request_model.clone(),
                    request_model: task.request_model.clone(),
                    route: task.route.clone(),
                    aliases: vec![],
                    hidden: false,
                    capabilities: Default::default(),
                };
                return Ok(ImageTaskSnapshot::Succeeded {
                    response: Box::new(decode(api, body, &model, profile)?),
                });
            }
            let mut data = body.clone();
            data["data"] = match result {
                Some(Value::Array(items)) => json!(items),
                Some(Value::Object(map)) => map
                    .get("results")
                    .or_else(|| map.get("data"))
                    .cloned()
                    .unwrap_or_else(|| json!([map])),
                _ => json!([]),
            };
            let model = ImageModelProfile {
                display_model: task.request_model.clone(),
                request_model: task.request_model.clone(),
                route: task.route.clone(),
                aliases: vec![],
                hidden: false,
                capabilities: Default::default(),
            };
            ImageTaskSnapshot::Succeeded {
                response: Box::new(decode(api, &data, &model, profile)?),
            }
        }
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
