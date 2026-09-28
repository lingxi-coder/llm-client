use crate::images::wire::*;
use crate::images::{ImageAdapter, ImageError};
use crate::protocol::*;
use crate::transport::HttpRequest;
use serde_json::Value;

use base64::Engine;
use bytes::Bytes;
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
pub(crate) struct OpenAiImages;
impl ImageAdapter for OpenAiImages {
    fn api(&self) -> ImageApi {
        ImageApi::OpenAi
    }
    fn validate(
        &self,
        req: &ImageRequest,
        model: &ImageModelProfile,
        _route: &ImageRouteConfig,
        asynchronous: bool,
    ) -> Result<(), ImageError> {
        validate(
            req,
            model,
            &ImageRules {
                options: &["background", "moderation", "output_compression"],
                ratio: false,
                size: true,
                count: true,
                format: true,
                quality: true,
            },
            asynchronous,
        )?;

        if let ImageRequest::Edit(edit) = req {
            if edit.mask.as_ref().is_some_and(|mask| mask.image_index != 0) {
                return Err(unsupported("OpenAI masks apply only to the first input image").into());
            }
            if edit
                .images
                .iter()
                .any(|image| matches!(image, ImageInput::Url { .. }))
            {
                return Err(unsupported("OpenAI edit requires Base64 or an attachment").into());
            }
        } else if !request_images(req).is_empty() {
            return Err(unsupported("OpenAI reference images require edit()").into());
        }
        require_size(
            req,
            |size| !matches!(size, ImageSize::Tier { .. }),
            "this image route cannot represent the selected size",
        )?;
        Ok(())
    }
    fn encode(
        &self,
        req: &ImageRequest,
        model: &ImageModelProfile,
        route: &ImageRouteConfig,
        asynchronous: bool,
    ) -> Result<HttpRequest, ImageError> {
        if asynchronous {
            return Err(unsupported("OpenAI image route has no native task endpoint").into());
        }
        if matches!(req, ImageRequest::Edit(_)) {
            return openai_edit(req, model, route);
        }
        post(
            format!(
                "{}/images/generations",
                route.base_url.trim_end_matches('/')
            ),
            common_body(req, model),
            json_headers(),
        )
    }
    fn decode(
        &self,
        body: &Value,
        model: &ImageModelProfile,
        profile: &ProviderProfile,
    ) -> Result<ImageResponse, ImageError> {
        response(
            body,
            model,
            profile,
            openai_images(body),
            None,
            false,
            false,
        )
    }
}
