use crate::images::wire::*;
use crate::images::{ImageAdapter, ImageError};
use crate::protocol::*;
use crate::transport::HttpRequest;
use serde_json::{json, Value};

fn gemini_image(image: &ImageInput) -> Result<Value, ImageError> {
    match image {
        ImageInput::Base64 { media_type, data } => {
            Ok(json!({"inlineData": {"mimeType": media_type, "data": data}}))
        }
        ImageInput::Url { url } => Ok(json!({"fileData": {"fileUri": url}})),
        ImageInput::Attachment { .. } => Err(invalid("unresolved image attachment").into()),
    }
}
pub(crate) struct GoogleImages;
impl ImageAdapter for GoogleImages {
    fn api(&self) -> ImageApi {
        ImageApi::Gemini
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
                options: &[],
                ratio: true,
                size: true,
                count: false,
                format: false,
                quality: false,
            },
            asynchronous,
        )?;

        require_size(
            req,
            |size| {
                matches!(size, ImageSize::Auto)
                    || matches!(size,ImageSize::Tier{name} if matches!(name.as_str(),"1K"|"2K"|"4K"))
            },
            "Gemini image size uses a 1K, 2K, or 4K resolution tier",
        )?;
        if request_images(req)
            .iter()
            .any(|image| matches!(image, ImageInput::Url { .. }))
        {
            return Err(unsupported("Gemini image inputs require Base64 or an attachment").into());
        }
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
            return Err(unsupported("Gemini image route has no native task endpoint").into());
        }
        let (prompt, output) = input_output(req);
        let mut parts = vec![json!({"text":prompt})];
        for image in request_images(req) {
            parts.push(gemini_image(image)?);
        }
        let mut config = serde_json::Map::new();
        config.insert("responseModalities".into(), json!(["IMAGE"]));
        let mut common = common_body(req, model);
        if let Some(ratio) = common.remove("aspect_ratio") {
            config.insert("imageConfig".into(), json!({"aspectRatio":ratio}));
        }
        if !matches!(output.size, Some(ImageSize::Auto)) {
            if let Some(size) = common.remove("size") {
                config.entry("imageConfig").or_insert_with(|| json!({}))["imageSize"] = size;
            }
        }
        let body = json!({"contents":[{"role":"user","parts":parts}],"generationConfig":config});
        post(
            format!(
                "{}/models/{}:generateContent",
                route.base_url.trim_end_matches('/'),
                model.request_model
            ),
            body.as_object().unwrap().clone(),
            json_headers(),
        )
    }
    fn decode(
        &self,
        body: &Value,
        model: &ImageModelProfile,
        profile: &ProviderProfile,
    ) -> Result<ImageResponse, ImageError> {
        let mut images = Vec::new();
        let mut text = None;
        let mut blocked;
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
                    if let Some(data) = part.pointer("/inlineData/data").and_then(Value::as_str) {
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
        response(body, model, profile, images, text, blocked, false)
    }
}
