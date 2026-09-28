use crate::images::wire::*;
use crate::images::{ImageAdapter, ImageError};
use crate::protocol::*;
use crate::transport::HttpRequest;
use serde_json::{json, Value};

pub(crate) struct MinimaxImages;
impl ImageAdapter for MinimaxImages {
    fn api(&self) -> ImageApi {
        ImageApi::Minimax
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
                options: &["seed", "prompt_optimizer", "response_format"],
                ratio: true,
                size: true,
                count: true,
                format: false,
                quality: false,
            },
            asynchronous,
        )?;

        if matches!(req, ImageRequest::Edit(_)) {
            return Err(
                unsupported("MiniMax character references are generation, not editing").into(),
            );
        }
        if let ImageRequest::Generate(request) = req {
            if request
                .references
                .iter()
                .any(|r| r.kind != ImageReferenceKind::Character)
            {
                return Err(unsupported("MiniMax supports character references only").into());
            }
        }
        if request_images(req)
            .iter()
            .any(|image| !matches!(image, ImageInput::Url { .. }))
        {
            return Err(unsupported("MiniMax character reference requires a public URL").into());
        }
        require_size(
            req,
            |size| matches!(size, ImageSize::Pixels { .. }),
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
            return Err(unsupported("MiniMax image route has no native task endpoint").into());
        }
        let mut body = common_body(req, model);
        let (_, output) = input_output(req);
        if let Some(ImageSize::Pixels { width, height }) = &output.size {
            body.remove("size");
            body.insert("width".into(), json!(width));
            body.insert("height".into(), json!(height));
        }
        let references = request_images(req)
            .into_iter()
            .map(|image| {
                image_data_uri(image).map(|url| json!({"type":"character","image_file":url}))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !references.is_empty() {
            body.insert("subject_reference".into(), json!(references));
        }
        post(
            format!("{}/image_generation", route.base_url.trim_end_matches('/')),
            body,
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
        let partial = body
            .pointer("/metadata/failed_count")
            .and_then(Value::as_str)
            .is_some_and(|v| v != "0");
        response(body, model, profile, images, None, false, partial)
    }
    fn check_response(&self, body: &Value) -> Result<(), ImageError> {
        if body
            .pointer("/base_resp/status_code")
            .and_then(Value::as_i64)
            .is_some_and(|code| code != 0)
        {
            return Err(LlmError::ProviderInternal {
                message: body
                    .pointer("/base_resp/status_msg")
                    .and_then(Value::as_str)
                    .unwrap_or("image provider error")
                    .into(),
            }
            .into());
        }
        Ok(())
    }
}
