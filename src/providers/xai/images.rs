use crate::images::wire::*;
use crate::images::{ImageAdapter, ImageError};
use crate::protocol::*;
use crate::transport::HttpRequest;
use serde_json::{json, Value};

pub(crate) struct XaiImages;
impl ImageAdapter for XaiImages {
    fn api(&self) -> ImageApi {
        ImageApi::Xai
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
                options: &["resolution"],
                ratio: true,
                size: false,
                count: true,
                format: false,
                quality: true,
            },
            asynchronous,
        )?;

        if input_output(req).1.quality == Some(ImageQuality::High) {
            return Err(unsupported("xAI image quality accepts low, medium, or auto").into());
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
            return Err(unsupported("xAI image route has no native task endpoint").into());
        }
        let mut body = common_body(req, model);
        let operation = if let ImageRequest::Edit(edit) = req {
            let image = edit
                .images
                .first()
                .ok_or_else(|| invalid("edit requires an image"))?;
            body.insert(
                "image".into(),
                json!({"type":"image_url","url":image_data_uri(image)?}),
            );
            "edits"
        } else {
            "generations"
        };
        post(
            format!(
                "{}/images/{operation}",
                route.base_url.trim_end_matches('/')
            ),
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
