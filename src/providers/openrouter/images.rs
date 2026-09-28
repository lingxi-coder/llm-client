use crate::images::wire::*;
use crate::images::{ImageAdapter, ImageError};
use crate::protocol::*;
use crate::transport::HttpRequest;
use serde_json::{json, Value};

pub(crate) struct OpenRouterImages;
impl ImageAdapter for OpenRouterImages {
    fn api(&self) -> ImageApi {
        ImageApi::OpenRouter
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
                options: &["seed", "background", "output_compression", "provider"],
                ratio: true,
                size: true,
                count: true,
                format: true,
                quality: true,
            },
            asynchronous,
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
            return Err(unsupported("OpenRouter image route has no native task endpoint").into());
        }
        let mut body = common_body(req, model);
        let refs = request_images(req)
            .into_iter()
            .map(|image| {
                image_data_uri(image).map(|url| json!({"type":"image_url","image_url":{"url":url}}))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !refs.is_empty() {
            body.insert("input_references".into(), json!(refs));
        }
        post(
            format!("{}/images", route.base_url.trim_end_matches('/')),
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
