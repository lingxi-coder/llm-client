use crate::images::wire::*;
use crate::images::{ImageAdapter, ImageError};
use crate::protocol::*;
use crate::transport::HttpRequest;
use serde_json::{json, Value};

pub(crate) struct ZhipuImages;
impl ImageAdapter for ZhipuImages {
    fn api(&self) -> ImageApi {
        ImageApi::Zai
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
                options: &["user_id"],
                ratio: false,
                size: true,
                count: false,
                format: false,
                quality: true,
            },
            asynchronous,
        )?;

        if matches!(req, ImageRequest::Edit(_)) || !request_images(req).is_empty() {
            return Err(unsupported("Z.AI image route accepts a text prompt only").into());
        }
        if input_output(req)
            .1
            .quality
            .is_some_and(|quality| quality != ImageQuality::High)
        {
            return Err(unsupported("GLM-Image accepts only HD quality").into());
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
        let mut body = common_body(req, model);
        if body.contains_key("quality") {
            body.insert("quality".into(), json!("hd"));
        }
        post(
            format!(
                "{}/{}images/generations",
                route.base_url.trim_end_matches('/'),
                if asynchronous { "async/" } else { "" }
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
    fn task_id(&self, body: &Value) -> Result<String, ImageError> {
        task_id_value(body.get("id"))
    }
    fn task_query(&self, route: &ImageRouteConfig, id: &str) -> Result<HttpRequest, ImageError> {
        task_get(
            format!("{}/async-result/{id}", route.base_url.trim_end_matches('/')),
            id,
        )
    }
    fn decode_task(
        &self,
        body: &Value,
        task: &ImageTaskRef,
        profile: &ProviderProfile,
    ) -> Result<ImageTaskSnapshot, ImageError> {
        task_status(body, body.get("task_status"), || {
            let result = body.get("image_result").or_else(|| body.get("data"));
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
            self.decode(&data, &task_model(task), profile)
        })
    }
}
