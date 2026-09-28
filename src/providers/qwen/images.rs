use crate::images::wire::*;
use crate::images::{ImageAdapter, ImageError};
use crate::protocol::*;
use crate::transport::HttpRequest;
use serde_json::{json, Value};

use serde_json::Map;
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
pub(crate) struct QwenImages;
impl ImageAdapter for QwenImages {
    fn api(&self) -> ImageApi {
        ImageApi::Qwen
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
                options: &[
                    "negative_prompt",
                    "seed",
                    "prompt_extend",
                    "prompt_extend_mode",
                    "enable_thinking",
                    "watermark",
                ],
                ratio: false,
                size: true,
                count: true,
                format: false,
                quality: false,
            },
            asynchronous,
        )?;

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
        let mut body = common_body(req, model);
        let mut headers = json_headers();
        let endpoint = if asynchronous {
            headers.push(("X-DashScope-Async".into(), "enable".into()));
            body = qwen_task_body(req, model, &body)?;
            format!(
                "{}/services/aigc/image-generation/generation",
                route
                    .task_base_url
                    .as_ref()
                    .ok_or_else(|| unsupported("native task endpoint not configured"))?
                    .trim_end_matches('/')
            )
        } else {
            let images = request_images(req)
                .into_iter()
                .map(image_data_uri)
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
            format!(
                "{}/images/generations",
                route.base_url.trim_end_matches('/')
            )
        };
        post(endpoint, body, headers)
    }
    fn decode(
        &self,
        body: &Value,
        model: &ImageModelProfile,
        profile: &ProviderProfile,
    ) -> Result<ImageResponse, ImageError> {
        let mut images = openai_images(body);
        images.extend(choices_images(body));
        response(body, model, profile, images, None, false, false)
    }
    fn task_id(&self, body: &Value) -> Result<String, ImageError> {
        task_id_value(body.pointer("/output/task_id"))
    }
    fn task_query(&self, route: &ImageRouteConfig, id: &str) -> Result<HttpRequest, ImageError> {
        task_get(
            format!(
                "{}/tasks/{id}",
                route
                    .task_base_url
                    .as_ref()
                    .ok_or_else(|| unsupported("task API root missing"))?
                    .trim_end_matches('/')
            ),
            id,
        )
    }
    fn decode_task(
        &self,
        body: &Value,
        task: &ImageTaskRef,
        profile: &ProviderProfile,
    ) -> Result<ImageTaskSnapshot, ImageError> {
        task_status(body, body.pointer("/output/task_status"), || {
            self.decode(body, &task_model(task), profile)
        })
    }
    fn check_response(&self, body: &Value) -> Result<(), ImageError> {
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
        Ok(())
    }
}

pub(crate) struct WanImages;
impl ImageAdapter for WanImages {
    fn api(&self) -> ImageApi {
        ImageApi::Wan
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
                options: &["watermark", "thinking_mode"],
                ratio: false,
                size: true,
                count: true,
                format: false,
                quality: false,
            },
            asynchronous,
        )?;

        require_size(
            req,
            |size| {
                matches!(size, ImageSize::Pixels { .. })
                    || matches!(size,ImageSize::Tier{name} if matches!(name.as_str(),"1K"|"2K"|"4K"))
            },
            "Wan image size must be pixels, 1K, 2K, or 4K",
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
        let (prompt, _) = input_output(req);
        let mut content = request_images(req)
            .into_iter()
            .map(|image| image_data_uri(image).map(|url| json!({"image":url})))
            .collect::<Result<Vec<_>, _>>()?;
        content.push(json!({"text":prompt}));
        let mut parameters = common_body(req, model);
        parameters.remove("model");
        parameters.remove("prompt");
        if let Some(size) = parameters.get_mut("size") {
            if let Some(s) = size.as_str() {
                *size = json!(s.replace('x', "*"));
            }
        }
        let body = json!({"model":model.request_model,"input":{"messages":[{"role":"user","content":content}]},"parameters":parameters});
        let root = if asynchronous {
            route.task_base_url.as_ref().unwrap_or(&route.base_url)
        } else {
            &route.base_url
        };
        let mut headers = json_headers();
        if asynchronous {
            headers.push(("X-DashScope-Async".into(), "enable".into()));
        }
        post(
            format!(
                "{}/services/aigc/{}/generation",
                root.trim_end_matches('/'),
                if asynchronous {
                    "image-generation"
                } else {
                    "multimodal-generation"
                }
            ),
            body.as_object().unwrap().clone(),
            headers,
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
            choices_images(body),
            None,
            false,
            false,
        )
    }
    fn task_id(&self, body: &Value) -> Result<String, ImageError> {
        task_id_value(body.pointer("/output/task_id"))
    }
    fn task_query(&self, route: &ImageRouteConfig, id: &str) -> Result<HttpRequest, ImageError> {
        task_get(
            format!(
                "{}/tasks/{id}",
                route
                    .task_base_url
                    .as_ref()
                    .ok_or_else(|| unsupported("task API root missing"))?
                    .trim_end_matches('/')
            ),
            id,
        )
    }
    fn decode_task(
        &self,
        body: &Value,
        task: &ImageTaskRef,
        profile: &ProviderProfile,
    ) -> Result<ImageTaskSnapshot, ImageError> {
        task_status(body, body.pointer("/output/task_status"), || {
            self.decode(body, &task_model(task), profile)
        })
    }
    fn check_response(&self, body: &Value) -> Result<(), ImageError> {
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
        Ok(())
    }
}
