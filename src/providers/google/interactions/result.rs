//! Translate portable tool content into Interactions function-result content.
use crate::protocol::LlmError;
use base64::Engine;
use serde_json::{json, Value};

fn image_data(data: &Value, mime: &Value) -> Result<(), LlmError> {
    let valid = data
        .as_str()
        .zip(mime.as_str())
        .is_some_and(|(data, mime)| {
            !data.is_empty()
                && data.len() <= 16 * 1024 * 1024
                && matches!(
                    mime,
                    "image/png" | "image/jpeg" | "image/webp" | "image/gif"
                )
                && base64::engine::general_purpose::STANDARD
                    .decode(data)
                    .is_ok_and(|bytes| !bytes.is_empty())
        });
    if valid {
        Ok(())
    } else {
        Err(invalid_image())
    }
}

fn image_uri(uri: &Value) -> Result<(), LlmError> {
    let valid = uri
        .as_str()
        .and_then(|uri| url::Url::parse(uri).ok())
        .is_some_and(|uri| {
            matches!(uri.scheme(), "https" | "gs")
                && uri.host_str().is_some()
                && uri.username().is_empty()
                && uri.password().is_none()
                && uri.fragment().is_none()
        });
    if valid {
        Ok(())
    } else {
        Err(invalid_image())
    }
}

fn invalid_image() -> LlmError {
    LlmError::InvalidRequest {
        message: "malformed Interactions image data, MIME type or URI".into(),
    }
}

pub(crate) fn encode_result_block(block: &Value) -> Result<Value, LlmError> {
    let invalid = || LlmError::InvalidRequest {
        message: "unsupported or malformed Interactions function-result content".into(),
    };
    match block["type"].as_str() {
        Some("text") if block["text"].is_string() => Ok(block.clone()),
        Some("image") => {
            let source = &block["source"];
            match source["type"].as_str() {
                Some("base64")
                    if source["data"].is_string() && source["media_type"].is_string() =>
                {
                    image_data(&source["data"], &source["media_type"])?;
                    Ok(
                        json!({"type":"image","data":source["data"],"mime_type":source["media_type"]}),
                    )
                }
                Some("url") if source["url"].is_string() => {
                    image_uri(&source["url"])?;
                    Ok(json!({"type":"image","uri":source["url"]}))
                }
                None if block["data"].is_string() && block["mime_type"].is_string() => {
                    if block.get("uri").is_some() {
                        return Err(invalid());
                    }
                    image_data(&block["data"], &block["mime_type"])?;
                    Ok(block.clone())
                }
                None if block["uri"].is_string() => {
                    if block.get("data").is_some() {
                        return Err(invalid());
                    }
                    if let Some(mime) = block.get("mime_type") {
                        if !matches!(
                            mime.as_str(),
                            Some("image/png" | "image/jpeg" | "image/webp" | "image/gif")
                        ) {
                            return Err(invalid_image());
                        }
                    }
                    image_uri(&block["uri"])?;
                    Ok(block.clone())
                }
                _ => Err(invalid()),
            }
        }
        _ => Err(invalid()),
    }
}
