//! Translate portable tool content into Interactions function-result content.
use crate::protocol::{JsonPayload, LlmError};
use crate::response_json::FlatJsonKind;
use base64::Engine;

fn image_data(data: &str, mime: &str) -> Result<(), LlmError> {
    let valid = !data.is_empty()
        && data.len() <= 16 * 1024 * 1024
        && matches!(
            mime,
            "image/png" | "image/jpeg" | "image/webp" | "image/gif"
        )
        && base64::engine::general_purpose::STANDARD
            .decode(data)
            .is_ok_and(|bytes| !bytes.is_empty());
    if valid {
        Ok(())
    } else {
        Err(invalid_image())
    }
}

fn image_uri(uri: &str) -> Result<(), LlmError> {
    let valid = url::Url::parse(uri).ok().is_some_and(|uri| {
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

pub(crate) fn encode_result_block(block: &JsonPayload) -> Result<JsonPayload, LlmError> {
    let invalid = || LlmError::InvalidRequest {
        message: "unsupported or malformed Interactions function-result content".into(),
    };
    match block.string_at("/type").as_deref() {
        Some("text") if block.kind_at("/text") == Some(FlatJsonKind::String) => Ok(block.clone()),
        Some("image") => match block.string_at("/source/type").as_deref() {
            Some("base64") => {
                let data = block.string_at("/source/data").ok_or_else(invalid)?;
                let mime = block.string_at("/source/media_type").ok_or_else(invalid)?;
                image_data(&data, &mime)?;
                let source = JsonPayload::parse(r#"{"type":"image","data":null,"mime_type":null}"#)
                    .map_err(|_| invalid())?;
                let source = source.with_object_field("data", &JsonPayload::string(&data))?;
                source.with_object_field("mime_type", &JsonPayload::string(&mime))
            }
            Some("url") => {
                let uri = block.string_at("/source/url").ok_or_else(invalid)?;
                image_uri(&uri)?;
                JsonPayload::parse(r#"{"type":"image","uri":null}"#)
                    .map_err(|_| invalid())?
                    .with_object_field("uri", &JsonPayload::string(&uri))
            }
            None if block.kind_at("/source").is_none() => {
                let data = block.string_at("/data");
                let mime = block.string_at("/mime_type");
                let uri = block.string_at("/uri");
                if (block.kind_at("/data").is_some() && data.is_none())
                    || (block.kind_at("/mime_type").is_some() && mime.is_none())
                    || (block.kind_at("/uri").is_some() && uri.is_none())
                {
                    return Err(invalid());
                }
                match (data.as_deref(), mime.as_deref(), uri.as_deref()) {
                    (Some(data), Some(mime), None) => image_data(data, mime)?,
                    (None, _, Some(uri)) => {
                        if let Some(mime) = mime.as_deref() {
                            if !matches!(
                                mime,
                                "image/png" | "image/jpeg" | "image/webp" | "image/gif"
                            ) {
                                return Err(invalid_image());
                            }
                        }
                        image_uri(uri)?;
                    }
                    _ => return Err(invalid()),
                }
                if data.is_some() && uri.is_some() {
                    return Err(invalid());
                }
                Ok(block.clone())
            }
            _ => Err(invalid()),
        },
        _ => Err(invalid()),
    }
}
