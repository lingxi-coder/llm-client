//! Read-only listing and retrieval of voices available through xAI text-to-speech.

use super::{
    invalid, invalid_response, parse_error_body, response_id, validate_credential,
    XaiAudioCredentials, XaiAudioError, XaiAudioScope, XaiAudioService, MAX_JSON_RESPONSE_BYTES,
};
use crate::{
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest},
};
use bytes::Bytes;
use serde_json::Value;
use std::collections::HashSet;
use url::Url;

/// One voice returned by xAI's TTS voice list or detail endpoint.
///
/// `native` retains the complete provider object, including fields this client
/// does not interpret. `language` is optional because the TTS guide's example
/// uses `voice_id` and `name` without requiring a language field.
#[derive(Debug, Clone, PartialEq)]
pub struct XaiVoice {
    pub voice_id: String,
    pub name: String,
    pub language: Option<String>,
    pub native: Value,
}

/// A scoped snapshot returned by xAI's read-only TTS voice-list endpoint.
#[derive(Debug, Clone, PartialEq)]
pub struct XaiVoiceList {
    pub scope: XaiAudioScope,
    pub voices: Vec<XaiVoice>,
    /// Complete response JSON, preserving fields unknown to this client.
    pub native: Value,
    pub request_id: Option<String>,
}

/// One voice returned by xAI's `GET /v1/tts/voices/{voice_id}` endpoint.
/// The typed voice and complete response are both retained for forward
/// compatibility with provider-added metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct XaiVoiceDetails {
    pub scope: XaiAudioScope,
    pub voice: XaiVoice,
    pub native: Value,
    pub request_id: Option<String>,
}

impl<'a> XaiAudioService<'a> {
    /// List the voices currently available from xAI text-to-speech.
    ///
    /// This calls `GET /v1/tts/voices` once. The operation is read-only and is
    /// never retried automatically.
    pub async fn list_voices(
        &self,
        credentials: &XaiAudioCredentials,
    ) -> Result<XaiVoiceList, XaiAudioError> {
        validate_credential(&credentials.api_key)?;

        // Start from the already validated `/tts` route and append the fixed
        // `voices` segment through URL path APIs rather than accepting a path.
        let mut url = Url::parse(&self.route_url("tts")?)
            .map_err(|_| invalid("configured TTS route URL is invalid"))?;
        url.path_segments_mut()
            .map_err(|_| invalid("configured TTS route URL cannot accept path segments"))?
            .push("voices");

        let deadline = Deadline::after(Some(self.config.request_timeout));
        let request = HttpRequest {
            method: "GET".into(),
            url: url.into(),
            headers: vec![(
                "authorization".into(),
                format!("Bearer {}", credentials.api_key.expose_secret()),
            )],
            body: Bytes::new(),
            timeout: deadline.remaining()?,
        };
        // A failed GET has no provider-side mutation outcome to report.
        let response = HttpExecutor::new(self.transport)
            .with_deadline(deadline)
            .send(request)
            .await
            .map_err(XaiAudioError::Llm)?;
        let request_id = response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned);

        if !(200..300).contains(&response.status) {
            let response = HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES))
                .await
                .map_err(XaiAudioError::Llm)?;
            return Err(XaiAudioError::Provider {
                operation: "list-voices",
                status: response.status,
                request_id,
                body: Box::new(parse_error_body(&response.body)),
            });
        }

        let response = HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES))
            .await
            .map_err(XaiAudioError::Llm)?;
        let native = serde_json::from_slice::<Value>(&response.body).map_err(|_| {
            invalid_response(
                "list-voices",
                "response body is not JSON",
                request_id.clone(),
                Value::Null,
            )
        })?;
        decode_voice_list(native, self.scope.clone(), request_id)
    }

    /// Fetch one voice's details by its exact xAI voice identifier.
    ///
    /// This calls `GET /v1/tts/voices/{voice_id}` once. The path parameter is
    /// appended as one URL segment, and the returned identifier must exactly
    /// match the requested value. This read-only operation is never retried.
    pub async fn get_voice(
        &self,
        voice_id: &str,
        credentials: &XaiAudioCredentials,
    ) -> Result<XaiVoiceDetails, XaiAudioError> {
        validate_voice_id_segment(voice_id)?;
        validate_credential(&credentials.api_key)?;

        let mut url = Url::parse(&self.route_url("tts")?)
            .map_err(|_| invalid("configured TTS route URL is invalid"))?;
        url.path_segments_mut()
            .map_err(|_| invalid("configured TTS route URL cannot accept path segments"))?
            .push("voices")
            .push(voice_id);

        let deadline = Deadline::after(Some(self.config.request_timeout));
        let response = HttpExecutor::new(self.transport)
            .with_deadline(deadline)
            .send(HttpRequest {
                method: "GET".into(),
                url: url.into(),
                headers: vec![(
                    "authorization".into(),
                    format!("Bearer {}", credentials.api_key.expose_secret()),
                )],
                body: Bytes::new(),
                timeout: deadline.remaining()?,
            })
            .await
            .map_err(XaiAudioError::Llm)?;
        let request_id = response_id(&response.headers);
        if !(200..300).contains(&response.status) {
            let response =
                HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES)).await?;
            return Err(XaiAudioError::Provider {
                operation: "get-voice",
                status: response.status,
                request_id,
                body: Box::new(parse_error_body(&response.body)),
            });
        }
        let response =
            HttpExecutor::collect_response(response, Some(MAX_JSON_RESPONSE_BYTES)).await?;
        let native = serde_json::from_slice::<Value>(&response.body).map_err(|_| {
            invalid_response(
                "get-voice",
                "response body is not JSON",
                request_id.clone(),
                Value::String(String::from_utf8_lossy(&response.body).into_owned()),
            )
        })?;
        decode_voice_details(native, voice_id, self.scope.clone(), request_id)
    }
}

fn validate_voice_id_segment(voice_id: &str) -> Result<(), XaiAudioError> {
    if voice_id.is_empty()
        || voice_id.trim() != voice_id
        || voice_id.chars().any(char::is_control)
        || voice_id.contains('/')
        || voice_id.contains('\\')
        || voice_id.contains('%')
        || voice_id == "."
        || voice_id == ".."
    {
        return Err(invalid(
            "voice_id must be nonempty and safe as one URL path segment without encoded path syntax",
        ));
    }
    Ok(())
}

fn decode_voice_details(
    native: Value,
    expected_voice_id: &str,
    scope: XaiAudioScope,
    request_id: Option<String>,
) -> Result<XaiVoiceDetails, XaiAudioError> {
    let Some(voice_id) = native
        .get("voice_id")
        .and_then(Value::as_str)
        .filter(|voice_id| !voice_id.trim().is_empty())
    else {
        return Err(invalid_response(
            "get-voice",
            "response has no nonempty voice_id",
            request_id,
            native,
        ));
    };
    if voice_id != expected_voice_id {
        return Err(invalid_response(
            "get-voice",
            "response voice_id does not exactly match the requested voice",
            request_id,
            native,
        ));
    }
    let Some(name) = native
        .get("name")
        .and_then(Value::as_str)
        .filter(|name| !name.trim().is_empty())
    else {
        return Err(invalid_response(
            "get-voice",
            "response has no nonempty name",
            request_id,
            native,
        ));
    };
    let language = match native.get("language") {
        None | Some(Value::Null) => None,
        Some(Value::String(language)) if !language.trim().is_empty() => Some(language.clone()),
        Some(_) => {
            return Err(invalid_response(
                "get-voice",
                "response language must be a nonempty string when present",
                request_id,
                native,
            ));
        }
    };
    let voice = XaiVoice {
        voice_id: voice_id.to_owned(),
        name: name.to_owned(),
        language,
        native: native.clone(),
    };
    Ok(XaiVoiceDetails {
        scope,
        voice,
        native,
        request_id,
    })
}

fn decode_voice_list(
    native: Value,
    scope: XaiAudioScope,
    request_id: Option<String>,
) -> Result<XaiVoiceList, XaiAudioError> {
    let Some(voice_values) = native.get("voices").and_then(Value::as_array) else {
        return Err(invalid_response(
            "list-voices",
            "response has no voices array",
            request_id,
            native,
        ));
    };

    let mut seen_ids = HashSet::with_capacity(voice_values.len());
    let mut voices = Vec::with_capacity(voice_values.len());
    for (index, value) in voice_values.iter().enumerate() {
        let Some(voice_id) = value
            .get("voice_id")
            .and_then(Value::as_str)
            .filter(|id| !id.trim().is_empty())
        else {
            return Err(invalid_response(
                "list-voices",
                &format!("voices[{index}] has no nonempty voice_id"),
                request_id.clone(),
                native.clone(),
            ));
        };
        if !seen_ids.insert(voice_id.to_ascii_lowercase()) {
            return Err(invalid_response(
                "list-voices",
                &format!("voices contains duplicate voice_id `{voice_id}`"),
                request_id.clone(),
                native.clone(),
            ));
        }
        let Some(name) = value
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.trim().is_empty())
        else {
            return Err(invalid_response(
                "list-voices",
                &format!("voices[{index}] has no nonempty name"),
                request_id.clone(),
                native.clone(),
            ));
        };
        let language = match value.get("language") {
            None | Some(Value::Null) => None,
            Some(Value::String(language)) if !language.trim().is_empty() => Some(language.clone()),
            Some(_) => {
                return Err(invalid_response(
                    "list-voices",
                    &format!("voices[{index}].language must be a nonempty string when present"),
                    request_id.clone(),
                    native.clone(),
                ));
            }
        };
        voices.push(XaiVoice {
            voice_id: voice_id.to_owned(),
            name: name.to_owned(),
            language,
            native: value.clone(),
        });
    }

    Ok(XaiVoiceList {
        scope,
        voices,
        native,
        request_id,
    })
}
