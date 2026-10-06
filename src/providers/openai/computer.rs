//! Typed OpenAI Responses computer-use messages.
//!
//! A completed `computer_call` means the model finished generating its
//! requested actions. The caller still decides whether and how to execute
//! them, and must only return a screenshot output after handling the call.

use crate::protocol::{ChatRequest, ContentBlock, LlmError, NativeExtension, NativeType};
use base64::Engine;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use url::Url;

pub const OPENAI_COMPUTER_TOOL_FORMAT: &str = "openai.responses.computer_tool.v1";
pub const OPENAI_COMPUTER_CALL_FORMAT: &str = "openai.responses.computer_call.v1";
pub const OPENAI_COMPUTER_CALL_OUTPUT_FORMAT: &str = "openai.responses.computer_call_output.v1";

pub const MAX_COMPUTER_ACTIONS: usize = 128;
pub const MAX_COMPUTER_DRAG_POINTS: usize = 256;
pub const MAX_COMPUTER_KEYS: usize = 16;
pub const MAX_COMPUTER_KEY_BYTES: usize = 64;
pub const MAX_COMPUTER_TEXT_BYTES: usize = 64 * 1024;
pub const MAX_COMPUTER_ID_BYTES: usize = 64;
pub const MAX_COMPUTER_SAFETY_CHECKS: usize = 32;
pub const MAX_COMPUTER_SAFETY_TEXT_BYTES: usize = 4096;
pub const MAX_COMPUTER_SCREENSHOT_URL_BYTES: usize = 16 * 1024 * 1024;
pub const MAX_COMPUTER_COORDINATE_ABS: f64 = 1_000_000.0;

/// Marker stored in `ChatRequest.native_options`; Responses encodes it as
/// `{ "type": "computer" }`. It is deliberately not a `HostedTool` because
/// the caller, not OpenAI, executes the returned actions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiComputerToolConfig {}

impl NativeType for OpenAiComputerToolConfig {
    const FORMAT: &'static str = OPENAI_COMPUTER_TOOL_FORMAT;
}

impl ChatRequest {
    /// Return the OpenAI Responses computer-tool declaration, if enabled.
    pub fn openai_computer_tool(&self) -> Option<&OpenAiComputerToolConfig> {
        self.native_options
            .iter()
            .find_map(|extension| extension.decode::<OpenAiComputerToolConfig>().ok())
    }

    /// Enable or remove the caller-executed OpenAI Responses computer tool.
    pub fn set_openai_computer_tool(&mut self, config: Option<OpenAiComputerToolConfig>) {
        self.native_options
            .retain(|extension| !extension.is::<OpenAiComputerToolConfig>());
        if let Some(config) = config {
            self.native_options.push(
                NativeExtension::from_typed(config)
                    .expect("OpenAI computer tool config is serializable"),
            );
        }
    }
}

/// Status of model-side action generation, not a report of caller execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputerCallStatus {
    InProgress,
    Completed,
    Incomplete,
}

/// A returned computer output can also report a failed input item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputerCallOutputStatus {
    InProgress,
    Completed,
    Incomplete,
    Failed,
}

/// Fixed item tag for an OpenAI Responses `computer_call`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiComputerCallType {
    ComputerCall,
}

/// A safety check reported by OpenAI. Presence here is not consent or
/// acknowledgement; callers must preserve it and handle it explicitly.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComputerSafetyCheck {
    pub id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

/// One generated caller-executed action in a Responses computer call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ComputerAction {
    Click {
        button: ComputerMouseButton,
        x: f64,
        y: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
    },
    DoubleClick {
        x: f64,
        y: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
    },
    Drag {
        path: Vec<ComputerPoint>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
    },
    Move {
        x: f64,
        y: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
    },
    Scroll {
        scroll_x: f64,
        scroll_y: f64,
        x: f64,
        y: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        keys: Option<Vec<String>>,
    },
    Keypress {
        keys: Vec<String>,
    },
    Type {
        text: String,
    },
    Wait,
    Screenshot,
}

impl ComputerAction {
    pub fn validate(&self) -> Result<(), LlmError> {
        match self {
            Self::Click { x, y, keys, .. }
            | Self::DoubleClick { x, y, keys }
            | Self::Move { x, y, keys } => {
                validate_point(*x, *y)?;
                if let Some(keys) = keys {
                    validate_keys(keys, true)?;
                }
            }
            Self::Drag { path, keys } => {
                if !(2..=MAX_COMPUTER_DRAG_POINTS).contains(&path.len()) {
                    return Err(invalid_request(format!(
                        "computer drag path must contain 2..={MAX_COMPUTER_DRAG_POINTS} points"
                    )));
                }
                for point in path {
                    point.validate()?;
                }
                if let Some(keys) = keys {
                    validate_keys(keys, true)?;
                }
            }
            Self::Scroll {
                scroll_x,
                scroll_y,
                x,
                y,
                keys,
            } => {
                validate_point(*x, *y)?;
                validate_coordinate(*scroll_x)?;
                validate_coordinate(*scroll_y)?;
                if let Some(keys) = keys {
                    validate_keys(keys, true)?;
                }
            }
            Self::Keypress { keys } => validate_keys(keys, false)?,
            Self::Type { text } => {
                if text.len() > MAX_COMPUTER_TEXT_BYTES || text.contains('\0') {
                    return Err(invalid_request(format!(
                        "computer type text must be at most {MAX_COMPUTER_TEXT_BYTES} bytes and contain no NUL"
                    )));
                }
            }
            Self::Wait | Self::Screenshot => {}
        }
        Ok(())
    }
}

/// Mouse button names accepted by the Responses computer action schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputerMouseButton {
    Left,
    Right,
    Wheel,
    Back,
    Forward,
}

/// One point in the ordered `drag.path` array.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComputerPoint {
    pub x: f64,
    pub y: f64,
}

impl ComputerPoint {
    pub fn validate(&self) -> Result<(), LlmError> {
        validate_point(self.x, self.y)
    }
}

/// A generated `computer_call` item. `status == Completed` marks the end of
/// model generation only; it does not claim the caller has executed actions.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiComputerCall {
    #[serde(rename = "type")]
    pub r#type: OpenAiComputerCallType,
    pub id: String,
    pub call_id: String,
    pub pending_safety_checks: Vec<ComputerSafetyCheck>,
    pub status: ComputerCallStatus,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub actions: Vec<ComputerAction>,
}

impl NativeType for OpenAiComputerCall {
    const FORMAT: &'static str = OPENAI_COMPUTER_CALL_FORMAT;
}

impl OpenAiComputerCall {
    pub fn validate(&self) -> Result<(), LlmError> {
        validate_id("computer call_id", &self.call_id)?;
        validate_id("computer call id", &self.id)?;
        validate_safety_checks(&self.pending_safety_checks)?;
        if self.actions.len() > MAX_COMPUTER_ACTIONS {
            return Err(invalid_request(format!(
                "computer call may contain at most {MAX_COMPUTER_ACTIONS} actions"
            )));
        }
        for action in &self.actions {
            action.validate()?;
        }
        if self.status == ComputerCallStatus::Completed && self.actions.is_empty() {
            return Err(invalid_request(
                "completed computer call requires a nonempty actions array",
            ));
        }
        Ok(())
    }

    /// Validate a complete generated call with actions. This only validates
    /// the model's generation; it does not grant execution permission or
    /// acknowledge any pending safety check.
    pub fn validate_completed_generation(&self) -> Result<(), LlmError> {
        self.validate()?;
        if self.status != ComputerCallStatus::Completed || self.actions.is_empty() {
            return Err(invalid_request(
                "computer call must have completed generation with at least one action",
            ));
        }
        Ok(())
    }

    pub fn from_response_item(value: &Value) -> Result<Self, LlmError> {
        if value.get("type").and_then(Value::as_str) != Some("computer_call") {
            return Err(invalid_request(
                "OpenAI computer call item must have type `computer_call`",
            ));
        }
        let call: Self = serde_json::from_value(value.clone())
            .map_err(|error| invalid_request(format!("invalid OpenAI computer call: {error}")))?;
        call.validate()?;
        Ok(call)
    }

    pub fn from_extension(extension: &NativeExtension) -> Result<&Self, LlmError> {
        let call = extension.decode::<Self>()?;
        call.validate()?;
        Ok(call)
    }

    pub fn from_content_block(block: &ContentBlock) -> Result<&Self, LlmError> {
        match block {
            ContentBlock::Native { value } => Self::from_extension(value),
            _ => Err(invalid_request(
                "content block is not a typed OpenAI computer call",
            )),
        }
    }

    pub fn into_native_extension(self) -> Result<NativeExtension, LlmError> {
        self.validate()?;
        NativeExtension::from_typed(self)
    }

    pub fn into_content_block(self) -> Result<ContentBlock, LlmError> {
        Ok(ContentBlock::Native {
            value: self.into_native_extension()?,
        })
    }
}

/// Fixed screenshot subtype accepted in `computer_call_output.output`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputerScreenshotType {
    ComputerScreenshot,
}

/// Current guide screenshot detail value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputerScreenshotDetail {
    Original,
}

/// Screenshot payload returned by the caller for a computer call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComputerScreenshot {
    #[serde(rename = "type")]
    pub r#type: ComputerScreenshotType,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ComputerScreenshotDetail>,
}

impl ComputerScreenshot {
    pub fn validate(&self) -> Result<(), LlmError> {
        if self.file_id.is_some() == self.image_url.is_some() {
            return Err(invalid_request(
                "computer screenshot requires exactly one of file_id or image_url",
            ));
        }
        self.validate_sources()
    }

    fn validate_sources(&self) -> Result<(), LlmError> {
        if self
            .file_id
            .as_deref()
            .is_some_and(|id| !bounded_text(id, 512))
        {
            return Err(invalid_request("computer screenshot file_id is invalid"));
        }
        if let Some(image_url) = &self.image_url {
            if image_url.len() > MAX_COMPUTER_SCREENSHOT_URL_BYTES
                || image_url.chars().any(char::is_control)
            {
                return Err(invalid_request(
                    "computer screenshot image_url is unbounded",
                ));
            }
            if let Some(data_uri) = image_url.strip_prefix("data:") {
                let Some((metadata, payload)) = data_uri.split_once(',') else {
                    return Err(invalid_request(
                        "computer screenshot data URI must include image data",
                    ));
                };
                let Some((media_metadata, encoding)) = metadata.rsplit_once(';') else {
                    return Err(invalid_request(
                        "computer screenshot data URI must use base64 encoding",
                    ));
                };
                let media_type = media_metadata.split(';').next().unwrap_or_default();
                if !media_type
                    .get(..6)
                    .is_some_and(|prefix| prefix.eq_ignore_ascii_case("image/"))
                    || media_type.len() <= 6
                    || !encoding.eq_ignore_ascii_case("base64")
                    || payload.is_empty()
                {
                    return Err(invalid_request(
                        "computer screenshot data URI must contain nonempty base64 image data",
                    ));
                }
                base64::engine::general_purpose::STANDARD
                    .decode(payload)
                    .map_err(|_| invalid_request("computer screenshot contains invalid base64"))?;
            } else {
                let url = Url::parse(image_url)
                    .map_err(|_| invalid_request("computer screenshot image_url must be a URI"))?;
                if url.scheme() != "https"
                    || url.host_str().is_none()
                    || !url.username().is_empty()
                    || url.password().is_some()
                    || url.fragment().is_some()
                {
                    return Err(invalid_request(
                        "computer screenshot image_url must be HTTPS without credentials or fragment",
                    ));
                }
            }
        }
        Ok(())
    }

    fn validate_readback(&self) -> Result<(), LlmError> {
        // Retrieved items may omit the image URL or include it alongside a file ID.
        self.validate_sources()
    }
}

/// Fixed item tag for an OpenAI Responses `computer_call_output`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpenAiComputerCallOutputType {
    ComputerCallOutput,
}

/// A caller-generated screenshot result paired to one model-generated call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpenAiComputerCallOutput {
    #[serde(rename = "type")]
    pub r#type: OpenAiComputerCallOutputType,
    pub call_id: String,
    pub output: ComputerScreenshot,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<ComputerCallOutputStatus>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub created_by: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub acknowledged_safety_checks: Vec<ComputerSafetyCheck>,
}

impl NativeType for OpenAiComputerCallOutput {
    const FORMAT: &'static str = OPENAI_COMPUTER_CALL_OUTPUT_FORMAT;
}

impl OpenAiComputerCallOutput {
    pub fn validate(&self) -> Result<(), LlmError> {
        validate_id("computer output call_id", &self.call_id)?;
        if let Some(id) = &self.id {
            validate_id("computer output id", id)?;
        }
        self.output.validate()?;
        validate_safety_checks(&self.acknowledged_safety_checks)
    }

    fn validate_readback(&self) -> Result<(), LlmError> {
        validate_id("computer output call_id", &self.call_id)?;
        if let Some(id) = &self.id {
            validate_id("computer output id", id)?;
        }
        self.output.validate_readback()?;
        validate_safety_checks(&self.acknowledged_safety_checks)
    }

    /// Returned output metadata is readable but cannot be sent as caller input.
    pub fn validate_for_submission(&self) -> Result<(), LlmError> {
        self.validate()?;
        if self.status == Some(ComputerCallOutputStatus::Failed) || self.created_by.is_some() {
            return Err(invalid_request(
                "returned computer output status and created_by cannot be submitted as input",
            ));
        }
        Ok(())
    }

    /// Ensure this output names the call it follows and only acknowledges
    /// safety checks that the model actually reported. This does not create
    /// acknowledgements or establish that local execution succeeded.
    pub fn validate_for_call(&self, call: &OpenAiComputerCall) -> Result<(), LlmError> {
        self.validate_for_submission()?;
        call.validate_completed_generation()?;
        if self.call_id != call.call_id {
            return Err(invalid_request(
                "computer_call_output.call_id must match the generated computer_call.call_id",
            ));
        }
        if self.acknowledged_safety_checks.iter().any(|check| {
            !call.pending_safety_checks.iter().any(|pending| {
                pending.id == check.id
                    && check
                        .code
                        .as_ref()
                        .is_none_or(|code| pending.code.as_ref() == Some(code))
                    && check
                        .message
                        .as_ref()
                        .is_none_or(|message| pending.message.as_ref() == Some(message))
            })
        }) {
            return Err(invalid_request(
                "computer output can only acknowledge safety checks reported by its call",
            ));
        }
        Ok(())
    }

    pub fn from_response_item(value: &Value) -> Result<Self, LlmError> {
        if value.get("type").and_then(Value::as_str) != Some("computer_call_output") {
            return Err(invalid_request(
                "OpenAI computer output item must have type `computer_call_output`",
            ));
        }
        let output: Self = serde_json::from_value(value.clone()).map_err(|error| {
            invalid_request(format!("invalid OpenAI computer call output: {error}"))
        })?;
        output.validate_readback()?;
        Ok(output)
    }

    pub fn from_extension(extension: &NativeExtension) -> Result<&Self, LlmError> {
        let output = extension.decode::<Self>()?;
        output.validate()?;
        Ok(output)
    }

    pub fn from_content_block(block: &ContentBlock) -> Result<&Self, LlmError> {
        match block {
            ContentBlock::Native { value } => Self::from_extension(value),
            _ => Err(invalid_request(
                "content block is not a typed OpenAI computer call output",
            )),
        }
    }

    pub fn into_native_extension(self) -> Result<NativeExtension, LlmError> {
        self.validate_for_submission()?;
        NativeExtension::from_typed(self)
    }

    pub fn into_content_block(self) -> Result<ContentBlock, LlmError> {
        Ok(ContentBlock::Native {
            value: self.into_native_extension()?,
        })
    }

    pub fn into_content_block_for(
        self,
        call: &OpenAiComputerCall,
    ) -> Result<ContentBlock, LlmError> {
        self.validate_for_call(call)?;
        Ok(ContentBlock::Native {
            value: NativeExtension::from_typed(self)?,
        })
    }
}

fn validate_safety_checks(checks: &[ComputerSafetyCheck]) -> Result<(), LlmError> {
    if checks.len() > MAX_COMPUTER_SAFETY_CHECKS {
        return Err(invalid_request(format!(
            "computer call may contain at most {MAX_COMPUTER_SAFETY_CHECKS} safety checks"
        )));
    }
    let mut ids = std::collections::BTreeSet::new();
    for check in checks {
        validate_id("computer safety check id", &check.id)?;
        if !ids.insert(check.id.as_str())
            || check
                .code
                .as_deref()
                .is_some_and(|code| !bounded_text(code, MAX_COMPUTER_KEY_BYTES))
            || check.message.as_deref().is_some_and(|message| {
                message.len() > MAX_COMPUTER_SAFETY_TEXT_BYTES
                    || message.chars().any(char::is_control)
            })
        {
            return Err(invalid_request(
                "computer safety checks require unique IDs and bounded code/message fields",
            ));
        }
    }
    Ok(())
}

fn validate_keys(keys: &[String], allow_empty: bool) -> Result<(), LlmError> {
    if (!allow_empty && keys.is_empty())
        || keys.len() > MAX_COMPUTER_KEYS
        || keys
            .iter()
            .any(|key| !bounded_text(key, MAX_COMPUTER_KEY_BYTES))
    {
        return Err(invalid_request(format!(
            "computer key list must contain {}..={MAX_COMPUTER_KEYS} bounded key names",
            if allow_empty { 0 } else { 1 }
        )));
    }
    Ok(())
}

fn validate_point(x: f64, y: f64) -> Result<(), LlmError> {
    validate_coordinate(x)?;
    validate_coordinate(y)
}

fn validate_coordinate(value: f64) -> Result<(), LlmError> {
    if !value.is_finite() || value.abs() > MAX_COMPUTER_COORDINATE_ABS {
        return Err(invalid_request(format!(
            "computer coordinates and scroll deltas must be finite and within +/-{MAX_COMPUTER_COORDINATE_ABS}"
        )));
    }
    Ok(())
}

fn validate_id(field: &str, value: &str) -> Result<(), LlmError> {
    if !bounded_text(value, MAX_COMPUTER_ID_BYTES) {
        return Err(invalid_request(format!(
            "{field} must be nonempty, control-free, and at most {MAX_COMPUTER_ID_BYTES} bytes"
        )));
    }
    Ok(())
}

fn bounded_text(value: &str, max_bytes: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max_bytes && !value.chars().any(char::is_control)
}

fn invalid_request(message: impl Into<String>) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(status: ComputerCallStatus, actions: Vec<ComputerAction>) -> OpenAiComputerCall {
        OpenAiComputerCall {
            r#type: OpenAiComputerCallType::ComputerCall,
            id: "cci_1".into(),
            call_id: "call_1".into(),
            pending_safety_checks: vec![],
            status,
            actions,
        }
    }

    fn screenshot() -> ComputerScreenshot {
        ComputerScreenshot {
            r#type: ComputerScreenshotType::ComputerScreenshot,
            file_id: None,
            image_url: Some("data:image/png;base64,AA==".into()),
            detail: Some(ComputerScreenshotDetail::Original),
        }
    }

    #[test]
    fn computer_call_actions_are_typed_ordered_and_bounded() {
        let value = json!({
            "type":"computer_call",
            "id":"cci_1",
            "call_id":"call_1",
            "pending_safety_checks":[],
            "status":"completed",
            "actions":[
                {"type":"click","button":"left","x":12.5,"y":19,"keys":["SHIFT"]},
                {"type":"drag","path":[{"x":1,"y":2},{"x":3,"y":4}]},
                {"type":"scroll","scroll_x":0,"scroll_y":120,"x":10,"y":20},
                {"type":"screenshot"}
            ]
        });
        let parsed = OpenAiComputerCall::from_response_item(&value).unwrap();
        parsed.validate_completed_generation().unwrap();
        assert_eq!(parsed.actions.len(), 4);
        assert!(matches!(
            parsed.actions.last(),
            Some(ComputerAction::Screenshot)
        ));
    }

    #[test]
    fn completed_generation_requires_actions_and_screenshot_only_is_valid() {
        let screenshot_call = call(
            ComputerCallStatus::Completed,
            vec![ComputerAction::Screenshot],
        );
        screenshot_call.validate_completed_generation().unwrap();

        let incomplete = call(ComputerCallStatus::InProgress, vec![]);
        assert!(incomplete.validate().is_ok());
        assert!(incomplete.validate_completed_generation().is_err());
        let no_actions = call(ComputerCallStatus::Completed, vec![]);
        assert!(no_actions.validate().is_err());
    }

    #[test]
    fn call_and_output_native_extensions_round_trip_without_auto_acknowledging() {
        let mut call = call(
            ComputerCallStatus::Completed,
            vec![ComputerAction::Screenshot],
        );
        call.pending_safety_checks = vec![ComputerSafetyCheck {
            id: "check_1".into(),
            code: Some("confirm".into()),
            message: Some("Confirm before continuing".into()),
        }];
        let call_block = call.clone().into_content_block().unwrap();
        let decoded = OpenAiComputerCall::from_content_block(&call_block).unwrap();
        assert_eq!(decoded, &call);

        let output = OpenAiComputerCallOutput {
            r#type: OpenAiComputerCallOutputType::ComputerCallOutput,
            call_id: "call_1".into(),
            output: screenshot(),
            id: None,
            status: None,
            created_by: None,
            acknowledged_safety_checks: vec![],
        };
        output.validate_for_call(&call).unwrap();
        assert!(output.acknowledged_safety_checks.is_empty());
        let output_block = output.into_content_block_for(&call).unwrap();
        assert_eq!(
            OpenAiComputerCallOutput::from_content_block(&output_block)
                .unwrap()
                .call_id,
            "call_1"
        );
    }

    #[test]
    fn output_call_id_must_match_and_safety_checks_are_not_fabricated() {
        let call = call(
            ComputerCallStatus::Completed,
            vec![ComputerAction::Screenshot],
        );
        let output = OpenAiComputerCallOutput {
            r#type: OpenAiComputerCallOutputType::ComputerCallOutput,
            call_id: "different_call".into(),
            output: screenshot(),
            id: None,
            status: None,
            created_by: None,
            acknowledged_safety_checks: vec![],
        };
        assert!(output.validate_for_call(&call).is_err());
        assert!(output.acknowledged_safety_checks.is_empty());
    }

    #[test]
    fn optional_modifier_keys_may_be_empty_but_keypress_may_not() {
        assert!(ComputerAction::Click {
            button: ComputerMouseButton::Left,
            x: 1.0,
            y: 2.0,
            keys: Some(vec![]),
        }
        .validate()
        .is_ok());
        assert!(ComputerAction::Keypress { keys: vec![] }
            .validate()
            .is_err());
    }

    #[test]
    fn provided_safety_acknowledgement_fields_must_match_the_reported_check() {
        let call = OpenAiComputerCall {
            pending_safety_checks: vec![ComputerSafetyCheck {
                id: "check_1".into(),
                code: Some("confirm".into()),
                message: Some("Confirm before continuing".into()),
            }],
            ..call(
                ComputerCallStatus::Completed,
                vec![ComputerAction::Screenshot],
            )
        };
        let output = OpenAiComputerCallOutput {
            r#type: OpenAiComputerCallOutputType::ComputerCallOutput,
            call_id: "call_1".into(),
            output: screenshot(),
            id: None,
            status: None,
            created_by: None,
            acknowledged_safety_checks: vec![ComputerSafetyCheck {
                id: "check_1".into(),
                code: Some("different-code".into()),
                message: Some("Confirm before continuing".into()),
            }],
        };
        assert!(output.validate_for_call(&call).is_err());
        let mut wrong_message = output.clone();
        wrong_message.acknowledged_safety_checks[0].code = Some("confirm".into());
        wrong_message.acknowledged_safety_checks[0].message = Some("Different message".into());
        assert!(wrong_message.validate_for_call(&call).is_err());
        wrong_message.acknowledged_safety_checks[0].message =
            Some("Confirm before continuing".into());
        wrong_message.acknowledged_safety_checks[0].id = "unknown_check".into();
        assert!(wrong_message.validate_for_call(&call).is_err());
    }

    #[test]
    fn screenshot_urls_are_bounded_https_or_image_data_without_credentials_or_fragment() {
        let mut screenshot = screenshot();
        assert!(screenshot.validate().is_ok());
        screenshot.image_url = Some("https://example.test/capture.png".into());
        assert!(screenshot.validate().is_ok());
        for invalid in [
            "http://example.test/capture.png",
            "https://user:pass@example.test/capture.png",
            "https://example.test/capture.png#fragment",
            "data:text/html;base64,AA==",
            "data:image/png;base64,%%%",
            "data:image/png;base64,AA=",
            "data:image/png,AA==",
        ] {
            screenshot.image_url = Some(invalid.into());
            assert!(screenshot.validate().is_err(), "accepted {invalid}");
        }
    }

    #[test]
    fn unknown_tags_fields_and_nonfinite_or_out_of_bounds_values_fail() {
        assert!(OpenAiComputerCall::from_response_item(&json!({
            "type":"computer_call",
            "call_id":"call_1",
            "status":"completed",
            "actions":[{"type":"unknown"}]
        }))
        .is_err());
        assert!(OpenAiComputerCall::from_response_item(&json!({
            "type":"computer_call",
            "call_id":"call_1",
            "status":"completed",
            "unexpected":true,
            "actions":[{"type":"screenshot"}]
        }))
        .is_err());
        assert!(ComputerAction::Click {
            button: ComputerMouseButton::Left,
            x: f64::INFINITY,
            y: 1.0,
            keys: None,
        }
        .validate()
        .is_err());
        assert!(ComputerAction::Move {
            x: MAX_COMPUTER_COORDINATE_ABS + 1.0,
            y: 1.0,
            keys: None,
        }
        .validate()
        .is_err());
        assert!(ComputerAction::Drag {
            path: vec![ComputerPoint { x: 1.0, y: 1.0 }],
            keys: None,
        }
        .validate()
        .is_err());
    }
}
