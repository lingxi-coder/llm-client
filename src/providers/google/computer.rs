//! Data-only Gemini Interactions Desktop declaration, operation and receipt helpers.
//! Coordinates are integers in 0..=999, scaled by screenshot dimension / 1000.
//! Pixel scrolling defaults to 300; wait defaults to one second.
use crate::protocol::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;
pub const CALL_FORMAT: &str = "google.interactions.computer_call.v1";
pub const RESULT_FORMAT: &str = "google.interactions.computer_result.v1";
pub const DESKTOP_PROTOCOL_VERSION: &str = "interactions.desktop.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiComputerToolConfig {
    pub excluded_predefined_functions: Vec<String>,
    pub enable_prompt_injection_detection: bool,
}
impl NativeType for GeminiComputerToolConfig {
    const FORMAT: &'static str = "google.interactions.computer_tool.v1";
}
impl GeminiComputerToolConfig {
    pub fn validate(&self) -> Result<(), LlmError> {
        let mut seen = BTreeSet::new();
        if !self.enable_prompt_injection_detection {
            return Err(invalid(
                "desktop declaration requires prompt injection detection",
            ));
        }
        for name in &self.excluded_predefined_functions {
            if !is_desktop_function(name) || !seen.insert(name) {
                return Err(invalid("unknown or duplicate excluded desktop function"));
            }
        }
        Ok(())
    }
    pub fn tool(&self) -> Value {
        json!({"type":"computer_use","environment":"desktop","excluded_predefined_functions":self.excluded_predefined_functions,"enable_prompt_injection_detection":self.enable_prompt_injection_detection})
    }
}
const FUNCTIONS: &[(&str, ComputerOperationKind)] = &[
    ("click", ComputerOperationKind::Click),
    ("double_click", ComputerOperationKind::Click),
    ("triple_click", ComputerOperationKind::Click),
    ("right_click", ComputerOperationKind::Click),
    ("middle_click", ComputerOperationKind::Click),
    ("move", ComputerOperationKind::Move),
    ("mouse_down", ComputerOperationKind::MouseDown),
    ("mouse_up", ComputerOperationKind::MouseUp),
    ("type", ComputerOperationKind::Type),
    ("drag_and_drop", ComputerOperationKind::Drag),
    ("scroll", ComputerOperationKind::Scroll),
    ("press_key", ComputerOperationKind::Key),
    ("hotkey", ComputerOperationKind::Key),
    ("key_down", ComputerOperationKind::KeyDown),
    ("key_up", ComputerOperationKind::KeyUp),
    ("wait", ComputerOperationKind::Wait),
    ("take_screenshot", ComputerOperationKind::Screenshot),
];
pub(crate) fn is_desktop_function(name: &str) -> bool {
    FUNCTIONS.iter().any(|(function, _)| *function == name)
}
fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
pub fn declare(
    request: &mut ChatRequest,
    capabilities: &ComputerCapabilities,
    frame: &ComputerFrame,
) -> Result<(), LlmError> {
    frame.validate()?;
    if !capabilities.supports(ComputerOperationKind::Screenshot) {
        return Err(invalid("Gemini desktop requires screenshot capability"));
    }
    if request
        .tools
        .iter()
        .any(|tool| tool.name == "computer" || is_desktop_function(&tool.name))
    {
        return Err(invalid(
            "Gemini native desktop and management/member functions cannot coexist in a turn",
        ));
    }
    if request
        .native_options
        .iter()
        .any(|extension| extension.is::<GeminiComputerToolConfig>())
    {
        return Err(invalid("duplicate Gemini desktop declaration"));
    }
    request
        .native_options
        .push(NativeExtension::from_typed(GeminiComputerToolConfig {
            excluded_predefined_functions: FUNCTIONS
                .iter()
                .filter(|(_, kind)| !capabilities.supports(*kind))
                .map(|(name, _)| name.to_string())
                .collect(),
            enable_prompt_injection_detection: true,
        })?);
    Ok(())
}
fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str, LlmError> {
    value[key]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid(format!("Gemini desktop argument {key} is required")))
}
fn coordinate(value: &Value, key: &str, dimension: u32) -> Result<f64, LlmError> {
    let coordinate = value[key].as_u64().filter(|n| *n <= 999).ok_or_else(|| {
        invalid(format!(
            "Gemini desktop {key} must be an integer within 0..=999"
        ))
    })?;
    Ok(coordinate as f64 / 1000.0 * dimension as f64)
}
fn point(
    value: &Value,
    x: &str,
    y: &str,
    frame: &ComputerFrame,
) -> Result<ComputerPoint, LlmError> {
    Ok(ComputerPoint {
        x: coordinate(value, x, frame.width)?,
        y: coordinate(value, y, frame.height)?,
    })
}
fn operation(
    name: &str,
    args: &Value,
    frame: &ComputerFrame,
) -> Result<ComputerOperation, LlmError> {
    let target = || point(args, "x", "y", frame).map(|point| ComputerTarget::Position { point });
    let modifiers = vec![];
    Ok(match name {
        "click" | "double_click" | "triple_click" | "right_click" | "middle_click" => {
            ComputerOperation::Click {
                target: target()?,
                button: match name {
                    "right_click" => ComputerMouseButton::Right,
                    "middle_click" => ComputerMouseButton::Middle,
                    _ => ComputerMouseButton::Left,
                },
                count: match name {
                    "double_click" => 2,
                    "triple_click" => 3,
                    _ => 1,
                },
                modifiers,
            }
        }
        "move" => ComputerOperation::Move {
            point: point(args, "x", "y", frame)?,
            modifiers,
        },
        "mouse_down" => ComputerOperation::MouseDown {
            target: Some(point(args, "x", "y", frame)?),
            modifiers,
        },
        "mouse_up" => ComputerOperation::MouseUp {
            target: Some(point(args, "x", "y", frame)?),
            modifiers,
        },
        "type" => ComputerOperation::Type {
            text: args["text"]
                .as_str()
                .ok_or_else(|| invalid("Gemini type requires text"))?
                .into(),
            press_enter: match args.get("press_enter") {
                None => false,
                Some(value) => value
                    .as_bool()
                    .ok_or_else(|| invalid("press_enter must be a boolean"))?,
            },
        },
        "drag_and_drop" => ComputerOperation::Drag {
            path: vec![
                point(args, "start_x", "start_y", frame)?,
                point(args, "end_x", "end_y", frame)?,
            ],
            modifiers,
        },
        "scroll" => {
            let magnitude = match args.get("magnitude_in_pixels") {
                None => 300,
                Some(value) => value
                    .as_u64()
                    .filter(|n| *n <= 999)
                    .ok_or_else(|| invalid("Gemini scroll magnitude must be 0..=999 pixels"))?,
            } as f64;
            let (delta_x, delta_y) = match string(args, "direction")? {
                "up" => (0.0, -magnitude),
                "down" => (0.0, magnitude),
                "left" => (-magnitude, 0.0),
                "right" => (magnitude, 0.0),
                _ => return Err(invalid("unknown Gemini scroll direction")),
            };
            ComputerOperation::Scroll {
                target: target()?,
                delta_x,
                delta_y,
                modifiers,
            }
        }
        "press_key" => ComputerOperation::Key {
            keys: vec![string(args, "key")?.into()],
        },
        "hotkey" => ComputerOperation::Key {
            keys: args["keys"]
                .as_array()
                .ok_or_else(|| invalid("Gemini hotkey requires keys array"))?
                .iter()
                .map(|key| {
                    key.as_str()
                        .map(str::to_owned)
                        .ok_or_else(|| invalid("Gemini hotkey keys must be strings"))
                })
                .collect::<Result<_, _>>()?,
        },
        "key_down" => ComputerOperation::KeyDown {
            key: string(args, "key")?.into(),
        },
        "key_up" => ComputerOperation::KeyUp {
            key: string(args, "key")?.into(),
        },
        "wait" => {
            ComputerOperation::Wait {
                duration_seconds: match args.get("seconds") {
                    None => 1.0,
                    Some(value) => value.as_u64().filter(|n| *n <= 300).ok_or_else(|| {
                        invalid("Gemini wait seconds must be integer within 0..=300")
                    })? as f64,
                },
            }
        }
        "take_screenshot" => ComputerOperation::Screenshot,
        _ => {
            return Err(invalid(format!(
                "unsupported Gemini desktop function {name}"
            )))
        }
    })
}
pub fn decode(
    blocks: &[ContentBlock],
    continuation: Option<&ContinuationRef>,
    frame: &ComputerFrame,
) -> Result<Vec<NativeComputerCall>, LlmError> {
    frame.validate()?;
    if continuation
        .is_some_and(|reference| reference.protocol != ProtocolFamily::GeminiInteractions)
    {
        return Err(invalid(
            "Gemini computer continuation must use Interactions",
        ));
    }
    let mut calls = vec![];
    let mut ids = BTreeSet::new();
    for block in blocks {
        let ContentBlock::Native { value } = block else {
            continue;
        };
        if value.format() != CALL_FORMAT {
            continue;
        }
        let step = value.data();
        if step["type"] != "function_call" || !step["arguments"].is_object() {
            return Err(invalid("malformed Gemini native desktop call"));
        }
        let id = string(step, "id")?;
        let name = string(step, "name")?;
        if !ids.insert(id) {
            return Err(invalid("duplicate Gemini native desktop call id"));
        }
        let action = operation(name, &step["arguments"], frame)?;
        action.validate(frame)?;
        let safety = step["arguments"].get("safety_decision");
        let pending_safety_checks = match safety {
            None => vec![],
            Some(value)
                if value["decision"] == "require_confirmation"
                    && value["explanation"].is_string() =>
            {
                vec![value.clone()]
            }
            Some(value) if value["decision"] == "allowed" => vec![],
            Some(_) => return Err(invalid("unknown Gemini desktop safety decision")),
        };
        let mut operations = vec![action];
        if !matches!(operations[0], ComputerOperation::Screenshot) {
            operations.push(ComputerOperation::Screenshot);
        }
        calls.push(NativeComputerCall {
            context: NativeCallContext {
                provider: NativeComputerProvider::Gemini,
                protocol_version: DESKTOP_PROTOCOL_VERSION.into(),
                call_id: id.into(),
                item_id: None,
                member_name: Some(name.into()),
                call_index: calls.len(),
                action_count: operations.len(),
                pending_safety_checks,
                continuation: continuation.cloned(),
                opaque: step.clone(),
            },
            operations,
            requires_screenshot: true,
        });
    }
    Ok(calls)
}
pub fn encode_receipt(
    call: &NativeComputerCall,
    input: &ComputerReceiptInput,
) -> Result<ContentBlock, LlmError> {
    input.validate_for(call)?;
    if call.context.provider != NativeComputerProvider::Gemini
        || call.context.protocol_version != DESKTOP_PROTOCOL_VERSION
        || call.context.opaque["id"] != call.context.call_id
        || call.context.opaque["name"].as_str() != call.context.member_name.as_deref()
    {
        return Err(invalid(
            "Gemini computer receipt identity differs from its original call",
        ));
    }
    if input
        .results
        .iter()
        .any(|result| result.status == NativeExecutionStatus::OutcomeUnknown)
    {
        return Err(invalid(
            "Gemini cannot continue an unknown computer execution outcome",
        ));
    }
    let success = input
        .results
        .iter()
        .all(|result| result.status == NativeExecutionStatus::Succeeded);
    let executed = input
        .results
        .first()
        .is_some_and(|result| result.status == NativeExecutionStatus::Succeeded);
    if executed && call.context.pending_safety_checks != input.acknowledged_safety_checks {
        return Err(invalid(
            "successful Gemini receipt requires the exact confirmed safety decision",
        ));
    }
    if !executed && !input.acknowledged_safety_checks.is_empty() {
        return Err(invalid(
            "failed Gemini action cannot claim safety acknowledgement",
        ));
    }
    let mut result = vec![];
    let mut terminal_image = false;
    for item in &input.results {
        result.push(json!({"type":"text","text":if item.status==NativeExecutionStatus::Succeeded{item.content.clone()}else{format!("Computer execution {:?}: {}",item.status,item.content)}}));
        if let Some(blocks) = &item.blocks {
            for block in blocks {
                let encoded = super::interactions::encode_result_block(block)?;
                terminal_image |=
                    item.operation_index + 1 == call.operations.len() && encoded["type"] == "image";
                result.push(encoded);
            }
        }
    }
    if success && call.requires_screenshot && !terminal_image {
        return Err(invalid(
            "Gemini computer receipt requires a final screenshot after host hooks",
        ));
    }
    if !call.context.pending_safety_checks.is_empty() && executed {
        result.insert(
            0,
            json!({"type":"text","text":json!({"safety_acknowledgement":true}).to_string()}),
        );
    }
    let mut step = json!({"type":"function_result","name":call.context.member_name,"call_id":call.context.call_id,"result":result});
    if let Some(reference) = &call.context.continuation {
        if reference.protocol != ProtocolFamily::GeminiInteractions {
            return Err(invalid(
                "Gemini receipt has a foreign continuation protocol",
            ));
        }
        // SDK-only association. The codec validates it and removes it from the wire.
        step["_sdk_continuation"] = json!(reference);
    }
    Ok(ContentBlock::Native {
        value: NativeExtension::new(RESULT_FORMAT, step)?,
    })
}
