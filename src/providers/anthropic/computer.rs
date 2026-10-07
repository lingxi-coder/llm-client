//! Stable Claude Computer client-toolset member normalization.
use super::toolsets::{
    AnthropicClientToolConfig, AnthropicClientToolset, AnthropicComputerMember,
    AnthropicComputerToolsetConfig,
};
use crate::protocol::{computer::invalid, *};
use base64::Engine;
use serde_json::{Map, Value};

pub const COMPUTER_TOOLSET_VERSION: &str = "computer_toolset_20260801";
pub const SKIPPED_COMPUTER_ACTION: &str =
    "Not executed: an earlier computer action in this turn failed.";
pub fn declare(
    request: &mut ChatRequest,
    capabilities: &ComputerCapabilities,
    frame: &ComputerFrame,
) -> Result<(), LlmError> {
    frame.validate()?;
    if request.tools.iter().any(|tool| tool.name == "computer") {
        return Err(invalid(
            "native computer and the computer function must be declared in separate rounds",
        ));
    }
    let mut config = AnthropicComputerToolsetConfig::default();
    for member in AnthropicComputerMember::ALL {
        let kind = member_kind(*member);
        config.configs.insert(
            *member,
            AnthropicClientToolConfig {
                enabled: Some(capabilities.supports(kind)),
                defer_loading: None,
            },
        );
    }
    if !capabilities.supports(ComputerOperationKind::Screenshot) {
        return Err(LlmError::UnsupportedCapability {
            message: "Claude computer requires host screenshot capability".into(),
        });
    }
    let mut toolsets = request.anthropic_client_toolsets().to_vec();
    toolsets.retain(|toolset| !matches!(toolset, AnthropicClientToolset::Computer(_)));
    toolsets.push(AnthropicClientToolset::Computer(config));
    request.set_anthropic_client_toolsets(toolsets);
    Ok(())
}
fn member_kind(member: AnthropicComputerMember) -> ComputerOperationKind {
    use AnthropicComputerMember as M;
    use ComputerOperationKind as K;
    match member {
        M::Screenshot => K::Screenshot,
        M::Zoom => K::Zoom,
        M::LeftClick | M::RightClick | M::MiddleClick | M::DoubleClick | M::TripleClick => K::Click,
        M::LeftClickDrag => K::Drag,
        M::MouseMove => K::Move,
        M::LeftMouseDown => K::MouseDown,
        M::LeftMouseUp => K::MouseUp,
        M::CursorPosition => K::CursorPosition,
        M::Scroll => K::ScrollWheel,
        M::Type => K::Type,
        M::Key => K::Key,
        M::HoldKey => K::HoldKey,
        M::Wait => K::Wait,
    }
}
pub fn decode(
    blocks: &[ContentBlock],
    continuation: Option<&ContinuationRef>,
    frame: &ComputerFrame,
) -> Result<Vec<NativeComputerCall>, LlmError> {
    let mut calls = Vec::new();
    for (call_index, block) in blocks.iter().enumerate() {
        let ContentBlock::ToolUse {
            id,
            name,
            input,
            toolset_name,
            caller,
            ..
        } = block
        else {
            continue;
        };
        if toolset_name.as_deref() != Some("computer") {
            continue;
        }
        if caller.as_ref().is_some_and(|caller| {
            !caller.is_null() && caller.get("type").and_then(Value::as_str) != Some("direct")
        }) {
            return Err(invalid("Claude computer members require a direct caller"));
        }
        let operations = normalize(name, input)?;
        for operation in &operations {
            operation.validate(frame)?;
        }
        let requires_screenshot = matches!(name.as_str(), "screenshot" | "zoom");
        calls.push(NativeComputerCall {
            context: NativeCallContext {
                provider: NativeComputerProvider::Anthropic,
                protocol_version: COMPUTER_TOOLSET_VERSION.into(),
                call_id: id.as_str().to_owned(),
                item_id: None,
                member_name: Some(name.clone()),
                call_index,
                action_count: operations.len(),
                pending_safety_checks: vec![],
                continuation: continuation.cloned(),
                opaque: serde_json::to_value(block).map_err(|e| invalid(e.to_string()))?,
            },
            operations,
            requires_screenshot,
        });
    }
    Ok(calls)
}
fn normalize(name: &str, input: &Value) -> Result<Vec<ComputerOperation>, LlmError> {
    use ComputerOperation as O;
    let map = input
        .as_object()
        .ok_or_else(|| invalid("Claude computer member input must be an object"))?;
    let allowed: &[&str] = match name {
        "screenshot" | "cursor_position" | "left_mouse_down" | "left_mouse_up" => &[],
        "zoom" => &["region"],
        "left_click" | "right_click" | "middle_click" | "double_click" | "triple_click" => {
            &["coordinate", "text"]
        }
        "left_click_drag" => &["start_coordinate", "coordinate", "text"],
        "mouse_move" => &["coordinate"],
        "scroll" => &["scroll_direction", "scroll_amount", "coordinate", "text"],
        "type" => &["text"],
        "key" => &["text", "repeat"],
        "hold_key" => &["text", "duration"],
        "wait" => &["duration"],
        _ => {
            return Err(invalid(format!(
                "unsupported Claude computer member {name:?}"
            )))
        }
    };
    if map.keys().any(|key| !allowed.contains(&key.as_str())) {
        return Err(invalid(
            "Claude computer member has an unrepresentable field",
        ));
    }
    let target = || -> Result<ComputerTarget, LlmError> {
        Ok(if map.contains_key("coordinate") {
            ComputerTarget::Position {
                point: point(map, "coordinate")?,
            }
        } else {
            ComputerTarget::CurrentCursor
        })
    };
    let modifiers = || -> Result<Vec<String>, LlmError> {
        let Some(value) = map.get("text") else {
            return Ok(vec![]);
        };
        let text = value
            .as_str()
            .ok_or_else(|| invalid("mouse modifiers must be a string"))?;
        let keys = keys(text)?;
        if keys
            .iter()
            .any(|key| !matches!(key.as_str(), "shift" | "ctrl" | "alt" | "super"))
        {
            return Err(invalid(
                "Claude mouse modifiers must be shift, ctrl, alt or super",
            ));
        }
        Ok(keys)
    };
    let operation = match name {
        "screenshot" => O::Screenshot,
        "zoom" => {
            let region = map
                .get("region")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("zoom requires a region"))?;
            if region.len() != 4 {
                return Err(invalid("zoom requires four pixel values"));
            }
            O::Zoom {
                region: [
                    number(&region[0])?,
                    number(&region[1])?,
                    number(&region[2])?,
                    number(&region[3])?,
                ],
            }
        }
        "left_click" | "right_click" | "middle_click" | "double_click" | "triple_click" => {
            O::Click {
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
                modifiers: modifiers()?,
            }
        }
        "left_click_drag" => O::Drag {
            path: vec![point(map, "start_coordinate")?, point(map, "coordinate")?],
            modifiers: modifiers()?,
        },
        "mouse_move" => O::Move {
            point: point(map, "coordinate")?,
            modifiers: vec![],
        },
        "left_mouse_down" => O::MouseDown {
            target: None,
            modifiers: vec![],
        },
        "left_mouse_up" => O::MouseUp {
            target: None,
            modifiers: vec![],
        },
        "cursor_position" => O::CursorPosition,
        "scroll" => O::ScrollWheel {
            target: target()?,
            direction: match string(map, "scroll_direction")? {
                "up" => ComputerScrollDirection::Up,
                "down" => ComputerScrollDirection::Down,
                "left" => ComputerScrollDirection::Left,
                "right" => ComputerScrollDirection::Right,
                _ => return Err(invalid("unknown Claude scroll direction")),
            },
            amount: integer(map, "scroll_amount", 1, 100)? as u32,
            modifiers: modifiers()?,
        },
        "type" => O::Type {
            text: string(map, "text")?.into(),
            press_enter: false,
        },
        "key" => {
            let keys = keys(string(map, "text")?)?;
            let repeat = if map.contains_key("repeat") {
                integer(map, "repeat", 1, 100)?
            } else {
                1
            };
            return Ok((0..repeat).map(|_| O::Key { keys: keys.clone() }).collect());
        }
        "hold_key" => O::HoldKey {
            keys: keys(string(map, "text")?)?,
            duration_seconds: number(
                map.get("duration")
                    .ok_or_else(|| invalid("hold_key requires duration"))?,
            )?,
        },
        "wait" => O::Wait {
            duration_seconds: number(
                map.get("duration")
                    .ok_or_else(|| invalid("wait requires duration"))?,
            )?,
        },
        _ => unreachable!(),
    };
    Ok(vec![operation])
}
fn keys(text: &str) -> Result<Vec<String>, LlmError> {
    // A literal plus key is representable, while empty components in a chord
    // are ambiguous and rejected before host execution.
    let values = if text == "+" {
        vec![text.to_owned()]
    } else {
        text.split('+').map(str::to_owned).collect()
    };
    crate::protocol::computer::validate_keys(&values, false)?;
    Ok(values)
}
fn string<'a>(map: &'a Map<String, Value>, key: &str) -> Result<&'a str, LlmError> {
    map.get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("computer {key} must be a string")))
}
fn number(value: &Value) -> Result<f64, LlmError> {
    value
        .as_f64()
        .filter(|n| n.is_finite())
        .ok_or_else(|| invalid("computer numeric value must be finite"))
}
fn point(map: &Map<String, Value>, key: &str) -> Result<ComputerPoint, LlmError> {
    let v = map
        .get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid(format!("computer {key} must be a pixel pair")))?;
    if v.len() != 2 {
        return Err(invalid("computer coordinate must have two values"));
    }
    Ok(ComputerPoint {
        x: number(&v[0])?,
        y: number(&v[1])?,
    })
}
fn integer(map: &Map<String, Value>, key: &str, min: u64, max: u64) -> Result<u64, LlmError> {
    map.get(key)
        .and_then(Value::as_u64)
        .filter(|n| (*n >= min) && (*n <= max))
        .ok_or_else(|| invalid(format!("computer {key} must be {min}..={max}")))
}
pub fn encode_receipt(
    call: &NativeComputerCall,
    input: &ComputerReceiptInput,
) -> Result<ContentBlock, LlmError> {
    input.validate_for(call)?;
    if call.context.provider != NativeComputerProvider::Anthropic
        || call.context.protocol_version != COMPUTER_TOOLSET_VERSION
        || !input.acknowledged_safety_checks.is_empty()
    {
        return Err(invalid(
            "Claude computer receipt has incompatible protocol or safety metadata",
        ));
    }
    let original: ContentBlock =
        serde_json::from_value(call.context.opaque.clone()).map_err(|e| invalid(e.to_string()))?;
    let ContentBlock::ToolUse {
        id,
        name,
        toolset_name,
        ..
    } = original
    else {
        return Err(invalid("Claude receipt lost its original tool-use block"));
    };
    if id.as_str() != call.context.call_id
        || Some(&name) != call.context.member_name.as_ref()
        || toolset_name.as_deref() != Some("computer")
    {
        return Err(invalid(
            "Claude receipt identity differs from its original member call",
        ));
    }
    let is_error = input
        .results
        .iter()
        .any(|r| r.status != NativeExecutionStatus::Succeeded);
    let skipped = input
        .results
        .iter()
        .all(|r| r.status == NativeExecutionStatus::Skipped);
    let content = if skipped {
        SKIPPED_COMPUTER_ACTION.into()
    } else {
        input
            .results
            .iter()
            .filter(|r| r.status != NativeExecutionStatus::Skipped)
            .map(|r| r.content.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    };
    let mut blocks = Vec::new();
    if !skipped {
        for result in &input.results {
            if result.status == NativeExecutionStatus::Skipped {
                continue;
            }
            if let Some(values) = &result.blocks {
                for value in values {
                    match value.get("type").and_then(Value::as_str) {
                        Some("text") if value.get("text").is_some_and(Value::is_string) => {}
                        Some("image") => validate_image(value)?,
                        _ => {
                            return Err(invalid(
                                "Claude computer results accept only text and image blocks",
                            ))
                        }
                    }
                    blocks.push(value.clone());
                }
            } else if !result.content.is_empty() {
                blocks.push(serde_json::json!({"type":"text","text":result.content}));
            }
        }
    }
    if !is_error
        && call.requires_screenshot
        && !blocks
            .iter()
            .any(|b| b.get("type").and_then(Value::as_str) == Some("image"))
    {
        return Err(invalid(
            "Claude screenshot/zoom receipt requires a final model-visible image",
        ));
    }
    Ok(ContentBlock::ToolResult {
        tool_use_id: id,
        content,
        is_error: Some(is_error),
        blocks: if skipped || blocks.is_empty() {
            None
        } else {
            Some(blocks)
        },
        toolset_name: Some("computer".into()),
    })
}

fn validate_image(value: &Value) -> Result<(), LlmError> {
    let block: ContentBlock = serde_json::from_value(value.clone())
        .map_err(|error| invalid(format!("invalid Claude computer result image: {error}")))?;
    match block {
        ContentBlock::Image {
            source: ImageSource::Base64 { media_type, data },
        } => {
            if !matches!(
                media_type.as_str(),
                "image/png" | "image/jpeg" | "image/gif" | "image/webp"
            ) || data.is_empty()
                || data.len() > 16 * 1024 * 1024
            {
                return Err(invalid(
                    "Claude computer result image has an unsupported media type or size",
                ));
            }
            base64::engine::general_purpose::STANDARD
                .decode(data)
                .map_err(|_| invalid("Claude computer result image contains invalid base64"))?;
        }
        ContentBlock::Image {
            source: ImageSource::Url { url },
        } => {
            let parsed = url::Url::parse(&url)
                .map_err(|_| invalid("Claude computer image URL is invalid"))?;
            if parsed.scheme() != "https"
                || parsed.host_str().is_none()
                || !parsed.username().is_empty()
                || parsed.password().is_some()
                || parsed.fragment().is_some()
            {
                return Err(invalid(
                    "Claude computer image URL must be HTTPS without credentials or fragment",
                ));
            }
        }
        _ => {
            return Err(invalid(
                "Claude computer result image source is not representable",
            ))
        }
    }
    Ok(())
}
