//! Normalized host-executed OpenAI computer calls and final-result receipts.
use super::computer::{
    ComputerAction as A, ComputerMouseButton as B, ComputerScreenshot, ComputerScreenshotDetail,
    ComputerScreenshotType, OpenAiComputerCall, OpenAiComputerCallOutput,
    OpenAiComputerCallOutputType, OpenAiComputerToolConfig, OPENAI_COMPUTER_CALL_FORMAT,
};
use crate::protocol::{computer::invalid, *};

pub fn declare(
    request: &mut ChatRequest,
    capabilities: &ComputerCapabilities,
    frame: &ComputerFrame,
) -> Result<(), LlmError> {
    frame.validate()?;
    use ComputerOperationKind as K;
    // Responses has no per-action disable controls. Withhold the declaration
    // unless the host supports every action it can emit.
    if [
        K::Click,
        K::Move,
        K::Drag,
        K::Scroll,
        K::Key,
        K::Type,
        K::Wait,
        K::Screenshot,
    ]
    .iter()
    .any(|kind| !capabilities.supports(*kind))
    {
        return Err(LlmError::UnsupportedCapability {
            message: "OpenAI computer requires the complete Responses action capability set".into(),
        });
    }
    if request.tools.iter().any(|tool| tool.name == "computer") {
        return Err(invalid(
            "native computer and the computer function must be declared in separate rounds",
        ));
    }
    request.set_openai_computer_tool(Some(OpenAiComputerToolConfig::default()));
    Ok(())
}
pub fn decode(
    blocks: &[ContentBlock],
    continuation: Option<&ContinuationRef>,
    frame: &ComputerFrame,
) -> Result<Vec<NativeComputerCall>, LlmError> {
    let mut calls = Vec::new();
    for (call_index, block) in blocks.iter().enumerate() {
        let ContentBlock::Native { value } = block else {
            continue;
        };
        if value.format() != OPENAI_COMPUTER_CALL_FORMAT {
            continue;
        }
        let original = OpenAiComputerCall::from_extension(value)?;
        original.validate_completed_generation()?;
        if continuation.is_none_or(|value| value.protocol != ProtocolFamily::OpenAiResponses) {
            return Err(invalid(
                "OpenAI native computer call requires its scoped continuation",
            ));
        }
        let mut operations = original.actions.iter().map(normalize).collect::<Vec<_>>();
        if !matches!(operations.last(), Some(ComputerOperation::Screenshot)) {
            operations.push(ComputerOperation::Screenshot);
        }
        for operation in &operations {
            operation.validate(frame)?;
        }
        calls.push(NativeComputerCall {
            context: NativeCallContext {
                provider: NativeComputerProvider::OpenAi,
                protocol_version: OPENAI_COMPUTER_CALL_FORMAT.into(),
                call_id: original.call_id.clone(),
                item_id: Some(original.id.clone()),
                member_name: None,
                call_index,
                action_count: operations.len(),
                pending_safety_checks: original
                    .pending_safety_checks
                    .iter()
                    .map(|check| serde_json::to_value(check).expect("serializable safety check"))
                    .collect(),
                continuation: continuation.cloned(),
                opaque: value.data().clone(),
            },
            operations,
            requires_screenshot: true,
        });
    }
    Ok(calls)
}
fn normalize(action: &A) -> ComputerOperation {
    let target = |x, y| ComputerTarget::Position {
        point: ComputerPoint { x, y },
    };
    let modifiers = |keys: &Option<Vec<String>>| keys.clone().unwrap_or_default();
    match action {
        A::Click { button, x, y, keys } => ComputerOperation::Click {
            target: target(*x, *y),
            button: match button {
                B::Left => crate::protocol::ComputerMouseButton::Left,
                B::Right => crate::protocol::ComputerMouseButton::Right,
                B::Wheel => crate::protocol::ComputerMouseButton::Middle,
                B::Back => crate::protocol::ComputerMouseButton::Back,
                B::Forward => crate::protocol::ComputerMouseButton::Forward,
            },
            count: 1,
            modifiers: modifiers(keys),
        },
        A::DoubleClick { x, y, keys } => ComputerOperation::Click {
            target: target(*x, *y),
            button: crate::protocol::ComputerMouseButton::Left,
            count: 2,
            modifiers: modifiers(keys),
        },
        A::Drag { path, keys } => ComputerOperation::Drag {
            path: path
                .iter()
                .map(|p| crate::protocol::ComputerPoint { x: p.x, y: p.y })
                .collect(),
            modifiers: modifiers(keys),
        },
        A::Move { x, y, keys } => ComputerOperation::Move {
            point: crate::protocol::ComputerPoint { x: *x, y: *y },
            modifiers: modifiers(keys),
        },
        A::Scroll {
            x,
            y,
            scroll_x,
            scroll_y,
            keys,
        } => ComputerOperation::Scroll {
            target: target(*x, *y),
            delta_x: *scroll_x,
            delta_y: *scroll_y,
            modifiers: modifiers(keys),
        },
        A::Keypress { keys } => ComputerOperation::Key { keys: keys.clone() },
        A::Type { text } => ComputerOperation::Type {
            text: text.clone(),
            press_enter: false,
        },
        A::Wait => ComputerOperation::Wait {
            duration_seconds: 2.0,
        },
        A::Screenshot => ComputerOperation::Screenshot,
    }
}
pub fn encode_receipt(
    call: &NativeComputerCall,
    input: &ComputerReceiptInput,
) -> Result<ContentBlock, LlmError> {
    input.validate_for(call)?;
    if call.context.provider != NativeComputerProvider::OpenAi
        || call.context.protocol_version != OPENAI_COMPUTER_CALL_FORMAT
        || call
            .context
            .continuation
            .as_ref()
            .is_none_or(|value| value.protocol != ProtocolFamily::OpenAiResponses)
    {
        return Err(invalid(
            "OpenAI computer receipt lost its original protocol binding or continuation",
        ));
    }
    let original = OpenAiComputerCall::from_response_item(&call.context.opaque)?;
    if original.call_id != call.context.call_id
        || Some(&original.id) != call.context.item_id.as_ref()
    {
        return Err(invalid(
            "OpenAI computer receipt identity differs from its original call",
        ));
    }
    if input
        .results
        .iter()
        .any(|result| result.status != NativeExecutionStatus::Succeeded)
    {
        return Err(LlmError::UnsupportedCapability{message:"OpenAI screenshot receipt cannot represent a failed, denied, cancelled, skipped or unknown execution; re-observe through the host recovery path".into()});
    }
    let result = input
        .results
        .last()
        .ok_or_else(|| invalid("OpenAI computer receipt requires a final observation"))?;
    if !matches!(call.operations.last(), Some(ComputerOperation::Screenshot)) {
        return Err(invalid(
            "OpenAI computer receipt must end with an explicit observation",
        ));
    }
    let image = result
        .blocks
        .as_ref()
        .and_then(|blocks| {
            blocks.iter().rev().find(|block| {
                block.get("type").and_then(serde_json::Value::as_str) == Some("image")
            })
        })
        .ok_or_else(|| {
            invalid("OpenAI computer receipt requires an image in the final model-visible result")
        })?;
    let source = image.get("source").unwrap_or(image);
    let image_url = match source.get("type").and_then(serde_json::Value::as_str) {
        Some("base64") => format!(
            "data:{};base64,{}",
            source
                .get("media_type")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid("screenshot has no media type"))?,
            source
                .get("data")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid("screenshot has no image data"))?
        ),
        Some("url") => source
            .get("url")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| invalid("screenshot has no URL"))?
            .to_owned(),
        _ => image
            .get("image_url")
            .and_then(|v| {
                v.as_str()
                    .or_else(|| v.get("url").and_then(serde_json::Value::as_str))
            })
            .ok_or_else(|| invalid("screenshot source cannot be represented by OpenAI"))?
            .to_owned(),
    };
    let output = OpenAiComputerCallOutput {
        r#type: OpenAiComputerCallOutputType::ComputerCallOutput,
        call_id: call.context.call_id.clone(),
        output: ComputerScreenshot {
            r#type: ComputerScreenshotType::ComputerScreenshot,
            file_id: None,
            image_url: Some(image_url),
            detail: Some(ComputerScreenshotDetail::Original),
        },
        id: None,
        status: None,
        created_by: None,
        acknowledged_safety_checks: input
            .acknowledged_safety_checks
            .iter()
            .map(|v| {
                serde_json::from_value(v.clone())
                    .map_err(|e| invalid(format!("invalid safety acknowledgement: {e}")))
            })
            .collect::<Result<Vec<_>, _>>()?,
    };
    output.into_content_block_for(&original)
}
