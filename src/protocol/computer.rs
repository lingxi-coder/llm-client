//! Data-only normalization of host-executed native computer protocols.
//! The host owns permissions, execution, observation and final result publication.
use super::{ChatRequest, ContentBlock, ContinuationRef, LlmError};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ComputerPoint {
    pub x: f64,
    pub y: f64,
}
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ComputerTarget {
    Position { point: ComputerPoint },
    CurrentCursor,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputerMouseButton {
    Left,
    Right,
    Middle,
    Back,
    Forward,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputerScrollDirection {
    Up,
    Down,
    Left,
    Right,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ComputerOperation {
    Click {
        target: ComputerTarget,
        button: ComputerMouseButton,
        count: u8,
        modifiers: Vec<String>,
    },
    Move {
        point: ComputerPoint,
        modifiers: Vec<String>,
    },
    Drag {
        path: Vec<ComputerPoint>,
        modifiers: Vec<String>,
    },
    /// Deltas are pixels, with positive axes pointing right and down.
    Scroll {
        target: ComputerTarget,
        delta_x: f64,
        delta_y: f64,
        modifiers: Vec<String>,
    },
    /// Wheel clicks retain their original unit; they are never guessed as pixels.
    ScrollWheel {
        target: ComputerTarget,
        direction: ComputerScrollDirection,
        amount: u32,
        modifiers: Vec<String>,
    },
    Key {
        keys: Vec<String>,
    },
    KeyDown {
        key: String,
    },
    KeyUp {
        key: String,
    },
    HoldKey {
        keys: Vec<String>,
        duration_seconds: f64,
    },
    Type {
        text: String,
        press_enter: bool,
    },
    Wait {
        duration_seconds: f64,
    },
    Screenshot,
    Zoom {
        region: [f64; 4],
    },
    MouseDown {
        target: Option<ComputerPoint>,
        modifiers: Vec<String>,
    },
    MouseUp {
        target: Option<ComputerPoint>,
        modifiers: Vec<String>,
    },
    CursorPosition,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputerOperationKind {
    Click,
    Move,
    Drag,
    Scroll,
    ScrollWheel,
    Key,
    KeyDown,
    KeyUp,
    HoldKey,
    Type,
    Wait,
    Screenshot,
    Zoom,
    MouseDown,
    MouseUp,
    CursorPosition,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputerCapabilities {
    pub operations: Vec<ComputerOperationKind>,
}
impl ComputerCapabilities {
    pub fn supports(&self, kind: ComputerOperationKind) -> bool {
        self.operations.contains(&kind)
    }
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputerFrame {
    pub width: u32,
    pub height: u32,
    pub geometry_version: String,
}
impl ComputerFrame {
    pub fn validate(&self) -> Result<(), LlmError> {
        if self.width == 0 || self.height == 0 || self.geometry_version.trim().is_empty() {
            return Err(invalid(
                "computer frame requires positive dimensions and a geometry version",
            ));
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeComputerProvider {
    OpenAi,
    Anthropic,
    Gemini,
}
/// Identify a native computer call without interpreting an ordinary function
/// that happens to share a member name. Full decoding remains mandatory before
/// execution; this helper only associates original model history blocks.
pub fn computer_call_id(block: &ContentBlock) -> Option<&str> {
    match block {
        ContentBlock::ToolUse {
            id, toolset_name, ..
        } if toolset_name.as_deref() == Some("computer") => Some(id.as_str()),
        ContentBlock::Native { value }
            if value.format()
                == crate::providers::openai::computer::OPENAI_COMPUTER_CALL_FORMAT =>
        {
            value.data().get("call_id").and_then(Value::as_str)
        }
        ContentBlock::Native { value }
            if value.format() == crate::providers::google::computer::CALL_FORMAT =>
        {
            value
                .data()
                .get("id")
                .and_then(Value::as_str)
                .or_else(|| value.data().get("call_id").and_then(Value::as_str))
        }
        _ => None,
    }
}
/// Original provider data is preserved verbatim for receipt association.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeCallContext {
    pub provider: NativeComputerProvider,
    pub protocol_version: String,
    pub call_id: String,
    pub item_id: Option<String>,
    pub member_name: Option<String>,
    pub call_index: usize,
    pub action_count: usize,
    pub pending_safety_checks: Vec<Value>,
    pub continuation: Option<ContinuationRef>,
    pub opaque: Value,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeComputerCall {
    pub context: NativeCallContext,
    pub operations: Vec<ComputerOperation>,
    pub requires_screenshot: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeExecutionStatus {
    Succeeded,
    Failed,
    Denied,
    Cancelled,
    Skipped,
    OutcomeUnknown,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NativeComputerResult {
    pub operation_index: usize,
    pub status: NativeExecutionStatus,
    /// Text and blocks after all host result hooks. No cached images are read.
    pub content: String,
    pub blocks: Option<Vec<Value>>,
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComputerReceiptInput {
    pub results: Vec<NativeComputerResult>,
    pub acknowledged_safety_checks: Vec<Value>,
}

impl ComputerOperation {
    pub fn kind(&self) -> ComputerOperationKind {
        use ComputerOperationKind as K;
        match self {
            Self::Click { .. } => K::Click,
            Self::Move { .. } => K::Move,
            Self::Drag { .. } => K::Drag,
            Self::Scroll { .. } => K::Scroll,
            Self::ScrollWheel { .. } => K::ScrollWheel,
            Self::Key { .. } => K::Key,
            Self::KeyDown { .. } => K::KeyDown,
            Self::KeyUp { .. } => K::KeyUp,
            Self::HoldKey { .. } => K::HoldKey,
            Self::Type { .. } => K::Type,
            Self::Wait { .. } => K::Wait,
            Self::Screenshot => K::Screenshot,
            Self::Zoom { .. } => K::Zoom,
            Self::MouseDown { .. } => K::MouseDown,
            Self::MouseUp { .. } => K::MouseUp,
            Self::CursorPosition => K::CursorPosition,
        }
    }
    pub fn validate(&self, frame: &ComputerFrame) -> Result<(), LlmError> {
        frame.validate()?;
        let point = |p: &ComputerPoint| -> Result<(), LlmError> {
            if !p.x.is_finite()
                || !p.y.is_finite()
                || p.x < 0.0
                || p.y < 0.0
                || p.x >= frame.width as f64
                || p.y >= frame.height as f64
            {
                Err(invalid("computer point is outside the current screenshot"))
            } else {
                Ok(())
            }
        };
        let target = |t: &ComputerTarget| -> Result<(), LlmError> {
            if let ComputerTarget::Position { point: p } = t {
                point(p)?;
            }
            Ok(())
        };
        match self {
            Self::Click {
                target: t,
                count,
                modifiers,
                ..
            } => {
                target(t)?;
                if !(1..=3).contains(count) {
                    return Err(invalid("computer click count must be 1..=3"));
                }
                validate_keys(modifiers, true)?;
            }
            Self::Move {
                point: p,
                modifiers,
            } => {
                point(p)?;
                validate_keys(modifiers, true)?;
            }
            Self::Drag { path, modifiers } => {
                if !(2..=256).contains(&path.len()) {
                    return Err(invalid("computer drag path must contain 2..=256 points"));
                }
                for p in path {
                    point(p)?;
                }
                validate_keys(modifiers, true)?;
            }
            Self::Scroll {
                target: t,
                delta_x,
                delta_y,
                modifiers,
            } => {
                target(t)?;
                if [*delta_x, *delta_y]
                    .iter()
                    .any(|n| !n.is_finite() || n.abs() > 1_000_000.0)
                {
                    return Err(invalid("computer pixel scroll delta is unbounded"));
                }
                validate_keys(modifiers, true)?;
            }
            Self::ScrollWheel {
                target: t,
                amount,
                modifiers,
                ..
            } => {
                target(t)?;
                if *amount == 0 || *amount > 100 {
                    return Err(invalid(
                        "computer scroll amount must be 1..=100 wheel clicks",
                    ));
                }
                validate_keys(modifiers, true)?;
            }
            Self::Key { keys } => validate_keys(keys, false)?,
            Self::KeyDown { key } | Self::KeyUp { key } => {
                validate_keys(std::slice::from_ref(key), false)?
            }
            Self::HoldKey {
                keys,
                duration_seconds,
            } => {
                validate_keys(keys, false)?;
                validate_duration(*duration_seconds)?;
            }
            Self::Type { text, .. } => {
                if text.len() > 64 * 1024 || text.contains('\0') {
                    return Err(invalid("computer text is unbounded or contains NUL"));
                }
            }
            Self::Wait { duration_seconds } => validate_duration(*duration_seconds)?,
            Self::Zoom { region } => {
                point(&ComputerPoint {
                    x: region[0],
                    y: region[1],
                })?;
                if !region[2].is_finite()
                    || !region[3].is_finite()
                    || region[2] <= region[0]
                    || region[3] <= region[1]
                    || region[2] > frame.width as f64
                    || region[3] > frame.height as f64
                {
                    return Err(invalid(
                        "computer zoom region is outside the current screenshot",
                    ));
                }
            }
            Self::MouseDown {
                target: t,
                modifiers,
            }
            | Self::MouseUp {
                target: t,
                modifiers,
            } => {
                if let Some(p) = t {
                    point(p)?;
                }
                validate_keys(modifiers, true)?;
            }
            Self::Screenshot | Self::CursorPosition => {}
        }
        Ok(())
    }
}
pub(crate) fn validate_keys(keys: &[String], allow_empty: bool) -> Result<(), LlmError> {
    if (!allow_empty && keys.is_empty())
        || keys.len() > 16
        || keys
            .iter()
            .any(|k| k.trim().is_empty() || k.len() > 64 || k.chars().any(char::is_control))
    {
        return Err(invalid(
            "computer key names require a bounded nonempty array",
        ));
    }
    Ok(())
}
pub(crate) fn validate_duration(seconds: f64) -> Result<(), LlmError> {
    if !seconds.is_finite() || !(0.0..=300.0).contains(&seconds) {
        return Err(invalid(
            "computer duration must be finite seconds within 0..=300",
        ));
    }
    Ok(())
}
pub(crate) fn invalid(message: impl Into<String>) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
impl ComputerReceiptInput {
    pub fn validate_for(&self, call: &NativeComputerCall) -> Result<(), LlmError> {
        if call.operations.is_empty()
            || call.context.action_count != call.operations.len()
            || self.results.len() != call.operations.len()
        {
            return Err(invalid(
                "native computer receipt must contain exactly one result per normalized operation",
            ));
        }
        let mut stopped = false;
        for (index, result) in self.results.iter().enumerate() {
            if result.operation_index != index {
                return Err(invalid(
                    "native computer results must preserve operation order and indices",
                ));
            }
            if stopped && result.status != NativeExecutionStatus::Skipped {
                return Err(invalid(
                    "native computer operations after a failure must be skipped",
                ));
            }
            if result.status != NativeExecutionStatus::Succeeded {
                stopped = true;
            }
        }
        Ok(())
    }
}
/// Declare only the current control round; callers select an authorized host tool.
pub fn declare_computer_tool(
    request: &mut ChatRequest,
    provider: NativeComputerProvider,
    capabilities: &ComputerCapabilities,
    frame: &ComputerFrame,
) -> Result<(), LlmError> {
    frame.validate()?;
    match provider {
        NativeComputerProvider::OpenAi => {
            crate::providers::openai::computer_adapter::declare(request, capabilities, frame)
        }
        NativeComputerProvider::Anthropic => {
            crate::providers::anthropic::computer::declare(request, capabilities, frame)
        }
        NativeComputerProvider::Gemini => {
            crate::providers::google::computer::declare(request, capabilities, frame)
        }
    }
}
/// Validate the complete model output before returning any executable operation.
pub fn decode_computer_calls(
    provider: NativeComputerProvider,
    blocks: &[ContentBlock],
    continuation: Option<&ContinuationRef>,
    frame: &ComputerFrame,
) -> Result<Vec<NativeComputerCall>, LlmError> {
    frame.validate()?;
    let calls = match provider {
        NativeComputerProvider::OpenAi => {
            crate::providers::openai::computer_adapter::decode(blocks, continuation, frame)?
        }
        NativeComputerProvider::Anthropic => {
            crate::providers::anthropic::computer::decode(blocks, continuation, frame)?
        }
        NativeComputerProvider::Gemini => {
            crate::providers::google::computer::decode(blocks, continuation, frame)?
        }
    };
    let mut ids = std::collections::HashSet::new();
    for call in &calls {
        if call.context.provider != provider
            || call.context.call_id.trim().is_empty()
            || !ids.insert(&call.context.call_id)
            || call.operations.is_empty()
            || call.context.action_count != call.operations.len()
        {
            return Err(invalid(
                "native computer call has invalid identity or operation count",
            ));
        }
        for operation in &call.operations {
            operation.validate(frame)?;
        }
    }
    Ok(calls)
}
pub fn encode_computer_receipt(
    call: &NativeComputerCall,
    input: &ComputerReceiptInput,
) -> Result<ContentBlock, LlmError> {
    input.validate_for(call)?;
    match call.context.provider {
        NativeComputerProvider::OpenAi => {
            crate::providers::openai::computer_adapter::encode_receipt(call, input)
        }
        NativeComputerProvider::Anthropic => {
            crate::providers::anthropic::computer::encode_receipt(call, input)
        }
        NativeComputerProvider::Gemini => {
            crate::providers::google::computer::encode_receipt(call, input)
        }
    }
}
