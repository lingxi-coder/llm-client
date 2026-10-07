//! JavaScript JSON identity strings, including unpaired UTF-16 keys/values.
//! An arena and explicit stacks keep valid deeply nested identity strings usable.
use crate::exact_json::{javascript_number, write_json_string_from_utf16};

#[derive(Debug)]
enum Node {
    Null,
    Bool(bool),
    Number(f64),
    String(Vec<u16>),
    Array(Vec<usize>),
    Object(Vec<(Vec<u16>, usize)>),
}
#[derive(Clone, Copy)]
enum State {
    FirstArray,
    ArrayValue,
    ArrayComma,
    FirstObject,
    Key,
    Colon,
    ObjectValue,
    ObjectComma,
}
struct Frame {
    node: usize,
    state: State,
    key: Option<Vec<u16>>,
}
struct Parser<'a> {
    input: &'a str,
    offset: usize,
    nodes: Vec<Node>,
}
impl Parser<'_> {
    fn whitespace(&mut self) {
        while self
            .input
            .as_bytes()
            .get(self.offset)
            .is_some_and(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
        {
            self.offset += 1;
        }
    }
    fn take(&mut self, byte: u8) -> bool {
        self.whitespace();
        if self.input.as_bytes().get(self.offset) == Some(&byte) {
            self.offset += 1;
            true
        } else {
            false
        }
    }
    fn string(&mut self) -> Option<Vec<u16>> {
        if !self.take(b'"') {
            return None;
        }
        let mut units = Vec::new();
        loop {
            match *self.input.as_bytes().get(self.offset)? {
                b'"' => {
                    self.offset += 1;
                    return Some(units);
                }
                b'\\' => {
                    self.offset += 1;
                    let escape = *self.input.as_bytes().get(self.offset)?;
                    self.offset += 1;
                    units.push(match escape {
                        b'"' => 34,
                        b'\\' => 92,
                        b'/' => 47,
                        b'b' => 8,
                        b'f' => 12,
                        b'n' => 10,
                        b'r' => 13,
                        b't' => 9,
                        b'u' => {
                            let mut unit = 0;
                            for _ in 0..4 {
                                let digit = char::from(*self.input.as_bytes().get(self.offset)?)
                                    .to_digit(16)?;
                                unit = (unit << 4) | digit as u16;
                                self.offset += 1;
                            }
                            unit
                        }
                        _ => return None,
                    });
                }
                0..=31 => return None,
                _ => {
                    let ch = self.input[self.offset..].chars().next()?;
                    self.offset += ch.len_utf8();
                    let mut pair = [0; 2];
                    units.extend_from_slice(ch.encode_utf16(&mut pair));
                }
            }
        }
    }
    fn literal(&mut self, text: &str) -> bool {
        if self.input[self.offset..].starts_with(text) {
            self.offset += text.len();
            true
        } else {
            false
        }
    }
    fn digits(&mut self) -> usize {
        let start = self.offset;
        while self
            .input
            .as_bytes()
            .get(self.offset)
            .is_some_and(u8::is_ascii_digit)
        {
            self.offset += 1;
        }
        self.offset - start
    }
    fn atom(&mut self) -> Option<(usize, Option<State>)> {
        self.whitespace();
        let next = *self.input.as_bytes().get(self.offset)?;
        let (node, state) = match next {
            b'{' => {
                self.offset += 1;
                (Node::Object(Vec::new()), Some(State::FirstObject))
            }
            b'[' => {
                self.offset += 1;
                (Node::Array(Vec::new()), Some(State::FirstArray))
            }
            b'"' => (Node::String(self.string()?), None),
            b'n' if self.literal("null") => (Node::Null, None),
            b't' if self.literal("true") => (Node::Bool(true), None),
            b'f' if self.literal("false") => (Node::Bool(false), None),
            b'-' | b'0'..=b'9' => {
                let start = self.offset;
                if next == b'-' {
                    self.offset += 1;
                }
                if self.input.as_bytes().get(self.offset) == Some(&b'0') {
                    self.offset += 1;
                } else if self.digits() == 0 {
                    return None;
                }
                if self.input.as_bytes().get(self.offset) == Some(&b'.') {
                    self.offset += 1;
                    if self.digits() == 0 {
                        return None;
                    }
                }
                if self
                    .input
                    .as_bytes()
                    .get(self.offset)
                    .is_some_and(|b| matches!(b, b'e' | b'E'))
                {
                    self.offset += 1;
                    if self
                        .input
                        .as_bytes()
                        .get(self.offset)
                        .is_some_and(|b| matches!(b, b'+' | b'-'))
                    {
                        self.offset += 1;
                    }
                    if self.digits() == 0 {
                        return None;
                    }
                }
                (
                    Node::Number(self.input[start..self.offset].parse().ok()?),
                    None,
                )
            }
            _ => return None,
        };
        let id = self.nodes.len();
        self.nodes.push(node);
        Some((id, state))
    }
    fn document(mut self) -> Option<Vec<Node>> {
        let (root, state) = self.atom()?;
        debug_assert_eq!(root, 0);
        let mut stack = Vec::new();
        if let Some(state) = state {
            stack.push(Frame {
                node: root,
                state,
                key: None,
            });
        }
        while let Some(frame) = stack.last() {
            let id = frame.node;
            match frame.state {
                State::FirstArray if self.take(b']') => {
                    stack.pop();
                }
                State::FirstArray | State::ArrayValue => {
                    let (child, state) = self.atom()?;
                    let Node::Array(items) = &mut self.nodes[id] else {
                        return None;
                    };
                    items.push(child);
                    stack.last_mut()?.state = State::ArrayComma;
                    if let Some(state) = state {
                        stack.push(Frame {
                            node: child,
                            state,
                            key: None,
                        });
                    }
                }
                State::ArrayComma => {
                    if self.take(b']') {
                        stack.pop();
                    } else if self.take(b',') {
                        stack.last_mut()?.state = State::ArrayValue;
                    } else {
                        return None;
                    }
                }
                State::FirstObject if self.take(b'}') => {
                    stack.pop();
                }
                State::FirstObject | State::Key => {
                    let key = self.string()?;
                    let frame = stack.last_mut()?;
                    frame.key = Some(key);
                    frame.state = State::Colon;
                }
                State::Colon => {
                    if !self.take(b':') {
                        return None;
                    }
                    stack.last_mut()?.state = State::ObjectValue;
                }
                State::ObjectValue => {
                    let (child, state) = self.atom()?;
                    let frame = stack.last_mut()?;
                    let key = frame.key.take()?;
                    frame.state = State::ObjectComma;
                    let Node::Object(fields) = &mut self.nodes[id] else {
                        return None;
                    };
                    if let Some((_, value)) =
                        fields.iter_mut().find(|(existing, _)| *existing == key)
                    {
                        *value = child;
                    } else {
                        fields.push((key, child));
                    }
                    if let Some(state) = state {
                        stack.push(Frame {
                            node: child,
                            state,
                            key: None,
                        });
                    }
                }
                State::ObjectComma => {
                    if self.take(b'}') {
                        stack.pop();
                    } else if self.take(b',') {
                        stack.last_mut()?.state = State::Key;
                    } else {
                        return None;
                    }
                }
            }
        }
        self.whitespace();
        (self.offset == self.input.len()).then_some(self.nodes)
    }
}
fn array_index(key: &[u16]) -> Option<u32> {
    if key.is_empty() || key.len() > 10 || key.len() > 1 && key[0] == 48 {
        return None;
    }
    let mut number = 0u32;
    for &unit in key {
        if !(48..=57).contains(&unit) {
            return None;
        }
        number = number.checked_mul(10)?.checked_add(u32::from(unit - 48))?;
    }
    (number != u32::MAX).then_some(number)
}
enum Write<'a> {
    Node(usize),
    Byte(u8),
    String(&'a [u16]),
}
fn stringify(nodes: &[Node]) -> String {
    let mut out = Vec::new();
    let mut stack = vec![Write::Node(0)];
    while let Some(job) = stack.pop() {
        match job {
            Write::Byte(byte) => out.push(byte),
            Write::String(units) => write_json_string_from_utf16(&mut out, units),
            Write::Node(id) => match &nodes[id] {
                Node::Null => out.extend_from_slice(b"null"),
                Node::Bool(value) => out.extend_from_slice(if *value { b"true" } else { b"false" }),
                Node::Number(value) => {
                    if value.is_finite() {
                        out.extend_from_slice(javascript_number(*value).as_bytes());
                    } else {
                        out.extend_from_slice(b"null");
                    }
                }
                Node::String(units) => write_json_string_from_utf16(&mut out, units),
                Node::Array(items) => {
                    out.push(b'[');
                    stack.push(Write::Byte(b']'));
                    for (index, &item) in items.iter().enumerate().rev() {
                        stack.push(Write::Node(item));
                        if index > 0 {
                            stack.push(Write::Byte(b','));
                        }
                    }
                }
                Node::Object(fields) => {
                    let mut ordered: Vec<_> = fields.iter().collect();
                    ordered.sort_by_key(|(key, _)| array_index(key).map_or((1, 0), |n| (0, n)));
                    out.push(b'{');
                    stack.push(Write::Byte(b'}'));
                    for (index, (key, child)) in ordered.into_iter().enumerate().rev() {
                        stack.push(Write::Node(*child));
                        stack.push(Write::Byte(b':'));
                        stack.push(Write::String(key));
                        if index > 0 {
                            stack.push(Write::Byte(b','));
                        }
                    }
                }
            },
        }
    }
    String::from_utf8(out).expect("JSON strings emit valid UTF-8")
}
/// None preserves the original identity for invalid/nonobject/no-tk input.
pub(super) fn remove_identity_tk(input: &str) -> Option<String> {
    let mut nodes = Parser {
        input: input.strip_prefix('\u{FEFF}').unwrap_or(input),
        offset: 0,
        nodes: Vec::new(),
    }
    .document()?;
    let Node::Object(fields) = &mut nodes[0] else {
        return None;
    };
    let original = fields.len();
    fields.retain(|(key, _)| key.as_slice() != [116, 107]);
    (fields.len() != original).then(|| stringify(&nodes))
}
