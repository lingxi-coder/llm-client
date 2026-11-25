//! System prompt sources and their exact UTF-16 text representation.

use super::SystemBlock;

/// Native separator marker carried as a standalone source-vector element.
pub const DYNAMIC_BOUNDARY: &str = "__SYSTEM_PROMPT_DYNAMIC_BOUNDARY__";

const SECTION_SEP: &str = "\n\n";
/// One JavaScript string carried across the host boundary without losing
/// lone UTF-16 surrogates. `display_text` is the ECMAScript well-formed view
/// used by the provider wire request; `utf16_code_units` remains the source
/// value for Native array comparisons and source-vector joins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptText {
    display_text: String,
    utf16_code_units: Vec<u16>,
}

impl PromptText {
    /// Build an exact prompt string from UTF-16 units, using the same U+FFFD
    /// display conversion as Native's final provider request egress.
    #[must_use]
    pub fn from_utf16(utf16_code_units: Vec<u16>) -> Self {
        let display_text = String::from_utf16_lossy(&utf16_code_units);
        Self {
            display_text,
            utf16_code_units,
        }
    }

    /// Build a prompt string from a Rust string without changing its value.
    #[must_use]
    pub fn from_string(display_text: impl Into<String>) -> Self {
        let display_text = display_text.into();
        let utf16_code_units = display_text.encode_utf16().collect();
        Self {
            display_text,
            utf16_code_units,
        }
    }

    /// Display-safe UTF-8 view used by the provider's JSON body.
    #[must_use]
    pub fn display_text(&self) -> &str {
        &self.display_text
    }

    /// Exact JavaScript string units used for source-vector classification.
    #[must_use]
    pub fn utf16_code_units(&self) -> &[u16] {
        &self.utf16_code_units
    }

    /// Whether the JavaScript string has no code units.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.utf16_code_units.is_empty()
    }

    pub fn equals_ascii(&self, value: &str) -> bool {
        self.utf16_code_units
            .iter()
            .copied()
            .eq(value.encode_utf16())
    }

    pub(crate) fn starts_with_ascii(&self, value: &str) -> bool {
        self.utf16_code_units
            .starts_with(&value.encode_utf16().collect::<Vec<_>>())
    }
}

impl From<&str> for PromptText {
    fn from(value: &str) -> Self {
        Self::from_string(value)
    }
}

impl From<String> for PromptText {
    fn from(value: String) -> Self {
        Self::from_string(value)
    }
}

/// Current prompt input. Static source elements and per-query system context
/// remain separate until provider-specific source-vector egress. Custom prompt
/// parsing is performed by the Host at the field's Native input boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SystemPromptInput {
    /// Ordered static sections plus the separately recomputed system context.
    SourceVector {
        elements: Vec<PromptText>,
        dynamic_context: Option<PromptText>,
        /// Host-owned identity member; never inserted into prompt text.
        /// Native brand treatment applies only to an exact UTF-16 member match.
        host_branding_identity: Option<PromptText>,
    },
    /// Opaque custom text, such as Native `overrideSystemPrompt`, which feeds
    /// one source element to the provider projector without marker parsing.
    CustomPrompt { text: PromptText },
    /// A Native `customSystemPrompt` value that the Host already split with
    /// Native's current line-marker rule. The original is retained for display;
    /// every provider projector consumes `source_elements`.
    NativeCustomPrompt {
        text: PromptText,
        source_elements: Vec<PromptText>,
    },
}

impl SystemPromptInput {
    /// Create a source-vector request.
    ///
    /// `host_branding_identity` is non-content Host metadata. The Anthropic
    /// projector classifies a final vector member as this brand block only
    /// when its full UTF-16 sequence exactly matches the identity. `None`
    /// leaves only Native's built-in exact brand identities classified specially.
    #[must_use]
    pub fn source_vector(
        elements: Vec<PromptText>,
        dynamic_context: Option<PromptText>,
        host_branding_identity: Option<PromptText>,
    ) -> Self {
        Self::SourceVector {
            elements,
            dynamic_context,
            host_branding_identity,
        }
    }

    /// Create a one-element Native CustomPrompt request.
    #[must_use]
    pub fn custom_prompt(text: PromptText) -> Self {
        Self::CustomPrompt { text }
    }

    /// Create a current Native customSystemPrompt request after Host-side
    /// source-vector parsing. The SDK never splits the original string.
    #[must_use]
    pub fn native_custom_prompt(text: PromptText, source_elements: Vec<PromptText>) -> Self {
        Self::NativeCustomPrompt {
            text,
            source_elements,
        }
    }

    /// Visible combined text used by host previews and approximate counting.
    #[must_use]
    pub fn display_text(&self) -> String {
        let elements = match self {
            Self::SourceVector {
                elements,
                dynamic_context,
                ..
            } => {
                let mut elements = elements.clone();
                if let Some(dynamic_context) =
                    dynamic_context.as_ref().filter(|text| !text.is_empty())
                {
                    elements.push(dynamic_context.clone());
                }
                elements
            }
            Self::CustomPrompt { text } => vec![text.clone()],
            Self::NativeCustomPrompt { text, .. } => vec![text.clone()],
        };
        elements
            .iter()
            .map(PromptText::display_text)
            .collect::<Vec<_>>()
            .join(SECTION_SEP)
    }

    /// Project provider-neutral system text, retaining exact source units.
    ///
    /// Source section markers are omitted. In a literal custom prompt the
    /// same text is ordinary content.
    pub fn to_system_block(&self) -> Option<SystemBlock> {
        let mut elements = self.source_elements();
        if !matches!(self, Self::CustomPrompt { .. }) {
            elements.retain(|element| !element.equals_ascii(DYNAMIC_BOUNDARY));
        }
        let units = join_source_units(elements.iter().map(PromptText::utf16_code_units));
        (!units.is_empty()).then(|| SystemBlock::from_utf16_code_units(&units))
    }

    pub(crate) fn host_branding_identity(&self) -> Option<&PromptText> {
        match self {
            Self::SourceVector {
                host_branding_identity,
                ..
            } => host_branding_identity.as_ref(),
            Self::CustomPrompt { .. } | Self::NativeCustomPrompt { .. } => None,
        }
    }

    pub(crate) fn source_elements(&self) -> Vec<PromptText> {
        match self {
            Self::SourceVector {
                elements,
                dynamic_context,
                ..
            } => {
                let mut result = elements.clone();
                if let Some(dynamic_context) =
                    dynamic_context.as_ref().filter(|text| !text.is_empty())
                {
                    result.push(dynamic_context.clone());
                }
                result
            }
            Self::CustomPrompt { text } => vec![text.clone()],
            Self::NativeCustomPrompt {
                source_elements, ..
            } => source_elements.clone(),
        }
    }
}

pub(crate) fn join_source_units<'a>(elements: impl IntoIterator<Item = &'a [u16]>) -> Vec<u16> {
    let mut units = Vec::new();
    let mut first = true;
    for element in elements.into_iter().filter(|element| !element.is_empty()) {
        if !first {
            units.extend(SECTION_SEP.encode_utf16());
        }
        units.extend_from_slice(element);
        first = false;
    }
    units
}
