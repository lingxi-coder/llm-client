//! Media inside a tool result, for wires whose tool-output slot holds text.
//!
//! The host supplies MCP blocks as source-backed JSON. Classification reads
//! only the fields needed to route media; all text and unknown blocks retain
//! their original JSON/UTF-16 source until a provider encoder writes them.

use crate::protocol::{JsonPayload, ModelProfile};

/// One tool-result block as a text-only result slot sees it.
pub(crate) enum Piece {
    Text(Vec<u16>),
    /// Any other block. The slot carries its JSON spelling as text.
    Json(JsonPayload),
    Image(Image),
    /// A base64 document; Read returns PDFs this way.
    Document {
        media_type: String,
        data: String,
        title: Option<Vec<u16>>,
    },
}

pub(crate) enum Image {
    Base64 { media_type: String, data: String },
    Url(String),
}

impl Piece {
    pub(crate) fn is_media(&self) -> bool {
        matches!(self, Self::Image(_) | Self::Document { .. })
    }
}

/// The classified blocks, or `None` when none is media and the codec's plain
/// result encoding applies unchanged.
pub(crate) fn pieces(blocks: &JsonPayload) -> Option<Vec<Piece>> {
    if !blocks.is_root_array() {
        return None;
    }
    let len = blocks.array_len("")?;
    let pieces = (0..len)
        .map(|index| blocks.array_item(index).map(|block| piece(&block)))
        .collect::<Option<Vec<_>>>()?;
    pieces.iter().any(Piece::is_media).then_some(pieces)
}

/// The blocks a text-only slot carries as derived text: [`pieces`], or every
/// block when all are text. Text-only results should not pay for the array's
/// quoting and escapes on every replay.
pub(crate) fn slot_pieces(blocks: &JsonPayload) -> Option<Vec<Piece>> {
    pieces(blocks).or_else(|| {
        if !blocks.is_root_array() {
            return None;
        }
        let len = blocks.array_len("")?;
        let pieces = (0..len)
            .map(|index| blocks.array_item(index).map(|block| piece(&block)))
            .collect::<Option<Vec<_>>>()?;
        pieces
            .iter()
            .all(|piece| matches!(piece, Piece::Text(_)))
            .then_some(pieces)
    })
}

pub(crate) fn has_media(blocks: &JsonPayload) -> bool {
    pieces(blocks).is_some()
}

pub(crate) fn piece(block: &JsonPayload) -> Piece {
    let json = || Piece::Json(block.clone());
    if block.string_is("/type", "text") {
        return block
            .string_units_at("/text")
            .map_or_else(json, Piece::Text);
    }
    match (
        block.string_at("/type").as_deref(),
        block.string_at("/source/type").as_deref(),
    ) {
        (Some("image"), Some("base64")) => match (
            scalar_string_at(block, "/source/media_type"),
            scalar_string_at(block, "/source/data"),
        ) {
            (Some(media_type), Some(data)) => Piece::Image(Image::Base64 { media_type, data }),
            _ => json(),
        },
        (Some("image"), Some("url")) => scalar_string_at(block, "/source/url")
            .map_or_else(json, |url| Piece::Image(Image::Url(url))),
        (Some("document"), Some("base64")) => match (
            scalar_string_at(block, "/source/media_type"),
            scalar_string_at(block, "/source/data"),
        ) {
            (Some(media_type), Some(data)) => Piece::Document {
                media_type,
                data,
                title: block.string_units_at("/title"),
            },
            _ => json(),
        },
        _ => json(),
    }
}

fn scalar_string_at(value: &JsonPayload, pointer: &str) -> Option<String> {
    String::from_utf16(&value.string_units_at(pointer)?).ok()
}

/// The text slot: one line per block, the next note standing in for each
/// media block in order. Source text keeps its exact UTF-16 units.
pub(crate) fn text<'n>(pieces: &[Piece], notes: impl IntoIterator<Item = &'n str>) -> JsonPayload {
    let mut notes = notes.into_iter();
    let mut units = Vec::new();
    for (index, piece) in pieces.iter().enumerate() {
        if index > 0 {
            units.push(b'\n' as u16);
        }
        match piece {
            Piece::Text(text) => units.extend_from_slice(text),
            Piece::Json(value) => {
                let json = value
                    .canonical_json(crate::exact_json::JsonEncoding::Serde)
                    .unwrap_or_else(|_| value.as_json_source().to_owned());
                units.extend(json.encode_utf16());
            }
            Piece::Image(_) | Piece::Document { .. } => {
                units.extend(notes.next().unwrap_or_default().encode_utf16())
            }
        }
    }
    JsonPayload::string_from_utf16_code_units(&units)
}

/// A selected catalog row that lists input modalities without images rules
/// image parts out. Rows without that metadata keep the image the tool chose
/// to return.
pub(crate) fn accepts_images(models: &[ModelProfile]) -> bool {
    models.iter().all(|model| {
        model.metadata.input_modalities.is_empty()
            || crate::files::model_declares_image_input(model)
    })
}

/// Documents need an explicit declaration: a file part the endpoint rejects
/// fails every later request that replays the result.
pub(crate) fn accepts_documents(models: &[ModelProfile]) -> bool {
    !models.is_empty() && models.iter().all(declares_documents)
}

/// A wire that carries tool-result media as-is drops it only when a selected
/// row lists input modalities without it. Rows without that metadata, such as
/// a custom Claude profile, keep what the tool returned.
pub(crate) fn rules_out_documents(models: &[ModelProfile]) -> bool {
    models
        .iter()
        .any(|model| !model.metadata.input_modalities.is_empty() && !declares_documents(model))
}

fn declares_documents(model: &ModelProfile) -> bool {
    model.metadata.input_modalities.iter().any(|modality| {
        matches!(
            modality.to_ascii_lowercase().as_str(),
            "pdf" | "file" | "files" | "document"
        )
    }) || model.capability_support_for(crate::protocol::ModelCapability::Documents)
        == crate::protocol::CapabilitySupport::Supported
}
