//! Media inside a tool result, for wires whose tool-output slot holds text.
//!
//! Tool results arrive in the Anthropic content shapes Claude Code sends
//! (`text`, `image`, `document`), and Anthropic Messages carries them as-is
//! unless the selected model's catalog row rules the media out.
//! An OpenAI Chat `tool` message and a Gemini `functionResponse` hold text, so
//! a content array placed there is billed as text, base64 included: one
//! 900×900 PNG Read cost ~135k input tokens on DeepSeek, which bills at most
//! 1,024 for the same image sent as an image part. Those codecs keep the
//! result's text in the slot and send each media block as its own part.

use crate::protocol::ModelProfile;
use serde_json::Value;

/// One tool-result block, as a text-only result slot sees it.
pub(crate) enum Piece<'a> {
    Text(&'a str),
    /// Any other block. The slot carries its JSON, as it did the whole array.
    Json(&'a Value),
    Image(Image<'a>),
    /// A base64 document; Read returns PDFs this way.
    Document {
        media_type: &'a str,
        data: &'a str,
        title: Option<&'a str>,
    },
}

pub(crate) enum Image<'a> {
    Base64 { media_type: &'a str, data: &'a str },
    Url(&'a str),
}

impl Piece<'_> {
    pub(crate) fn is_media(&self) -> bool {
        matches!(self, Self::Image(_) | Self::Document { .. })
    }
}

/// The classified blocks, or `None` when none is media and the codec's plain
/// result encoding applies unchanged.
pub(crate) fn pieces(blocks: &[Value]) -> Option<Vec<Piece<'_>>> {
    let pieces: Vec<_> = blocks.iter().map(piece).collect();
    pieces.iter().any(Piece::is_media).then_some(pieces)
}

/// The blocks a text-only slot carries as derived text: [`pieces`], or every
/// block when all are text. Sent as their JSON spelling, text blocks would
/// pay for the array's quoting and escapes on every replay.
pub(crate) fn slot_pieces(blocks: &[Value]) -> Option<Vec<Piece<'_>>> {
    pieces(blocks).or_else(|| {
        let pieces: Vec<_> = blocks.iter().map(piece).collect();
        pieces
            .iter()
            .all(|piece| matches!(piece, Piece::Text(_)))
            .then_some(pieces)
    })
}

pub(crate) fn has_media(blocks: &[Value]) -> bool {
    blocks.iter().any(|block| piece(block).is_media())
}

pub(crate) fn piece(block: &Value) -> Piece<'_> {
    let source = &block["source"];
    let base64 = || source["media_type"].as_str().zip(source["data"].as_str());
    match (block["type"].as_str(), source["type"].as_str()) {
        (Some("text"), _) => block["text"]
            .as_str()
            .map_or(Piece::Json(block), Piece::Text),
        (Some("image"), Some("base64")) => base64()
            .map_or(Piece::Json(block), |(media_type, data)| {
                Piece::Image(Image::Base64 { media_type, data })
            }),
        (Some("image"), Some("url")) => source["url"]
            .as_str()
            .map_or(Piece::Json(block), |url| Piece::Image(Image::Url(url))),
        (Some("document"), Some("base64")) => {
            base64().map_or(Piece::Json(block), |(media_type, data)| Piece::Document {
                media_type,
                data,
                title: block["title"].as_str(),
            })
        }
        _ => Piece::Json(block),
    }
}

/// The slot text: one line per block, the next of `notes` standing in for
/// each media block in order.
pub(crate) fn text<'n>(pieces: &[Piece<'_>], notes: impl IntoIterator<Item = &'n str>) -> String {
    let mut notes = notes.into_iter();
    let mut text = String::new();
    for (index, piece) in pieces.iter().enumerate() {
        if index > 0 {
            text.push('\n');
        }
        match piece {
            Piece::Text(value) => text.push_str(value),
            Piece::Json(value) => text.push_str(&value.to_string()),
            _ => text.push_str(notes.next().unwrap_or_default()),
        }
    }
    text
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
/// fails every later request that replays it.
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
