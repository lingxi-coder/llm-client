//! MiniMax file purpose for video understanding.
use crate::{client::AttachmentKind, files::FilePurpose};
pub(crate) fn file_purpose(kind: AttachmentKind) -> FilePurpose {
    if kind == AttachmentKind::Video {
        FilePurpose::VideoUnderstanding
    } else {
        FilePurpose::ModelInput
    }
}
