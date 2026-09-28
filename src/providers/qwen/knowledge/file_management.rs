//! Scoped list, delete, and tag-update operations for Qwen data-center files.
//!
//! These are data-center assets, distinct from documents already imported
//! into a knowledge base. All calls are single requests; pagination and any
//! follow-up cleanup remain caller-managed.

use super::{
    files::{invalid_file_response, response_outcome_unknown, FileEnvelope},
    *,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{HashMap, HashSet},
    fmt,
};

const LIST_FILES_PATH: &str = "/api/v1/connector/dash/listFile";
const DELETE_FILE_PATH: &str = "/api/v1/connector/dash/deleteFile";
const UPDATE_FILE_TAGS_PATH: &str = "/api/v1/connector/dash/batchUpdateFileTag";

/// Query one page of data-center files in a required category.
///
/// `file_name` is passed exactly as supplied. The provider contract expects a
/// filename without its extension and performs an exact match.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeFileListRequest {
    category_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    file_name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    file_ids: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    next_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_result: Option<u32>,
}

impl QwenKnowledgeFileListRequest {
    pub fn new(category_id: impl Into<String>) -> Self {
        Self {
            category_id: category_id.into(),
            file_name: None,
            file_ids: Vec::new(),
            next_token: None,
            max_result: None,
        }
    }

    /// Set an exact file-name filter. Supply the filename without extension.
    pub fn with_file_name(mut self, file_name: impl Into<String>) -> Self {
        self.file_name = Some(file_name.into());
        self
    }

    /// Set the documented batch file-ID filter (at most 20 IDs).
    pub fn with_file_ids(mut self, file_ids: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.file_ids = file_ids.into_iter().map(Into::into).collect();
        self
    }

    /// Continue a prior page using its opaque `nextToken`.
    pub fn with_next_token(mut self, next_token: impl Into<String>) -> Self {
        self.next_token = Some(next_token.into());
        self
    }

    /// Override the provider's default page size of 20.
    pub fn with_max_result(mut self, max_result: u32) -> Self {
        self.max_result = Some(max_result);
        self
    }

    pub fn category_id(&self) -> &str {
        &self.category_id
    }

    pub fn next_token(&self) -> Option<&str> {
        self.next_token.as_deref()
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        if !valid_resource_id(&self.category_id) {
            return Err(invalid(
                "category_id must be a valid Qwen data-center category ID",
            ));
        }
        if self
            .file_name
            .as_ref()
            .is_some_and(|value| value.trim().is_empty() || value.chars().any(char::is_control))
        {
            return Err(invalid(
                "file_name must be nonempty and contain no control characters",
            ));
        }
        if self.file_ids.len() > 20 || self.file_ids.iter().any(|value| !valid_resource_id(value)) {
            return Err(invalid(
                "file_ids must contain valid IDs and cannot exceed the documented 20-item limit",
            ));
        }
        if self
            .next_token
            .as_ref()
            .is_some_and(|value| value.trim().is_empty() || value.chars().any(char::is_control))
        {
            return Err(invalid("next_token must be nonempty when specified"));
        }
        Ok(())
    }

    fn to_body(&self) -> Value {
        let mut body = json!({"categoryId": self.category_id});
        if let Some(value) = &self.file_name {
            body["fileName"] = json!(value);
        }
        if !self.file_ids.is_empty() {
            body["fileIds"] = json!(self.file_ids);
        }
        if let Some(value) = &self.next_token {
            body["nextToken"] = json!(value);
        }
        if let Some(value) = self.max_result {
            body["maxResult"] = json!(value);
        }
        body
    }
}

impl fmt::Debug for QwenKnowledgeFileListRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenKnowledgeFileListRequest")
            .field("category_id", &self.category_id)
            .field("file_name", &self.file_name)
            .field("file_ids", &self.file_ids)
            .field(
                "next_token",
                &self.next_token.as_ref().map(|_| "<redacted>"),
            )
            .field("max_result", &self.max_result)
            .finish()
    }
}

/// One data-center asset returned by `listFile`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeDataFile {
    pub reference: QwenKnowledgeFileRef,
    pub file_name: Option<String>,
    pub file_type: Option<String>,
    pub status: Option<String>,
    pub parser: Option<String>,
    pub size_bytes: Option<u64>,
    pub upload_time: Option<String>,
    pub category: Option<String>,
    pub auto_id: Option<u64>,
    pub parse_error_message: Option<String>,
    /// Complete file row, including provider fields not projected above.
    pub native: Value,
}

/// One page from the data-center `listFile` endpoint.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeFileListResult {
    pub files: Vec<QwenKnowledgeDataFile>,
    pub has_next: bool,
    /// Present only when the provider reports another page. It can be passed
    /// unchanged to [`QwenKnowledgeFileListRequest::with_next_token`].
    pub next_token: Option<String>,
    pub max_result: Option<u64>,
    pub total_count: Option<u64>,
    pub max_id: Option<u64>,
    pub request_id: Option<String>,
    pub native: Value,
}

impl fmt::Debug for QwenKnowledgeFileListResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenKnowledgeFileListResult")
            .field("files", &self.files)
            .field("has_next", &self.has_next)
            .field(
                "next_token",
                &self.next_token.as_ref().map(|_| "<redacted>"),
            )
            .field("max_result", &self.max_result)
            .field("total_count", &self.total_count)
            .field("max_id", &self.max_id)
            .field("request_id", &self.request_id)
            .field("native", &"<preserved>")
            .finish()
    }
}

/// Delete a data-center file and retain the provider response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeFileDeleteResult {
    pub reference: QwenKnowledgeFileRef,
    /// Optional because the official success example returns `data: {}`.
    pub deleted_file_id: Option<String>,
    /// Optional status when the provider includes the documented `DELETED` value.
    pub status: Option<String>,
    pub request_id: Option<String>,
    pub native: Value,
}

/// The provider's documented tag replacement or append behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum QwenKnowledgeFileTagUpdateMode {
    Overwrite,
    Append,
}

impl QwenKnowledgeFileTagUpdateMode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Overwrite => "OVERWRITE",
            Self::Append => "APPEND",
        }
    }
}

/// Tags to assign to one scoped data-center file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeFileTagUpdate {
    file: QwenKnowledgeFileRef,
    tags: Vec<String>,
}

impl QwenKnowledgeFileTagUpdate {
    pub fn new(
        file: QwenKnowledgeFileRef,
        tags: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        Self {
            file,
            tags: tags.into_iter().map(Into::into).collect(),
        }
    }

    pub fn file(&self) -> &QwenKnowledgeFileRef {
        &self.file
    }

    pub fn tags(&self) -> &[String] {
        &self.tags
    }
}

/// One batch of file-tag updates. The API accepts 1–20 file entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeFileTagUpdateRequest {
    files: Vec<QwenKnowledgeFileTagUpdate>,
    update_mode: Option<QwenKnowledgeFileTagUpdateMode>,
}

impl QwenKnowledgeFileTagUpdateRequest {
    pub fn new(files: impl IntoIterator<Item = QwenKnowledgeFileTagUpdate>) -> Self {
        Self {
            files: files.into_iter().collect(),
            update_mode: None,
        }
    }

    pub fn with_update_mode(mut self, mode: QwenKnowledgeFileTagUpdateMode) -> Self {
        self.update_mode = Some(mode);
        self
    }

    pub fn files(&self) -> &[QwenKnowledgeFileTagUpdate] {
        &self.files
    }

    pub fn update_mode(&self) -> Option<QwenKnowledgeFileTagUpdateMode> {
        self.update_mode
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        if !(1..=20).contains(&self.files.len()) {
            return Err(invalid("file tag updates require 1 to 20 file entries"));
        }
        for file in &self.files {
            if file.tags.len() > 100
                || file.tags.iter().any(|tag| tag.chars().count() > 32)
                || file
                    .tags
                    .iter()
                    .map(|tag| tag.chars().count())
                    .sum::<usize>()
                    > 700
            {
                return Err(invalid(
                    "each file allows at most 100 tags, 32 characters per tag, and 700 tag characters total",
                ));
            }
        }
        Ok(())
    }

    fn to_body(&self) -> Value {
        let mut body = json!({
            "fileInfos": self.files.iter().map(|file| json!({
                "fileId": file.file.file_id(),
                "tags": file.tags,
            })).collect::<Vec<_>>(),
        });
        if let Some(mode) = self.update_mode {
            body["updateMode"] = json!(mode.as_str());
        }
        body
    }
}

/// One file's result in a batch tag update.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeFileTagUpdateItemResult {
    pub reference: QwenKnowledgeFileRef,
    pub success: bool,
    /// Complete per-file response row for future provider fields.
    pub native: Value,
}

/// Result of `batchUpdateFileTag`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeFileTagUpdateResult {
    /// `None` is retained when the valid success envelope contains the
    /// documented empty `data` object and omits per-file results.
    pub per_file_results: Option<Vec<QwenKnowledgeFileTagUpdateItemResult>>,
    pub request_id: Option<String>,
    pub native: Value,
}

impl<'a> QwenKnowledgeService<'a> {
    /// List one page of files in a category. Pass a returned `next_token` to
    /// request the next page; this method never paginates or retries itself.
    pub async fn list_files(
        &self,
        request: &QwenKnowledgeFileListRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeFileListResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        request.validate()?;
        let envelope = pinned_service
            .file_json_request(
                LIST_FILES_PATH,
                request.to_body(),
                "list_files",
                false,
                false,
                request_options,
            )
            .await?;
        decode_file_list(&pinned_service, request, envelope)
    }

    /// Permanently delete one data-center file. This does not delete a
    /// knowledge-base resource separately or rebuild its index.
    pub async fn delete_file(
        &self,
        file: &QwenKnowledgeFileRef,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeFileDeleteResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_file_ref(file)?;
        let envelope = pinned_service
            .file_json_request(
                DELETE_FILE_PATH,
                json!({"fileId": file.file_id()}),
                "delete_file",
                true,
                false,
                request_options,
            )
            .await?;
        let data = require_mutation_data_object(&envelope, "delete_file")?;
        let deleted_file_id = optional_string_field(data, "fileId").map_err(|message| {
            response_outcome_unknown(
                "delete_file",
                message,
                envelope.request_id.clone(),
                envelope.native.clone(),
            )
        })?;
        let status = optional_string_field(data, "status").map_err(|message| {
            response_outcome_unknown(
                "delete_file",
                message,
                envelope.request_id.clone(),
                envelope.native.clone(),
            )
        })?;
        if deleted_file_id
            .as_deref()
            .is_some_and(|returned_id| returned_id != file.file_id())
            || status.as_deref().is_some_and(|value| value != "DELETED")
        {
            return Err(response_outcome_unknown(
                "delete_file",
                "successful delete response did not confirm the requested file",
                envelope.request_id,
                envelope.native,
            ));
        }
        Ok(QwenKnowledgeFileDeleteResult {
            reference: file.clone(),
            deleted_file_id,
            status,
            request_id: envelope.request_id,
            native: envelope.native,
        })
    }

    /// Apply one API batch request for the supplied file-tag entries.
    /// Provider per-file failures remain in the returned `success` fields.
    pub async fn update_file_tags(
        &self,
        request: &QwenKnowledgeFileTagUpdateRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeFileTagUpdateResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        request.validate()?;
        for item in &request.files {
            pinned_service.validate_file_ref(&item.file)?;
        }
        let envelope = pinned_service
            .file_json_request(
                UPDATE_FILE_TAGS_PATH,
                request.to_body(),
                "update_file_tags",
                true,
                false,
                request_options,
            )
            .await?;
        decode_tag_update_result(request, envelope)
    }
}

fn decode_file_list(
    service: &QwenKnowledgeService<'_>,
    request: &QwenKnowledgeFileListRequest,
    envelope: FileEnvelope,
) -> Result<QwenKnowledgeFileListResult, QwenKnowledgeError> {
    let data = envelope
        .native
        .get("data")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            invalid_file_response(
                "list_files",
                "successful response omitted object data",
                envelope.request_id.clone(),
                envelope.native.clone(),
            )
        })?;
    let fail = |message: &str| {
        invalid_file_response(
            "list_files",
            message,
            envelope.request_id.clone(),
            envelope.native.clone(),
        )
    };
    let has_next = data
        .get("hasNext")
        .and_then(Value::as_bool)
        .ok_or_else(|| fail("data.hasNext must be a boolean"))?;
    let raw_token = data.get("nextToken");
    if raw_token.is_some_and(|value| !value.is_string()) {
        return Err(fail("data.nextToken must be a string when present"));
    }
    let next_token = raw_token
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty());
    if has_next {
        let Some(next_token) = next_token else {
            return Err(fail("hasNext=true requires a nonempty nextToken"));
        };
        if request.next_token.as_deref() == Some(next_token) {
            return Err(fail("nextToken did not advance to another page"));
        }
    }
    let rows = data
        .get("fileList")
        .and_then(Value::as_array)
        .ok_or_else(|| fail("successful response omitted data.fileList"))?;
    let mut seen = HashSet::with_capacity(rows.len());
    let mut files = Vec::with_capacity(rows.len());
    for row in rows {
        let row_object = row
            .as_object()
            .ok_or_else(|| fail("fileList items must be objects"))?;
        let file_id = row_object
            .get("fileId")
            .and_then(Value::as_str)
            .filter(|value| valid_resource_id(value))
            .ok_or_else(|| fail("fileList item omitted a valid fileId"))?;
        if !seen.insert(file_id) {
            return Err(fail("fileList contained a duplicate fileId"));
        }
        files.push(QwenKnowledgeDataFile {
            reference: service.scope().file_ref(file_id)?,
            file_name: string_field(row, "fileName"),
            file_type: string_field(row, "fileType"),
            status: string_field(row, "status"),
            parser: string_field(row, "parser"),
            size_bytes: optional_u64_field(row_object, "sizeBytes").map_err(fail)?,
            upload_time: string_field(row, "uploadTime"),
            category: string_field(row, "category"),
            auto_id: optional_u64_field(row_object, "autoId").map_err(fail)?,
            parse_error_message: string_field(row, "parseErrorMessage"),
            native: row.clone(),
        });
    }
    Ok(QwenKnowledgeFileListResult {
        files,
        has_next,
        next_token: next_token.map(str::to_owned),
        max_result: optional_u64_field(data, "maxResult").map_err(fail)?,
        total_count: optional_u64_field(data, "totalCount").map_err(fail)?,
        max_id: optional_u64_field(data, "maxId").map_err(fail)?,
        request_id: envelope.request_id,
        native: envelope.native,
    })
}

fn decode_tag_update_result(
    request: &QwenKnowledgeFileTagUpdateRequest,
    envelope: FileEnvelope,
) -> Result<QwenKnowledgeFileTagUpdateResult, QwenKnowledgeError> {
    let fail = |message: &str| {
        response_outcome_unknown(
            "update_file_tags",
            message,
            envelope.request_id.clone(),
            envelope.native.clone(),
        )
    };
    let data = envelope
        .native
        .get("data")
        .and_then(Value::as_object)
        .ok_or_else(|| fail("successful response omitted object data"))?;
    let Some(raw_results) = data.get("results") else {
        // The current API's success example returns `data: {}`; keep the
        // per-file result layer explicitly unknown in that documented shape.
        return Ok(QwenKnowledgeFileTagUpdateResult {
            per_file_results: None,
            request_id: envelope.request_id,
            native: envelope.native,
        });
    };
    let rows = raw_results
        .as_array()
        .ok_or_else(|| fail("data.results must be an array when present"))?;
    let mut requested = request
        .files
        .iter()
        .map(|item| (item.file.file_id().to_owned(), &item.file))
        .collect::<HashMap<_, _>>();
    let mut seen = HashSet::with_capacity(rows.len());
    let mut results = Vec::with_capacity(rows.len());
    for row in rows {
        let file_id = row
            .get("fileId")
            .and_then(Value::as_str)
            .filter(|value| valid_resource_id(value))
            .ok_or_else(|| fail("data.results item omitted a valid fileId"))?;
        if !seen.insert(file_id) {
            return Err(fail("data.results contained a duplicate fileId"));
        }
        let Some(reference) = requested.remove(file_id) else {
            return Err(fail("data.results returned a file that was not requested"));
        };
        let success = row
            .get("success")
            .and_then(Value::as_bool)
            .ok_or_else(|| fail("data.results item omitted boolean success"))?;
        results.push(QwenKnowledgeFileTagUpdateItemResult {
            reference: reference.clone(),
            success,
            native: row.clone(),
        });
    }
    Ok(QwenKnowledgeFileTagUpdateResult {
        per_file_results: Some(results),
        request_id: envelope.request_id,
        native: envelope.native,
    })
}

fn require_mutation_data_object<'a>(
    envelope: &'a FileEnvelope,
    operation: &'static str,
) -> Result<&'a serde_json::Map<String, Value>, QwenKnowledgeError> {
    envelope
        .native
        .get("data")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            response_outcome_unknown(
                operation,
                "successful response omitted object data",
                envelope.request_id.clone(),
                envelope.native.clone(),
            )
        })
}

fn optional_string_field(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<String>, &'static str> {
    match object.get(field) {
        None => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err("successful response contained a non-string optional field"),
    }
}

fn optional_u64_field(
    object: &serde_json::Map<String, Value>,
    field: &str,
) -> Result<Option<u64>, &'static str> {
    match object.get(field) {
        None => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or("successful response contained a non-integer optional field"),
    }
}

fn string_field(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(Value::as_str).map(str::to_owned)
}
