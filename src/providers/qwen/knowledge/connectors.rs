//! Data connectors and imports from an already-authorized OSS bucket.
//!
//! These operations use the Model Studio data-import routes. The caller owns
//! OSS bucket authorization and object management; this module only submits
//! bucket names and object keys to the RAG workspace API.

use super::{
    files::{invalid_file_response, response_outcome_unknown, FileEnvelope},
    *,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::fmt;

const CREATE_CONNECTOR_PATH: &str = "/api/v1/connector/dash/addConnector";
const GET_CONNECTOR_PATH: &str = "/api/v1/connector/dash/getConnector";
const IMPORT_FROM_OSS_PATH: &str = "/api/v1/connector/dash/addFilesFromAuthorizedOss";
const MAX_OSS_IMPORT_FILES: usize = 10;
const MAX_OSS_IMPORT_TAGS: usize = 10;

/// Storage managed by a Model Studio file connector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum QwenKnowledgeConnectorStoreType {
    /// Platform-managed storage.
    Platform,
    /// A caller-owned OSS bucket.
    Custom,
}

impl QwenKnowledgeConnectorStoreType {
    fn as_str(self) -> &'static str {
        match self {
            Self::Platform => "PLATFORM",
            Self::Custom => "CUSTOM",
        }
    }
}

/// Opaque identity for a connector in one Model Studio workspace/account.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeConnectorRef {
    scope: QwenKnowledgeScope,
    endpoint_fingerprint: String,
    connector_id: String,
}

impl QwenKnowledgeConnectorRef {
    /// Bind a connector ID obtained from this exact Model Studio scope.
    pub fn from_scope(
        scope: &QwenKnowledgeScope,
        connector_id: impl Into<String>,
    ) -> Result<Self, QwenKnowledgeError> {
        scope.validate()?;
        let connector_id = connector_id.into();
        if connector_id.is_empty() {
            return Err(invalid("Qwen connector ID is invalid"));
        }
        Ok(Self {
            scope: scope.clone(),
            endpoint_fingerprint: scope.endpoint_fingerprint(),
            connector_id,
        })
    }

    pub fn scope(&self) -> &QwenKnowledgeScope {
        &self.scope
    }

    pub fn connector_id(&self) -> &str {
        &self.connector_id
    }
}

impl QwenKnowledgeScope {
    /// Bind a connector ID obtained from this exact data-center scope.
    pub fn connector_ref(
        &self,
        connector_id: impl Into<String>,
    ) -> Result<QwenKnowledgeConnectorRef, QwenKnowledgeError> {
        QwenKnowledgeConnectorRef::from_scope(self, connector_id)
    }
}

/// Create the documented `FILE` connector.
///
/// Model Studio currently documents only `FILE` for `connectorType`, so the
/// request does not expose an unsupported connector type selector.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeConnectorCreateRequest {
    connector_name: String,
    description: String,
    store_type: QwenKnowledgeConnectorStoreType,
    region_id: Option<String>,
    bucket_name: Option<String>,
}

impl QwenKnowledgeConnectorCreateRequest {
    pub fn new(connector_name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            connector_name: connector_name.into(),
            description: description.into(),
            store_type: QwenKnowledgeConnectorStoreType::Platform,
            region_id: None,
            bucket_name: None,
        }
    }

    /// Configure a connector for the caller's own OSS bucket.
    pub fn with_custom_oss(
        mut self,
        region_id: impl Into<String>,
        bucket_name: impl Into<String>,
    ) -> Self {
        self.store_type = QwenKnowledgeConnectorStoreType::Custom;
        self.region_id = Some(region_id.into());
        self.bucket_name = Some(bucket_name.into());
        self
    }

    pub fn connector_name(&self) -> &str {
        &self.connector_name
    }

    pub fn description(&self) -> &str {
        &self.description
    }

    pub fn store_type(&self) -> QwenKnowledgeConnectorStoreType {
        self.store_type
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        let name_len = self.connector_name.chars().count();
        if !(1..=20).contains(&name_len) {
            return Err(invalid("connector_name must contain 1 to 20 characters"));
        }
        let description_len = self.description.chars().count();
        if !(1..=200).contains(&description_len) {
            return Err(invalid("description must contain 1 to 200 characters"));
        }
        match self.store_type {
            QwenKnowledgeConnectorStoreType::Platform
                if self.region_id.is_some() || self.bucket_name.is_some() =>
            {
                Err(invalid(
                    "region_id and bucket_name are only allowed for CUSTOM connector storage",
                ))
            }
            QwenKnowledgeConnectorStoreType::Custom => {
                let (Some(region_id), Some(bucket_name)) =
                    (self.region_id.as_deref(), self.bucket_name.as_deref())
                else {
                    return Err(invalid(
                        "CUSTOM connector storage requires region_id and bucket_name",
                    ));
                };
                if !valid_nonempty_text(region_id) || !valid_nonempty_text(bucket_name) {
                    return Err(invalid(
                        "CUSTOM connector region_id and bucket_name must be nonempty",
                    ));
                }
                Ok(())
            }
            QwenKnowledgeConnectorStoreType::Platform => Ok(()),
        }
    }

    fn to_body(&self) -> Value {
        let mut config = Map::new();
        config.insert("storeType".into(), json!(self.store_type.as_str()));
        if let Some(region_id) = &self.region_id {
            config.insert("regionId".into(), json!(region_id));
        }
        if let Some(bucket_name) = &self.bucket_name {
            config.insert("bucketName".into(), json!(bucket_name));
        }
        json!({
            "connectorType":"FILE",
            "connectorName":self.connector_name,
            "description":self.description,
            "fileConnectorConfig":Value::Object(config)
        })
    }
}

/// Result of creating a connector. The full provider envelope remains in
/// `native` for fields added by Model Studio.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeConnectorCreateResult {
    pub reference: QwenKnowledgeConnectorRef,
    pub request_id: Option<String>,
    pub native: Value,
}

/// Select a connector by ID, name, or both.
///
/// The provider requires at least one selector and gives `connector_id`
/// precedence when both are present.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeConnectorLookup {
    connector_id: Option<QwenKnowledgeConnectorRef>,
    connector_name: Option<String>,
}

impl QwenKnowledgeConnectorLookup {
    pub fn by_id(reference: QwenKnowledgeConnectorRef) -> Self {
        Self {
            connector_id: Some(reference),
            connector_name: None,
        }
    }

    pub fn by_name(connector_name: impl Into<String>) -> Self {
        Self {
            connector_id: None,
            connector_name: Some(connector_name.into()),
        }
    }

    pub fn with_id(mut self, reference: QwenKnowledgeConnectorRef) -> Self {
        self.connector_id = Some(reference);
        self
    }

    pub fn with_name(mut self, connector_name: impl Into<String>) -> Self {
        self.connector_name = Some(connector_name.into());
        self
    }

    pub fn connector_id(&self) -> Option<&QwenKnowledgeConnectorRef> {
        self.connector_id.as_ref()
    }

    pub fn connector_name(&self) -> Option<&str> {
        self.connector_name.as_deref()
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        if self.connector_id.is_none() && self.connector_name.is_none() {
            return Err(invalid(
                "get connector requires a connector ID or connector name",
            ));
        }
        if self
            .connector_name
            .as_ref()
            .is_some_and(|name| name.chars().count() > 20)
        {
            return Err(invalid("connector_name must contain at most 20 characters"));
        }
        Ok(())
    }

    fn to_body(&self) -> Value {
        let mut body = Map::new();
        if let Some(reference) = &self.connector_id {
            body.insert("connectorId".into(), json!(reference.connector_id));
        }
        if let Some(name) = &self.connector_name {
            body.insert("connectorName".into(), json!(name));
        }
        Value::Object(body)
    }
}

impl fmt::Debug for QwenKnowledgeConnectorLookup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenKnowledgeConnectorLookup")
            .field("connector_id", &self.connector_id)
            .field("connector_name", &self.connector_name)
            .finish()
    }
}

/// Connector metadata returned by `getConnector`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeConnectorDetails {
    pub reference: QwenKnowledgeConnectorRef,
    pub connector_name: Option<String>,
    pub connector_type: Option<String>,
    pub connector_sub_type: Option<String>,
    pub description: Option<String>,
    pub created_at: Option<String>,
    pub modified_at: Option<String>,
    pub request_id: Option<String>,
    pub native: Value,
}

/// One file to import from a caller-authorized OSS bucket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenKnowledgeOssImportFile {
    file_name: String,
    oss_key: String,
    parser: Option<QwenKnowledgeParser>,
    parser_config: Option<QwenKnowledgeParserConfig>,
}

impl QwenKnowledgeOssImportFile {
    pub fn new(file_name: impl Into<String>, oss_key: impl Into<String>) -> Self {
        Self {
            file_name: file_name.into(),
            oss_key: oss_key.into(),
            parser: None,
            parser_config: None,
        }
    }

    pub fn with_parser(mut self, parser: QwenKnowledgeParser) -> Self {
        self.parser = Some(parser);
        self
    }

    pub fn with_parser_config(mut self, config: QwenKnowledgeParserConfig) -> Self {
        self.parser_config = Some(config);
        self
    }

    pub fn file_name(&self) -> &str {
        &self.file_name
    }

    pub fn oss_key(&self) -> &str {
        &self.oss_key
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        let file_name_len = self.file_name.chars().count();
        if !(1..=500).contains(&file_name_len) {
            return Err(invalid(
                "OSS import file_name must contain 1 to 500 characters",
            ));
        }
        let oss_key_len = self.oss_key.chars().count();
        if !(1..=256).contains(&oss_key_len) {
            return Err(invalid(
                "OSS import oss_key must contain 1 to 256 characters",
            ));
        }
        match self.parser {
            Some(QwenKnowledgeParser::DashQwenVlParser) => {
                let Some(config) = &self.parser_config else {
                    return Err(invalid(
                        "DASH_QWEN_VL_PARSER requires parser_config with a model prompt",
                    ));
                };
                let prompt_len = config.model_prompt().chars().count();
                if !(1..=1500).contains(&prompt_len) {
                    return Err(invalid(
                        "Qwen VL parser model_prompt must contain 1 to 1500 characters",
                    ));
                }
                Ok(())
            }
            Some(_) if self.parser_config.is_some() => Err(invalid(
                "parser_config is only allowed for DASH_QWEN_VL_PARSER",
            )),
            _ if self.parser_config.is_some() => Err(invalid(
                "parser_config requires the DASH_QWEN_VL_PARSER parser",
            )),
            _ => Ok(()),
        }
    }

    fn to_value(&self) -> Value {
        let mut value = Map::new();
        value.insert("fileName".into(), json!(self.file_name));
        value.insert("ossKey".into(), json!(self.oss_key));
        if let Some(parser) = self.parser {
            value.insert("parser".into(), json!(parser_as_str(parser)));
        }
        if let Some(config) = &self.parser_config {
            value.insert(
                "parserConfig".into(),
                json!({"modelName":"qwen3-vl-plus", "modelPrompt":config.model_prompt()}),
            );
        }
        Value::Object(value)
    }
}

/// Batch-import up to ten explicit object keys from an authorized OSS bucket.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QwenKnowledgeOssImportRequest {
    category_id: String,
    category: Option<QwenKnowledgeCategoryRef>,
    category_type: QwenKnowledgeCategoryType,
    oss_bucket: String,
    oss_region_id: String,
    file_details: Vec<QwenKnowledgeOssImportFile>,
    tags: Vec<String>,
    overwrite_file_by_oss_key: Option<bool>,
}

impl QwenKnowledgeOssImportRequest {
    pub fn new(
        category_id: impl Into<String>,
        oss_bucket: impl Into<String>,
        oss_region_id: impl Into<String>,
        file_details: impl IntoIterator<Item = QwenKnowledgeOssImportFile>,
    ) -> Self {
        Self {
            category_id: category_id.into(),
            category: None,
            category_type: QwenKnowledgeCategoryType::Unstructured,
            oss_bucket: oss_bucket.into(),
            oss_region_id: oss_region_id.into(),
            file_details: file_details.into_iter().collect(),
            tags: Vec::new(),
            overwrite_file_by_oss_key: None,
        }
    }

    /// Create an import request using a category reference returned by this
    /// Qwen workspace. Its account scope is checked again at dispatch time.
    pub fn for_category(
        category: QwenKnowledgeCategoryRef,
        oss_bucket: impl Into<String>,
        oss_region_id: impl Into<String>,
        file_details: impl IntoIterator<Item = QwenKnowledgeOssImportFile>,
    ) -> Self {
        Self {
            category_id: category.category_id().to_owned(),
            category_type: category.category_type(),
            category: Some(category),
            oss_bucket: oss_bucket.into(),
            oss_region_id: oss_region_id.into(),
            file_details: file_details.into_iter().collect(),
            tags: Vec::new(),
            overwrite_file_by_oss_key: None,
        }
    }

    pub fn with_category_type(mut self, category_type: QwenKnowledgeCategoryType) -> Self {
        self.category_type = category_type;
        self
    }

    pub fn with_tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_overwrite_file_by_oss_key(mut self, overwrite: bool) -> Self {
        self.overwrite_file_by_oss_key = Some(overwrite);
        self
    }

    pub fn category_id(&self) -> &str {
        &self.category_id
    }

    pub fn category_ref(&self) -> Option<&QwenKnowledgeCategoryRef> {
        self.category.as_ref()
    }

    pub fn file_details(&self) -> &[QwenKnowledgeOssImportFile] {
        &self.file_details
    }

    fn validate(&self, scope: &QwenKnowledgeScope) -> Result<(), QwenKnowledgeError> {
        if self.category_id.is_empty() {
            return Err(invalid("OSS import category_id must be nonempty"));
        }
        if let Some(category) = &self.category {
            let canonical =
                QwenKnowledgeCategoryRef::from_scope(category.scope(), category.category_id())?;
            if category.scope() != scope
                || canonical != *category
                || category.category_id() != self.category_id
                || category.category_type() != self.category_type
            {
                return Err(LlmError::PermissionDenied {
                    message: "Qwen OSS import category belongs to another profile, account, region, workspace, endpoint, or category namespace".into(),
                }
                .into());
            }
        }
        if !valid_nonempty_text(&self.oss_bucket) || !valid_nonempty_text(&self.oss_region_id) {
            return Err(invalid(
                "OSS import requires nonempty oss_bucket and oss_region_id",
            ));
        }
        if !(1..=MAX_OSS_IMPORT_FILES).contains(&self.file_details.len()) {
            return Err(invalid(
                "OSS import file_details must contain 1 to 10 files",
            ));
        }
        for file in &self.file_details {
            file.validate()?;
            if self.category_type == QwenKnowledgeCategoryType::SessionFile
                && (file.parser.is_some() || file.parser_config.is_some())
            {
                return Err(invalid(
                    "SESSION_FILE imports use the default parser and cannot specify parser settings",
                ));
            }
        }
        if self.tags.len() > MAX_OSS_IMPORT_TAGS {
            return Err(invalid("OSS import accepts at most 10 tags"));
        }
        Ok(())
    }

    fn to_body(&self) -> Value {
        let mut body = json!({
            "categoryId":self.category_id,
            "categoryType":category_type_as_str(self.category_type),
            "ossBucket":self.oss_bucket,
            "ossRegionId":self.oss_region_id,
            "fileDetails":self.file_details.iter().map(QwenKnowledgeOssImportFile::to_value).collect::<Vec<_>>()
        });
        if !self.tags.is_empty() {
            body["tags"] = json!(self.tags);
        }
        if let Some(overwrite) = self.overwrite_file_by_oss_key {
            body["overWriteFileByOssKey"] = json!(overwrite);
        }
        body
    }
}

/// Result of one OSS import request.
///
/// The official success example returns an empty `data` object even though
/// the response-field table describes `fileIds`. `imported_files` is thus
/// optional; no job identity or completion status is inferred from this
/// response.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeOssImportResult {
    pub imported_files: Option<Vec<QwenKnowledgeFileRef>>,
    pub request_id: Option<String>,
    pub native: Value,
}

impl<'a> QwenKnowledgeService<'a> {
    /// Create a Model Studio `FILE` data connector.
    pub async fn create_connector(
        &self,
        request: &QwenKnowledgeConnectorCreateRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeConnectorCreateResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        request.validate()?;
        let envelope = pinned_service
            .file_json_request(
                CREATE_CONNECTOR_PATH,
                request.to_body(),
                "create_connector",
                true,
                false,
                request_options,
            )
            .await?;
        decode_created_connector(&pinned_service.scope, envelope)
    }

    /// Look up a connector once by ID, name, or both.
    pub async fn get_connector(
        &self,
        lookup: &QwenKnowledgeConnectorLookup,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeConnectorDetails, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        lookup.validate()?;
        if let Some(reference) = &lookup.connector_id {
            pinned_service.validate_connector_ref(reference)?;
        }
        let envelope = pinned_service
            .file_json_request(
                GET_CONNECTOR_PATH,
                lookup.to_body(),
                "get_connector",
                false,
                false,
                request_options,
            )
            .await?;
        decode_connector_details(&pinned_service.scope, lookup, envelope)
    }

    /// Submit one OSS import request for explicit object keys.
    ///
    /// OSS service-linked-role authorization must already be in place. This
    /// method neither checks nor changes bucket permissions, uploads bytes,
    /// retries the request, nor polls for processing completion.
    pub async fn import_files_from_oss(
        &self,
        request: &QwenKnowledgeOssImportRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeOssImportResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        request.validate(&pinned_service.scope)?;
        let envelope = pinned_service
            .file_json_request(
                IMPORT_FROM_OSS_PATH,
                request.to_body(),
                "import_files_from_oss",
                true,
                false,
                request_options,
            )
            .await?;
        decode_oss_import_result(&pinned_service.scope, envelope)
    }

    pub(super) fn validate_connector_ref(
        &self,
        reference: &QwenKnowledgeConnectorRef,
    ) -> Result<(), QwenKnowledgeError> {
        let belongs_to_scope =
            QwenKnowledgeConnectorRef::from_scope(self.scope(), reference.connector_id())
                .is_ok_and(|expected| expected == *reference);
        if !belongs_to_scope {
            return Err(LlmError::PermissionDenied {
                message: "Qwen connector reference belongs to another profile, account, region, workspace, or endpoint".into(),
            }
            .into());
        }
        Ok(())
    }
}

fn decode_created_connector(
    scope: &QwenKnowledgeScope,
    envelope: FileEnvelope,
) -> Result<QwenKnowledgeConnectorCreateResult, QwenKnowledgeError> {
    let data = envelope.native.get("data");
    let Some(connector_id) = data
        .and_then(|value| value.get("connectorId"))
        .and_then(Value::as_str)
    else {
        return Err(response_outcome_unknown(
            "create_connector",
            "successful response omitted data.connectorId",
            envelope.request_id,
            envelope.native,
        ));
    };
    let reference = QwenKnowledgeConnectorRef::from_scope(scope, connector_id).map_err(|_| {
        response_outcome_unknown(
            "create_connector",
            "successful response contained an invalid data.connectorId",
            envelope.request_id.clone(),
            envelope.native.clone(),
        )
    })?;
    Ok(QwenKnowledgeConnectorCreateResult {
        reference,
        request_id: envelope.request_id,
        native: envelope.native,
    })
}

fn decode_connector_details(
    scope: &QwenKnowledgeScope,
    lookup: &QwenKnowledgeConnectorLookup,
    envelope: FileEnvelope,
) -> Result<QwenKnowledgeConnectorDetails, QwenKnowledgeError> {
    let data = envelope.native.get("data");
    let Some(connector_id) = data
        .and_then(|value| value.get("connectorId"))
        .and_then(Value::as_str)
    else {
        return Err(invalid_file_response(
            "get_connector",
            "successful response omitted data.connectorId",
            envelope.request_id,
            envelope.native,
        ));
    };
    if lookup
        .connector_id
        .as_ref()
        .is_some_and(|expected| expected.connector_id != connector_id)
    {
        return Err(invalid_file_response(
            "get_connector",
            "response connectorId did not match the requested connector ID",
            envelope.request_id,
            envelope.native,
        ));
    }
    let reference = QwenKnowledgeConnectorRef::from_scope(scope, connector_id).map_err(|_| {
        invalid_file_response(
            "get_connector",
            "successful response contained an invalid data.connectorId",
            envelope.request_id.clone(),
            envelope.native.clone(),
        )
    })?;
    let data = data.expect("connector ID was extracted from data");
    Ok(QwenKnowledgeConnectorDetails {
        reference,
        connector_name: string_field(data, "connectorName"),
        connector_type: string_field(data, "connectorType"),
        connector_sub_type: string_field(data, "connectorSubType"),
        description: string_field(data, "description"),
        created_at: string_field(data, "gmtCreate"),
        modified_at: string_field(data, "gmtModified"),
        request_id: envelope.request_id,
        native: envelope.native,
    })
}

fn decode_oss_import_result(
    scope: &QwenKnowledgeScope,
    envelope: FileEnvelope,
) -> Result<QwenKnowledgeOssImportResult, QwenKnowledgeError> {
    let file_ids = envelope
        .native
        .get("data")
        .and_then(|value| value.get("fileIds"))
        .cloned();
    let imported_files = match file_ids {
        None => None,
        Some(Value::Array(ids)) => {
            let mut files = Vec::with_capacity(ids.len());
            for id in ids {
                let Some(id) = id.as_str() else {
                    return Err(response_outcome_unknown(
                        "import_files_from_oss",
                        "successful response contained a non-string data.fileIds entry",
                        envelope.request_id,
                        envelope.native,
                    ));
                };
                let file = QwenKnowledgeFileRef::from_scope(scope, id).map_err(|_| {
                    response_outcome_unknown(
                        "import_files_from_oss",
                        "successful response contained an invalid data.fileIds entry",
                        envelope.request_id.clone(),
                        envelope.native.clone(),
                    )
                })?;
                files.push(file);
            }
            Some(files)
        }
        Some(_) => {
            return Err(response_outcome_unknown(
                "import_files_from_oss",
                "successful response data.fileIds was not an array",
                envelope.request_id,
                envelope.native,
            ));
        }
    };
    Ok(QwenKnowledgeOssImportResult {
        imported_files,
        request_id: envelope.request_id,
        native: envelope.native,
    })
}

fn string_field(value: &Value, field: &str) -> Option<String> {
    value.get(field).and_then(Value::as_str).map(str::to_owned)
}

fn category_type_as_str(category_type: QwenKnowledgeCategoryType) -> &'static str {
    match category_type {
        QwenKnowledgeCategoryType::Unstructured => "UNSTRUCTURED",
        QwenKnowledgeCategoryType::SessionFile => "SESSION_FILE",
    }
}

fn parser_as_str(parser: QwenKnowledgeParser) -> &'static str {
    match parser {
        QwenKnowledgeParser::AutoSelect => "AUTO_SELECT",
        QwenKnowledgeParser::Docmind => "DOCMIND",
        QwenKnowledgeParser::DocmindDigital => "DOCMIND_DIGITAL",
        QwenKnowledgeParser::DocmindLlmVersion => "DOCMIND_LLM_VERSION",
        QwenKnowledgeParser::DashQwenVlParser => "DASH_QWEN_VL_PARSER",
        QwenKnowledgeParser::DocmindLlmVersionMedia => "DOCMIND_LLM_VERSION_MEDIA",
    }
}

fn valid_nonempty_text(value: &str) -> bool {
    !value.is_empty()
}

fn invalid(message: impl Into<String>) -> QwenKnowledgeError {
    QwenKnowledgeError::InvalidInput(message.into())
}
