//! Scoped category listing and lifecycle operations for Qwen data-center files.
//!
//! Categories organize data-center files. Deleting one makes its files
//! uncategorized; it does not delete the files or their knowledge-base data.

use super::{
    files::{invalid_file_response, response_outcome_unknown, FileEnvelope},
    *,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::HashSet, fmt};

const LIST_CATEGORIES_PATH: &str = "/api/v1/connector/dash/listCategory";
const CREATE_CATEGORY_PATH: &str = "/api/v1/connector/dash/addCategory";
const DELETE_CATEGORY_PATH: &str = "/api/v1/connector/dash/deleteCategory";

/// A data-center category ID bound to one Qwen workspace and connection scope.
///
/// The RAG category endpoints currently support the `UNSTRUCTURED` category
/// namespace. Other category namespaces, including session files, are not
/// accepted by these methods.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeCategoryRef {
    scope: QwenKnowledgeScope,
    endpoint_fingerprint: String,
    category_id: String,
    category_type: QwenKnowledgeCategoryType,
}

impl QwenKnowledgeCategoryRef {
    /// Bind a category ID obtained from this exact Qwen knowledge scope.
    pub fn from_scope(
        scope: &QwenKnowledgeScope,
        category_id: impl Into<String>,
    ) -> Result<Self, QwenKnowledgeError> {
        scope.validate()?;
        let category_id = category_id.into();
        if !valid_resource_id(&category_id) {
            return Err(invalid("Qwen knowledge category ID is invalid"));
        }
        Ok(Self {
            scope: scope.clone(),
            endpoint_fingerprint: scope.endpoint_fingerprint(),
            category_id,
            category_type: QwenKnowledgeCategoryType::Unstructured,
        })
    }

    pub fn scope(&self) -> &QwenKnowledgeScope {
        &self.scope
    }

    pub fn category_id(&self) -> &str {
        &self.category_id
    }

    pub fn category_type(&self) -> QwenKnowledgeCategoryType {
        self.category_type
    }
}

impl QwenKnowledgeScope {
    /// Bind a category ID obtained from this workspace and account scope.
    pub fn category_ref(
        &self,
        category_id: impl Into<String>,
    ) -> Result<QwenKnowledgeCategoryRef, QwenKnowledgeError> {
        QwenKnowledgeCategoryRef::from_scope(self, category_id)
    }
}

/// Filters and pagination options for one page of categories.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeCategoryListRequest {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parent_category: Option<QwenKnowledgeCategoryRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    category_name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    connector_ref: Option<QwenKnowledgeConnectorRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    next_token: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    max_result: Option<u32>,
}

impl QwenKnowledgeCategoryListRequest {
    pub fn new() -> Self {
        Self {
            parent_category: None,
            category_name: None,
            connector_ref: None,
            next_token: None,
            max_result: None,
        }
    }

    /// Filter for child categories of this scoped category.
    pub fn with_parent_category(mut self, parent: QwenKnowledgeCategoryRef) -> Self {
        self.parent_category = Some(parent);
        self
    }

    /// Filter by the provider's exact category-name match.
    pub fn with_category_name(mut self, category_name: impl Into<String>) -> Self {
        self.category_name = Some(category_name.into());
        self
    }

    /// Filter categories associated with this scoped provider connector.
    pub fn with_connector_ref(mut self, connector: QwenKnowledgeConnectorRef) -> Self {
        self.connector_ref = Some(connector);
        self
    }

    /// Continue a page using the opaque `nextToken` returned by the provider.
    pub fn with_next_token(mut self, next_token: impl Into<String>) -> Self {
        self.next_token = Some(next_token.into());
        self
    }

    /// Set the provider page size. The documented default is 20.
    pub fn with_max_result(mut self, max_result: u32) -> Self {
        self.max_result = Some(max_result);
        self
    }

    pub fn parent_category(&self) -> Option<&QwenKnowledgeCategoryRef> {
        self.parent_category.as_ref()
    }

    pub fn category_name(&self) -> Option<&str> {
        self.category_name.as_deref()
    }

    pub fn connector_ref(&self) -> Option<&QwenKnowledgeConnectorRef> {
        self.connector_ref.as_ref()
    }

    pub fn next_token(&self) -> Option<&str> {
        self.next_token.as_deref()
    }

    pub fn max_result(&self) -> Option<u32> {
        self.max_result
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        if self
            .next_token
            .as_ref()
            .is_some_and(|token| token.trim().is_empty() || token.chars().any(char::is_control))
        {
            return Err(invalid("next_token must be nonempty when specified"));
        }
        Ok(())
    }

    fn to_body(&self) -> Value {
        let mut body = json!({"type":"UNSTRUCTURED"});
        if let Some(parent) = &self.parent_category {
            body["parentId"] = json!(parent.category_id());
        }
        if let Some(value) = &self.category_name {
            body["categoryName"] = json!(value);
        }
        if let Some(reference) = &self.connector_ref {
            body["connectorId"] = json!(reference.connector_id());
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

impl Default for QwenKnowledgeCategoryListRequest {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for QwenKnowledgeCategoryListRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenKnowledgeCategoryListRequest")
            .field("parent_category", &self.parent_category)
            .field("category_name", &self.category_name)
            .field("connector_ref", &self.connector_ref)
            .field(
                "next_token",
                &self.next_token.as_ref().map(|_| "<redacted>"),
            )
            .field("max_result", &self.max_result)
            .finish()
    }
}

/// A category returned by `listCategory`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeCategory {
    pub reference: QwenKnowledgeCategoryRef,
    pub category_name: String,
    pub category_type: QwenKnowledgeCategoryType,
    pub is_default: bool,
    /// Complete category row, including provider fields not projected above.
    pub native: Value,
}

/// One page from the data-center `listCategory` endpoint.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeCategoryPage {
    pub categories: Vec<QwenKnowledgeCategory>,
    pub has_next: bool,
    /// Opaque token for the next page, when the provider reports one.
    pub next_token: Option<String>,
    pub max_result: Option<u64>,
    pub total_count: Option<u64>,
    pub max_id: Option<u64>,
    pub request_id: Option<String>,
    pub native: Value,
}

impl fmt::Debug for QwenKnowledgeCategoryPage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("QwenKnowledgeCategoryPage")
            .field("categories", &self.categories)
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

/// Request to create one unstructured data-center category.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeCategoryCreateRequest {
    category_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    parent_category: Option<QwenKnowledgeCategoryRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    connector_ref: Option<QwenKnowledgeConnectorRef>,
}

impl QwenKnowledgeCategoryCreateRequest {
    pub fn new(category_name: impl Into<String>) -> Self {
        Self {
            category_name: category_name.into(),
            parent_category: None,
            connector_ref: None,
        }
    }

    /// Create a child category under this scoped parent category.
    pub fn with_parent_category(mut self, parent: QwenKnowledgeCategoryRef) -> Self {
        self.parent_category = Some(parent);
        self
    }

    /// Associate the category with this scoped provider connector.
    pub fn with_connector_ref(mut self, connector: QwenKnowledgeConnectorRef) -> Self {
        self.connector_ref = Some(connector);
        self
    }

    pub fn category_name(&self) -> &str {
        &self.category_name
    }

    pub fn parent_category(&self) -> Option<&QwenKnowledgeCategoryRef> {
        self.parent_category.as_ref()
    }

    pub fn connector_ref(&self) -> Option<&QwenKnowledgeConnectorRef> {
        self.connector_ref.as_ref()
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        let length = self.category_name.chars().count();
        if !(1..=20).contains(&length) {
            return Err(invalid(
                "category_name must contain between 1 and 20 characters",
            ));
        }
        Ok(())
    }

    fn to_body(&self) -> Value {
        let mut body = json!({
            "categoryName": self.category_name,
            "categoryType": "UNSTRUCTURED",
        });
        if let Some(parent) = &self.parent_category {
            body["parentCategoryId"] = json!(parent.category_id());
        }
        if let Some(reference) = &self.connector_ref {
            body["connectorId"] = json!(reference.connector_id());
        }
        body
    }
}

/// Result of successfully creating one category.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeCategoryCreateResult {
    pub reference: QwenKnowledgeCategoryRef,
    /// The returned name, when included by the service.
    pub category_name: Option<String>,
    pub request_id: Option<String>,
    pub native: Value,
}

/// Result of successfully deleting one category.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeCategoryDeleteResult {
    pub reference: QwenKnowledgeCategoryRef,
    pub request_id: Option<String>,
    pub native: Value,
}

impl<'a> QwenKnowledgeService<'a> {
    /// List one page of unstructured categories. Pagination is caller-managed.
    pub async fn list_categories(
        &self,
        request: &QwenKnowledgeCategoryListRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeCategoryPage, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        request.validate()?;
        if let Some(parent) = &request.parent_category {
            pinned_service.validate_category_ref(parent)?;
        }
        if let Some(connector) = &request.connector_ref {
            pinned_service.validate_connector_ref(connector)?;
        }
        let envelope = pinned_service
            .file_json_request(
                LIST_CATEGORIES_PATH,
                request.to_body(),
                "list_categories",
                false,
                false,
                request_options,
            )
            .await?;
        decode_category_page(&pinned_service, request, envelope)
    }

    /// Create an unstructured data-center category.
    pub async fn create_category(
        &self,
        request: &QwenKnowledgeCategoryCreateRequest,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeCategoryCreateResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        request.validate()?;
        if let Some(parent) = &request.parent_category {
            pinned_service.validate_category_ref(parent)?;
        }
        if let Some(connector) = &request.connector_ref {
            pinned_service.validate_connector_ref(connector)?;
        }
        let envelope = pinned_service
            .file_json_request(
                CREATE_CATEGORY_PATH,
                request.to_body(),
                "create_category",
                true,
                false,
                request_options,
            )
            .await?;
        decode_created_category(&pinned_service, envelope)
    }

    /// Permanently delete a category. Its data-center files become uncategorized.
    pub async fn delete_category(
        &self,
        category: &QwenKnowledgeCategoryRef,
        request_options: &crate::RequestOptions,
    ) -> Result<QwenKnowledgeCategoryDeleteResult, QwenKnowledgeError> {
        let pinned_service = self.pin()?;
        pinned_service.validate_category_ref(category)?;
        let envelope = pinned_service
            .file_json_request(
                DELETE_CATEGORY_PATH,
                json!({"categoryId": category.category_id()}),
                "delete_category",
                true,
                false,
                request_options,
            )
            .await?;
        decode_deleted_category(category, envelope)
    }

    fn validate_category_ref(
        &self,
        category: &QwenKnowledgeCategoryRef,
    ) -> Result<(), QwenKnowledgeError> {
        if category.scope != self.scope
            || category.endpoint_fingerprint != self.scope.endpoint_fingerprint()
            || category.category_type != QwenKnowledgeCategoryType::Unstructured
            || !valid_resource_id(&category.category_id)
        {
            return Err(crate::protocol::LlmError::PermissionDenied {
                message: "Qwen category reference belongs to another profile, account, region, workspace, endpoint, or category namespace".into(),
            }
            .into());
        }
        Ok(())
    }
}

fn decode_category_page(
    service: &QwenKnowledgeService<'_>,
    request: &QwenKnowledgeCategoryListRequest,
    envelope: FileEnvelope,
) -> Result<QwenKnowledgeCategoryPage, QwenKnowledgeError> {
    let fail = |message: &str| {
        invalid_file_response(
            "list_categories",
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
        .get("categoryList")
        .and_then(Value::as_array)
        .ok_or_else(|| fail("successful response omitted data.categoryList"))?;
    let mut seen = HashSet::with_capacity(rows.len());
    let mut categories = Vec::with_capacity(rows.len());
    for row in rows {
        let Some(object) = row.as_object() else {
            return Err(fail("categoryList items must be objects"));
        };
        let category_id = object
            .get("categoryId")
            .and_then(Value::as_str)
            .filter(|value| valid_resource_id(value))
            .ok_or_else(|| fail("categoryList item omitted a valid categoryId"))?;
        if !seen.insert(category_id) {
            return Err(fail("categoryList contained a duplicate categoryId"));
        }
        let category_name = object
            .get("categoryName")
            .and_then(Value::as_str)
            .ok_or_else(|| fail("categoryList item omitted string categoryName"))?
            .to_owned();
        let category_type = object
            .get("type")
            .and_then(Value::as_str)
            .ok_or_else(|| fail("categoryList item omitted string type"))?;
        if category_type != "UNSTRUCTURED" {
            return Err(fail("categoryList returned an unsupported category type"));
        }
        let is_default = object
            .get("isDefault")
            .and_then(Value::as_bool)
            .ok_or_else(|| fail("categoryList item omitted boolean isDefault"))?;
        let reference = QwenKnowledgeCategoryRef::from_scope(service.scope(), category_id)
            .map_err(|_| fail("categoryList item contained an invalid categoryId"))?;
        categories.push(QwenKnowledgeCategory {
            reference,
            category_name,
            category_type: QwenKnowledgeCategoryType::Unstructured,
            is_default,
            native: row.clone(),
        });
    }
    Ok(QwenKnowledgeCategoryPage {
        categories,
        has_next,
        next_token: next_token.map(str::to_owned),
        max_result: optional_u64_field(data, "maxResult").map_err(fail)?,
        total_count: optional_u64_field(data, "totalCount").map_err(fail)?,
        max_id: optional_u64_field(data, "maxId").map_err(fail)?,
        request_id: envelope.request_id,
        native: envelope.native,
    })
}

fn decode_created_category(
    service: &QwenKnowledgeService<'_>,
    envelope: FileEnvelope,
) -> Result<QwenKnowledgeCategoryCreateResult, QwenKnowledgeError> {
    let fail = |message: &str| {
        response_outcome_unknown(
            "create_category",
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
    let category_id = data
        .get("categoryId")
        .and_then(Value::as_str)
        .filter(|value| valid_resource_id(value))
        .ok_or_else(|| fail("successful response omitted a valid data.categoryId"))?;
    let category_name = optional_string_field(data, "categoryName").map_err(fail)?;
    let reference = QwenKnowledgeCategoryRef::from_scope(service.scope(), category_id)
        .map_err(|_| fail("successful response contained an invalid category ID"))?;
    Ok(QwenKnowledgeCategoryCreateResult {
        reference,
        category_name,
        request_id: envelope.request_id,
        native: envelope.native,
    })
}

fn decode_deleted_category(
    category: &QwenKnowledgeCategoryRef,
    envelope: FileEnvelope,
) -> Result<QwenKnowledgeCategoryDeleteResult, QwenKnowledgeError> {
    let fail = |message: &str| {
        response_outcome_unknown(
            "delete_category",
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
    let returned_id = data
        .get("categoryId")
        .and_then(Value::as_str)
        .filter(|value| valid_resource_id(value))
        .ok_or_else(|| fail("successful response omitted a valid data.categoryId"))?;
    if returned_id != category.category_id() {
        return Err(fail(
            "successful response confirmed a different category ID",
        ));
    }
    Ok(QwenKnowledgeCategoryDeleteResult {
        reference: category.clone(),
        request_id: envelope.request_id,
        native: envelope.native,
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
