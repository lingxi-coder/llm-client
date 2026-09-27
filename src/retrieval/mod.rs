//! Independent managed retrieval routes and scoped vector-store resources.

use crate::{
    client::{ClientSnapshot, ClientSource, RequestOptions},
    files::{provider_file_endpoint_fingerprint, ProviderFileRef},
    protocol::{
        LlmError, ProtocolFamily, ProviderId, ProviderProfile, ServiceAuth, ServiceSetting,
    },
    runtime::Deadline,
    transport::{HttpExecutor, HttpRequest},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

const MAX_BODY: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetrievalApi {
    OpenAi,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalRoute {
    pub api: RetrievalApi,
    /// Complete vector-stores collection URL, e.g. `/v1/vector_stores`.
    pub endpoint: String,
    pub auth: ServiceAuth,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalStoreRef {
    pub provider_id: ProviderId,
    pub profile_name: String,
    pub endpoint_fingerprint: String,
    pub account_scope: String,
    pub store_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalFileRef {
    pub store: RetrievalStoreRef,
    pub file_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndexStatus {
    InProgress,
    Completed,
    Failed,
    Cancelled,
    Other(String),
}
impl IndexStatus {
    pub fn is_ready(&self) -> bool {
        *self == Self::Completed
    }
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalStore {
    pub reference: RetrievalStoreRef,
    pub name: Option<String>,
    pub status: Option<String>,
    pub file_counts: Option<Value>,
    pub native: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexTask {
    pub reference: RetrievalFileRef,
    pub status: IndexStatus,
    pub last_error: Option<Value>,
    pub native: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FileChunking {
    Auto,
    Static {
        max_chunk_size_tokens: u16,
        chunk_overlap_tokens: u16,
    },
}

impl FileChunking {
    fn validate(self) -> Result<(), LlmError> {
        if let Self::Static {
            max_chunk_size_tokens,
            chunk_overlap_tokens,
        } = self
        {
            if !(100..=4096).contains(&max_chunk_size_tokens)
                || chunk_overlap_tokens > max_chunk_size_tokens / 2
            {
                return Err(invalid("retrieval static chunk size or overlap is invalid"));
            }
        }
        Ok(())
    }

    fn wire(self) -> Value {
        match self {
            Self::Auto => json!({"type":"auto"}),
            Self::Static {
                max_chunk_size_tokens,
                chunk_overlap_tokens,
            } => json!({
                "type":"static",
                "static":{
                    "max_chunk_size_tokens":max_chunk_size_tokens,
                    "chunk_overlap_tokens":chunk_overlap_tokens
                }
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BatchFileInput {
    pub file: ProviderFileRef,
    pub attributes: Option<BTreeMap<String, Value>>,
    pub chunking: Option<FileChunking>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexBatchRef {
    pub store: RetrievalStoreRef,
    pub batch_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexFileCounts {
    pub in_progress: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub total: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexBatchTask {
    pub reference: IndexBatchRef,
    pub status: IndexStatus,
    pub file_counts: IndexFileCounts,
    pub native: Value,
}

impl IndexBatchTask {
    /// A completed batch can still contain failed files; inspect both states.
    pub fn all_files_ready(&self) -> bool {
        self.status == IndexStatus::Completed
            && self.file_counts.failed == 0
            && self.file_counts.cancelled == 0
            && self.file_counts.in_progress == 0
            && self.file_counts.completed == self.file_counts.total
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexFilePage {
    pub files: Vec<IndexTask>,
    pub has_more: bool,
    pub last_id: Option<String>,
}

/// Parameters for one page of files attached to an OpenAI vector store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexFileListRequest {
    pub limit: u8,
    pub after: Option<String>,
    pub before: Option<String>,
    pub status: Option<IndexStatus>,
    pub order: Option<IndexFileOrder>,
}

impl Default for IndexFileListRequest {
    fn default() -> Self {
        Self {
            limit: 20,
            after: None,
            before: None,
            status: None,
            order: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexFileOrder {
    Asc,
    Desc,
}

impl IndexFileListRequest {
    fn validate(&self) -> Result<(), RetrievalError> {
        if !(1..=100).contains(&self.limit)
            || self.after.as_deref().is_some_and(|id| !valid_id(id))
            || self.before.as_deref().is_some_and(|id| !valid_id(id))
            || (self.after.is_some() && self.before.is_some())
            || matches!(self.status, Some(IndexStatus::Other(_)))
        {
            return Err(invalid("retrieval file list options are invalid").into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalHit {
    pub file: RetrievalFileRef,
    pub filename: Option<String>,
    pub score: Option<f64>,
    pub attributes: Value,
    /// Provider-native chunks, including types this version does not model.
    pub content: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalSearchResult {
    pub search_query: Value,
    pub hits: Vec<RetrievalHit>,
    pub has_more: bool,
    pub next_page: Option<String>,
    pub request_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RetrievalFilter {
    Eq { key: String, value: Value },
    Ne { key: String, value: Value },
    Gt { key: String, value: Value },
    Gte { key: String, value: Value },
    Lt { key: String, value: Value },
    Lte { key: String, value: Value },
    In { key: String, value: Value },
    Nin { key: String, value: Value },
    And { filters: Vec<RetrievalFilter> },
    Or { filters: Vec<RetrievalFilter> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchRanker {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "default-2024-11-15")]
    Default20241115,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub struct SearchRanking {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ranker: Option<SearchRanker>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score_threshold: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchRequest {
    pub query: String,
    pub max_results: u8,
    pub filters: Option<RetrievalFilter>,
    pub ranking: Option<SearchRanking>,
    pub rewrite_query: Option<bool>,
}

impl SearchRequest {
    pub fn new(query: impl Into<String>, max_results: u8) -> Self {
        Self {
            query: query.into(),
            max_results,
            filters: None,
            ranking: None,
            rewrite_query: None,
        }
    }

    fn validate(&self) -> Result<(), LlmError> {
        if self.query.trim().is_empty()
            || self.query.len() > 128 * 1024
            || !(1..=50).contains(&self.max_results)
        {
            return Err(invalid("retrieval query or max_results is invalid"));
        }
        if let Some(filter) = &self.filters {
            validate_filter(filter, 0)?;
        }
        if let Some(ranking) = &self.ranking {
            if ranking
                .score_threshold
                .is_some_and(|score| !score.is_finite() || !(0.0..=1.0).contains(&score))
            {
                return Err(invalid("retrieval score_threshold must be between 0 and 1"));
            }
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RetrievalError {
    #[error(transparent)]
    Llm(#[from] LlmError),
    #[error("invalid retrieval response: {0}")]
    InvalidResponse(String),
    #[error("retrieval provider returned HTTP {status}")]
    Provider {
        status: u16,
        request_id: Option<String>,
        body: Value,
    },
    /// A state-changing request may have reached the provider. Reconcile its
    /// resource state before deciding whether to submit another request.
    #[error("outcome of retrieval {operation} is unknown: {source}")]
    OutcomeUnknown {
        operation: &'static str,
        #[source]
        source: LlmError,
    },
}

/// A service handle whose live configuration is captured once per operation.
#[derive(Clone, Copy)]
pub struct RetrievalService<'a> {
    source: ClientSource<'a>,
}

impl<'a> RetrievalService<'a> {
    pub(crate) fn new(source: ClientSource<'a>) -> Self {
        Self { source }
    }

    pub async fn create_store(
        self,
        profile_name: &str,
        name: &str,
        options: &RequestOptions,
    ) -> Result<RetrievalStore, RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .create_store(profile_name, name, options)
            .await
    }

    pub async fn list_stores(
        self,
        profile_name: &str,
        limit: u8,
        after: Option<&str>,
        options: &RequestOptions,
    ) -> Result<(Vec<RetrievalStore>, bool, Option<String>), RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .list_stores(profile_name, limit, after, options)
            .await
    }

    pub async fn get_store(
        self,
        reference: &RetrievalStoreRef,
        options: &RequestOptions,
    ) -> Result<RetrievalStore, RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .get_store(reference, options)
            .await
    }

    pub async fn delete_store(
        self,
        reference: &RetrievalStoreRef,
        options: &RequestOptions,
    ) -> Result<(), RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .delete_store(reference, options)
            .await
    }

    pub async fn attach_file(
        self,
        store: &RetrievalStoreRef,
        file: &ProviderFileRef,
        options: &RequestOptions,
    ) -> Result<IndexTask, RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .attach_file(store, file, options)
            .await
    }

    pub async fn attach_file_with_attributes(
        self,
        store: &RetrievalStoreRef,
        file: &ProviderFileRef,
        attributes: Option<&BTreeMap<String, Value>>,
        options: &RequestOptions,
    ) -> Result<IndexTask, RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .attach_file_with_attributes(store, file, attributes, options)
            .await
    }

    pub async fn attach_file_with_options(
        self,
        store: &RetrievalStoreRef,
        file: &ProviderFileRef,
        attributes: Option<&BTreeMap<String, Value>>,
        chunking: Option<FileChunking>,
        options: &RequestOptions,
    ) -> Result<IndexTask, RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .attach_file_with_options(store, file, attributes, chunking, options)
            .await
    }

    pub async fn update_file_attributes(
        self,
        reference: &RetrievalFileRef,
        attributes: &BTreeMap<String, Value>,
        options: &RequestOptions,
    ) -> Result<IndexTask, RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .update_file_attributes(reference, attributes, options)
            .await
    }

    pub async fn get_file(
        self,
        reference: &RetrievalFileRef,
        options: &RequestOptions,
    ) -> Result<IndexTask, RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .get_file(reference, options)
            .await
    }

    pub async fn delete_file(
        self,
        reference: &RetrievalFileRef,
        options: &RequestOptions,
    ) -> Result<(), RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .delete_file(reference, options)
            .await
    }

    pub async fn list_store_files(
        self,
        store: &RetrievalStoreRef,
        limit: u8,
        after: Option<&str>,
        options: &RequestOptions,
    ) -> Result<IndexFilePage, RetrievalError> {
        self.list_store_files_with(
            store,
            &IndexFileListRequest {
                limit,
                after: after.map(str::to_owned),
                ..Default::default()
            },
            options,
        )
        .await
    }

    pub async fn list_store_files_with(
        self,
        store: &RetrievalStoreRef,
        request: &IndexFileListRequest,
        options: &RequestOptions,
    ) -> Result<IndexFilePage, RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .list_store_files_with(store, request, options)
            .await
    }

    /// Submit up to 2000 already-uploaded files for asynchronous indexing.
    /// A transport failure can leave an accepted batch; callers must reconcile.
    pub async fn create_file_batch(
        self,
        store: &RetrievalStoreRef,
        files: &[BatchFileInput],
        options: &RequestOptions,
    ) -> Result<IndexBatchTask, RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .create_file_batch(store, files, options)
            .await
    }

    pub async fn get_file_batch(
        self,
        reference: &IndexBatchRef,
        options: &RequestOptions,
    ) -> Result<IndexBatchTask, RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .get_file_batch(reference, options)
            .await
    }

    pub async fn cancel_file_batch(
        self,
        reference: &IndexBatchRef,
        options: &RequestOptions,
    ) -> Result<IndexBatchTask, RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .cancel_file_batch(reference, options)
            .await
    }

    pub async fn list_batch_files(
        self,
        reference: &IndexBatchRef,
        limit: u8,
        after: Option<&str>,
        options: &RequestOptions,
    ) -> Result<IndexFilePage, RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .list_batch_files(reference, limit, after, options)
            .await
    }

    pub async fn search(
        self,
        store: &RetrievalStoreRef,
        query: &str,
        max_results: u8,
        options: &RequestOptions,
    ) -> Result<RetrievalSearchResult, RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .search(store, query, max_results, options)
            .await
    }

    pub async fn search_with(
        self,
        store: &RetrievalStoreRef,
        request: &SearchRequest,
        options: &RequestOptions,
    ) -> Result<RetrievalSearchResult, RetrievalError> {
        let snapshot = self.source.snapshot();
        PinnedRetrievalService { client: &snapshot }
            .search_with(store, request, options)
            .await
    }
}

#[derive(Clone, Copy)]
struct PinnedRetrievalService<'a> {
    client: &'a ClientSnapshot,
}

impl<'a> PinnedRetrievalService<'a> {
    fn resolve<'s, 'o>(
        &'s self,
        profile_name: &str,
        options: &'o RequestOptions,
    ) -> Result<(&'s ProviderProfile, &'s RetrievalRoute, &'o str), RetrievalError> {
        let profile = self
            .client
            .provider(profile_name)
            .ok_or_else(|| invalid("unknown retrieval profile"))?;
        if !profile.supports_region(self.client.region()) {
            return Err(invalid("retrieval profile is unavailable in this region").into());
        }
        let ServiceSetting::Enabled(route) = &profile.retrieval else {
            return Err(LlmError::UnsupportedCapability {
                message: "profile has no enabled retrieval route".into(),
            }
            .into());
        };
        let scope = options
            .account_scope
            .as_deref()
            .filter(|scope| !scope.trim().is_empty())
            .ok_or_else(|| invalid("retrieval requires a non-secret account_scope"))?;
        Ok((profile, route, scope))
    }

    fn checked<'s>(
        &'s self,
        reference: &RetrievalStoreRef,
        options: &RequestOptions,
    ) -> Result<(&'s ProviderProfile, &'s RetrievalRoute), RetrievalError> {
        let (profile, route, scope) = self.resolve(&reference.profile_name, options)?;
        if reference.provider_id != profile.provider_id
            || reference.profile_name != profile.profile_name
            || reference.endpoint_fingerprint != provider_file_endpoint_fingerprint(&route.endpoint)
            || reference.account_scope != scope
            || !valid_id(&reference.store_id)
        {
            return Err(LlmError::PermissionDenied {
                message:
                    "retrieval store belongs to another provider, profile, endpoint or account"
                        .into(),
            }
            .into());
        }
        Ok((profile, route))
    }

    async fn create_store(
        self,
        profile_name: &str,
        name: &str,
        options: &RequestOptions,
    ) -> Result<RetrievalStore, RetrievalError> {
        let (profile, route, scope) = self.resolve(profile_name, options)?;
        if name.trim().is_empty() || name.len() > 256 {
            return Err(invalid("retrieval store name must contain 1–256 bytes").into());
        }
        let body = json!({"name":name});
        let (value, _) = self
            .send(route, options, "POST", &[], Some(body), "create_store")
            .await?;
        decode_store(value, profile, route, scope, None)
    }

    async fn list_stores(
        self,
        profile_name: &str,
        limit: u8,
        after: Option<&str>,
        options: &RequestOptions,
    ) -> Result<(Vec<RetrievalStore>, bool, Option<String>), RetrievalError> {
        let (profile, route, scope) = self.resolve(profile_name, options)?;
        if !(1..=100).contains(&limit) || after.is_some_and(|id| !valid_id(id)) {
            return Err(invalid("retrieval list limit or cursor is invalid").into());
        }
        let mut url = route_url(route, &[])?;
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("limit", &limit.to_string());
            if let Some(after) = after {
                query.append_pair("after", after);
            }
        }
        let (value, _) = self
            .send_url(route, options, "GET", url, None, "list_stores")
            .await?;
        let rows = value["data"]
            .as_array()
            .ok_or_else(|| bad("store list has no data array"))?;
        let stores = rows
            .iter()
            .map(|row| decode_store(row.clone(), profile, route, scope, None))
            .collect::<Result<Vec<_>, _>>()?;
        let has_more = value["has_more"]
            .as_bool()
            .ok_or_else(|| bad("store list has no has_more boolean"))?;
        let last_id = value["last_id"].as_str().map(str::to_owned);
        if has_more && last_id.as_deref().is_none_or(|id| !valid_id(id)) {
            return Err(bad("store list has no valid next cursor"));
        }
        Ok((stores, has_more, last_id))
    }

    async fn get_store(
        self,
        reference: &RetrievalStoreRef,
        options: &RequestOptions,
    ) -> Result<RetrievalStore, RetrievalError> {
        let (profile, route) = self.checked(reference, options)?;
        let (value, _) = self
            .send(
                route,
                options,
                "GET",
                &[&reference.store_id],
                None,
                "get_store",
            )
            .await?;
        decode_store(
            value,
            profile,
            route,
            &reference.account_scope,
            Some(&reference.store_id),
        )
    }

    async fn delete_store(
        self,
        reference: &RetrievalStoreRef,
        options: &RequestOptions,
    ) -> Result<(), RetrievalError> {
        let (_, route) = self.checked(reference, options)?;
        let (value, _) = self
            .send(
                route,
                options,
                "DELETE",
                &[&reference.store_id],
                None,
                "delete_store",
            )
            .await?;
        check_deleted(&value, &reference.store_id)
    }

    async fn attach_file(
        self,
        store: &RetrievalStoreRef,
        file: &ProviderFileRef,
        options: &RequestOptions,
    ) -> Result<IndexTask, RetrievalError> {
        self.attach_file_with_attributes(store, file, None, options)
            .await
    }

    async fn attach_file_with_attributes(
        self,
        store: &RetrievalStoreRef,
        file: &ProviderFileRef,
        attributes: Option<&BTreeMap<String, Value>>,
        options: &RequestOptions,
    ) -> Result<IndexTask, RetrievalError> {
        self.attach_file_with_options(store, file, attributes, None, options)
            .await
    }

    async fn attach_file_with_options(
        self,
        store: &RetrievalStoreRef,
        file: &ProviderFileRef,
        attributes: Option<&BTreeMap<String, Value>>,
        chunking: Option<FileChunking>,
        options: &RequestOptions,
    ) -> Result<IndexTask, RetrievalError> {
        let (profile, route) = self.checked(store, options)?;
        check_uploaded_file(profile, store, file)?;
        validate_attributes(attributes)?;
        if let Some(chunking) = chunking {
            chunking.validate()?;
        }
        let mut body = json!({"file_id":file.file_id});
        if let Some(attributes) = attributes {
            body["attributes"] = serde_json::to_value(attributes)
                .map_err(|_| invalid("retrieval file attributes cannot be serialized"))?;
        }
        if let Some(chunking) = chunking {
            body["chunking_strategy"] = chunking.wire();
        }
        let (value, _) = self
            .send(
                route,
                options,
                "POST",
                &[&store.store_id, "files"],
                Some(body),
                "attach_file",
            )
            .await?;
        decode_task(value, store, Some(&file.file_id))
    }

    async fn update_file_attributes(
        self,
        reference: &RetrievalFileRef,
        attributes: &BTreeMap<String, Value>,
        options: &RequestOptions,
    ) -> Result<IndexTask, RetrievalError> {
        let (_, route) = self.checked(&reference.store, options)?;
        if !valid_id(&reference.file_id) {
            return Err(invalid("invalid retrieval file ID").into());
        }
        validate_attributes(Some(attributes))?;
        let (value, _) = self
            .send(
                route,
                options,
                "POST",
                &[&reference.store.store_id, "files", &reference.file_id],
                Some(json!({"attributes":attributes})),
                "update_file_attributes",
            )
            .await?;
        decode_task(value, &reference.store, Some(&reference.file_id))
    }

    async fn get_file(
        self,
        reference: &RetrievalFileRef,
        options: &RequestOptions,
    ) -> Result<IndexTask, RetrievalError> {
        let (_, route) = self.checked(&reference.store, options)?;
        if !valid_id(&reference.file_id) {
            return Err(invalid("invalid retrieval file ID").into());
        }
        let (value, _) = self
            .send(
                route,
                options,
                "GET",
                &[&reference.store.store_id, "files", &reference.file_id],
                None,
                "get_file",
            )
            .await?;
        decode_task(value, &reference.store, Some(&reference.file_id))
    }

    async fn delete_file(
        self,
        reference: &RetrievalFileRef,
        options: &RequestOptions,
    ) -> Result<(), RetrievalError> {
        let (_, route) = self.checked(&reference.store, options)?;
        if !valid_id(&reference.file_id) {
            return Err(invalid("invalid retrieval file ID").into());
        }
        let (value, _) = self
            .send(
                route,
                options,
                "DELETE",
                &[&reference.store.store_id, "files", &reference.file_id],
                None,
                "delete_file",
            )
            .await?;
        check_deleted(&value, &reference.file_id)
    }

    async fn list_store_files_with(
        self,
        store: &RetrievalStoreRef,
        request: &IndexFileListRequest,
        options: &RequestOptions,
    ) -> Result<IndexFilePage, RetrievalError> {
        let (_, route) = self.checked(store, options)?;
        request.validate()?;
        let mut url = route_url(route, &[&store.store_id, "files"])?;
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("limit", &request.limit.to_string());
            if let Some(after) = &request.after {
                query.append_pair("after", after);
            }
            if let Some(before) = &request.before {
                query.append_pair("before", before);
            }
            if let Some(status) = &request.status {
                let value = match status {
                    IndexStatus::InProgress => "in_progress",
                    IndexStatus::Completed => "completed",
                    IndexStatus::Failed => "failed",
                    IndexStatus::Cancelled => "cancelled",
                    IndexStatus::Other(_) => unreachable!("validated above"),
                };
                query.append_pair("filter", value);
            }
            if let Some(order) = request.order {
                query.append_pair(
                    "order",
                    match order {
                        IndexFileOrder::Asc => "asc",
                        IndexFileOrder::Desc => "desc",
                    },
                );
            }
        }
        let (value, _) = self
            .send_url(route, options, "GET", url, None, "list_store_files")
            .await?;
        decode_file_page(value, store)
    }

    /// Submit up to 2000 already-uploaded files for asynchronous indexing.
    /// A transport failure can leave an accepted batch; callers must reconcile.
    async fn create_file_batch(
        self,
        store: &RetrievalStoreRef,
        files: &[BatchFileInput],
        options: &RequestOptions,
    ) -> Result<IndexBatchTask, RetrievalError> {
        let (profile, route) = self.checked(store, options)?;
        if files.is_empty() || files.len() > 2000 {
            return Err(invalid("retrieval batch requires 1–2000 files").into());
        }
        let mut seen = BTreeSet::new();
        let mut rows = Vec::with_capacity(files.len());
        for entry in files {
            check_uploaded_file(profile, store, &entry.file)?;
            if !seen.insert(entry.file.file_id.as_str()) {
                return Err(invalid("retrieval batch has duplicate file IDs").into());
            }
            validate_attributes(entry.attributes.as_ref())?;
            if let Some(chunking) = entry.chunking {
                chunking.validate()?;
            }
            let mut row = json!({"file_id":entry.file.file_id});
            if let Some(attributes) = &entry.attributes {
                row["attributes"] = json!(attributes);
            }
            if let Some(chunking) = entry.chunking {
                row["chunking_strategy"] = chunking.wire();
            }
            rows.push(row);
        }
        let (value, _) = self
            .send(
                route,
                options,
                "POST",
                &[&store.store_id, "file_batches"],
                Some(json!({"files":rows})),
                "create_file_batch",
            )
            .await?;
        decode_batch(value, store, None)
    }

    async fn get_file_batch(
        self,
        reference: &IndexBatchRef,
        options: &RequestOptions,
    ) -> Result<IndexBatchTask, RetrievalError> {
        let (_, route) = self.checked_batch(reference, options)?;
        let (value, _) = self
            .send(
                route,
                options,
                "GET",
                &[
                    &reference.store.store_id,
                    "file_batches",
                    &reference.batch_id,
                ],
                None,
                "get_file_batch",
            )
            .await?;
        decode_batch(value, &reference.store, Some(&reference.batch_id))
    }

    async fn cancel_file_batch(
        self,
        reference: &IndexBatchRef,
        options: &RequestOptions,
    ) -> Result<IndexBatchTask, RetrievalError> {
        let (_, route) = self.checked_batch(reference, options)?;
        let (value, _) = self
            .send(
                route,
                options,
                "POST",
                &[
                    &reference.store.store_id,
                    "file_batches",
                    &reference.batch_id,
                    "cancel",
                ],
                None,
                "cancel_file_batch",
            )
            .await?;
        decode_batch(value, &reference.store, Some(&reference.batch_id))
    }

    async fn list_batch_files(
        self,
        reference: &IndexBatchRef,
        limit: u8,
        after: Option<&str>,
        options: &RequestOptions,
    ) -> Result<IndexFilePage, RetrievalError> {
        let (_, route) = self.checked_batch(reference, options)?;
        if !(1..=100).contains(&limit) || after.is_some_and(|id| !valid_id(id)) {
            return Err(invalid("retrieval batch list limit or cursor is invalid").into());
        }
        let mut url = route_url(
            route,
            &[
                &reference.store.store_id,
                "file_batches",
                &reference.batch_id,
                "files",
            ],
        )?;
        {
            let mut query = url.query_pairs_mut();
            query.append_pair("limit", &limit.to_string());
            if let Some(after) = after {
                query.append_pair("after", after);
            }
        }
        let (value, _) = self
            .send_url(route, options, "GET", url, None, "list_batch_files")
            .await?;
        decode_file_page(value, &reference.store)
    }

    fn checked_batch<'s>(
        &'s self,
        reference: &IndexBatchRef,
        options: &RequestOptions,
    ) -> Result<(&'s ProviderProfile, &'s RetrievalRoute), RetrievalError> {
        let connection = self.checked(&reference.store, options)?;
        if !valid_id(&reference.batch_id) {
            return Err(invalid("invalid retrieval batch ID").into());
        }
        Ok(connection)
    }

    async fn search(
        self,
        store: &RetrievalStoreRef,
        query: &str,
        max_results: u8,
        options: &RequestOptions,
    ) -> Result<RetrievalSearchResult, RetrievalError> {
        self.search_with(store, &SearchRequest::new(query, max_results), options)
            .await
    }

    async fn search_with(
        self,
        store: &RetrievalStoreRef,
        request: &SearchRequest,
        options: &RequestOptions,
    ) -> Result<RetrievalSearchResult, RetrievalError> {
        let (_, route) = self.checked(store, options)?;
        request.validate()?;
        let mut body = json!({"query":request.query,"max_num_results":request.max_results});
        if let Some(filters) = &request.filters {
            body["filters"] = serde_json::to_value(filters)
                .map_err(|_| invalid("retrieval filter cannot be serialized"))?;
        }
        if let Some(ranking) = &request.ranking {
            body["ranking_options"] = serde_json::to_value(ranking)
                .map_err(|_| invalid("retrieval ranking cannot be serialized"))?;
        }
        if let Some(rewrite_query) = request.rewrite_query {
            body["rewrite_query"] = Value::Bool(rewrite_query);
        }
        let (value, request_id) = self
            .send(
                route,
                options,
                "POST",
                &[&store.store_id, "search"],
                Some(body),
                "search",
            )
            .await?;
        let rows = value["data"]
            .as_array()
            .ok_or_else(|| bad("retrieval search has no data array"))?;
        let mut hits = Vec::with_capacity(rows.len());
        for row in rows {
            let file_id = row["file_id"]
                .as_str()
                .filter(|id| valid_id(id))
                .ok_or_else(|| bad("retrieval hit has no valid file ID"))?;
            let score = row
                .get("score")
                .filter(|v| !v.is_null())
                .map(|v| {
                    v.as_f64()
                        .filter(|score| score.is_finite())
                        .ok_or_else(|| bad("retrieval hit has invalid score"))
                })
                .transpose()?;
            let content = row["content"]
                .as_array()
                .ok_or_else(|| bad("retrieval hit has no content array"))?
                .clone();
            hits.push(RetrievalHit {
                file: RetrievalFileRef {
                    store: store.clone(),
                    file_id: file_id.into(),
                },
                filename: row["filename"].as_str().map(str::to_owned),
                score,
                attributes: row.get("attributes").cloned().unwrap_or(Value::Null),
                content,
            });
        }
        let has_more = value["has_more"]
            .as_bool()
            .ok_or_else(|| bad("retrieval search has no has_more boolean"))?;
        let next_page = value["next_page"].as_str().map(str::to_owned);
        if has_more && next_page.as_deref().is_none_or(str::is_empty) {
            return Err(bad("retrieval search has no next page cursor"));
        }
        Ok(RetrievalSearchResult {
            search_query: value.get("search_query").cloned().unwrap_or(Value::Null),
            hits,
            has_more,
            next_page,
            request_id,
        })
    }

    async fn send(
        self,
        route: &RetrievalRoute,
        options: &RequestOptions,
        method: &'static str,
        path: &[&str],
        body: Option<Value>,
        operation: &'static str,
    ) -> Result<(Value, Option<String>), RetrievalError> {
        self.send_url(
            route,
            options,
            method,
            route_url(route, path)?,
            body,
            operation,
        )
        .await
    }

    async fn send_url(
        self,
        route: &RetrievalRoute,
        options: &RequestOptions,
        method: &'static str,
        url: url::Url,
        body: Option<Value>,
        operation: &'static str,
    ) -> Result<(Value, Option<String>), RetrievalError> {
        let bytes = body
            .map(|value| serde_json::to_vec(&value))
            .transpose()
            .map_err(|_| invalid("retrieval request cannot be serialized"))?
            .unwrap_or_default();
        if bytes.len() > MAX_BODY {
            return Err(invalid("retrieval request exceeds 64 MiB").into());
        }
        let mut headers = vec![
            ("content-type".into(), "application/json".into()),
            ("openai-beta".into(), "assistants=v2".into()),
        ];
        match &route.auth {
            ServiceAuth::None => {}
            ServiceAuth::Bearer | ServiceAuth::ApiKey { .. } => {
                let secret =
                    options
                        .credential
                        .as_ref()
                        .ok_or_else(|| LlmError::Authentication {
                            message: "retrieval route requires a credential".into(),
                        })?;
                match &route.auth {
                    ServiceAuth::Bearer => headers.push((
                        "authorization".into(),
                        format!("Bearer {}", secret.expose_secret()),
                    )),
                    ServiceAuth::ApiKey { header } => {
                        headers.push((header.clone(), secret.expose_secret().clone()))
                    }
                    ServiceAuth::None => unreachable!(),
                }
            }
        }
        let deadline = Deadline::after(Some(
            options.total_timeout.unwrap_or(Duration::from_secs(120)),
        ));
        let request = HttpRequest {
            method: method.into(),
            url: url.into(),
            headers,
            body: bytes.into(),
            timeout: deadline.remaining()?,
        };
        let response = HttpExecutor::new(self.client.runtime.http.as_ref())
            .with_deadline(deadline)
            .execute_bounded(request, MAX_BODY)
            .await
            .map_err(|source| {
                if matches!(
                    operation,
                    "create_store"
                        | "attach_file"
                        | "update_file_attributes"
                        | "create_file_batch"
                        | "cancel_file_batch"
                        | "delete_store"
                        | "delete_file"
                ) && matches!(
                    source,
                    LlmError::Transport { .. } | LlmError::TransportTimeout { .. }
                ) {
                    RetrievalError::OutcomeUnknown { operation, source }
                } else {
                    RetrievalError::Llm(source)
                }
            })?;
        let request_id = response
            .header("x-request-id")
            .or_else(|| response.header("request-id"))
            .map(str::to_owned);
        let value = match serde_json::from_slice(&response.body) {
            Ok(value) => value,
            Err(_) if !(200..300).contains(&response.status) => {
                Value::String(String::from_utf8_lossy(&response.body).into_owned())
            }
            Err(_) => return Err(bad("retrieval success body is not JSON")),
        };
        if !(200..300).contains(&response.status) {
            return Err(RetrievalError::Provider {
                status: response.status,
                request_id,
                body: value,
            });
        }
        Ok((value, request_id))
    }
}

fn invalid(message: &str) -> LlmError {
    LlmError::InvalidRequest {
        message: message.into(),
    }
}
fn bad(message: &str) -> RetrievalError {
    RetrievalError::InvalidResponse(message.into())
}
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 256
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
}
fn check_uploaded_file(
    profile: &ProviderProfile,
    store: &RetrievalStoreRef,
    file: &ProviderFileRef,
) -> Result<(), RetrievalError> {
    if !valid_id(&file.file_id)
        || file.provider_id != profile.provider_id
        || file.profile_name != profile.profile_name
        || file.protocol != ProtocolFamily::OpenAiResponses
        || file.endpoint_fingerprint != provider_file_endpoint_fingerprint(&profile.base_url)
        || file.account_scope.as_deref() != Some(store.account_scope.as_str())
    {
        return Err(LlmError::PermissionDenied {
            message: "provider file belongs to another connection or account".into(),
        }
        .into());
    }
    Ok(())
}
fn validate_attributes(attributes: Option<&BTreeMap<String, Value>>) -> Result<(), LlmError> {
    if let Some(attributes) = attributes {
        if attributes.len() > 16
            || !attributes
                .iter()
                .all(|(key, value)| valid_filter_key(key) && valid_filter_scalar(value))
        {
            return Err(invalid("retrieval file attributes are invalid"));
        }
    }
    Ok(())
}
fn validate_filter(filter: &RetrievalFilter, depth: usize) -> Result<(), LlmError> {
    if depth > 8 {
        return Err(invalid("retrieval filter nesting exceeds 8 levels"));
    }
    match filter {
        RetrievalFilter::Eq { key, value }
        | RetrievalFilter::Ne { key, value }
        | RetrievalFilter::Gt { key, value }
        | RetrievalFilter::Gte { key, value }
        | RetrievalFilter::Lt { key, value }
        | RetrievalFilter::Lte { key, value } => {
            if !valid_filter_key(key) || !valid_filter_scalar(value) {
                return Err(invalid("retrieval comparison filter is invalid"));
            }
        }
        RetrievalFilter::In { key, value } | RetrievalFilter::Nin { key, value } => {
            let Some(items) = value.as_array() else {
                return Err(invalid("retrieval set filter requires an array"));
            };
            if !valid_filter_key(key)
                || items.is_empty()
                || items.len() > 100
                || !items
                    .iter()
                    .all(|item| item.is_string() || item.is_number())
            {
                return Err(invalid("retrieval set filter is invalid"));
            }
        }
        RetrievalFilter::And { filters } | RetrievalFilter::Or { filters } => {
            if filters.is_empty() || filters.len() > 100 {
                return Err(invalid("retrieval compound filter is empty or too large"));
            }
            for filter in filters {
                validate_filter(filter, depth + 1)?;
            }
        }
    }
    Ok(())
}
fn valid_filter_key(key: &str) -> bool {
    !key.trim().is_empty() && key.len() <= 64
}
fn valid_filter_scalar(value: &Value) -> bool {
    value.as_str().is_some_and(|s| s.len() <= 512) || value.is_number() || value.is_boolean()
}
fn route_url(route: &RetrievalRoute, path: &[&str]) -> Result<url::Url, LlmError> {
    let mut url = url::Url::parse(&route.endpoint).map_err(|_| invalid("invalid retrieval URL"))?;
    {
        let mut segments = url
            .path_segments_mut()
            .map_err(|_| invalid("invalid retrieval URL"))?;
        for part in path {
            if !valid_id(part) {
                return Err(invalid("invalid retrieval path ID"));
            }
            segments.push(part);
        }
    }
    Ok(url)
}
pub fn validate_route(route: &RetrievalRoute) -> Result<(), LlmError> {
    let url = url::Url::parse(&route.endpoint).map_err(|_| invalid("invalid retrieval route"))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
        || !url.path().ends_with("/vector_stores")
    {
        return Err(invalid("retrieval endpoint must end in /vector_stores and contain no credentials, query or fragment"));
    }
    if let ServiceAuth::ApiKey { header } = &route.auth {
        if header.is_empty()
            || !header
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-')
            || [
                "host",
                "content-type",
                "content-length",
                "connection",
                "transfer-encoding",
                "openai-beta",
            ]
            .iter()
            .any(|reserved| header.eq_ignore_ascii_case(reserved))
        {
            return Err(invalid("invalid retrieval authentication header"));
        }
    }
    Ok(())
}
fn decode_store(
    value: Value,
    profile: &ProviderProfile,
    route: &RetrievalRoute,
    scope: &str,
    expected_id: Option<&str>,
) -> Result<RetrievalStore, RetrievalError> {
    let id = value["id"]
        .as_str()
        .filter(|id| valid_id(id))
        .ok_or_else(|| bad("vector store has no valid ID"))?;
    if expected_id.is_some_and(|expected| expected != id) {
        return Err(bad("vector store response ID does not match request"));
    }
    let reference = RetrievalStoreRef {
        provider_id: profile.provider_id.clone(),
        profile_name: profile.profile_name.clone(),
        endpoint_fingerprint: provider_file_endpoint_fingerprint(&route.endpoint),
        account_scope: scope.into(),
        store_id: id.into(),
    };
    Ok(RetrievalStore {
        reference,
        name: value["name"].as_str().map(str::to_owned),
        status: value["status"].as_str().map(str::to_owned),
        file_counts: value.get("file_counts").cloned(),
        native: value,
    })
}
fn decode_task(
    value: Value,
    store: &RetrievalStoreRef,
    expected_id: Option<&str>,
) -> Result<IndexTask, RetrievalError> {
    let id = value["id"]
        .as_str()
        .filter(|id| valid_id(id))
        .ok_or_else(|| bad("vector store file has no valid ID"))?;
    if expected_id.is_some_and(|expected| expected != id)
        || value["vector_store_id"].as_str() != Some(store.store_id.as_str())
    {
        return Err(bad("vector store file response does not match request"));
    }
    let status = match value["status"]
        .as_str()
        .ok_or_else(|| bad("vector store file has no status"))?
    {
        "in_progress" => IndexStatus::InProgress,
        "completed" => IndexStatus::Completed,
        "failed" => IndexStatus::Failed,
        "cancelled" => IndexStatus::Cancelled,
        other => IndexStatus::Other(other.into()),
    };
    Ok(IndexTask {
        reference: RetrievalFileRef {
            store: store.clone(),
            file_id: id.into(),
        },
        status,
        last_error: value.get("last_error").filter(|v| !v.is_null()).cloned(),
        native: value,
    })
}
fn decode_batch(
    value: Value,
    store: &RetrievalStoreRef,
    expected_id: Option<&str>,
) -> Result<IndexBatchTask, RetrievalError> {
    let id = value["id"]
        .as_str()
        .filter(|id| valid_id(id))
        .ok_or_else(|| bad("vector store file batch has no valid ID"))?;
    if expected_id.is_some_and(|expected| expected != id)
        || value["vector_store_id"].as_str() != Some(store.store_id.as_str())
    {
        return Err(bad(
            "vector store file batch response does not match request",
        ));
    }
    let status = match value["status"]
        .as_str()
        .ok_or_else(|| bad("vector store file batch has no status"))?
    {
        "in_progress" => IndexStatus::InProgress,
        "completed" => IndexStatus::Completed,
        "failed" => IndexStatus::Failed,
        "cancelled" => IndexStatus::Cancelled,
        other => IndexStatus::Other(other.into()),
    };
    let counts = serde_json::from_value::<IndexFileCounts>(value["file_counts"].clone())
        .map_err(|_| bad("vector store file batch has invalid file counts"))?;
    Ok(IndexBatchTask {
        reference: IndexBatchRef {
            store: store.clone(),
            batch_id: id.into(),
        },
        status,
        file_counts: counts,
        native: value,
    })
}
fn decode_file_page(
    value: Value,
    store: &RetrievalStoreRef,
) -> Result<IndexFilePage, RetrievalError> {
    let rows = value["data"]
        .as_array()
        .ok_or_else(|| bad("retrieval file list has no data array"))?;
    let files = rows
        .iter()
        .map(|row| decode_task(row.clone(), store, None))
        .collect::<Result<Vec<_>, _>>()?;
    let has_more = value["has_more"]
        .as_bool()
        .ok_or_else(|| bad("retrieval file list has no has_more boolean"))?;
    let last_id = value["last_id"].as_str().map(str::to_owned);
    if has_more && last_id.as_deref().is_none_or(|id| !valid_id(id)) {
        return Err(bad("retrieval file list has no valid next cursor"));
    }
    Ok(IndexFilePage {
        files,
        has_more,
        last_id,
    })
}
fn check_deleted(value: &Value, expected_id: &str) -> Result<(), RetrievalError> {
    if value["id"].as_str() != Some(expected_id) || value["deleted"].as_bool() != Some(true) {
        return Err(bad("provider did not confirm the requested deletion"));
    }
    Ok(())
}
