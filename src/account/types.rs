//! Account query inputs and result data.
use super::*;
/// The kind of principal whose account is queried, independent of the wire's
/// authentication header or the connection's billing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountIdentity {
    ApiKey,
    AuthUser,
}

/// IDs needed to select the precise scope of a provider's account API.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AccountSelector {
    pub organization_id: Option<String>,
    pub project_id: Option<String>,
    pub workspace_id: Option<String>,
    pub team_id: Option<String>,
    pub api_key_id: Option<String>,
}

/// Optional Alibaba Cloud AccessKey credentials for Model Studio's signed
/// billing-trend API. These values belong to one query and are never stored.
#[derive(Debug, Clone)]
pub struct AlibabaAccessKey {
    pub id: String,
    pub secret: Secret<String>,
    pub security_token: Option<Secret<String>>,
}

/// Credentials belong to this query and are never persisted by the client.
#[derive(Debug, Clone)]
pub struct AccountQuery {
    pub execution: AccountExecutionOptions,
    pub identity: AccountIdentity,
    /// The ordinary API key or user token, when a source uses one directly.
    pub credential: Option<Secret<String>>,
    /// A separate Admin/Management credential, when required by a source.
    pub management_credential: Option<Secret<String>>,
    /// Local service bearer token, used by a host-provided Kimi Code server.
    pub service_credential: Option<Secret<String>>,
    /// Host-provided loopback URL for an official local service.
    pub service_url: Option<String>,
    /// Separate Alibaba Cloud RAM AccessKey used only to sign GetBillingTrend.
    pub alibaba_access_key: Option<AlibabaAccessKey>,
    pub selector: AccountSelector,
    /// Inclusive UTC Unix seconds. Defaults to 30 days before the query.
    pub since_unix: Option<u64>,
    /// Exclusive UTC Unix seconds. Defaults to the query time.
    pub until_unix: Option<u64>,
}

impl AccountQuery {
    pub fn new(identity: AccountIdentity) -> Self {
        Self {
            execution: AccountExecutionOptions::default(),
            identity,
            credential: None,
            management_credential: None,
            service_credential: None,
            service_url: None,
            alibaba_access_key: None,
            selector: AccountSelector::default(),
            since_unix: None,
            until_unix: None,
        }
    }

    pub(crate) fn range(&self, now: u64) -> Result<(u64, u64), AccountUsageError> {
        let until = self.until_unix.unwrap_or(now);
        let since = self
            .since_unix
            .unwrap_or_else(|| until.saturating_sub(30 * 86_400));
        if since >= until {
            return Err(AccountUsageError::InvalidTimeRange);
        }
        Ok((since, until))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum AccountUsageError {
    #[error("account execution timeouts must be greater than zero")]
    InvalidExecutionOptions,
    #[error("unknown provider profile {0:?}")]
    UnknownProfile(String),
    #[error("no account query was supplied for provider profile {0:?}")]
    MissingQuery(String),
    #[error("account source for provider profile {0:?} must be registered per profile")]
    AmbiguousAccountSource(String),
    #[error("account usage start must be earlier than end")]
    InvalidTimeRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountScopeKind {
    ApiKey,
    User,
    Project,
    Workspace,
    Organization,
    Team,
    Account,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountScope {
    pub kind: AccountScopeKind,
    pub id: Option<String>,
}

impl AccountScope {
    pub fn new(kind: AccountScopeKind, id: Option<String>) -> Self {
        Self { kind, id }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccountFailure {
    Timeout,
    Transport,
    Unauthorized,
    PermissionDenied,
    RateLimited,
    InvalidResponse,
    ProviderError,
}

/// Per-field availability. A failed balance fetch does not erase valid usage.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum AccountMetric<T> {
    Available {
        scope: AccountScope,
        source: String,
        value: T,
    },
    Unsupported,
    CredentialRequired,
    NotReported,
    Failed {
        reason: AccountFailure,
    },
}

impl<T> AccountMetric<T> {
    pub fn available(scope: AccountScope, source: &str, value: T) -> Self {
        Self::Available {
            scope,
            source: source.to_owned(),
            value,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountBalance {
    /// ISO currency code or the provider's documented credit unit.
    pub unit: String,
    /// Decimal text preserves the provider's monetary precision.
    pub remaining: String,
    pub total: Option<String>,
    /// Provider-reported ability to spend this balance, when exposed.
    pub is_available: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountTokenBucket {
    pub start_unix: u64,
    pub end_unix: u64,
    pub model: Option<String>,
    /// Total input, including cached input when the provider reports it so.
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    /// A subset of `input_tokens`, never an additional total.
    pub cached_input_tokens: Option<u64>,
    /// Cache-creation input, also a subset of `input_tokens` when reported.
    pub cache_write_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountTokenUsage {
    pub lifetime_tokens: Option<u64>,
    pub buckets: Vec<AccountTokenBucket>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountCostBucket {
    pub start_unix: u64,
    pub end_unix: u64,
    pub currency: String,
    /// Exact decimal amount in major currency units (for example USD).
    pub amount: String,
    pub model: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountCostUsage {
    pub buckets: Vec<AccountCostBucket>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccountQuotaWindow {
    pub name: String,
    pub duration_mins: Option<u64>,
    pub used_percent: Option<f64>,
    pub remaining_percent: Option<f64>,
    pub resets_at_unix: Option<u64>,
    pub limit: Option<u64>,
    pub used: Option<u64>,
    pub remaining: Option<u64>,
    /// Exact decimal quantities for fractional or monetary limits.
    pub limit_decimal: Option<String>,
    pub remaining_decimal: Option<String>,
    pub unit: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubscriptionStatus {
    VerifiedActive,
    VerifiedInactive,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccountSubscription {
    pub plan_name: Option<String>,
    pub status: SubscriptionStatus,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AccountSnapshot {
    pub profile_name: String,
    pub provider_id: ProviderId,
    pub identity: AccountIdentity,
    pub fetched_at_unix: u64,
    pub balance: AccountMetric<Vec<AccountBalance>>,
    pub token_usage: AccountMetric<AccountTokenUsage>,
    pub cost_usage: AccountMetric<AccountCostUsage>,
    pub quota_windows: AccountMetric<Vec<AccountQuotaWindow>>,
    pub subscription: AccountMetric<AccountSubscription>,
}

impl AccountSnapshot {
    pub fn unsupported(profile: &ProviderProfile, identity: AccountIdentity, now: u64) -> Self {
        Self {
            profile_name: profile.profile_name.clone(),
            provider_id: profile.provider_id.clone(),
            identity,
            fetched_at_unix: now,
            balance: AccountMetric::Unsupported,
            token_usage: AccountMetric::Unsupported,
            cost_usage: AccountMetric::Unsupported,
            quota_windows: AccountMetric::Unsupported,
            subscription: AccountMetric::Unsupported,
        }
    }
}
