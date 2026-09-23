use async_trait::async_trait;
use lingxi_llm_client::protocol::ProviderProfile;
use lingxi_llm_client::{
    builtin_providers, AccountBalance, AccountIdentity, AccountMetric, AccountQuery, AccountScope,
    AccountScopeKind, AccountSnapshot, AccountUsageError, AccountUsageSource, LlmClientBuilder,
    Transport,
};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Barrier;

static NEXT_ACCOUNT_DIR: AtomicU64 = AtomicU64::new(0);

fn account_config_dir() -> std::path::PathBuf {
    std::env::temp_dir().join(format!(
        "llm-client-account-binding-{}-{}",
        std::process::id(),
        NEXT_ACCOUNT_DIR.fetch_add(1, Ordering::Relaxed)
    ))
}

struct FakeSource;

struct BarrierSource(Arc<Barrier>);

#[async_trait]
impl AccountUsageSource for BarrierSource {
    async fn fetch(
        &self,
        profile: &ProviderProfile,
        query: &AccountQuery,
        _range: (u64, u64),
        now: u64,
        _http: &dyn Transport,
    ) -> AccountSnapshot {
        self.0.wait().await;
        AccountSnapshot::unsupported(profile, query.identity, now)
    }
}

#[async_trait]
impl AccountUsageSource for FakeSource {
    fn requires_profile_binding(&self, _query: &AccountQuery) -> bool {
        true
    }

    async fn fetch(
        &self,
        profile: &ProviderProfile,
        query: &AccountQuery,
        range: (u64, u64),
        now: u64,
        _http: &dyn Transport,
    ) -> AccountSnapshot {
        assert_eq!(range, (100, 200));
        let mut result = AccountSnapshot::unsupported(profile, query.identity, now);
        result.profile_name = "wrong-profile".into();
        result.balance = AccountMetric::available(
            AccountScope::new(AccountScopeKind::Account, None),
            "fake",
            vec![AccountBalance {
                unit: "USD".into(),
                remaining: "12.50".into(),
                total: None,
                is_available: None,
            }],
        );
        result
    }
}

fn client() -> lingxi_llm_client::LlmClient {
    let profiles = builtin_providers()
        .unwrap()
        .into_iter()
        .filter(|p| p.profile_name == "deepseek" || p.profile_name == "openai")
        .collect::<Vec<_>>();
    let mut builder = LlmClientBuilder::new(&profiles).unwrap();
    builder.register_account_source("deepseek", AccountIdentity::AuthUser, Arc::new(FakeSource));
    builder.build().unwrap()
}

#[tokio::test]
async fn explicit_identity_selects_source_and_preserves_profile_identity() {
    let mut query = AccountQuery::new(AccountIdentity::AuthUser);
    query.since_unix = Some(100);
    query.until_unix = Some(200);
    let result = client().account_usage("deepseek", &query).await.unwrap();
    assert_eq!(result.profile_name, "deepseek");
    assert_eq!(result.identity, AccountIdentity::AuthUser);
    assert!(matches!(result.balance, AccountMetric::Available { .. }));

    let key_query = AccountQuery {
        identity: AccountIdentity::ApiKey,
        ..query
    };
    let key_result = client()
        .account_usage("deepseek", &key_query)
        .await
        .unwrap();
    assert!(
        !matches!(key_result.balance, AccountMetric::Available { source, .. } if source == "fake")
    );
}

#[tokio::test]
async fn bulk_query_keeps_missing_and_invalid_profiles_independent() {
    let mut invalid = AccountQuery::new(AccountIdentity::ApiKey);
    invalid.since_unix = Some(200);
    invalid.until_unix = Some(100);
    let queries = BTreeMap::from([("openai".to_owned(), invalid)]);
    let results = client().accounts_usage(&queries).await;
    assert_eq!(results.len(), 2);
    assert!(results.iter().any(|(name, result)| {
        name == "openai" && matches!(result, Err(AccountUsageError::InvalidTimeRange))
    }));
    assert!(results.iter().any(|(name, result)| {
        name == "deepseek"
            && matches!(result, Err(AccountUsageError::MissingQuery(missing)) if missing == "deepseek")
    }));
    assert!(matches!(
        client()
            .account_usage(
                "does-not-exist",
                &AccountQuery::new(AccountIdentity::ApiKey)
            )
            .await,
        Err(AccountUsageError::UnknownProfile(_))
    ));
}

#[tokio::test]
async fn ambiguous_auth_source_requires_profile_binding() {
    let mut first = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "openai")
        .unwrap();
    first.profile_name = "openai-a".into();
    let mut second = first.clone();
    second.profile_name = "openai-b".into();
    let mut builder = LlmClientBuilder::new(&[first, second]).unwrap();
    builder.register_account_source("openai", AccountIdentity::AuthUser, Arc::new(FakeSource));
    let client = builder.build().unwrap();
    let mut query = AccountQuery::new(AccountIdentity::AuthUser);
    query.since_unix = Some(100);
    query.until_unix = Some(200);
    assert!(matches!(
        client.account_usage("openai-a", &query).await,
        Err(AccountUsageError::AmbiguousAccountSource(_))
    ));

    let mut first = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "openai")
        .unwrap();
    first.profile_name = "openai-a".into();
    let mut second = first.clone();
    second.profile_name = "openai-b".into();
    let mut builder = LlmClientBuilder::new(&[first, second]).unwrap();
    builder.register_account_source("openai", AccountIdentity::AuthUser, Arc::new(FakeSource));
    builder.register_profile_account_source(
        "openai-a",
        AccountIdentity::AuthUser,
        Arc::new(FakeSource),
    );
    let client = builder.build().unwrap();
    assert!(matches!(
        client.account_usage("openai-a", &query).await,
        Ok(AccountSnapshot {
            balance: AccountMetric::Available { .. },
            ..
        })
    ));
    assert!(matches!(
        client.account_usage("openai-b", &query).await,
        Err(AccountUsageError::AmbiguousAccountSource(_))
    ));
}

#[tokio::test]
async fn bulk_queries_progress_concurrently_and_keep_profile_order() {
    let profiles = builtin_providers()
        .unwrap()
        .into_iter()
        .filter(|p| p.profile_name == "deepseek" || p.profile_name == "openai")
        .collect::<Vec<_>>();
    let barrier = Arc::new(Barrier::new(2));
    let mut builder = LlmClientBuilder::new(&profiles).unwrap();
    for provider in ["deepseek", "openai"] {
        builder.register_account_source(
            provider,
            AccountIdentity::AuthUser,
            Arc::new(BarrierSource(barrier.clone())),
        );
    }
    let client = builder.build().unwrap();
    let queries = BTreeMap::from([
        (
            "deepseek".into(),
            AccountQuery::new(AccountIdentity::AuthUser),
        ),
        (
            "openai".into(),
            AccountQuery::new(AccountIdentity::AuthUser),
        ),
    ]);
    let results = tokio::time::timeout(
        std::time::Duration::from_secs(2),
        client.accounts_usage(&queries),
    )
    .await
    .expect("independent account fetches must overlap");
    assert_eq!(
        results
            .into_iter()
            .map(|(name, _)| name)
            .collect::<Vec<_>>(),
        profiles
            .iter()
            .map(|p| p.profile_name.clone())
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn replacing_a_profile_invalidates_its_signed_in_account_source() {
    let profile = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "openai")
        .unwrap();
    let mut builder = LlmClientBuilder::new(std::slice::from_ref(&profile)).unwrap();
    builder.register_profile_account_source(
        "openai",
        AccountIdentity::AuthUser,
        Arc::new(FakeSource),
    );
    let mut client = builder.build().unwrap();
    let dir = account_config_dir();
    client.set_config_dir(&dir).unwrap();
    let mut query = AccountQuery::new(AccountIdentity::AuthUser);
    query.since_unix = Some(100);
    query.until_unix = Some(200);
    assert!(matches!(
        client
            .account_usage("openai", &query)
            .await
            .unwrap()
            .balance,
        AccountMetric::Available { .. }
    ));

    let mut replacement = profile;
    replacement.base_url = "https://replacement.invalid/v1".into();
    client.add_provider(replacement).unwrap();
    assert!(matches!(
        client
            .account_usage("openai", &query)
            .await
            .unwrap()
            .balance,
        AccountMetric::Unsupported
    ));

    client
        .register_profile_account_source("openai", AccountIdentity::AuthUser, Arc::new(FakeSource))
        .unwrap();
    assert!(matches!(
        client
            .account_usage("openai", &query)
            .await
            .unwrap()
            .balance,
        AccountMetric::Available { .. }
    ));
    drop(client);
    std::fs::remove_dir_all(dir).unwrap();
}

#[tokio::test]
async fn reloading_a_changed_profile_invalidates_its_signed_in_account_source() {
    let profile = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "openai")
        .unwrap();
    let dir = account_config_dir();
    let mut builder = LlmClientBuilder::new(std::slice::from_ref(&profile)).unwrap();
    builder.register_profile_account_source(
        "openai",
        AccountIdentity::AuthUser,
        Arc::new(FakeSource),
    );
    let mut client = builder.build().unwrap();
    client.set_config_dir(&dir).unwrap();
    let mut other = LlmClientBuilder::new(std::slice::from_ref(&profile))
        .unwrap()
        .build()
        .unwrap();
    other.set_config_dir(&dir).unwrap();
    let mut replacement = profile;
    replacement.base_url = "https://replacement.invalid/v1".into();
    other.add_provider(replacement).unwrap();

    client.set_config_dir(&dir).unwrap();
    let mut query = AccountQuery::new(AccountIdentity::AuthUser);
    query.since_unix = Some(100);
    query.until_unix = Some(200);
    assert!(matches!(
        client
            .account_usage("openai", &query)
            .await
            .unwrap()
            .balance,
        AccountMetric::Unsupported
    ));
    drop(other);
    drop(client);
    std::fs::remove_dir_all(dir).unwrap();
}
