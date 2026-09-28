use async_trait::async_trait;
use lingxi_llm_client::{
    builtin_providers, AccountBalance, AccountIdentity, AccountMetric, AccountQuery, AccountScope,
    AccountScopeKind, AccountSnapshot, AccountUsageError, AccountUsageSource, LlmClientBuilder,
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
        context: &lingxi_llm_client::AccountFetchContext<'_>,
        _report: &mut lingxi_llm_client::AccountReport,
    ) -> Result<(), lingxi_llm_client::AccountFailure> {
        let _range = context.range;

        self.0.wait().await;
        Ok(())
    }
}

#[async_trait]
impl AccountUsageSource for FakeSource {
    fn requires_profile_binding(&self, _query: &AccountQuery) -> bool {
        true
    }

    async fn fetch(
        &self,
        context: &lingxi_llm_client::AccountFetchContext<'_>,
        report: &mut lingxi_llm_client::AccountReport,
    ) -> Result<(), lingxi_llm_client::AccountFailure> {
        let range = context.range;

        assert_eq!(range, (100, 200));

        report.balance = Some(AccountMetric::available(
            AccountScope::new(AccountScopeKind::Account, None),
            "fake",
            vec![AccountBalance {
                unit: "USD".into(),
                remaining: "12.50".into(),
                total: None,
                is_available: None,
            }],
        ));
        Ok(())
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
    builder
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap()
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
    let client = builder
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
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
    let client = builder
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
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
    let client = builder
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build()
        .unwrap();
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
    let (client, client_config) = builder
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    let dir = account_config_dir();
    client_config.set_config_dir(&dir).await.unwrap();
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
    replacement.background = lingxi_llm_client::protocol::ServiceSetting::Disabled;
    client_config.add_provider(replacement).await.unwrap();
    assert!(matches!(
        client
            .account_usage("openai", &query)
            .await
            .unwrap()
            .balance,
        AccountMetric::Unsupported
    ));

    client_config
        .register_profile_account_source("openai", AccountIdentity::AuthUser, Arc::new(FakeSource))
        .await
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
    let (client, client_config) = builder
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    let (other, other_config) = LlmClientBuilder::new(std::slice::from_ref(&profile))
        .unwrap()
        .with_region(lingxi_llm_client::protocol::Region::International)
        .build_managed()
        .unwrap();
    other_config.set_config_dir(&dir).await.unwrap();
    let mut replacement = profile;
    replacement.base_url = "https://replacement.invalid/v1".into();
    replacement.background = lingxi_llm_client::protocol::ServiceSetting::Disabled;
    other_config.add_provider(replacement).await.unwrap();

    client_config.set_config_dir(&dir).await.unwrap();
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

struct QueryBoundSource(Arc<std::sync::atomic::AtomicUsize>);

#[async_trait]
impl AccountUsageSource for QueryBoundSource {
    fn requires_profile_binding(&self, query: &AccountQuery) -> bool {
        query.credential.is_none()
    }

    async fn fetch(
        &self,
        context: &lingxi_llm_client::AccountFetchContext<'_>,
        _report: &mut lingxi_llm_client::AccountReport,
    ) -> Result<(), lingxi_llm_client::AccountFailure> {
        let _range = context.range;

        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

#[tokio::test]
async fn implicit_session_binding_expires_on_account_changes_but_token_queries_still_work() {
    use lingxi_llm_client::protocol::{CredentialConfig, Region, Secret};
    use std::sync::atomic::{AtomicUsize, Ordering};
    for action in ["replace", "readd", "remove", "restore", "switch", "reload"] {
        let mut profile = builtin_providers()
            .unwrap()
            .into_iter()
            .find(|p| p.profile_name == "openai")
            .unwrap();
        profile.connection.connection_id = Some("alice".into());
        profile.credential = CredentialConfig::HostManaged {
            key: "alice-login".into(),
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let mut builder = LlmClientBuilder::new(std::slice::from_ref(&profile)).unwrap();
        builder.register_account_source(
            "openai",
            AccountIdentity::AuthUser,
            Arc::new(QueryBoundSource(calls.clone())),
        );
        let (client, client_config) = builder
            .with_region(Region::International)
            .build_managed()
            .unwrap();
        let dir = account_config_dir();
        let second_dir = account_config_dir();
        client_config.set_config_dir(&dir).await.unwrap();
        client_config.set_config_dir(&dir).await.unwrap();
        let query = AccountQuery::new(AccountIdentity::AuthUser);
        client.account_usage("openai", &query).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        let mut replacement = profile.clone();
        replacement.connection.connection_id = Some("bob".into());
        replacement.credential = CredentialConfig::HostManaged {
            key: "bob-login".into(),
        };
        match action {
            "replace" => client_config.add_provider(replacement).await.unwrap(),
            "readd" => client_config.add_provider(profile.clone()).await.unwrap(),
            "remove" => {
                client_config.remove_provider("openai").await.unwrap();
                client_config.add_provider(profile.clone()).await.unwrap();
            }
            "restore" => client_config.restore_builtin("openai").await.unwrap(),
            "switch" => client_config.set_config_dir(&second_dir).await.unwrap(),
            "reload" => {
                let (_other, other_config) = LlmClientBuilder::new(std::slice::from_ref(&profile))
                    .unwrap()
                    .with_region(Region::International)
                    .build_managed()
                    .unwrap();
                other_config.set_config_dir(&dir).await.unwrap();
                other_config.add_provider(replacement).await.unwrap();
                client_config.set_config_dir(&dir).await.unwrap();
            }
            _ => unreachable!(),
        }
        for _ in 0..2 {
            assert!(
                matches!(
                    client.account_usage("openai", &query).await,
                    Err(AccountUsageError::AmbiguousAccountSource(_))
                ),
                "{action}"
            );
            let token_query = AccountQuery {
                credential: Some(Secret::new("bob-token".into())),
                ..query.clone()
            };
            client.account_usage("openai", &token_query).await.unwrap();
        }
        assert_eq!(
            calls.load(Ordering::SeqCst),
            3,
            "only explicitly scoped queries may call the old source: {action}"
        );
        let rebound = Arc::new(AtomicUsize::new(0));
        client_config
            .register_profile_account_source(
                "openai",
                AccountIdentity::AuthUser,
                Arc::new(QueryBoundSource(rebound.clone())),
            )
            .await
            .unwrap();
        client.account_usage("openai", &query).await.unwrap();
        assert_eq!(rebound.load(Ordering::SeqCst), 1);
        let client_view = client.snapshot();
        let current = client_view.profile("openai").unwrap().clone();
        client_config.add_provider(current).await.unwrap();
        assert!(matches!(
            client.account_usage("openai", &query).await,
            Err(AccountUsageError::AmbiguousAccountSource(_))
        ));
        std::fs::remove_dir_all(dir).unwrap();
        if second_dir.exists() {
            std::fs::remove_dir_all(second_dir).unwrap();
        }
    }
}

#[tokio::test]
async fn model_only_and_other_provider_changes_preserve_implicit_session_binding() {
    use lingxi_llm_client::protocol::Region;
    use std::sync::atomic::{AtomicUsize, Ordering};
    let profile = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "openai")
        .unwrap();
    let model = profile.models[0].request_model.clone();
    let calls = Arc::new(AtomicUsize::new(0));
    let mut builder = LlmClientBuilder::new(std::slice::from_ref(&profile)).unwrap();
    builder.register_account_source(
        "openai",
        AccountIdentity::AuthUser,
        Arc::new(QueryBoundSource(calls.clone())),
    );
    let (client, client_config) = builder
        .with_region(Region::International)
        .build_managed()
        .unwrap();
    let dir = account_config_dir();
    client_config.set_config_dir(&dir).await.unwrap();
    client_config
        .set_tracked_models("openai", [model.clone()])
        .await
        .unwrap();
    client_config
        .set_model_visibility("openai", &model, false)
        .await
        .unwrap();
    let other = builtin_providers()
        .unwrap()
        .into_iter()
        .find(|p| p.profile_name == "deepseek")
        .unwrap();
    client_config.add_provider(other).await.unwrap();
    client_config.set_config_dir(&dir).await.unwrap();
    client
        .account_usage("openai", &AccountQuery::new(AccountIdentity::AuthUser))
        .await
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    std::fs::remove_dir_all(dir).unwrap();
}
