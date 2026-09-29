//! Provider authentication for borrowed, host-owned credential material.
//! Storage, refresh scheduling and account selection stay with the caller.
//! Explicit token-exchange and refresh protocols are available in `auth::oauth`.
use super::{header_policy, sigv4};
use crate::protocol::{AuthStrategy, LlmError, ProtocolFamily, ProviderProfile};
use crate::HttpRequest;
use std::collections::BTreeMap;
use std::time::SystemTime;

/// A borrowed credential. Deliberately does not implement Debug or serialization.
///
/// Credential material cannot accidentally enter a debug log:
/// ```compile_fail
/// use lingxi_llm_client::auth::CredentialRef;
/// let credential = CredentialRef::ChatGpt {
///     access_token: "secret", account_id: None, fedramp: false,
/// };
/// let _ = format!("{credential:?}");
/// ```
#[derive(Clone, Copy)]
pub enum CredentialRef<'a> {
    Token(&'a str),
    Aws {
        access_key_id: &'a str,
        secret_access_key: &'a str,
        session_token: Option<&'a str>,
    },
    ChatGpt {
        access_token: &'a str,
        account_id: Option<&'a str>,
        fedramp: bool,
    },
}

/// Host application identity for protocols that require editor identification.
#[derive(Debug, Clone, Copy)]
pub struct ClientIdentity<'a> {
    pub user_agent: &'a str,
    pub editor_version: &'a str,
    pub plugin_version: &'a str,
}

/// Apply one provider's authentication to the exact outgoing bytes. No secret
/// is retained after this call. Validation errors never include credential data.
pub fn apply_credential(
    request: &mut HttpRequest,
    profile: &ProviderProfile,
    credential: CredentialRef<'_>,
    identity: ClientIdentity<'_>,
    now: SystemTime,
) -> Result<(), LlmError> {
    let mut headers: BTreeMap<_, _> = request.headers.iter().cloned().collect();
    let mismatch = || {
        let message = format!(
            "credential material does not match {:?} authentication",
            profile.auth
        );
        if profile.auth == AuthStrategy::ChatGptOAuth {
            LlmError::Authentication { message }
        } else {
            LlmError::InvalidRequest { message }
        }
    };
    match profile.auth {
        AuthStrategy::None => return Ok(()),
        AuthStrategy::AwsSigV4 => {
            let CredentialRef::Aws {
                access_key_id,
                secret_access_key,
                session_token,
            } = credential
            else {
                return Err(mismatch());
            };
            let signing = profile
                .signing
                .as_ref()
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "AwsSigV4 authentication requires signing configuration".into(),
                })?;
            let region = signing
                .region
                .as_deref()
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "AwsSigV4 authentication requires a signing region".into(),
                })?;
            let service = signing
                .service
                .as_deref()
                .ok_or_else(|| LlmError::InvalidRequest {
                    message: "AwsSigV4 authentication requires a signing service".into(),
                })?;
            headers.retain(|name, _| {
                ![
                    "authorization",
                    "x-amz-date",
                    "x-amz-content-sha256",
                    "x-amz-security-token",
                ]
                .iter()
                .any(|key| name.eq_ignore_ascii_case(key))
            });
            let date: chrono::DateTime<chrono::Utc> = now.into();
            let signed = sigv4::sign_request(
                &request.method,
                &request.url,
                &headers,
                &request.body,
                access_key_id,
                secret_access_key,
                session_token,
                region,
                service,
                &date.format("%Y%m%dT%H%M%SZ").to_string(),
            )
            .map_err(|_| LlmError::InvalidRequest {
                message: "could not sign provider request".into(),
            })?;
            headers.insert("x-amz-date".into(), signed.x_amz_date);
            headers.insert("x-amz-content-sha256".into(), signed.x_amz_content_sha256);
            if let Some(token) = signed.x_amz_security_token {
                headers.insert("x-amz-security-token".into(), token);
            }
            headers.insert("Authorization".into(), signed.authorization);
        }
        AuthStrategy::ChatGptOAuth => {
            let CredentialRef::ChatGpt {
                access_token,
                account_id,
                fedramp,
            } = credential
            else {
                return Err(mismatch());
            };
            // Only Responses generation bodies have this restriction. File and
            // resource operations may be binary and must keep their exact bytes.
            if profile.protocol == ProtocolFamily::OpenAiResponses && !request.body.is_empty() {
                if let Ok(mut body) = serde_json::from_slice::<serde_json::Value>(&request.body) {
                    if body.get("model").is_some() && body.get("input").is_some() {
                        header_policy::chatgpt_body(&mut body);
                        request.body = serde_json::to_vec(&body)
                            .map_err(|_| LlmError::InvalidRequest {
                                message: "could not encode authenticated request".into(),
                            })?
                            .into();
                    }
                }
            }
            header_policy::chatgpt(&mut headers, access_token, account_id, fedramp);
        }
        strategy => {
            let CredentialRef::Token(token) = credential else {
                return Err(mismatch());
            };
            match strategy {
                AuthStrategy::ApiKey => {
                    let name = super::key_header(profile);
                    if name.eq_ignore_ascii_case("authorization") {
                        header_policy::bearer(&mut headers, token);
                    } else {
                        header_policy::api_key(&mut headers, name, token);
                    }
                }
                AuthStrategy::AzureToken => header_policy::api_key(&mut headers, "api-key", token),
                AuthStrategy::CopilotBearer => header_policy::copilot(
                    &mut headers,
                    token,
                    identity.user_agent,
                    identity.editor_version,
                    identity.plugin_version,
                ),
                AuthStrategy::Bearer | AuthStrategy::OAuthBearer | AuthStrategy::GcpToken => {
                    header_policy::bearer(&mut headers, token);
                    if strategy == AuthStrategy::OAuthBearer
                        && profile.protocol == ProtocolFamily::AnthropicMessages
                    {
                        header_policy::anthropic_oauth(&mut headers);
                    }
                }
                _ => unreachable!("specialized strategies handled above"),
            }
        }
    }
    request.headers = headers.into_iter().collect();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn profile(auth: &str, protocol: &str) -> ProviderProfile {
        serde_json::from_value(serde_json::json!({"provider_id":"test","profile_name":"test","protocol":protocol,"auth":auth,"base_url":"https://example.com","models":[],"signing":{"region":"us-east-1","service":"bedrock"}})).unwrap()
    }
    fn identity() -> ClientIdentity<'static> {
        ClientIdentity {
            user_agent: "test",
            editor_version: "test/1",
            plugin_version: "test/1",
        }
    }
    fn request(body: &[u8]) -> HttpRequest {
        HttpRequest {
            method: "POST".into(),
            url: "https://example.com/model/m/invoke".into(),
            headers: vec![],
            body: body.to_vec().into(),
            timeout: None,
        }
    }
    #[test]
    fn aws_signs_exact_bytes_and_replaces_previous_account() {
        let body = br#"{"text":"\ud800"}"#;
        let mut req = request(body);
        req.headers
            .push(("X-Amz-Security-Token".into(), "old-account".into()));
        apply_credential(
            &mut req,
            &profile("aws_sig_v4", "bedrock_claude"),
            CredentialRef::Aws {
                access_key_id: "id",
                secret_access_key: "secret",
                session_token: None,
            },
            identity(),
            SystemTime::UNIX_EPOCH,
        )
        .unwrap();
        assert_eq!(&req.body[..], body);
        assert!(!req
            .headers
            .iter()
            .any(|(n, v)| n.eq_ignore_ascii_case("x-amz-security-token") || v == "old-account"));
        assert!(req
            .headers
            .iter()
            .any(|(n, v)| n == "x-amz-date" && v == "19700101T000000Z"));
    }
    #[test]
    fn mismatch_does_not_expose_secret_or_modify_request() {
        let mut req = request(b"binary");
        let err = apply_credential(
            &mut req,
            &profile("aws_sig_v4", "bedrock_claude"),
            CredentialRef::Token("secret-value"),
            identity(),
            SystemTime::UNIX_EPOCH,
        )
        .unwrap_err();
        assert!(!err.to_string().contains("secret-value"));
        assert!(req.headers.is_empty());
        assert_eq!(&req.body[..], b"binary");
    }
    #[test]
    fn chatgpt_rejects_api_keys_as_authentication_failure() {
        let mut req = request(b"{}");
        let error = apply_credential(
            &mut req,
            &profile("chat_gpt_o_auth", "open_ai_responses"),
            CredentialRef::Token("api-key-secret"),
            identity(),
            SystemTime::UNIX_EPOCH,
        )
        .unwrap_err();
        assert!(matches!(error, LlmError::Authentication { .. }));
        assert!(!error.to_string().contains("api-key-secret"));
        assert!(req.headers.is_empty());
    }

    #[test]
    fn chatgpt_resource_binary_is_preserved() {
        let mut req = request(&[0, 255, 4]);
        apply_credential(
            &mut req,
            &profile("chat_gpt_o_auth", "open_ai_responses"),
            CredentialRef::ChatGpt {
                access_token: "token",
                account_id: Some("account"),
                fedramp: false,
            },
            identity(),
            SystemTime::UNIX_EPOCH,
        )
        .unwrap();
        assert_eq!(&req.body[..], &[0, 255, 4]);
    }
    #[test]
    fn chatgpt_sets_account_and_removes_stale_api_key() {
        let mut req = request(b"{}");
        req.headers.push(("x-api-key".into(), "leftover".into()));
        apply_credential(
            &mut req,
            &profile("chat_gpt_o_auth", "open_ai_responses"),
            CredentialRef::ChatGpt {
                access_token: "tok-123",
                account_id: Some("acc_9"),
                fedramp: false,
            },
            identity(),
            SystemTime::UNIX_EPOCH,
        )
        .unwrap();
        let headers: BTreeMap<_, _> = req.headers.into_iter().collect();
        assert_eq!(
            headers.get("Authorization").map(String::as_str),
            Some("Bearer tok-123")
        );
        assert_eq!(
            headers.get("ChatGPT-Account-ID").map(String::as_str),
            Some("acc_9")
        );
        assert!(!headers.contains_key("X-OpenAI-Fedramp"));
        assert!(!headers.contains_key("x-api-key"));
    }

    #[test]
    fn chatgpt_replaces_all_case_insensitive_credentials() {
        let mut req = request(b"{}");
        for name in [
            "authorization",
            "AUTHORIZATION",
            "X-API-Key",
            "api-key",
            "chatgpt-account-id",
            "x-openai-fedramp",
        ] {
            req.headers.push((name.into(), "stale".into()));
        }
        req.headers.push(("originator".into(), "lingxi".into()));
        apply_credential(
            &mut req,
            &profile("chat_gpt_o_auth", "open_ai_responses"),
            CredentialRef::ChatGpt {
                access_token: "oauth-token",
                account_id: None,
                fedramp: false,
            },
            identity(),
            SystemTime::UNIX_EPOCH,
        )
        .unwrap();
        assert!(req
            .headers
            .iter()
            .any(|(name, value)| name == "Authorization" && value == "Bearer oauth-token"));
        assert!(req
            .headers
            .iter()
            .any(|(name, value)| name == "originator" && value == "lingxi"));
        assert!(!req.headers.iter().any(|(_, value)| value == "stale"));
    }

    #[test]
    fn chatgpt_sets_fedramp_only_when_flagged() {
        let mut req = request(b"{}");
        apply_credential(
            &mut req,
            &profile("chat_gpt_o_auth", "open_ai_responses"),
            CredentialRef::ChatGpt {
                access_token: "token",
                account_id: None,
                fedramp: true,
            },
            identity(),
            SystemTime::UNIX_EPOCH,
        )
        .unwrap();
        assert!(req
            .headers
            .iter()
            .any(|(name, value)| name == "X-OpenAI-Fedramp" && value == "true"));
        assert!(!req
            .headers
            .iter()
            .any(|(name, _)| name.eq_ignore_ascii_case("ChatGPT-Account-ID")));
    }
}

#[cfg(test)]
mod copilot_contract_tests {
    use crate::auth::oauth::copilot::CopilotSecret;
    const COPILOT_USER_AGENT: &str = "example-editor";
    const COPILOT_EDITOR_VERSION: &str = "example/1.0";
    const COPILOT_EDITOR_PLUGIN_VERSION: &str = "example-plugin/1.0";
    use crate::auth::{apply_credential, ClientIdentity, CredentialRef};
    use serde_json::json;

    #[test]
    fn injects_copilot_headers_and_strips_x_api_key() {
        let mut request = crate::HttpRequest {
            method: "POST".into(),
            url: "https://api.githubcopilot.com/chat/completions".into(),
            headers: vec![("x-api-key".into(), "leftover".into())],
            body: serde_json::to_vec(&json!({"model":"gpt-5.4-nano"}))
                .unwrap()
                .into(),
            timeout: None,
        };
        let profile = serde_json::from_value(json!({
            "provider_id":"github-copilot", "profile_name":"github-copilot",
            "base_url":"https://api.githubcopilot.com", "protocol":"open_ai_chat",
            "auth":"copilot_bearer", "models":[]
        }))
        .unwrap();
        apply_credential(
            &mut request,
            &profile,
            CredentialRef::Token("ght_token"),
            ClientIdentity {
                user_agent: COPILOT_USER_AGENT,
                editor_version: COPILOT_EDITOR_VERSION,
                plugin_version: COPILOT_EDITOR_PLUGIN_VERSION,
            },
            std::time::SystemTime::UNIX_EPOCH,
        )
        .unwrap();
        let headers: std::collections::BTreeMap<_, _> = request.headers.into_iter().collect();

        assert_eq!(
            headers.get("Authorization"),
            Some(&"Bearer ght_token".to_string())
        );
        assert_eq!(
            headers.get("X-GitHub-Api-Version"),
            Some(&"2026-06-01".to_string())
        );
        assert_eq!(
            headers.get("Openai-Intent"),
            Some(&"conversation-edits".to_string())
        );
        assert_eq!(
            headers.get("User-Agent"),
            Some(&COPILOT_USER_AGENT.to_string())
        );
        assert_eq!(headers.get("x-initiator"), Some(&"agent".to_string()));
        assert_eq!(
            headers.get("Copilot-Integration-Id"),
            Some(&"vscode-chat".to_string())
        );
        assert_eq!(
            headers.get("Editor-Version"),
            Some(&COPILOT_EDITOR_VERSION.to_string())
        );
        assert_eq!(
            headers.get("Editor-Plugin-Version"),
            Some(&COPILOT_EDITOR_PLUGIN_VERSION.to_string())
        );
        assert!(!headers.contains_key("x-api-key"));
    }

    #[test]
    fn token_for_storage_returns_raw_token_for_persistence() {
        // Frozen-crate (§10) exception: the /connect device-flow MUST persist the
        // GitHub token under `github-copilot`. The Debug stays redacting; only
        // this explicit, doc-hidden accessor exposes the raw token.
        let s = CopilotSecret::new("ght_live_token");
        assert_eq!(s.token_for_storage(), "ght_live_token");
        // Debug is still redacting (no regression).
        assert!(!format!("{s:?}").contains("ght_live_token"));
    }

    #[test]
    fn debug_does_not_leak_token() {
        let dbg_secret = format!("{:?}", CopilotSecret::new("supersecret"));
        assert!(!dbg_secret.contains("supersecret"));
    }
}
