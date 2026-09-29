//! Provider header policies, independent of credential storage.
use std::collections::BTreeMap;

pub fn api_key(headers: &mut BTreeMap<String, String>, name: &str, secret: &str) {
    headers.retain(|key, _| !key.eq_ignore_ascii_case(name));
    headers.insert(name.into(), secret.into());
}
pub fn bearer(headers: &mut BTreeMap<String, String>, token: &str) {
    api_key(headers, "Authorization", &format!("Bearer {token}"));
}
pub fn chatgpt(
    headers: &mut BTreeMap<String, String>,
    token: &str,
    account: Option<&str>,
    fedramp: bool,
) {
    headers.retain(|name, _| {
        ![
            "authorization",
            "x-api-key",
            "api-key",
            "chatgpt-account-id",
            "x-openai-fedramp",
        ]
        .iter()
        .any(|key| name.eq_ignore_ascii_case(key))
    });
    bearer(headers, token);
    if let Some(account) = account {
        headers.insert("ChatGPT-Account-ID".into(), account.into());
    }
    if fedramp {
        headers.insert("X-OpenAI-Fedramp".into(), "true".into());
    }
}
pub fn chatgpt_body(body: &mut serde_json::Value) {
    if let Some(body) = body.as_object_mut() {
        for key in ["max_output_tokens", "temperature", "top_p"] {
            body.remove(key);
        }
        body.insert("store".into(), false.into());
        body.entry("instructions").or_insert_with(|| "".into());
    }
}
pub fn copilot(
    headers: &mut BTreeMap<String, String>,
    token: &str,
    user_agent: &str,
    editor: &str,
    plugin: &str,
) {
    headers.retain(|name, _| !name.eq_ignore_ascii_case("x-api-key"));
    bearer(headers, token);
    for (name, value) in [
        ("User-Agent", user_agent),
        ("Openai-Intent", "conversation-edits"),
        ("X-GitHub-Api-Version", "2026-06-01"),
        ("x-initiator", "agent"),
        ("Copilot-Integration-Id", "vscode-chat"),
        ("Editor-Version", editor),
        ("Editor-Plugin-Version", plugin),
    ] {
        api_key(headers, name, value);
    }
}
pub fn oauth_beta_value<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    let mut tokens = Vec::new();
    for value in values {
        for token in value
            .split(',')
            .map(str::trim)
            .filter(|token| !token.is_empty())
        {
            if !tokens.contains(&token) {
                tokens.push(token);
            }
        }
    }
    if !tokens.contains(&"oauth-2025-04-20") {
        tokens.push("oauth-2025-04-20");
    }
    tokens.join(",")
}
pub fn anthropic_oauth(headers: &mut BTreeMap<String, String>) {
    let value = oauth_beta_value(
        headers
            .iter()
            .filter(|(name, _)| name.eq_ignore_ascii_case("anthropic-beta"))
            .map(|(_, value)| value.as_str()),
    );
    headers.retain(|name, _| !name.eq_ignore_ascii_case("anthropic-beta"));
    headers.insert("anthropic-beta".into(), value);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn oauth_merges_duplicate_case_variants_without_losing_custom_betas() {
        let mut headers = BTreeMap::from([
            ("Anthropic-Beta".into(), "custom-a".into()),
            ("anthropic-beta".into(), "custom-b,oauth-2025-04-20".into()),
        ]);
        anthropic_oauth(&mut headers);
        assert_eq!(headers.len(), 1);
        assert_eq!(
            headers["anthropic-beta"],
            "custom-a,custom-b,oauth-2025-04-20"
        );
    }
}
