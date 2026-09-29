//! Copilot model-specific inference protocol selection.
use crate::protocol::ProtocolFamily;

/// GitHub Copilot serves its GPT-5.x and `codex` models ONLY through the OpenAI
/// Responses endpoint (`api.githubcopilot.com/responses`); older models use
/// `/chat/completions`. Our `github-copilot` preset declares a single
/// `OpenAiChat` protocol for the whole provider, so those newer models 400 with
/// "model … is not accessible via the /chat/completions endpoint". When we
/// detect one we route it through a Responses codec bound to the SAME host — the
/// Copilot bearer authorizes both paths. Mirrors the fix other Copilot gateways
/// adopted (cherry-studio #13637, opencode #5866): non-`codex` GPT-5+ models
/// must use `/responses`.
///
/// Returns `Some(OpenAiResponses)` only for the `github-copilot` profile on an
/// `OpenAiChat` route whose model needs Responses; `None` leaves routing intact
/// (so `openai`/`openrouter`/`deepseek`/etc. are never affected).
pub fn responses_protocol_override(
    profile_name: &str,
    protocol: &ProtocolFamily,
    request_model: &str,
) -> Option<ProtocolFamily> {
    if profile_name != "github-copilot" || !matches!(protocol, ProtocolFamily::OpenAiChat) {
        return None;
    }
    let model = request_model.to_ascii_lowercase();
    (model.contains("codex") || is_gpt5_or_newer(&model)).then_some(ProtocolFamily::OpenAiResponses)
}

/// `true` for `gpt-<major>[…]` with `major >= 5` (`gpt-5`, `gpt-5.5`,
/// `gpt-5-mini`, `gpt-6`, …); `false` for `gpt-4o`, `gpt-4.1`, non-`gpt-` ids.
fn is_gpt5_or_newer(model_lower: &str) -> bool {
    let Some(rest) = model_lower.strip_prefix("gpt-") else {
        return false;
    };
    let major: String = rest.chars().take_while(char::is_ascii_digit).collect();
    major.parse::<u32>().is_ok_and(|n| n >= 5)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gpt5_plus_detection_covers_majors_and_variants() {
        for yes in [
            "gpt-5",
            "gpt-5.5",
            "gpt-5-mini",
            "gpt-5.2-codex",
            "gpt-6",
            "gpt-10",
        ] {
            assert!(is_gpt5_or_newer(yes), "{yes} should be gpt-5+");
        }
        for no in [
            "gpt-4o",
            "gpt-4.1",
            "gpt-4-turbo",
            "o3",
            "claude-opus-4-8",
            "",
        ] {
            assert!(!is_gpt5_or_newer(no), "{no} should NOT be gpt-5+");
        }
    }

    #[test]
    fn copilot_override_fires_only_for_copilot_chat_gpt5_and_codex() {
        let chat = ProtocolFamily::OpenAiChat;
        // Fires: Copilot + OpenAiChat + gpt-5.x/codex.
        for m in ["gpt-5.5", "gpt-5-codex", "gpt-5.4-mini"] {
            assert_eq!(
                responses_protocol_override("github-copilot", &chat, m),
                Some(ProtocolFamily::OpenAiResponses),
                "{m} on copilot should override to Responses"
            );
        }
        // No override: older Copilot model.
        assert_eq!(
            responses_protocol_override("github-copilot", &chat, "gpt-4o"),
            None
        );
        // No override: different profile, even for a gpt-5 id.
        assert_eq!(
            responses_protocol_override("openrouter", &chat, "gpt-5.5"),
            None
        );
        // No override: already Responses (openai first-party) — nothing to fix.
        assert_eq!(
            responses_protocol_override(
                "github-copilot",
                &ProtocolFamily::OpenAiResponses,
                "gpt-5.5"
            ),
            None
        );
    }
}
