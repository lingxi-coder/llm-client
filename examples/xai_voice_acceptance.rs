//! Read-only live acceptance for the xAI voice catalog. No generation or mutation.
//! By default this only prints the proposed check; pass --run-read-only to send it.

use lingxi_llm_client::{
    protocol::Secret,
    providers::xai::audio::{XaiAudioConfig, XaiAudioCredentials, XaiAudioError, XaiAudioService},
    HttpTransport,
};
use serde_json::{json, Value};
use std::{
    process::ExitCode,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

fn failure(error: XaiAudioError) -> Value {
    // Provider bodies and transport errors can contain sensitive diagnostics.
    // The acceptance report deliberately includes only the category and HTTP status.
    match error {
        XaiAudioError::Provider { status, .. } => {
            json!({"status":"failed","category":"provider","http_status":status})
        }
        XaiAudioError::InvalidResponse { .. } => {
            json!({"status":"failed","category":"response_contract"})
        }
        XaiAudioError::InvalidRequest(_) => {
            json!({"status":"failed","category":"local_validation"})
        }
        XaiAudioError::Llm(_) => json!({"status":"failed","category":"transport"}),
        XaiAudioError::OutcomeUnknown { .. } => {
            json!({"status":"failed","category":"unexpected_unknown_outcome"})
        }
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() || args == ["--help"] {
        println!(
            "{}",
            json!({
                "status":"not_run", "provider":"xai", "operations":["voices.list", "voices.get"],
                "endpoint":"https://api.x.ai/v1/tts/voices", "method":"GET",
                "run":"cargo run --locked --offline --example xai_voice_acceptance -- --run-read-only",
                "credential_env":"XAI_API_KEY",
                "scope":"List the voice catalog, then get the first voice if present; at most two GET requests. No inference, custom voice management or audio playback."
            })
        );
        return ExitCode::SUCCESS;
    }
    if args != ["--run-read-only"] {
        eprintln!("Expected no arguments, --help, or --run-read-only");
        return ExitCode::from(2);
    }
    let Ok(key) = std::env::var("XAI_API_KEY") else {
        println!(
            "{}",
            json!({"status":"not_run","reason":"missing_XAI_API_KEY"})
        );
        return ExitCode::from(2);
    };
    if key.trim().is_empty() {
        println!(
            "{}",
            json!({"status":"not_run","reason":"empty_XAI_API_KEY"})
        );
        return ExitCode::from(2);
    }
    let Ok(transport) = HttpTransport::new() else {
        println!(
            "{}",
            json!({"status":"not_run","reason":"transport_initialization"})
        );
        return ExitCode::from(2);
    };
    let service = XaiAudioService::new(
        &transport,
        // This is a local routing label, not a verified provider account identity.
        XaiAudioConfig::new("xai-readonly-acceptance", "environment-xai-key")
            .with_request_timeout(Duration::from_secs(20)),
    )
    .expect("fixed official route and nonempty local scope");
    let credentials = XaiAudioCredentials::new(Secret::new(key));
    let started = Instant::now();
    let checked_at = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let mut checks = Vec::new();
    let passed = match service.list_voices(&credentials).await {
        Ok(list) => {
            checks.push(json!({"operation":"voices.list","status":"passed","voice_count":list.voices.len()}));
            if let Some(voice) = list.voices.first() {
                match service.get_voice(&voice.voice_id, &credentials).await {
                    Ok(_) => {
                        checks.push(json!({"operation":"voices.get","status":"passed"}));
                        true
                    }
                    Err(error) => {
                        let mut check = failure(error);
                        check["operation"] = json!("voices.get");
                        checks.push(check);
                        false
                    }
                }
            } else {
                checks.push(
                    json!({"operation":"voices.get","status":"not_run","reason":"empty_catalog"}),
                );
                false
            }
        }
        Err(error) => {
            let mut check = failure(error);
            check["operation"] = json!("voices.list");
            checks.push(check);
            checks
                .push(json!({"operation":"voices.get","status":"not_run","reason":"list_failed"}));
            false
        }
    };
    let mut report = json!({"status":if passed {"passed"} else {"incomplete"},"checks":checks});
    report["provider"] = json!("xai");
    report["checked_at_unix_seconds"] = json!(checked_at);
    report["elapsed_ms"] = json!(started.elapsed().as_millis());
    report["acceptance_scope"] = json!("at most two read-only catalog requests; no inference, account identity or audio validation");
    println!("{report}");
    if passed {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}
