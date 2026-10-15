use lingxi_llm_client::protocol::JsonPayload;
use lingxi_llm_client::providers::anthropic::fallback_response::*;
use serde_json::Value;

fn payload(value: &Value) -> JsonPayload {
    JsonPayload::from_generated_value(value).unwrap()
}

fn compare(actual: &Value, expected: &Value) {
    match (actual, expected) {
        (Value::Number(a), Value::Number(b)) => assert_eq!(a.as_f64(), b.as_f64()),
        (Value::Object(a), Value::Object(b)) => {
            assert_eq!(a.len(), b.len(), "{actual} != {expected}");
            for (key, value) in a {
                compare(value, &b[key]);
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len());
            for (a, b) in a.iter().zip(b) {
                compare(a, b);
            }
        }
        _ => assert_eq!(actual, expected),
    }
}

#[test]
fn actual_native_fallback_controls_and_usage_iterations_match() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/fallback_response_2_1_288.json")).unwrap();
    for case in fixture["blockCases"].as_array().unwrap() {
        let hop = complete_block(&payload(&case["input"]));
        compare(&serde_json::to_value(&hop).unwrap(), &case["expected"]);
        assert_eq!(
            hop.map(|hop| hop.materialized_block()),
            (!case["materialized"].is_null()).then(|| case["materialized"].clone())
        );
    }
    for case in fixture["streamCases"].as_array().unwrap() {
        compare(
            &serde_json::to_value(stream_start(&payload(&case["input"]))).unwrap(),
            &case["expected"],
        );
        assert_eq!(
            malformed_start(&payload(&case["input"])),
            case["malformed"].as_bool().unwrap()
        );
    }
    for case in fixture["usageCases"].as_array().unwrap() {
        compare(
            &serde_json::to_value(usage_iterations(&payload(&case["input"]))).unwrap(),
            &case["expected"],
        );
    }
}

#[tokio::test]
async fn sdk_collectors_preserve_controls_and_iterations_after_completion_or_interruption() {
    use async_trait::async_trait;
    use lingxi_llm_client::protocol::{ChatRequest, LlmError, ProviderProfile, Region};
    use lingxi_llm_client::{
        HttpRequest, HttpResponse, LlmClientBuilder, RequestOptions, StreamResponse, Transport,
    };
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    struct Once(Mutex<Option<StreamResponse>>);
    #[async_trait]
    impl Transport for Once {
        async fn send(&self, _: HttpRequest) -> Result<StreamResponse, LlmError> {
            Ok(self.0.lock().unwrap().take().unwrap())
        }
    }
    for scenario in [
        "interrupted",
        "completed_sparse_usage",
        "completed_empty_iterations",
    ] {
        let interrupted = scenario == "interrupted";
        let mut frames = vec![
            json!({"type":"message_start","message":{"id":"msg","model":"m"}}),
            json!({"type":"content_block_start","index":0,"content_block":{"type":"fallback","from":{"model":"m"},"to":{"model":"target"}}}),
            json!({"type":"content_block_start","index":1,"content_block":{"type":"fallback","from":{"model":"target"},"to":{"model":"next"},"trigger":{"type":"refusal","category":"bio"}}}),
            json!({"type":"message_delta","delta":{},"usage":{"iterations":[{"type":"fallback_message","model":"usage-target","input_tokens":1.5}]}}),
        ];
        if !interrupted {
            let terminal_usage = if scenario == "completed_empty_iterations" {
                json!({"output_tokens":1,"iterations":[]})
            } else {
                json!({"output_tokens":1})
            };
            frames.extend([
                json!({"type":"message_delta","delta":{"stop_reason":"end_turn","stop_details":{"category":"cyber"}},"usage":terminal_usage}),
                json!({"type":"message_stop"}),
            ]);
        }
        let bytes = frames
            .iter()
            .map(|frame| format!("data: {frame}\n\n"))
            .collect::<String>();
        let transport = Arc::new(Once(Mutex::new(Some(
            HttpResponse {
                status: 200,
                headers: vec![],
                body: bytes.into_bytes().into(),
            }
            .into(),
        ))));
        let profile:ProviderProfile=serde_json::from_value(json!({"provider_id":"custom","profile_name":"fixture","base_url":"https://fixture.invalid","protocol":"anthropic_messages","auth":"none","models":[{"display_model":"m","request_model":"m","billing_model":"m"}]})).unwrap();
        let client = LlmClientBuilder::with_transport(transport, &[profile])
            .with_region(Region::International)
            .build()
            .unwrap();
        let request: ChatRequest = serde_json::from_value(json!({
            "model":"m",
            "messages":[],
            "max_tokens":256
        }))
        .unwrap();
        let result = client
            .chat()
            .stream(&request, &RequestOptions::default())
            .await
            .unwrap()
            .collect_response()
            .await;
        assert_eq!(result.is_err(), interrupted);
        let assembly = match result {
            Ok(assembly) => assembly,
            Err(error) => *error.partial,
        };
        assert!(assembly.response.message.content.is_empty());
        let fallback = assembly.response.anthropic_fallback().unwrap();
        assert_eq!(fallback.hops.len(), 2);
        assert_eq!(fallback.hops[1].category.as_deref(), Some("bio"));
        assert_eq!(
            fallback
                .iterations
                .as_ref()
                .unwrap()
                .served_fallback_model
                .as_deref(),
            if scenario == "completed_empty_iterations" {
                None
            } else {
                Some("usage-target")
            }
        );
        assert_eq!(
            fallback.final_stop_reason.as_deref(),
            (!interrupted).then_some("end_turn")
        );
        if !interrupted {
            assert_eq!(
                assembly
                    .response
                    .anthropic_stop_details()
                    .unwrap()
                    .as_json_source(),
                r#"{"category":"cyber"}"#
            );
        }
        assert_eq!(assembly.response.model, "m");
    }
}
