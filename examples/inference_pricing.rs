//! Query capabilities/prices, choose controls, execute, and inspect actual charges.
//! Running this example sends a billable request using OPENAI_API_KEY.
use lingxi_llm_client::{builtin_providers, protocol::*, LlmClientBuilder, RequestOptions};

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let profiles = builtin_providers()?;
    let client = LlmClientBuilder::new(&profiles)?
        .with_region(Region::International)
        .build()?;
    let model = "gpt-5.6";
    let listing = client
        .models()
        .into_iter()
        .find(|row| row.profile_name == "openai" && row.request_model == model)
        .ok_or("model is absent from the catalog")?;
    println!("features: {:?}", listing.info.features);
    let context = PricingContext {
        service_tier: Some(ServiceTier::Fast),
        input_tokens: Some(1000),
        ..Default::default()
    };
    let quote = client.price_quote(model, Some("openai"), &context)?;
    println!("price: {quote:?}");
    if listing.info.features.fast != CapabilitySupport::Supported
        || quote.status != PriceStatus::Priced
    {
        return Err("fast support or its price is unknown".into());
    }
    let route = client.resolve_in(model, Some("openai"))?;
    let assumed_usage = Usage {
        input_tokens: 1000,
        output_tokens: 500,
        ..Default::default()
    };
    println!(
        "preflight: {:?}",
        client.estimate_cost(&route, &assumed_usage, &context)?
    );

    let mut request: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": model,
        "messages": [{"role":"user","content":[{"type":"text","text":"Explain why reasoning effort changes token consumption."}]}],
        "max_tokens": 2048
    }))?;
    request.thinking = Some(ThinkingConfig {
        effort: Some(ReasoningEffort::High),
        ..Default::default()
    });
    request.service_tier = context.service_tier;
    let options = RequestOptions {
        credential: Some(Secret::new(std::env::var("OPENAI_API_KEY")?)),
        ..Default::default()
    };
    let response = client.complete_in("openai", &request, &options).await?;
    println!("{}", response.message.text());
    println!("inference: {:?}", response.inference);
    println!("provider usage and reported cost: {:?}", response.usage);
    println!(
        "local estimate: {:?}",
        client.estimate_actual_cost(&route, &response, Submission::Interactive)?
    );
    Ok(())
}
