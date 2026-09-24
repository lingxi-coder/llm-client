//! Arithmetic over a selected price quote. All public estimates select rates
//! through the client price query before reaching this module.

use super::route::PricingModelRef;
use crate::protocol::{LlmError, PriceQuote, Submission, Usage};

/// Catalog estimate of token charges in the stated currency; no FX conversion.
#[derive(Debug, Clone, PartialEq)]
pub struct CostEstimate {
    pub currency: String,
    /// Pricing assumptions or the confirmed service tier; effort is not a rate key.
    pub context: crate::protocol::PricingContext,
    pub pricing_model: PricingModelRef,
    pub submission: Submission,
    pub input_cost: f64,
    pub output_cost: f64,
    pub cache_read_cost: f64,
    pub cache_write_cost: f64,
    pub reasoning_cost: f64,
    pub total_cost: f64,
    /// The catalog's provenance for the rates, when it records one.
    pub source: Option<String>,
}

/// Compute token charges only after the pricing engine selects a quote.
pub(super) fn estimate(
    quote: &PriceQuote,
    usage: &Usage,
    pricing_model: &PricingModelRef,
) -> Result<CostEstimate, LlmError> {
    let rates = quote
        .rates
        .as_ref()
        .ok_or_else(|| LlmError::CostUnavailable {
            message: quote
                .reason
                .clone()
                .unwrap_or_else(|| "this quote has no monetary token rates".into()),
        })?;
    if usage.cache_write_1h_tokens > usage.cache_write_tokens {
        return Err(LlmError::CostUnavailable {
            message: "one-hour cache-write tokens exceed total cache writes".to_owned(),
        });
    }
    if usage.reasoning_tokens > usage.output_tokens {
        return Err(LlmError::CostUnavailable {
            message: "reasoning tokens exceed output tokens".to_owned(),
        });
    }
    let model = pricing_model.billing_model.as_str();
    let input_cost = bucket(usage.input_tokens, rates.input_per_million, "input", model)?;
    let cache_read_cost = bucket(
        usage.cache_read_tokens,
        rates.cache_read_per_million,
        "cache_read",
        model,
    )?;
    let cache_write_cost = bucket(
        usage.cache_write_tokens - usage.cache_write_1h_tokens,
        rates.cache_write_per_million,
        "cache_write",
        model,
    )? + bucket(
        usage.cache_write_1h_tokens,
        rates.cache_write_1h_per_million,
        "cache_write_1h",
        model,
    )?;
    // Reasoning is a subset of output, never an addend: it is carved out of
    // the output bucket only when the catalog prices it on its own.
    let (output_cost, reasoning_cost) = match rates.reasoning_per_million {
        Some(rate) => {
            let reasoning = usage.reasoning_tokens;
            (
                bucket(
                    usage.output_tokens - reasoning,
                    rates.output_per_million,
                    "output",
                    model,
                )?,
                bucket(reasoning, Some(rate), "reasoning", model)?,
            )
        }
        None => (
            bucket(
                usage.output_tokens,
                rates.output_per_million,
                "output",
                model,
            )?,
            0.0,
        ),
    };
    let total_cost = input_cost + output_cost + cache_read_cost + cache_write_cost + reasoning_cost;
    if !total_cost.is_finite() {
        return Err(LlmError::CostUnavailable {
            message: format!("{model:?} cost exceeds the supported numeric range"),
        });
    }
    Ok(CostEstimate {
        currency: quote.currency.clone(),
        context: quote.context.clone(),
        pricing_model: pricing_model.clone(),
        submission: quote.context.submission,
        input_cost,
        output_cost,
        cache_read_cost,
        cache_write_cost,
        reasoning_cost,
        total_cost,
        source: quote.source.clone(),
    })
}

fn bucket(tokens: u64, rate: Option<f64>, name: &str, model: &str) -> Result<f64, LlmError> {
    if tokens == 0 {
        return Ok(0.0);
    }
    match rate {
        Some(rate) => {
            let cost = price(tokens, rate);
            if !rate.is_finite() || rate < 0.0 || !cost.is_finite() {
                return Err(LlmError::CostUnavailable {
                    message: format!("{model:?} has an invalid or overflowing {name} rate"),
                });
            }
            Ok(cost)
        }
        None => Err(LlmError::CostUnavailable {
            message: format!("{model:?} has no {name} rate for {tokens} {name} tokens"),
        }),
    }
}

#[allow(clippy::cast_precision_loss)]
fn price(tokens: u64, per_million: f64) -> f64 {
    (tokens as f64 / 1_000_000.0) * per_million
}
