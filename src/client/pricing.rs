//! Usage × the catalog's prices → a cost. Ported from the previous project's
//! `cost.rs` with its catalog folded away: the rates live on the model the
//! route resolved to, so there is nothing to look up by key.
//!
//! Every bucket is priced from its own rate. A batch submission takes each
//! bucket's batch rate where the vendor published one — one vendor halves
//! input and output but not the cache read, and an aggregator's batch rate can
//! be *dearer* than its interactive one (two sellers, not a discount) — so
//! nothing here assumes a direction or a multiplier. A bucket that carries
//! tokens but no rate makes the estimate unavailable rather than silently
//! zero: a number the catalog did not publish is a guess, and a guess is not a
//! bill.

use super::route::PricingModelRef;
use lingxi_agent_api::protocol::{LlmError, PeakSchedule, Submission, TokenPricing, Usage};

/// What one finished request cost, by bucket, in USD.
#[derive(Debug, Clone, PartialEq)]
pub struct CostEstimate {
    pub pricing_model: PricingModelRef,
    pub submission: Submission,
    pub input_usd: f64,
    pub output_usd: f64,
    pub cache_read_usd: f64,
    pub cache_write_usd: f64,
    pub reasoning_usd: f64,
    pub total_usd: f64,
    /// The catalog's provenance for the rates, when it records one.
    pub source: Option<String>,
}

/// Price `usage` at the rates in force for `submission` at `unix_seconds`.
pub fn estimate(
    pricing: &TokenPricing,
    submission: Submission,
    schedule: Option<&PeakSchedule>,
    unix_seconds: u64,
    usage: &Usage,
    pricing_model: &PricingModelRef,
) -> Result<CostEstimate, LlmError> {
    let rates = pricing.at(submission, schedule, unix_seconds);
    let model = pricing_model.billing_model.as_str();
    let input_usd = bucket(usage.input_tokens, rates.input_per_million, "input", model)?;
    let cache_read_usd = bucket(
        usage.cache_read_tokens,
        rates.cache_read_per_million,
        "cache_read",
        model,
    )?;
    let cache_write_usd = bucket(
        usage.cache_write_tokens,
        rates.cache_write_per_million,
        "cache_write",
        model,
    )?;
    // Reasoning is a subset of output, never an addend: it is carved out of
    // the output bucket only when the catalog prices it on its own.
    let (output_usd, reasoning_usd) = match rates.reasoning_per_million {
        Some(rate) => {
            let reasoning = usage.reasoning_tokens.min(usage.output_tokens);
            (
                bucket(
                    usage.output_tokens - reasoning,
                    rates.output_per_million,
                    "output",
                    model,
                )?,
                price(reasoning, rate),
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
    Ok(CostEstimate {
        pricing_model: pricing_model.clone(),
        submission,
        input_usd,
        output_usd,
        cache_read_usd,
        cache_write_usd,
        reasoning_usd,
        total_usd: input_usd + output_usd + cache_read_usd + cache_write_usd + reasoning_usd,
        source: rates.source,
    })
}

fn bucket(tokens: u64, rate: Option<f64>, name: &str, model: &str) -> Result<f64, LlmError> {
    if tokens == 0 {
        return Ok(0.0);
    }
    match rate {
        Some(rate) => Ok(price(tokens, rate)),
        None => Err(LlmError::CostUnavailable {
            message: format!("{model:?} has no {name} rate for {tokens} {name} tokens"),
        }),
    }
}

#[allow(clippy::cast_precision_loss)]
fn price(tokens: u64, per_million: f64) -> f64 {
    (tokens as f64 / 1_000_000.0) * per_million
}

#[cfg(test)]
mod tests {
    use super::*;
    use lingxi_agent_api::protocol::{BatchPricing, ProviderId};

    fn model() -> PricingModelRef {
        PricingModelRef {
            pricing_provider_id: ProviderId::new("acme"),
            billing_model: "m".into(),
            request_model: "m".into(),
            display_model: "m".into(),
        }
    }

    fn rates(input: f64, output: f64, cache_read: f64) -> TokenPricing {
        TokenPricing {
            input_per_million: Some(input),
            output_per_million: Some(output),
            cache_read_per_million: Some(cache_read),
            ..TokenPricing::default()
        }
    }

    fn usage(input: u64, output: u64, cache_read: u64) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            cache_read_tokens: cache_read,
            ..Usage::default()
        }
    }

    #[test]
    fn a_batch_rate_is_per_bucket_and_has_no_direction() {
        // Input and output halved, the cache read not: the interactive cache
        // read rate must not leak into the batch estimate, nor be halved.
        let mut p = rates(1.0, 2.0, 0.5);
        p.batch = Some(BatchPricing {
            input_per_million: Some(0.5),
            output_per_million: Some(1.0),
            cache_read_per_million: Some(0.5),
            ..BatchPricing::default()
        });
        let u = usage(1_000_000, 1_000_000, 1_000_000);
        let live = estimate(&p, Submission::Interactive, None, 0, &u, &model()).unwrap();
        let batch = estimate(&p, Submission::Batch, None, 0, &u, &model()).unwrap();
        assert_eq!(live.total_usd, 3.5);
        assert_eq!(batch.total_usd, 2.0);
        assert_eq!(batch.cache_read_usd, live.cache_read_usd);

        // Dearer as a batch stays dearer.
        p.batch = Some(BatchPricing {
            input_per_million: Some(4.0),
            output_per_million: Some(2.0),
            cache_read_per_million: Some(0.5),
            ..BatchPricing::default()
        });
        let batch = estimate(&p, Submission::Batch, None, 0, &u, &model()).unwrap();
        assert_eq!(batch.input_usd, 4.0, "no direction is assumed");
        assert_eq!(batch.submission, Submission::Batch);
    }

    #[test]
    fn reasoning_is_carved_out_of_output_only_when_priced() {
        let mut p = rates(1.0, 1.0, 0.0);
        let u = Usage {
            output_tokens: 10_000_000,
            reasoning_tokens: 4_000_000,
            ..Usage::default()
        };
        let plain = estimate(&p, Submission::Interactive, None, 0, &u, &model()).unwrap();
        assert_eq!((plain.output_usd, plain.reasoning_usd), (10.0, 0.0));
        p.reasoning_per_million = Some(3.0);
        let split = estimate(&p, Submission::Interactive, None, 0, &u, &model()).unwrap();
        assert_eq!((split.output_usd, split.reasoning_usd), (6.0, 12.0));
        assert_eq!(split.total_usd, 18.0);
    }

    #[test]
    fn a_bucket_with_tokens_and_no_rate_is_unavailable_not_free() {
        let p = TokenPricing {
            input_per_million: Some(1.0),
            output_per_million: Some(1.0),
            ..TokenPricing::default()
        };
        let ok = estimate(
            &p,
            Submission::Interactive,
            None,
            0,
            &usage(5, 5, 0),
            &model(),
        )
        .unwrap();
        assert!(ok.total_usd > 0.0);
        let err = estimate(
            &p,
            Submission::Interactive,
            None,
            0,
            &usage(5, 5, 1),
            &model(),
        )
        .unwrap_err();
        assert!(
            matches!(&err, LlmError::CostUnavailable { message } if message.contains("cache_read")),
            "{err:?}"
        );
    }
}
