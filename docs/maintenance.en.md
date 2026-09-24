# Catalog Maintenance and Publishing

[简体中文](maintenance.md)

## Updating the Static Catalog

`data/providers/*.toml` are static snapshots packaged with the crate. `build.rs` only generates a file list from the catalog. This repository does not include the original workspace's `scripts/vendor-catalog.py`; model and price updates must be made directly in these TOML files.

1. Check model IDs, capabilities, context windows, release dates, token rates, and effective dates against each provider's official model catalog and pricing documentation. Record the source links and the date checked. Leave prices that cannot be confirmed unset; do not guess.
2. Preserve the manually maintained connection URLs, protocols, authentication, billing, and provider display information before the first `[[model]]`; update the model blocks after it. Bare keys following a TOML table header belong to that table, so new routing fields must go before the first table header.
3. Check model aliases, search adapters, peak periods, and region-specific connections. Live catalog sync updates only the host's local configuration and does not rewrite the repository's static snapshot; maintaining the static catalog still requires checking upstream sources. Leave missing capability fields as unknown; write an explicit `false` only when a capability is known to be unsupported.
4. Run `cargo test --locked`, `cargo test --manifest-path tests/downstream-docs/Cargo.toml --doc --locked --offline --target-dir target`, `cargo fmt --all -- --check`, `cargo clippy --all-targets --locked -- -D warnings`, and `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --locked`. The independent downstream project compiles both languages' examples with only the documented direct dependencies. Also check price changes manually against provider documentation; tests cannot establish that live rates are still valid.

First-party Anthropic preset IDs use official wire IDs; former aggregator names remain local `aliases`. Check the [official ID rules](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions) and [retirement table](https://platform.claude.com/docs/en/about-claude/model-deprecations) during refreshes. Stripping an aggregator namespace does not produce a first-party model ID. On 2026-09-23, retired first-party Haiku 3, Opus 4/4.1, and Sonnet 4 entries were removed; aggregator entries were preserved.

Architecture changes also require `cargo test --all-features --locked`, independent checks for each `tokenizer-*` feature, and Clippy/Rustdoc for default and all features. `tests/downstream-docs` verifies both language guides and externally implemented Transport, WireCodec, StreamDecoder and AccountUsageSource traits. Run `cargo tree --no-default-features --edges normal` to check default dependency isolation. The [validation record](architecture-validation.md) gives the benchmark workload and commands.

Preserve model `features` and verified `pricing.rules` during catalog refreshes. Effort is not a unit-price dimension. Fast uses only model-specific published multipliers or fixed rates; preserve affected buckets, currency, context bands, validity dates, sources and verification dates. Leave unverified fast prices unset.

## Publishing 0.1.0

This repository publishes one crate, `lingxi-llm-client`, which includes its own protocol types. Publishing requires permission for this crate and network access to crates.io. Before publishing, check the lockfile, licenses, README, archive contents, and CI results. There is no automated publishing workflow.

1. Run `cargo package --list --locked` and inspect the contents. Protocol sources, guides, provider presets, and bundled tokenizer assets must be included; local runtime state and build artifacts must be excluded.
2. Run `cargo package --locked` to verify the complete standalone archive and its build, then `cargo publish --dry-run --locked`. Local verification of intentionally uncommitted changes can add `--allow-dirty` to `cargo package`; release verification uses a clean checkout.
3. After release review, publish with `cargo publish --locked`. No companion protocol crate needs to be published first.
4. If live acceptance testing is needed, use a test account to check real provider requests and the HTTPS trust chain separately. Offline tests and local HTTP tests do not cover either of these.

These commands describe a manual release; maintenance and CI do not publish automatically.
