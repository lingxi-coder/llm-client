# Catalog Maintenance and Publishing

[简体中文](maintenance.md)

## Updating the Static Catalog

`data/providers/*.toml` are static snapshots packaged with the crate. `build.rs` only generates a file list from the catalog. This repository does not include the original workspace's `scripts/vendor-catalog.py`; model and price updates must be made directly in these TOML files.

1. Check model IDs, capabilities, context windows, release dates, token rates, and effective dates against each provider's official model catalog and pricing documentation. Record the source links and the date checked. Leave prices that cannot be confirmed unset; do not guess.
2. Preserve the manually maintained connection URLs, protocols, authentication, billing, and provider display information before the first `[[model]]`; update the model blocks after it. Bare keys following a TOML table header belong to that table, so new routing fields must go before the first table header.
3. Check model aliases, search adapters, peak periods, and region-specific connections. Live catalog sync updates only the host's local configuration and does not rewrite the repository's static snapshot; maintaining the static catalog still requires checking upstream sources. Leave missing capability fields as unknown; write an explicit `false` only when a capability is known to be unsupported.
4. Run `cargo test --locked`, `cargo test --manifest-path tests/downstream-docs/Cargo.toml --doc --locked --offline --target-dir target`, `cargo fmt --package lingxi-llm-client -- --check`, `cargo clippy --all-targets --locked -- --no-deps -D warnings`, and `RUSTDOCFLAGS='-D warnings' cargo doc --no-deps --locked`. Clippy lints the SDK package's targets only; the independent downstream project compiles both languages' examples with only the documented direct dependencies. Also check price changes manually against provider documentation; tests cannot establish that live rates are still valid.

First-party Anthropic preset IDs use official wire IDs; former aggregator names remain local `aliases`. Check the [official ID rules](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions) and [retirement table](https://platform.claude.com/docs/en/about-claude/model-deprecations) during refreshes. Stripping an aggregator namespace does not produce a first-party model ID. On 2026-09-23, retired first-party Haiku 3, Opus 4/4.1, and Sonnet 4 entries were removed; aggregator entries were preserved.

Architecture changes also require `cargo test --all-features --locked`, independent checks for each `tokenizer-*` feature, and SDK-package Clippy/Rustdoc for default and all features. Clippy uses `--no-deps`; the workspace package gate verifies the vendored transport archives under the SDK-selected feature closure. `tests/downstream-docs` verifies both language guides and externally implemented Transport, WireCodec, StreamDecoder and AccountUsageSource traits. Run `cargo tree --no-default-features --edges normal` to check default dependency isolation. The [validation record](architecture-validation.md) gives the benchmark workload and commands.

Preserve model `features` and verified `pricing.rules` during catalog refreshes. Effort is not a unit-price dimension. Fast uses only model-specific published multipliers or fixed rates; preserve affected buckets, currency, context bands, validity dates, sources and verification dates. Leave unverified fast prices unset.

## Publishing 0.2.0

The public client API is in `lingxi-llm-client`; four renamed transport forks are separate workspace packages used to verify the local dependency closure. CI runs Clippy on all SDK targets with `--no-deps`; the workspace package gate separately builds and verifies all five archives, without applying the SDK's lint policy to vendored upstream sources. Local workspace packaging does not establish that crates.io serves those fork versions. Publishing the client requires the four matching fork versions to be available from crates.io first, plus permission and network access. There is no automated publishing workflow.

1. Run `cargo package --workspace --list --locked` and inspect all five package contents. Protocol sources, guides, provider presets, and bundled tokenizer assets must be included; local runtime state and build artifacts must be excluded.
2. Run `cargo package --workspace --locked --no-default-features --features 'lingxi-reqwest/rustls-tls,lingxi-reqwest/stream,lingxi-reqwest/json,lingxi-reqwest/http1-header-layout,lingxi-reqwest/gzip,lingxi-reqwest/brotli,lingxi-reqwest/zstd,lingxi-reqwest/deflate'` to create and verify each archive under the SDK's selected reqwest feature set. This is local package verification, not a registry availability check. The vendored reqwest fork's upstream native-TLS default is not usable with the renamed Hyper dependency graph and is not covered by this gate. Local verification of intentionally uncommitted changes can add `--allow-dirty`; release verification uses a clean checkout.
3. Before publishing `lingxi-llm-client`, make the matching transport fork versions available from crates.io and confirm registry resolution. Then run `cargo publish --dry-run --locked` and complete release review. Publish only after that review.
4. If live acceptance testing is needed, use a test account to check real provider requests and the HTTPS trust chain separately. Offline tests and local HTTP tests do not cover either of these.

These commands describe a manual release; maintenance and CI do not publish automatically.
