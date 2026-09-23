# Catalog Maintenance and Publishing

[简体中文](maintenance.md)

## Updating the Static Catalog

`data/providers/*.toml` are static snapshots packaged with the crate. `build.rs` only generates a file list from the catalog. This repository does not include the original workspace's `scripts/vendor-catalog.py`; model and price updates must be made directly in these TOML files.

1. Check model IDs, capabilities, context windows, release dates, token rates, and effective dates against each provider's official model catalog and pricing documentation. Record the source links and the date checked. Leave prices that cannot be confirmed unset; do not guess.
2. Preserve the manually maintained connection URLs, protocols, authentication, billing, and provider display information before the first `[[model]]`; update the model blocks after it. Bare keys following a TOML table header belong to that table, so new routing fields must go before the first table header.
3. Check model aliases, search adapters, peak periods, and region-specific connections. Live catalog sync updates only the host's local configuration and does not rewrite the repository's static snapshot; maintaining the static catalog still requires checking upstream sources. Leave missing capability fields as unknown; write an explicit `false` only when a capability is known to be unsupported.
4. Run `cargo test --workspace --locked`, `cargo test -p lingxi-agent-api --no-default-features --locked`, `cargo check -p lingxi-llm-client --locked`, `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets --locked -- -D warnings`, and `RUSTDOCFLAGS='-D warnings' cargo doc --workspace --no-deps --locked`. Check the minimal feature set separately so workspace default feature unification does not hide problems in LLM-only builds. Also check price changes manually against current provider documentation; tests cannot establish that live rates are still valid.

First-party Anthropic preset IDs use official wire IDs; former aggregator names remain local `aliases`. Check the [official ID rules](https://platform.claude.com/docs/en/about-claude/models/model-ids-and-versions) and [retirement table](https://platform.claude.com/docs/en/about-claude/model-deprecations) during refreshes. Stripping an aggregator namespace does not produce a first-party model ID. On 2026-09-23, retired first-party Haiku 3, Opus 4/4.1, and Sonnet 4 entries were removed; aggregator entries were preserved.

## Publishing 0.1.0

The two crates share version `0.1.0`; `lingxi-llm-client` depends on the same version of `lingxi-agent-api` from crates.io. Publishing requires permission to publish both crate names and a working network connection to crates.io. Before publishing, check `Cargo.lock`, licenses, the README, crate archive contents, and the CI results above. This repository has no automated publishing workflow.

1. In the publishing environment, run `cargo package -p lingxi-agent-api --locked` and `cargo package -p lingxi-llm-client --list --locked`, then inspect the package contents. The second command only lists files and cannot yet verify resolution of the shared crate from the registry.
2. First run `cargo publish -p lingxi-agent-api --dry-run --locked`; after confirming the result, publish with `cargo publish -p lingxi-agent-api --locked`.
3. Wait until the crates.io index can resolve `lingxi-agent-api = "0.1.0"`, then run `cargo package -p lingxi-llm-client --locked` and `cargo publish -p lingxi-llm-client --dry-run --locked`; after confirming the result, publish with `cargo publish -p lingxi-llm-client --locked`.
4. If live acceptance testing is needed, use a test account to check real provider requests and the HTTPS trust chain separately. Offline tests and local HTTP tests do not cover either of these.

The commands above describe the publishing steps; maintenance and CI do not run `cargo publish` automatically.
