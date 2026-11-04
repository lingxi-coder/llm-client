//! Reproducible local copy-cost probe; no credentials or network requests.
use lingxi_llm_client::protocol::JsonPayload;
use std::hint::black_box;
use std::time::Instant;

fn main() {
    for bytes in [64 * 1024, 1024 * 1024, 4 * 1024 * 1024] {
        let source = format!(
            "{{\"text\":\"{}\",\"exact\":\"\\ud800\",\"n\":9007199254740993}}",
            "x".repeat(bytes)
        );
        let started = Instant::now();
        let payload = JsonPayload::parse(&source).unwrap();
        let parse = started.elapsed();
        let started = Instant::now();
        let copies: Vec<_> = (0..64).map(|_| black_box(payload.clone())).collect();
        let clone = started.elapsed();
        let copied_source_bytes: usize = copies
            .iter()
            .filter(|copy| copy.as_json_source().as_ptr() != payload.as_json_source().as_ptr())
            .map(|copy| copy.as_json_source().len())
            .sum();
        println!(
            "source_bytes={} clones=64 parse_us={} clone_us={} copied_source_bytes={}",
            source.len(),
            parse.as_micros(),
            clone.as_micros(),
            copied_source_bytes
        );
        black_box(copies);
    }
}
