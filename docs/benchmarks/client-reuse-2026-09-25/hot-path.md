# Text request hot path follow-up

The same release benchmark was run before and after moving first-party endpoint
classification into the published configuration snapshot and encoding a
single-text Chat Completions message directly. Three alternating runs per
variant used 1,024 requests per worker and the same local mock transport. The
figures are medians of each run's p50; they measure preparation up to the mock
transport, not a provider round trip.

| Workers | Before p50 | After p50 | Change |
| ---: | ---: | ---: | ---: |
| 1 | 8.375 µs | 7.625 µs | −9.0% |
| 16 | 9.250 µs | 8.291 µs | −10.4% |
| 64 | 9.125 µs | 8.167 µs | −10.5% |

At 16 workers, the median p95 across runs changed from 22.291 to 19.792 µs.
Preparation allocations fell from 95 to 83 per request and cumulative
requested allocation bytes from 34,354 to 30,726. The benchmark request has
one text block, one configured model and no network, attachments or tools;
other request shapes need their own measurements.
