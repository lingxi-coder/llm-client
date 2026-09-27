# Gemini Chat audio input

`ContentBlock::Audio { format, data }` accepts user-message audio on Gemini GenerateContent and Vertex Gemini, encoded as `inlineData`. Supply non-empty standard Base64 without a data URI. Supported format declarations are wav, mp3, aiff, aac, ogg, flac, mpeg, m4a, l16, opus, alaw, mulaw, and webm. The codec validates format and encoding; it does not decode media or infer model/account access.

```rust,no_run
use lingxi_llm_client::protocol::ContentBlock;
let audio = ContentBlock::Audio {
    format: "wav".into(),
    data: "UklGRg==".into(), // Placeholder: replace with a complete audio file.
};
```

This is audio input; output follows the Chat request. OpenRouter-specific audio output settings are rejected on Gemini. The complete request containing this block has a 20 MB limit. Larger inputs can use the existing Files/Document provider-file path. The provider validates duration, model support and media contents.

Sources: [Google audio understanding](https://ai.google.dev/gemini-api/docs/generate-content/audio), [Vertex audio understanding](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/audio-understanding). Local codec and pre-dispatch regression tests pass; no live acceptance was performed.

For requests mixing typed inline audio with automatic attachments, the client checks the complete inline representation before uploading any attachment. If that representation exceeds 20 MB, explicitly upload the larger files first and pass their scoped file references. The final encoded request is checked again after file preparation.
