# xAI custom voices

`XaiAudioService` supports creating, listing pages of, reading, updating, and deleting xAI custom voices, plus streaming their original reference audio. These are team-owned resources. Each [`XaiCustomVoiceRef`] is bound to its profile, API endpoint, and `account_scope`, so a reference from another connection cannot be used accidentally. Pass `XaiAudioCredentials` on every call.

```rust,no_run
use lingxi_llm_client::{
    audio::AudioInput,
    xai_audio::{
        XaiAudioCredentials, XaiAudioService, XaiCustomVoiceAge,
        XaiCustomVoiceCreateRequest, XaiCustomVoiceGender,
        XaiCustomVoiceListRequest, XaiCustomVoicePatch, XaiCustomVoiceTone,
        XaiCustomVoiceUseCase,
    },
};

async fn manage_voice(
    service: &XaiAudioService<'_>,
    credentials: &XaiAudioCredentials,
) -> Result<(), Box<dyn std::error::Error>> {
    let audio = AudioInput::from_bytes(
        "reference.wav",
        "audio/wav",
        b"caller-owned audio bytes".to_vec(),
    );
    let request = XaiCustomVoiceCreateRequest::new(audio)
        .with_duration_seconds(90.0)
        .with_name("Friendly Narrator")
        .with_gender(XaiCustomVoiceGender::Female)
        .with_age(XaiCustomVoiceAge::Young)
        .with_language("en-US")
        .with_use_case(XaiCustomVoiceUseCase::Narration)
        .with_tone(XaiCustomVoiceTone::Warm);
    let created = service.create_custom_voice(request, credentials).await?;
    let reference = created.reference.clone();

    let first_page = service
        .list_custom_voices(&XaiCustomVoiceListRequest::new().with_limit(50), credentials)
        .await?;
    if let Some(cursor) = first_page.next_page {
        let _next_page = service
            .list_custom_voices(&XaiCustomVoiceListRequest::new().after(cursor), credentials)
            .await?;
    }

    let _current = service.get_custom_voice(&reference, credentials).await?;
    let patch = XaiCustomVoicePatch::new()
        .clear_description()
        .with_tone(XaiCustomVoiceTone::Calm);
    let _updated = service
        .update_custom_voice(&reference, &patch, credentials)
        .await?;

    // max_bytes is a caller-side bound. The stream is not written to disk.
    let mut audio = service
        .get_custom_voice_audio(&reference, 100_000_000, credentials)
        .await?;
    while let Some(chunk) = audio.next_chunk().await? {
        let _ = chunk;
    }

    let _receipt = service.delete_custom_voice(&reference, credentials).await?;
    Ok(())
}
```

Creation sends the required `file` and optional `name`, `description`, `gender`, `accent`, `age`, `language`, `use_case`, and `tone` fields as multipart form data. xAI lists WAV, MP3, FLAC, OGG, Opus, M4A, AAC, MKV, and MP4 as accepted formats and recommends WAV. The request sends the MIME type supplied by the caller. The official guide caps the reference clip at 120 seconds but publishes no byte-size limit. `with_duration_seconds` only validates caller-declared duration; it does not decode or measure media. If omitted, xAI validates the clip. The client does not guess MIME from a filename extension or apply the STT upload limit to custom voices.

`XaiCustomVoicePatch::with_*` sets a nonempty value; `clear_*` sends JSON `null` to clear metadata. Unset fields are omitted, and empty strings are rejected locally. Responses retain typed fields and the complete `native` JSON so callers can inspect provider fields added later.

`get_custom_voice_audio` explicitly requests `/audio` and returns a bounded stream. The caller supplies a positive `max_bytes`; the stream errors if a later chunk would exceed that bound and reports how many bytes it already delivered. It preserves the actual response `Content-Type`; it does not cache, save to disk, download automatically, or retry. xAI publishes no byte-size limit for this download route.

API creation requires an Enterprise plan. xAI currently says Custom Voices is available in the United States except Illinois and limits each team to 30 voices. The client does not infer entitlement from local region or plan; xAI checks access server-side. The guide documents no consent field or consent endpoint, so callers must obtain any necessary rights before submitting a reference recording. If a create, update, or delete loses its connection after dispatch, or a successful HTTP response does not contain a usable acknowledgement, the client returns `XaiAudioError::OutcomeUnknown` and does not retry. For an unusable success response, `response` retains its request ID and original body.

Source: [xAI Custom Voices documentation](https://docs.x.ai/developers/model-capabilities/audio/custom-voices).

After an error or EOF, the download wrapper immediately releases the underlying HTTP body even if the caller retains the wrapper.
