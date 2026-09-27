# MiniMax Voice Lifecycle

`minimax_voices` wraps MiniMax's native voice endpoints: cloning (`POST /v1/voice_clone`), voice design (`POST /v1/voice_design`), categorized listing (`POST /v1/get_voice`), and deletion (`POST /v1/delete_voice`). Results retain the provider's native JSON, preview URL or hex data, and request ID. The service does not download previews, activate voices, poll, or retry write operations.

```rust,no_run
# async fn example(api_key: String) -> Result<(), Box<dyn std::error::Error>> {
use lingxi_llm_client::{
    minimax_voices::{
        MiniMaxVoiceDesignRequest, MiniMaxVoiceListRequest, MiniMaxVoicesConfig,
        MiniMaxVoicesCredentials, MiniMaxVoicesRegion, MiniMaxVoicesService,
    },
    protocol::Secret,
    HttpTransport,
};

let transport = HttpTransport::new()?;
let service = MiniMaxVoicesService::new(
    &transport,
    MiniMaxVoicesConfig::new(
        "minimax-voice-profile",
        "team/account-17",
        MiniMaxVoicesRegion::ChinaMainland,
    ),
)?;
let credentials = MiniMaxVoicesCredentials::new(Secret::new(api_key));
let catalog = service
    .list_voices(&MiniMaxVoiceListRequest::default(), &credentials)
    .await?;

// Voice design uses the supplied text to generate a billed preview.
let paid_design = MiniMaxVoiceDesignRequest::new(
    "A calm, clear Mandarin narrator",
    "This text will be used to generate the paid preview audio.",
)
.with_aigc_watermark(false);
let designed = service.design_voice(&paid_design, &credentials).await?;
let _ = (catalog, designed.reference, designed.trial_audio);
# Ok(())
# }
```

Clone an uploaded recording without a preview, or explicitly delete a selected voice:

```rust,no_run
use lingxi_llm_client::{
    files::ProviderFileRef,
    minimax_voices::{
        MiniMaxVoiceCloneRequest, MiniMaxVoiceRef,
        MiniMaxVoicesCredentials, MiniMaxVoicesError, MiniMaxVoicesService,
    },
};

async fn clone_uploaded_audio(
    service: &MiniMaxVoicesService<'_>,
    credentials: &MiniMaxVoicesCredentials,
    audio: ProviderFileRef,
    new_voice_id: String,
) -> Result<MiniMaxVoiceRef, MiniMaxVoicesError> {
    let request = MiniMaxVoiceCloneRequest::new(audio, new_voice_id);
    Ok(service.clone_voice(&request, credentials).await?.reference)
}

async fn delete_selected_voice(
    service: &MiniMaxVoicesService<'_>,
    credentials: &MiniMaxVoicesCredentials,
    voice: &MiniMaxVoiceRef,
) -> Result<(), MiniMaxVoicesError> {
    service.delete_voice(voice, credentials).await?;
    Ok(())
}
```

`MiniMaxVoicesConfig` requires a stable profile name, a caller-defined account-scope label, and an explicit `International` or `ChinaMainland` region. The default roots are `https://api.minimax.io/v1` and `https://api.minimax.cn/v1`. The account scope is a routing label, not proof of which account owns a key. Supply the matching account's `MiniMaxVoicesCredentials` on every operation. MiniMax checks regional authentication, plan, and entitlement on its servers.

File inputs use `ProviderFileRef`; before sending, the service checks provider, profile, account scope, regional endpoint, and upload purpose. Clone recordings use purpose `voice_clone`; prompt recordings use `prompt_audio`. For a file reference to match, configure the `FileService` profile `base_url` with the same `/v1` root as the voice service, such as `https://api.minimax.cn/v1`. The service does not rewrite endpoints or rebind file references. MiniMax's upload docs specify MP3, M4A, or WAV files up to 20,000,000 bytes; clone recordings are 10 seconds to 5 minutes, and prompt recordings are shorter than 8 seconds. This service does not decode media. Optional duration values are caller declarations used only for local range checks.

For cloning, `voice_id` must satisfy MiniMax's rules for a new clone ID and be unique within the account. `MiniMaxVoiceClonePrompt` pairs a prompt recording with its transcript. A clone preview is opt-in: the client sends it only when both text and a `MiniMaxTtsModel` are explicitly set. Preview text is limited to 1,000 characters and may incur a charge. `accuracy` can only be set together with `text_validation`. If a successful clone response omits `voice_id`, the service creates its `MiniMaxVoiceRef` with the ID from the request and retains the original response. This module does not activate a new clone; MiniMax documents that an unused clone may be deleted after seven days.

Voice design requires caller-supplied `preview_text`, up to 500 characters, and generating the preview is billed. The returned `trial_audio` remains the original hex string; the service neither decodes nor downloads it. The optional design `voice_id` does not use the clone endpoint's new-ID rules. The Mainland API also documents the boolean `aigc_watermark`, which adds an audio rhythm marker to the preview's end and defaults to `false`. The client sends it only when `.with_aigc_watermark(...)` is explicitly used. The International docs do not document this field, so setting it for that region is rejected before dispatch.

Listing supports system, cloned, generated, and all categories. System voice IDs may contain spaces and parentheses, so they are not checked with the new-clone ID rules. Results include typed categories and the full native JSON. `delete_voice` accepts only a same-scope `MiniMaxVoiceRef` of kind `Cloned` or `Generated`; system voices cannot be deleted, and deleted IDs cannot be reused. Cloned and generated voices appear in their categories only after they have been used successfully for synthesis.

Clone, design, and delete are mutations. If dispatch is followed by a connection failure, timeout, or an unconfirmed success response, the error reports an unknown outcome; the service does not retry. Check `MiniMaxVoicesError::dispatch()` and avoid blindly repeating operations that may have side effects.

Official contract: [International voice clone](https://platform.minimax.io/docs/api-reference/voice-cloning-clone), [International voice design](https://platform.minimax.io/docs/api-reference/voice-design-design), [International list voices](https://platform.minimax.io/docs/api-reference/voice-management-get), and [International delete voice](https://platform.minimax.io/docs/api-reference/voice-management-delete). The Mainland design field is documented in [Mainland voice design](https://platform.minimax.cn/docs/api-reference/voice-design-design); the other Mainland routes and fields are documented under [clone](https://platform.minimax.cn/docs/api-reference/voice-cloning-clone), [list](https://platform.minimax.cn/docs/api-reference/voice-management-get), and [delete](https://platform.minimax.cn/docs/api-reference/voice-management-delete).
