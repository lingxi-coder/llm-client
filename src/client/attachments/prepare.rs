//! Execute a validated plan and own one cleanup lease per attempt.
use super::*;
impl AttachmentManager {
    pub(crate) async fn prepare_provider_file_inputs<'a>(
        &self,
        profile: &ProviderProfile,
        model: &str,
        request: &'a CompletionRequest,
        attachments: &[ResolvedAttachmentPayload],
        preparation: ProviderFilePreparation<'_>,
        validate: impl FnOnce(&[crate::codecs::ContentBinding<'a>]) -> Result<(), LlmError>,
    ) -> Result<PreparedAttachments<'a>, LlmError> {
        let ProviderFilePreparation {
            planning_profile,
            opts,
            request_deadline,
            stable_account_scope,
            inline_image_data_budget_bytes,
            authenticator,
            credential,
            automatic_cleanup,
        } = preparation;
        let plan = self.plan_provider_files(
            planning_profile,
            model,
            request,
            attachments,
            opts,
            inline_image_data_budget_bytes,
        )?;
        let projected = plan
            .uploads
            .iter()
            .map(|upload| {
                let file = files::projected_reference(
                    profile,
                    opts.file_account_scope.as_deref(),
                    &upload.payload.attachment.media_type,
                    upload.purpose,
                );
                provider_file_binding(request, upload.payload, file)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if !plan.uploads.is_empty() {
            validate(&projected)?;
        }
        let mut prepared_file_uses = Vec::new();
        let mut bindings = Vec::new();
        let mut local_files = BTreeMap::<(String, String, String), files::ProviderFileRef>::new();
        for upload in plan.uploads {
            let payload = upload.payload;
            let purpose = upload.purpose;
            let service = files::FileService::new(
                self.http.as_ref(),
                profile,
                authenticator,
                credential,
                opts.file_account_scope.as_deref(),
            );
            let service = if profile.protocol == ProtocolFamily::GeminiGenerateContent {
                if let Some(deadline) = request_deadline {
                    service.with_gemini_request_deadline(deadline)
                } else {
                    service
                }
            } else {
                service
            };
            let service = if let Some(cleanup) = automatic_cleanup.as_ref() {
                service.with_qwen_rate_limiter(cleanup.rate_limiter())
            } else {
                service
            };
            let purpose_key = match payload.kind {
                AttachmentKind::Image => "model_input:image",
                AttachmentKind::Document => "model_input:document",
                AttachmentKind::Video => "model_input:video",
            };
            let local_key = (
                payload.attachment.attachment_id.clone(),
                payload.attachment.revision.clone(),
                purpose_key.to_owned(),
            );
            let file = if let Some(file) = local_files.get(&local_key) {
                file.clone()
            } else {
                let cache_key = stable_account_scope
                    .filter(|_| automatic_cleanup.is_none())
                    .map(|scope| FileCacheKey {
                        attachment_id: payload.attachment.attachment_id.clone(),
                        revision: payload.attachment.revision.clone(),
                        filename: payload.attachment.filename.clone(),
                        media_type: payload.attachment.media_type.clone(),
                        profile_name: profile.profile_name.clone(),
                        provider_id: profile.provider_id.clone(),
                        protocol: profile.protocol,
                        base_url: profile.base_url.clone(),
                        account_scope: scope.to_owned(),
                        purpose: purpose_key.to_owned(),
                    });
                let upload_lock = cache_key.as_ref().map(|key| self.upload_lock(key));
                let _upload_guard = match upload_lock.as_ref() {
                    Some(lock) => Some(lock.lock().await),
                    None => None,
                };
                let cached = cache_key.as_ref().and_then(|key| self.cached_file(key));
                let file = if let Some(file) = cached {
                    local_files.insert(local_key.clone(), file.clone());
                    file
                } else {
                    let file = service
                        .upload_automatic(
                            &files::UploadFile {
                                filename: payload.attachment.filename.clone(),
                                media_type: payload.attachment.media_type.clone(),
                                bytes: payload.bytes.clone(),
                            },
                            purpose,
                        )
                        .await?;
                    if let Some(cleanup) = automatic_cleanup.as_ref() {
                        cleanup.track(file.clone());
                        service
                            .wait_for_qwen_file_ready(&file, opts.total_timeout)
                            .await?;
                    }
                    if let Some(cache_key) = cache_key.as_ref() {
                        self.cache_file(
                            cache_key.clone(),
                            file.clone(),
                            files::automatic_file_cache_ttl(profile).or(upload.retention),
                        );
                    }
                    local_files.insert(local_key.clone(), file.clone());
                    file
                };
                prepared_file_uses.push(PreparedProviderFileUse {
                    key: cache_key,
                    file_id: file.file_id.clone(),
                    uri: file.uri.clone(),
                });
                file
            };

            bindings.push(provider_file_binding(
                request,
                payload,
                file.model_reference(),
            )?);
        }
        Ok(PreparedAttachments {
            uses: prepared_file_uses,
            bindings,
            cleanup: automatic_cleanup,
        })
    }
}

impl AttachmentManager {
    pub(crate) fn cleanup_lease(
        &self,
        profile: &ProviderProfile,
        authenticator: Option<Arc<dyn Authenticator>>,
        credential: Option<&crate::protocol::Secret<String>>,
        account_scope: Option<String>,
        stable_scope: Option<&str>,
    ) -> Option<Arc<files::AutomaticFileCleanup>> {
        files::needs_automatic_cleanup(profile).then(|| {
            Arc::new(files::AutomaticFileCleanup::new(
                self.http.clone(),
                profile.clone(),
                authenticator,
                credential.cloned(),
                account_scope,
                self.qwen_file_rate_limiter(profile, stable_scope),
            ))
        })
    }
}
