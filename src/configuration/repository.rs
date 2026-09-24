//! File locking, version validation and durable atomic commits. No service effects.
use super::{model::*, ProviderStoreError};
use crate::protocol::{CredentialConfig, ProviderProfile};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};
const FILE_NAME: &str = "providers.json";
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
#[derive(Clone)]
pub(crate) struct Repository {
    pub path: PathBuf,
}
impl Repository {
    pub fn open(path: &Path) -> Result<Self, ProviderStoreError> {
        fs::create_dir_all(path)?;
        Ok(Self {
            path: path.canonicalize()?,
        })
    }
    pub fn lock(&self) -> Result<File, ProviderStoreError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.path.join(".providers.json.lock"))?;
        file.lock()?;
        Ok(file)
    }
    pub fn read(&self) -> Result<SavedConfig, ProviderStoreError> {
        let bytes = match fs::read(self.path.join(FILE_NAME)) {
            Ok(bytes) => bytes,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(SavedConfig::default()),
            Err(e) => return Err(e.into()),
        };
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        let version = value
            .get("version")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0);
        let result = match version {
            2 => serde_json::from_value(value)?,
            _ => return Err(ProviderStoreError::UnsupportedVersion(version as u32)),
        };
        validate(&result)?;
        Ok(result)
    }
    pub fn read_locked(&self) -> Result<SavedConfig, ProviderStoreError> {
        let _lock = self.lock()?;
        self.read()
    }
    pub fn write(&self, config: &SavedConfig) -> Result<(), ProviderStoreError> {
        validate(config)?;
        let bytes = serde_json::to_vec_pretty(&config.persisted())?;
        let path = self.path.join(FILE_NAME);

        let temp = self.path.join(format!(
            ".{FILE_NAME}.{}.{}.tmp",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        let result = (|| -> Result<(), std::io::Error> {
            let mut file = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temp)?;
            file.write_all(&bytes)?;
            file.sync_all()?;
            fs::rename(&temp, &path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = fs::remove_file(&temp);
        }
        result.map_err(Into::into)
    }
}
pub(crate) fn reject_secrets(profile: &ProviderProfile) -> Result<(), ProviderStoreError> {
    if matches!(profile.credential, CredentialConfig::Static { .. }) {
        return Err(ProviderStoreError::StaticCredential(
            profile.profile_name.clone(),
        ));
    }
    if profile
        .extra
        .get("headers")
        .and_then(serde_json::Value::as_object)
        .is_some_and(|headers| {
            headers
                .keys()
                .any(|name| crate::wire_options::is_credential_header(profile, name))
        })
    {
        return Err(ProviderStoreError::CredentialHeader(
            profile.profile_name.clone(),
        ));
    }
    Ok(())
}
fn validate(config: &SavedConfig) -> Result<(), ProviderStoreError> {
    let mut names = BTreeSet::new();
    for profile in &config.providers {
        if !names.insert(&profile.connection.profile_name) {
            return Err(ProviderStoreError::DuplicateProfile(
                profile.connection.profile_name.clone(),
            ));
        }
        reject_secrets(&profile.connection)?;
        reject_secrets(&profile.fallback)?;
        let mut rows = BTreeSet::new();
        for row in &profile.models {
            if !rows.insert(&row.id) {
                return Err(ProviderStoreError::InvalidModelOverride(
                    "duplicate model row id".into(),
                ));
            }
        }
    }
    Ok(())
}
