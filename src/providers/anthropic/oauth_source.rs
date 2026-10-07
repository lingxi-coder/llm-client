//! Native vK/rk selection of the access token and its actual scope source.
use std::borrow::Cow;

#[derive(Default)]
pub struct OAuthSourceInputs<'a> {
    pub environment_token: Option<&'a str>,
    pub environment_scopes: Option<&'a str>,
    pub descriptor_token: Option<&'a str>,
    pub descriptor_scopes: Option<&'a [String]>,
    pub from_background_snapshot: bool,
    pub host_managed: bool,
    pub stored_token: Option<&'a str>,
    pub stored_scopes: Option<&'a [String]>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OAuthSource {
    Environment,
    Descriptor,
    Store,
}
/// No Debug or serialization surface exposes the borrowed token.
pub struct OAuthSelection<'a> {
    pub access_token: &'a str,
    pub scopes: Cow<'a, [String]>,
    pub source: OAuthSource,
}
fn whitespace(c: char) -> bool {
    matches!(c,'\u{0009}'..='\u{000D}'|'\u{0020}'|'\u{00A0}'|'\u{1680}'|'\u{2000}'..='\u{200A}'|'\u{2028}'|'\u{2029}'|'\u{202F}'|'\u{205F}'|'\u{3000}'|'\u{FEFF}')
}
fn scopes(value: Option<&str>, fallback: &[&str]) -> Vec<String> {
    let parsed: Vec<String> = value
        .into_iter()
        .flat_map(|value| value.split(whitespace))
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect();
    if parsed.is_empty() {
        fallback.iter().map(|value| (*value).into()).collect()
    } else {
        parsed
    }
}
/// Parsed environment token first, then descriptor/store precedence from vK.
/// This selects supplied snapshots; it performs no descriptor or store I/O.
pub fn select_oauth_source(input: OAuthSourceInputs<'_>) -> Option<OAuthSelection<'_>> {
    if let Some(token) = input
        .environment_token
        .map(|value| value.trim_matches(whitespace))
        .filter(|value| !value.is_empty())
    {
        return Some(OAuthSelection {
            access_token: token,
            scopes: Cow::Owned(scopes(input.environment_scopes, &["user:inference"])),
            source: OAuthSource::Environment,
        });
    }
    let descriptor = input.descriptor_token.filter(|value| !value.is_empty());
    let from_descriptor = || {
        descriptor.map(|token| OAuthSelection {
            access_token: token,
            scopes: input
                .descriptor_scopes
                .map(Cow::Borrowed)
                .unwrap_or_else(|| {
                    Cow::Owned(scopes(
                        input.environment_scopes,
                        &["user:inference", "user:ccr_inference", "user:file_upload"],
                    ))
                }),
            source: OAuthSource::Descriptor,
        })
    };
    if descriptor.is_some() && (!input.from_background_snapshot || input.host_managed) {
        return from_descriptor();
    }
    if input.host_managed {
        return None;
    }
    if let Some(token) = input.stored_token.filter(|value| !value.is_empty()) {
        return Some(OAuthSelection {
            access_token: token,
            scopes: Cow::Borrowed(input.stored_scopes.unwrap_or_default()),
            source: OAuthSource::Store,
        });
    }
    from_descriptor()
}
