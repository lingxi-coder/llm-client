//! A credential value with redacted `Debug` and `Display` output.
//! Deserializable from caller-supplied configuration, never serializable.
//! The caller owns credential storage, lookup, and refresh.

use serde::{Deserialize, Deserializer};
use std::fmt;

#[derive(Clone, PartialEq, Eq)]
pub struct Secret<T>(T);

impl<T> Secret<T> {
    pub fn new(value: T) -> Self {
        Self(value)
    }

    /// The only way to read the value. Named so that call sites are greppable.
    pub fn expose_secret(&self) -> &T {
        &self.0
    }

    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> fmt::Debug for Secret<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

impl<T> fmt::Display for Secret<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("<redacted>")
    }
}

impl<T> From<T> for Secret<T> {
    fn from(value: T) -> Self {
        Self(value)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Secret<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        T::deserialize(d).map(Secret)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_and_display_never_print_the_value() {
        let s = Secret::new(String::from("sk-live-123"));
        assert_eq!(format!("{s:?}"), "<redacted>");
        assert_eq!(format!("{s}"), "<redacted>");
        assert_eq!(format!("{:?}", Some(&s)), "Some(<redacted>)");
        assert_eq!(s.expose_secret(), "sk-live-123");
    }

    #[test]
    fn deserializes_but_has_no_serialize_impl() {
        let s: Secret<String> = serde_json::from_str("\"k\"").unwrap();
        assert_eq!(s.expose_secret(), "k");
        // No `Serialize` impl exists; a struct holding a Secret cannot derive it.
    }
}
