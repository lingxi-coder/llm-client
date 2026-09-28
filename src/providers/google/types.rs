//! Provider-specific request, tool, and replay types.

use crate::protocol::LlmError;
use serde::{Deserialize, Serialize};

/// Optional widget-context support for Gemini Google Maps grounding.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiMapsGroundingConfig {
    /// Request the Maps widget context token when the caller renders the
    /// provider's map widget. `None` leaves the API default unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub enable_widget: Option<bool>,
    /// Optional user location supplied to Maps retrieval. The provider API
    /// encodes this at `toolConfig.retrievalConfig.latLng`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lat_lng: Option<GeminiLatLng>,
}

/// Latitude and longitude used as context for Gemini Maps grounding.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeminiLatLng {
    pub latitude: f64,
    pub longitude: f64,
}

impl GeminiLatLng {
    pub fn new(latitude: f64, longitude: f64) -> Result<Self, LlmError> {
        let location = Self {
            latitude,
            longitude,
        };
        location.validate()?;
        Ok(location)
    }

    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        if !self.latitude.is_finite() || !(-90.0..=90.0).contains(&self.latitude) {
            return Err(LlmError::InvalidRequest {
                message: "Gemini Maps latitude must be finite and within -90..=90".into(),
            });
        }
        if !self.longitude.is_finite() || !(-180.0..=180.0).contains(&self.longitude) {
            return Err(LlmError::InvalidRequest {
                message: "Gemini Maps longitude must be finite and within -180..=180".into(),
            });
        }
        Ok(())
    }
}

impl PartialEq for GeminiLatLng {
    fn eq(&self, other: &Self) -> bool {
        self.latitude.to_bits() == other.latitude.to_bits()
            && self.longitude.to_bits() == other.longitude.to_bits()
    }
}

impl Eq for GeminiLatLng {}

impl GeminiMapsGroundingConfig {
    pub(crate) fn validate(&self) -> Result<(), LlmError> {
        if let Some(location) = &self.lat_lng {
            location.validate()?;
        }
        Ok(())
    }
}
