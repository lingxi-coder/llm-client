//! Point-in-time monitoring queries for a scoped Qwen Model Studio knowledge base.

use super::*;

const MONITOR_PATH: &str = "/api/v1/indices/rag/index/monitor";
const MAX_MONITORING_RANGE_SECONDS: i128 = 30 * 24 * 60 * 60;
const MONITOR_OPERATION: &str = "get_knowledge_base_monitoring";

/// A monitoring window expressed as Unix timestamps in seconds.
///
/// The API accepts string or integer timestamps and recommends strings. The
/// service serializes these signed integer seconds as decimal strings and
/// limits the requested window to the documented maximum of 30 days.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QwenKnowledgeMonitoringRequest {
    start_timestamp_secs: i64,
    end_timestamp_secs: i64,
}

impl QwenKnowledgeMonitoringRequest {
    pub fn new(start_timestamp_secs: i64, end_timestamp_secs: i64) -> Self {
        Self {
            start_timestamp_secs,
            end_timestamp_secs,
        }
    }

    pub fn start_timestamp_secs(&self) -> i64 {
        self.start_timestamp_secs
    }

    pub fn end_timestamp_secs(&self) -> i64 {
        self.end_timestamp_secs
    }

    fn validate(&self) -> Result<(), QwenKnowledgeError> {
        let range_seconds =
            i128::from(self.end_timestamp_secs) - i128::from(self.start_timestamp_secs);
        if range_seconds < 0 {
            return Err(invalid(
                "Qwen knowledge monitoring end timestamp must not precede the start timestamp",
            ));
        }
        if range_seconds > MAX_MONITORING_RANGE_SECONDS {
            return Err(invalid(
                "Qwen knowledge monitoring range must not exceed 30 days",
            ));
        }
        Ok(())
    }
}

/// Monitoring data returned by Model Studio.
///
/// `data` and `native` intentionally retain the provider values. The current
/// documentation's example represents `storageMonitorData` as an array while
/// its field table describes it as an object, so this client does not impose a
/// guessed nested schema.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QwenKnowledgeMonitoringResult {
    pub knowledge: QwenKnowledgeRef,
    pub data: Value,
    pub request_id: Option<String>,
    pub native: Value,
}

impl<'a> QwenKnowledgeService<'a> {
    /// Fetch storage and QPS monitoring data for one knowledge base.
    ///
    /// This sends one request; time-window scheduling and repeated polling are
    /// caller-managed.
    pub async fn get_knowledge_base_monitoring(
        &self,
        knowledge: &QwenKnowledgeRef,
        request: &QwenKnowledgeMonitoringRequest,
    ) -> Result<QwenKnowledgeMonitoringResult, QwenKnowledgeError> {
        self.validate_knowledge_ref(knowledge)?;
        request.validate()?;

        let response = self
            .request_json(
                "POST",
                MONITOR_PATH,
                &[],
                Some(json!({
                    "indexId": knowledge.index_id(),
                    "startTimestamp": request.start_timestamp_secs.to_string(),
                    "endTimestamp": request.end_timestamp_secs.to_string(),
                })),
                MONITOR_OPERATION,
                false,
            )
            .await?;
        let data = response
            .native
            .get("data")
            .filter(|data| data.is_object())
            .cloned()
            .ok_or_else(|| {
                response_invalid(
                    MONITOR_OPERATION,
                    "successful response omitted the documented data object",
                    &response,
                    QwenKnowledgeDispatch::NotSent,
                )
            })?;

        Ok(QwenKnowledgeMonitoringResult {
            knowledge: knowledge.clone(),
            data,
            request_id: response.request_id,
            native: response.native,
        })
    }
}
