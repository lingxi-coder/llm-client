//! Shared monotonic deadlines. Wall-clock policy lives in `Clock`.

use crate::protocol::LlmError;
use std::{future::Future, time::Duration};
use tokio::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Deadline(Option<Instant>);

impl Deadline {
    pub(crate) fn after(timeout: Option<Duration>) -> Self {
        let now = Instant::now();
        Self(timeout.map(|duration| now.checked_add(duration).unwrap_or(now)))
    }

    pub(crate) fn at(instant: Option<std::time::Instant>) -> Self {
        Self(instant.map(Instant::from_std))
    }

    pub(crate) fn remaining(self) -> Result<Option<Duration>, LlmError> {
        self.0
            .map(|end| {
                end.checked_duration_since(Instant::now())
                    .filter(|duration| !duration.is_zero())
                    .ok_or_else(timeout_error)
            })
            .transpose()
    }

    pub(crate) fn cap(self, timeout: Option<Duration>) -> Self {
        let other = Self::after(timeout);
        Self(match (self.0, other.0) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left, right) => left.or(right),
        })
    }

    pub(crate) async fn run<F: Future>(self, future: F) -> Result<F::Output, LlmError> {
        let Some(remaining) = self.remaining()? else {
            return Ok(future.await);
        };
        match futures::future::select(Box::pin(future), Box::pin(delay(remaining))).await {
            futures::future::Either::Left((result, _)) => Ok(result),
            futures::future::Either::Right((_, _)) => Err(timeout_error()),
        }
    }
}

pub(crate) fn timeout_error() -> LlmError {
    LlmError::TransportTimeout {
        message: "request deadline elapsed".into(),
    }
}

/// Also works with an embedding application's non-Tokio executor. Dropping
/// the future interrupts the fallback sleeper instead of leaving it running.
pub(crate) async fn delay(duration: Duration) {
    if tokio::runtime::Handle::try_current().is_ok() {
        tokio::time::sleep(duration).await;
        return;
    }
    let (cancel_send, cancel_receive) = std::sync::mpsc::channel::<()>();
    let (send, receive) = futures::channel::oneshot::channel();
    std::thread::spawn(move || {
        let _ = cancel_receive.recv_timeout(duration);
        let _ = send.send(());
    });
    let _cancel_on_drop = cancel_send;
    let _ = receive.await;
}
