//! The run process's half of the operator controls: pause, stop and steer.
//!
//! A run executes here, and the operator is at a dashboard in another process. The dashboard holds
//! what they asked for; this asks it, at each node's turn boundaries, what that node should do
//! now. The direction is deliberate and matches the clarification rendezvous next door: the run
//! reaches out, so nothing outside it can reach in — into its worktree, its conversation, or the
//! store only it may write.
//!
//! At ordinary turn boundaries, **every failure means carry on.** An unreachable dashboard, a reply
//! that will not parse, or a run nobody is watching all answer `Continue`. A persisted provider
//! pause is different: it only resumes after an explicit dashboard response, so an unavailable
//! dashboard keeps it paused rather than silently spending more provider quota.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ratatoskr_agent::{
    Controller, PausedPoll, ProviderPauseAcknowledgement, ProviderPauseRegistration,
};
use ratatoskr_core::Control;

/// Ceiling on one control request. Short: this is a loopback call between two local processes, and
/// a node is waiting on it at every turn boundary.
const POLL_TIMEOUT: Duration = Duration::from_secs(5);

/// The durable disposition returned after a provider pause is registered.
#[derive(serde::Deserialize)]
struct ProviderPauseReply {
    directive: ProviderPauseDirective,
    generation: i64,
}

/// The current durable directive returned with a provider-pause acknowledgement.
#[derive(serde::Deserialize)]
struct ProviderPauseAcknowledgementReply {
    directive: ProviderPauseDirective,
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum ProviderPauseDirective {
    Hold,
    Continue,
    Stop,
}

/// A regular control response, optionally carrying a paused waiter's durable generation.
#[derive(serde::Deserialize)]
struct ProviderControlReply {
    #[serde(flatten)]
    control: Control,
    provider_pause_generation: Option<i64>,
}

/// Asks the dashboard that started this run what its nodes should do.
struct DashboardControl {
    dashboard: String,
    project: String,
    run_id: String,
    /// One client, so the connection to the dashboard is kept alive across a run's many polls
    /// rather than rebuilt at every turn boundary.
    client: reqwest::Client,
    /// The newest provider pause this child acknowledged. A fresh provider failure after it must
    /// allocate a new generation, even when another old waiter is still acknowledging.
    known_provider_resume_generation: Mutex<Option<i64>>,
    /// Each waiter needs its exact generation for an idempotent acknowledgement.
    provider_pause_generations: Mutex<HashMap<String, i64>>,
}

impl DashboardControl {
    fn known_provider_resume_generation(&self) -> Option<i64> {
        *self
            .known_provider_resume_generation
            .lock()
            .expect("provider pause generation mutex poisoned")
    }

    fn remember_provider_pause(&self, waiter: &str, generation: i64) {
        self.provider_pause_generations
            .lock()
            .expect("provider pause generation mutex poisoned")
            .insert(waiter.to_string(), generation);
    }

    fn acknowledge_provider_generation(&self, generation: i64) {
        let mut known = self
            .known_provider_resume_generation
            .lock()
            .expect("provider pause generation mutex poisoned");
        if known.is_none_or(|previous| generation > previous) {
            *known = Some(generation);
        }
    }

    async fn request_control(
        &self,
        node: &str,
        provider_pause_waiter: Option<&str>,
    ) -> Result<ProviderControlReply, reqwest::Error> {
        self.client
            .post(format!("{}/internal/control", self.dashboard))
            .json(&serde_json::json!({
                "project": self.project,
                "run_id": self.run_id,
                "node": node,
                "provider_pause_waiter": provider_pause_waiter,
                "known_provider_resume_generation": self.known_provider_resume_generation(),
            }))
            .timeout(POLL_TIMEOUT)
            .send()
            .await?
            .error_for_status()?
            .json::<ProviderControlReply>()
            .await
    }
}

impl Controller for DashboardControl {
    fn poll<'a>(
        &'a self,
        node: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Control> + Send + 'a>> {
        Box::pin(async move {
            self.request_control(node, None)
                .await
                .map(|reply| reply.control)
                .unwrap_or_else(|error| {
                    // Worth a line: a reply this side cannot read means the two processes disagree
                    // about the protocol, and the operator's buttons will look broken.
                    tracing::debug!("could not ask the dashboard for control: {error}");
                    Control::carry_on()
                })
        })
    }

    fn poll_while_paused<'a>(
        &'a self,
        node: &'a str,
        waiter: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = PausedPoll> + Send + 'a>> {
        Box::pin(async move {
            self.request_control(node, Some(waiter))
                .await
                .map(|reply| {
                    if let Some(generation) = reply.provider_pause_generation {
                        self.remember_provider_pause(waiter, generation);
                    }
                    PausedPoll::Response(reply.control)
                })
                .unwrap_or_else(|error| {
                    tracing::debug!(
                        "could not ask the dashboard to resume a provider-paused run: {error}"
                    );
                    PausedPoll::Unavailable
                })
        })
    }

    fn pause<'a>(
        &'a self,
        node: &'a str,
        waiter: &'a str,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = ProviderPauseRegistration> + Send + 'a>>
    {
        Box::pin(async move {
            let known_provider_resume_generation = self.known_provider_resume_generation();
            let asked = self
                .client
                .post(format!("{}/internal/control/pause", self.dashboard))
                .json(&serde_json::json!({
                    "project": self.project,
                    "run_id": self.run_id,
                    "node": node,
                    "provider_pause_waiter": waiter,
                    "known_provider_resume_generation": known_provider_resume_generation,
                }))
                .timeout(POLL_TIMEOUT)
                .send()
                .await;
            match asked {
                Ok(reply) if reply.status().is_success() => {
                    match reply.json::<ProviderPauseReply>().await {
                        Ok(reply) => {
                            self.remember_provider_pause(waiter, reply.generation);
                            match reply.directive {
                                ProviderPauseDirective::Hold => ProviderPauseRegistration::Paused,
                                ProviderPauseDirective::Continue => {
                                    ProviderPauseRegistration::Resumed
                                }
                                ProviderPauseDirective::Stop => ProviderPauseRegistration::Stopped,
                            }
                        }
                        Err(error) => {
                            tracing::debug!(
                                "could not read the automatic provider pause reply: {error}"
                            );
                            ProviderPauseRegistration::Uncertain
                        }
                    }
                }
                Ok(reply) => {
                    tracing::debug!(
                        status = %reply.status(),
                        "dashboard refused an automatic provider pause"
                    );
                    ProviderPauseRegistration::Uncertain
                }
                Err(e) => {
                    tracing::debug!("could not ask the dashboard to pause the run: {e}");
                    ProviderPauseRegistration::Uncertain
                }
            }
        })
    }

    fn acknowledge_provider_pause<'a>(
        &'a self,
        node: &'a str,
        waiter: &'a str,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = ProviderPauseAcknowledgement> + Send + 'a>,
    > {
        Box::pin(async move {
            let Some(generation) = self
                .provider_pause_generations
                .lock()
                .expect("provider pause generation mutex poisoned")
                .get(waiter)
                .copied()
            else {
                return ProviderPauseAcknowledgement::Unavailable;
            };
            let acknowledged = self
                .client
                .post(format!("{}/internal/control/pause/ack", self.dashboard))
                .json(&serde_json::json!({
                    "project": self.project,
                    "run_id": self.run_id,
                    "node": node,
                    "provider_pause_waiter": waiter,
                    "provider_pause_generation": generation,
                }))
                .timeout(POLL_TIMEOUT)
                .send()
                .await;
            match acknowledged {
                Ok(reply) if reply.status().is_success() => {
                    match reply.json::<ProviderPauseAcknowledgementReply>().await {
                        Ok(reply) => {
                            let acknowledgement = match reply.directive {
                                ProviderPauseDirective::Continue => {
                                    ProviderPauseAcknowledgement::Continue
                                }
                                ProviderPauseDirective::Stop => ProviderPauseAcknowledgement::Stop,
                                ProviderPauseDirective::Hold => {
                                    tracing::debug!(
                                        "dashboard returned Hold for a provider pause acknowledgement"
                                    );
                                    return ProviderPauseAcknowledgement::Unavailable;
                                }
                            };
                            self.acknowledge_provider_generation(generation);
                            self.provider_pause_generations
                                .lock()
                                .expect("provider pause generation mutex poisoned")
                                .remove(waiter);
                            acknowledgement
                        }
                        Err(error) => {
                            tracing::debug!(
                                "could not read the provider pause acknowledgement reply: {error}"
                            );
                            ProviderPauseAcknowledgement::Unavailable
                        }
                    }
                }
                Ok(reply) => {
                    tracing::debug!(
                        status = %reply.status(),
                        "dashboard refused a provider pause acknowledgement"
                    );
                    ProviderPauseAcknowledgement::Unavailable
                }
                Err(error) => {
                    tracing::debug!("could not acknowledge the provider pause: {error}");
                    ProviderPauseAcknowledgement::Unavailable
                }
            }
        })
    }
}

/// Point this run's nodes at the dashboard that started it, if one did.
///
/// Does nothing for a run started from the command line: there is no dashboard holding commands,
/// and a node that cannot be paused by anyone should not pay for asking.
pub fn install(run_id: &str) {
    let Ok(dashboard) = std::env::var("RATATOSKR_DASHBOARD") else {
        return;
    };
    ratatoskr_agent::configure_control(Arc::new(DashboardControl {
        dashboard,
        // Empty unless the dashboard spawned this run, which is also the only case in which it has
        // anything to say.
        project: std::env::var("RATATOSKR_PROJECT").unwrap_or_default(),
        run_id: run_id.to_string(),
        client: reqwest::Client::new(),
        known_provider_resume_generation: Mutex::new(None),
        provider_pause_generations: Mutex::new(HashMap::new()),
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn an_unreachable_dashboard_lets_the_node_carry_on() {
        // The defect this guards is a run that stalls because the thing watching it went away.
        // Port 1 is not listening, so this is the unreachable case without a fixture.
        let control = DashboardControl {
            dashboard: "http://127.0.0.1:1".to_string(),
            project: "p".to_string(),
            run_id: "r".to_string(),
            client: reqwest::Client::new(),
            known_provider_resume_generation: Mutex::new(None),
            provider_pause_generations: Mutex::new(HashMap::new()),
        };
        assert_eq!(control.poll("analyst").await, Control::carry_on());
        assert_eq!(
            control
                .poll_while_paused("analyst", "provider-pause-test")
                .await,
            PausedPoll::Unavailable
        );
        assert_eq!(
            control.pause("analyst", "provider-pause-test").await,
            ProviderPauseRegistration::Uncertain
        );
        assert!(matches!(
            control
                .acknowledge_provider_pause("analyst", "provider-pause-test")
                .await,
            ProviderPauseAcknowledgement::Unavailable
        ));
    }
}
