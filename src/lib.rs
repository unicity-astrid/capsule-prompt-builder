#![deny(unsafe_code)]
#![deny(clippy::all)]
#![deny(unreachable_pub)]

//! Prompt Builder capsule — assembles LLM prompts with plugin hook interception.
//!
//! This capsule owns the prompt assembly pipeline. When the react loop needs
//! a prompt assembled, it publishes to `prompt_builder.v1.assemble`. The prompt
//! builder then:
//!
//! 1. Fires `prompt_builder.v1.hook.before_build` to all plugin capsules via IPC
//! 2. Collects plugin responses (`prependSystemContext`, `appendSystemContext`,
//!    `systemPrompt` override, `prependContext`)
//! 3. Merges them according to OpenClaw-compatible semantics
//! 4. Returns the assembled prompt on `prompt_builder.v1.response.assemble`
//! 5. Fires `prompt_builder.v1.hook.after_build` as a notification
//!
//! # Merge Semantics
//!
//! 1. `prependContext` — concatenated in order, becomes `user_context_prefix`
//! 2. `systemPrompt` — last non-null value wins (full override)
//! 3. `prependSystemContext` — concatenated in order, prepended to system prompt
//! 4. `appendSystemContext` — concatenated in order, appended to system prompt

use astrid_sdk::prelude::*;
use serde::{Deserialize, Serialize};

pub(crate) mod tool_cache;

/// Runtime configuration loaded from capsule config at startup.
struct Config {
    /// Maximum time (in milliseconds) to wait for plugin hook responses — the
    /// OUTER cap for the multi-responder accumulation case.
    hook_timeout_ms: u64,
    /// First-response window (ms): how long to wait for the FIRST hook response
    /// before concluding no plugin will reply. Defaults to
    /// [`HOOK_FIRST_RESPONSE_MS`]; an operator with a genuinely slow plugin can
    /// raise it (per principal) instead of having it silently excluded.
    hook_first_response_ms: u64,
    /// Idle-grace window (ms) once at least one response has arrived. Defaults
    /// to [`HOOK_IDLE_GRACE_MS`].
    hook_idle_grace_ms: u64,
}

impl Config {
    /// Load configuration from the capsule's config store, falling back to
    /// defaults.
    ///
    /// Read per assemble invocation, **not** cached: `env::var` resolves the
    /// per-invocation, per-principal env overlay, so a global cache would pin
    /// one principal's tuning for every principal. The cost is a few host calls,
    /// negligible against the hook fan-out they gate.
    fn load() -> Self {
        Self {
            hook_timeout_ms: env_u64("hook_timeout_ms", DEFAULT_HOOK_POLL_TIMEOUT_MS),
            hook_first_response_ms: env_u64("hook_first_response_ms", HOOK_FIRST_RESPONSE_MS),
            hook_idle_grace_ms: env_u64("hook_idle_grace_ms", HOOK_IDLE_GRACE_MS),
        }
    }
}

/// Read a `u64` capsule config value from `env`, falling back to `default` when
/// the key is missing or unparseable. Trims surrounding whitespace and quotes.
fn env_u64(key: &str, default: u64) -> u64 {
    env::var(key)
        .ok()
        .and_then(|s| s.trim().trim_matches('"').parse::<u64>().ok())
        .unwrap_or(default)
}

/// Request from the react loop to assemble a prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct AssembleRequest {
    /// The conversation messages.
    #[serde(default)]
    messages: serde_json::Value,
    /// The current system prompt before plugin modifications.
    #[serde(default)]
    system_prompt: String,
    /// Unique request identifier for correlation.
    request_id: String,
    /// The target LLM model identifier.
    #[serde(default)]
    model: String,
    /// The LLM provider identifier.
    #[serde(default)]
    provider: String,
    /// Session ID echoed back for react loop correlation.
    #[serde(default)]
    session_id: Option<String>,
}

/// Response returned to the react loop with the assembled prompt.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct AssembleResponse {
    /// The final assembled system prompt.
    system_prompt: String,
    /// Text to prepend to the user's message (from `prependContext` hooks).
    user_context_prefix: String,
    /// The original request ID for correlation.
    request_id: String,
    /// Session ID echoed from the request for react loop correlation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    session_id: Option<String>,
    /// Collected tool schemas from all tool-providing capsules.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tools: Vec<serde_json::Value>,
    /// Session conversation history messages.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    messages: Vec<serde_json::Value>,
}

/// Payload sent to plugin capsules via the `prompt_builder.v1.hook.before_build` interceptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct BeforePromptBuildPayload {
    messages: serde_json::Value,
    system_prompt: String,
    request_id: String,
    model: String,
    provider: String,
    /// Topic where plugins should publish their hook responses.
    response_topic: String,
}

/// A single plugin's response to the `prompt_builder.v1.hook.before_build` hook.
///
/// All fields are optional. The prompt builder merges responses from
/// multiple plugins according to the documented merge semantics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct HookResponse {
    /// Text to prepend to the system prompt.
    #[serde(default)]
    prepend_system_context: Option<String>,
    /// Text to append to the system prompt.
    #[serde(default)]
    append_system_context: Option<String>,
    /// Full system prompt override (last non-null wins).
    #[serde(default)]
    system_prompt: Option<String>,
    /// Text to prepend to the user's message.
    #[serde(default)]
    prepend_context: Option<String>,
}

impl HookResponse {
    /// Returns `true` if at least one field is set.
    fn has_any_field(&self) -> bool {
        self.prepend_system_context.is_some()
            || self.append_system_context.is_some()
            || self.system_prompt.is_some()
            || self.prepend_context.is_some()
    }
}

/// Notification payload sent after prompt assembly completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
struct AfterPromptBuildPayload {
    system_prompt: String,
    user_context_prefix: String,
    request_id: String,
}

/// Default maximum time (in milliseconds) to wait for plugin hook responses.
/// Overridable via the `hook_timeout_ms` capsule config key.
const DEFAULT_HOOK_POLL_TIMEOUT_MS: u64 = 2000;

/// Maximum number of hook responses to collect before proceeding.
const MAX_HOOK_RESPONSES: usize = 50;

/// First-response window (ms): how long to wait for the FIRST hook
/// response before concluding no plugin is going to reply. Hook
/// responders are local pooled capsules that reply within a few bus
/// hops (~1ms), so the old behaviour of waiting the full
/// `hook_timeout_ms` (2s) for a first response was a pure per-prompt
/// floor — and the COMMON case is zero responders (no plugin injects
/// for a given prompt), which always hit that full backstop. Capping
/// the first-response wait un-floors throughput under the Store pool
/// (astrid#816: was pool_size / 2s). `hook_timeout_ms` is retained as
/// the OUTER cap for the multi-responder accumulation case.
const HOOK_FIRST_RESPONSE_MS: u64 = 250;

/// Idle-grace window (ms) for the hook fan-out once at least one
/// response has arrived: poll only this long for stragglers and return
/// as soon as responders go quiet, instead of waiting the full window.
const HOOK_IDLE_GRACE_MS: u64 = 100;

/// A hook response paired with its source capsule identifier.
///
/// Used for permission gating: plugins with `allowPromptInjection: false`
/// should have their prompt-mutating fields discarded.
struct SourcedHookResponse {
    /// The capsule/session ID that sent this response.
    source_id: Option<String>,
    /// The parsed hook response.
    response: HookResponse,
}

/// Filter hook responses based on prompt injection permissions.
///
/// Plugins without prompt injection permission retain only `prependContext`
/// (user-visible context), while `systemPrompt`, `prependSystemContext`,
/// and `appendSystemContext` are stripped.
///
/// The `has_permission` closure receives the `source_id` (if present) and
/// returns whether the source capsule is allowed to mutate the system
/// prompt. This closure is parameterized for testability - the production
/// call site queries the kernel via `capabilities::check`.
fn filter_by_permission(
    sourced: Vec<SourcedHookResponse>,
    mut has_permission: impl FnMut(Option<&str>) -> bool,
) -> Vec<HookResponse> {
    sourced
        .into_iter()
        .map(|s| {
            if has_permission(s.source_id.as_deref()) {
                s.response
            } else {
                // Strip prompt-mutating fields; only user-visible context passes.
                if s.response.system_prompt.is_some()
                    || s.response.prepend_system_context.is_some()
                    || s.response.append_system_context.is_some()
                {
                    log::warn(format!(
                        "Stripped prompt-mutating fields from capsule {:?} \
                         (missing allow_prompt_injection capability)",
                        s.source_id
                    ));
                }
                HookResponse {
                    prepend_context: s.response.prepend_context,
                    ..Default::default()
                }
            }
        })
        .collect()
}

/// Merge collected hook responses into a final assembled prompt.
///
/// Merge order (matches OpenClaw documented behaviour):
/// 1. `prependContext` — concatenated in interceptor order
/// 2. `systemPrompt` — last non-null value wins as full override
/// 3. `prependSystemContext` — concatenated, prepended to (possibly overridden) prompt
/// 4. `appendSystemContext` — concatenated, appended to system prompt
fn merge_hook_responses(original_system_prompt: &str, responses: &[HookResponse]) -> MergedPrompt {
    let mut prepend_contexts: Vec<&str> = Vec::new();
    let mut prepend_system_contexts: Vec<&str> = Vec::new();
    let mut append_system_contexts: Vec<&str> = Vec::new();
    let mut system_prompt_override: Option<&str> = None;

    for resp in responses {
        if let Some(ref ctx) = resp.prepend_context
            && !ctx.is_empty()
        {
            prepend_contexts.push(ctx);
        }
        if let Some(ref prompt) = resp.system_prompt
            && !prompt.is_empty()
        {
            // Last non-empty wins — intentionally overwrites previous overrides.
            // An empty string is treated as "no override" to prevent accidentally
            // wiping the system prompt.
            system_prompt_override = Some(prompt);
        }
        if let Some(ref ctx) = resp.prepend_system_context
            && !ctx.is_empty()
        {
            prepend_system_contexts.push(ctx);
        }
        if let Some(ref ctx) = resp.append_system_context
            && !ctx.is_empty()
        {
            append_system_contexts.push(ctx);
        }
    }

    // Step 2: Determine the base system prompt (override or original).
    let base_prompt = system_prompt_override.unwrap_or(original_system_prompt);

    // Step 3-4: Prepend + base + append, joined with newlines.
    let mut parts: Vec<&str> = Vec::new();
    parts.extend_from_slice(&prepend_system_contexts);
    if !base_prompt.is_empty() {
        parts.push(base_prompt);
    }
    parts.extend_from_slice(&append_system_contexts);
    let final_prompt = parts.join("\n");

    // Step 1: Build user context prefix.
    let user_context_prefix = prepend_contexts.join("\n");

    MergedPrompt {
        system_prompt: final_prompt,
        user_context_prefix,
    }
}

/// The result of merging all hook responses.
struct MergedPrompt {
    system_prompt: String,
    user_context_prefix: String,
}

/// Fire the `prompt_builder.v1.hook.before_build` interceptor and collect plugin responses.
///
/// Publishes the hook event on the `prompt_builder.v1.hook.before_build` IPC topic and polls
/// a dedicated response topic for plugin contributions. Returns all collected
/// responses within the timeout window, filtered by permission gating.
fn fire_before_prompt_build(request: &AssembleRequest, config: &Config) -> Vec<HookResponse> {
    let response_topic = format!("prompt_builder.v1.hook_response.{}", request.request_id);

    // Subscribe BEFORE publishing to avoid missing fast responses.
    let sub = match ipc::subscribe(&response_topic) {
        Ok(h) => h,
        Err(e) => {
            log::error(format!("Failed to subscribe to hook response topic: {e}"));
            return Vec::new();
        }
    };

    let payload = BeforePromptBuildPayload {
        messages: request.messages.clone(),
        system_prompt: request.system_prompt.clone(),
        request_id: request.request_id.clone(),
        model: request.model.clone(),
        provider: request.provider.clone(),
        response_topic: response_topic.clone(),
    };

    if let Err(e) = ipc::publish_json("prompt_builder.v1.hook.before_build", &payload) {
        log::error(format!(
            "Failed to publish prompt_builder.v1.hook.before_build event: {e}"
        ));
        // sub dropped on return — kernel-side resource released automatically.
        return Vec::new();
    }

    // Block-wait for hook responses within the configured timeout.
    // `std::time::Instant::now()` panics on `wasm32-unknown-unknown`
    // (the Astrid-canonical capsule target); track the deadline as a
    // host-monotonic instant via `astrid_sdk::time::monotonic`, which
    // routes through `astrid:sys.clock-monotonic-ns`.
    let mut sourced_responses = Vec::new();
    let start = astrid_sdk::time::monotonic();
    let timeout_dur = std::time::Duration::from_millis(config.hook_timeout_ms);

    while astrid_sdk::time::monotonic().saturating_sub(start) < timeout_dur
        && sourced_responses.len() < MAX_HOOK_RESPONSES
    {
        let elapsed = astrid_sdk::time::monotonic().saturating_sub(start);
        let remaining_ms = timeout_dur.saturating_sub(elapsed).as_millis();
        if remaining_ms == 0 {
            break;
        }
        let remaining = u64::try_from(remaining_ms).unwrap_or(u64::MAX);
        // First-response window: wait up to `hook_first_response_ms` for the
        // FIRST response, measured from the START of the loop (NOT reset each
        // iteration), so a stream of non-matching messages cannot extend it
        // past the cap; give up once it elapses with still no response (the
        // no-responder backstop). Once we have one, switch to a short
        // idle-grace so the loop returns shortly after responders fall quiet
        // instead of burning the whole window.
        let timeout = if sourced_responses.is_empty() {
            let elapsed_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
            let first_remaining = config.hook_first_response_ms.saturating_sub(elapsed_ms);
            if first_remaining == 0 {
                break;
            }
            first_remaining.min(remaining)
        } else {
            config.hook_idle_grace_ms.min(remaining)
        };

        match sub.recv(timeout) {
            Ok(result) => {
                if result.messages.is_empty() {
                    break;
                }
                for msg in &result.messages {
                    if let Some(new_responses) = parse_hook_message(msg) {
                        sourced_responses.extend(new_responses);
                    }
                }
            }
            _ => break,
        }
    }

    // sub drops here, releasing the kernel-side subscription.

    log::info(format!(
        "Collected {} hook responses for request {}",
        sourced_responses.len(),
        request.request_id
    ));

    // Cache capability results per-UUID to avoid redundant host function calls.
    // Multiple hook responses can come from the same capsule.
    let mut cache = std::collections::HashMap::<String, bool>::new();
    filter_by_permission(sourced_responses, |source_id| {
        let Some(uuid) = source_id else {
            return false;
        };
        *cache.entry(uuid.to_owned()).or_insert_with(|| {
            capabilities::check(uuid, "allow_prompt_injection")
                .inspect_err(|e| {
                    log::warn(format!("capability check failed for {uuid}: {e}, denying"));
                })
                .unwrap_or(false)
        })
    })
}

/// Parse a single IPC message and extract hook responses with source capsule IDs.
fn parse_hook_message(msg: &ipc::Message) -> Option<Vec<SourcedHookResponse>> {
    let payload: serde_json::Value = match serde_json::from_str(&msg.payload) {
        Ok(v) => v,
        Err(e) => {
            log::warn(format!("failed to deserialize hook response payload: {e}"));
            return None;
        }
    };

    let source_id = if msg.source_id.is_empty() {
        None
    } else {
        Some(msg.source_id.clone())
    };

    // Try to parse the payload directly as a HookResponse.
    // Since all fields are optional, an unrelated JSON object would
    // parse as an empty HookResponse — check `has_any_field()` to
    // distinguish real responses from false positives.
    // Plugins may wrap it in various IPC payload envelopes, so we
    // also check inside `data` for Custom payloads.
    let maybe_response = serde_json::from_value::<HookResponse>(payload.clone())
        .ok()
        .filter(HookResponse::has_any_field)
        .or_else(|| {
            payload
                .get("data")
                .and_then(|data| serde_json::from_value::<HookResponse>(data.clone()).ok())
                .filter(HookResponse::has_any_field)
        });

    if let Some(response) = maybe_response {
        Some(vec![SourcedHookResponse {
            source_id,
            response,
        }])
    } else {
        None
    }
}

/// Parse the poll envelope and extract hook responses with source capsule IDs.
///
/// Retained for unit tests that construct raw JSON envelopes.
#[cfg(test)]
fn parse_hook_responses(poll_bytes: &[u8]) -> Option<Vec<SourcedHookResponse>> {
    let envelope: serde_json::Value = match serde_json::from_slice(poll_bytes) {
        Ok(v) => v,
        Err(e) => {
            log::warn(format!("failed to deserialize hook response envelope: {e}"));
            return None;
        }
    };

    let messages = envelope.get("messages")?.as_array()?;
    let mut responses = Vec::new();

    for msg in messages {
        let payload = match msg.get("payload") {
            Some(p) => p,
            None => continue,
        };

        // Track the source capsule for permission gating.
        let source_id = msg
            .get("source_id")
            .and_then(|s| s.as_str())
            .map(String::from);

        let maybe_response = serde_json::from_value::<HookResponse>(payload.clone())
            .ok()
            .filter(HookResponse::has_any_field)
            .or_else(|| {
                payload
                    .get("data")
                    .and_then(|data| serde_json::from_value::<HookResponse>(data.clone()).ok())
                    .filter(HookResponse::has_any_field)
            });

        if let Some(response) = maybe_response {
            responses.push(SourcedHookResponse {
                source_id,
                response,
            });
        }
    }

    if responses.is_empty() {
        None
    } else {
        Some(responses)
    }
}

/// Fire the `prompt_builder.v1.hook.after_build` notification (fire-and-forget).
fn fire_after_prompt_build(system_prompt: &str, user_context_prefix: &str, request_id: &str) {
    let payload = AfterPromptBuildPayload {
        system_prompt: system_prompt.to_string(),
        user_context_prefix: user_context_prefix.to_string(),
        request_id: request_id.to_string(),
    };
    let _ = ipc::publish_json("prompt_builder.v1.hook.after_build", &payload);
}

/// Timeout (ms) for fetching session messages from the session capsule.
const SESSION_FETCH_TIMEOUT_MS: u64 = 5000;

/// Fetch session conversation history from the session capsule via IPC.
///
/// Uses [`ipc::request_response`], which generates a v4 correlation ID,
/// subscribes to the scoped reply topic *before* publishing, injects the
/// correlation ID into the request body, and tears the subscription down
/// on every return path via the [`ipc::Subscription`] Drop. Returns an
/// empty vec on timeout, parse failure, or transport error.
fn fetch_session_messages(session_id: &str) -> Vec<serde_json::Value> {
    let request = serde_json::json!({ "session_id": session_id });

    // Responder publishes onto `session.v1.response.get_messages.<corr_id>`.
    let raw: serde_json::Value = match ipc::request_response(
        "session.v1.request.get_messages",
        "session.v1.response.get_messages",
        &request,
        SESSION_FETCH_TIMEOUT_MS,
    ) {
        Ok(v) => v,
        Err(e) => {
            log::warn(format!(
                "session.v1.request.get_messages failed (timeout={SESSION_FETCH_TIMEOUT_MS}ms): {e}"
            ));
            return Vec::new();
        }
    };

    // The session capsule may wrap the array in a Custom `data` envelope.
    let messages_value = raw
        .get("messages")
        .or_else(|| raw.get("data").and_then(|d| d.get("messages")));

    let result = match messages_value {
        Some(messages) => serde_json::from_value::<Vec<serde_json::Value>>(messages.clone())
            .unwrap_or_else(|e| {
                log::warn(format!("Failed to parse session messages array: {e}"));
                Vec::new()
            }),
        None => {
            log::warn("Session response missing `messages` field");
            Vec::new()
        }
    };

    log::debug(format!(
        "Fetched {} session messages for session {session_id}",
        result.len()
    ));

    result
}

/// Assemble a single prompt for a `prompt_builder.v1.assemble` request.
fn assemble(payload: &serde_json::Value, config: &Config) {
    // Extract from Custom payload envelope or direct.
    let request_value = payload.get("data").unwrap_or(payload);

    let request: AssembleRequest = match serde_json::from_value(request_value.clone()) {
        Ok(r) => r,
        Err(e) => {
            log::error(format!("Failed to parse assemble request: {e}"));
            let _ = ipc::publish_json(
                "prompt_builder.v1.response.assemble",
                &serde_json::json!({"error": format!("invalid request: {e}")}),
            );
            return;
        }
    };

    if request.request_id.is_empty() {
        let _ = ipc::publish_json(
            "prompt_builder.v1.response.assemble",
            &serde_json::json!({"error": "missing request_id"}),
        );
        return;
    }

    // Fire interceptor hooks and collect responses.
    let hook_responses = fire_before_prompt_build(&request, config);

    // Merge all responses into the final prompt.
    let merged = merge_hook_responses(&request.system_prompt, &hook_responses);

    // Collect tool schemas (cached per capsule-set epoch).
    let tools = tool_cache::collect_tool_schemas();

    // Fetch session messages if a session_id was provided.
    let messages = request
        .session_id
        .as_deref()
        .map(fetch_session_messages)
        .unwrap_or_default();

    // Publish the assembled result.
    let response = AssembleResponse {
        system_prompt: merged.system_prompt.clone(),
        user_context_prefix: merged.user_context_prefix.clone(),
        request_id: request.request_id.clone(),
        session_id: request.session_id.clone(),
        tools,
        messages,
    };

    let _ = ipc::publish_json("prompt_builder.v1.response.assemble", &response);

    // Fire after_prompt_build notification (fire-and-forget).
    fire_after_prompt_build(
        &merged.system_prompt,
        &merged.user_context_prefix,
        &request.request_id,
    );
}

#[derive(Default)]
struct PromptBuilder;

#[capsule]
impl PromptBuilder {
    /// Assembles a prompt for a `prompt_builder.v1.assemble` request.
    ///
    /// Pooled interceptor: the kernel leases one of the capsule's pooled
    /// Stores per invocation, so concurrent prompts assemble in parallel
    /// (one in-flight assemble per Store). This replaces the former
    /// single-threaded `#[astrid::run]` recv loop, which drained
    /// `prompt_builder.v1.assemble` strictly one-at-a-time and so capped
    /// end-to-end orchestration throughput at one prompt per hook
    /// round-trip regardless of how many were in flight (astrid#816).
    /// The hook fan-out (`fire_before_prompt_build`) still does a nested
    /// `recv` — that works inside an interceptor exactly as it did in the
    /// run loop, and now overlaps across pooled invocations instead of
    /// serializing.
    #[astrid::interceptor("handle_assemble")]
    pub(crate) fn handle_assemble(&self, payload: serde_json::Value) -> Result<(), SysError> {
        assemble(&payload, &Config::load());
        Ok(())
    }
}

#[cfg(test)]
mod tests;
