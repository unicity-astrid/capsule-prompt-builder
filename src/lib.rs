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

/// KV key for cached tool schemas. First call populates it via IPC broadcast;
/// subsequent calls read directly from KV until invalidated.
const TOOL_SCHEMA_CACHE_KEY: &str = "__tool_schema_cache";

/// Invalidate the cached tool schemas in KV.
///
/// Called when the capsule set changes (`astrid.v1.capsules_loaded`) so the
/// next `collect_tool_schemas()` re-runs the describe fan-out instead of
/// serving a stale (KV-persisted, restart-surviving) list.
fn invalidate_tool_cache() {
    let _ = kv::delete(TOOL_SCHEMA_CACHE_KEY);
    log::info("Tool schema cache invalidated");
}

/// Timeout (ms) for fetching session messages from the session capsule.
const SESSION_FETCH_TIMEOUT_MS: u64 = 5000;

/// Installed-capsule manifests in the per-principal VFS. The number of these
/// that subscribe to `tool.v1.request.describe` is the EXPECTED responder count
/// the fan-out collects against (see `expected_tool_responder_count`). Same path
/// the system capsule walks for `list_capsules`.
const INSTALLED_CAPSULES_DIR: &str = "home://.local/capsules";

/// Overall hard ceiling (ms) for the tool-describe fan-out: re-firing and
/// collection stop once this elapses no matter what. A COMPLETE result is
/// cached (`__tool_schema_cache`) so this cost is paid once; it is only reached
/// when collection stays incomplete.
const TOOL_DESCRIBE_FANOUT_TIMEOUT_MS: u64 = 5000;

/// Per-round drain window (ms): after each (re)publish of the describe request,
/// collect responses for this long before re-firing. A single fan-out publish
/// reaches only a subset of responders (a kernel delivery race), so the request
/// is re-fired and the union of distinct responders accumulated until the
/// expected count has answered.
const TOOL_DESCRIBE_ROUND_MS: u64 = 300;

/// Fallback only. When the expected responder count cannot be derived from the
/// installed manifests (capsule dir unreadable, or none found) the loop has no
/// completion target, so it instead stops after this many consecutive rounds
/// add no new responders.
const TOOL_DESCRIBE_STABLE_ROUNDS: u32 = 3;

/// Maximum number of tool-describe responses to collect before proceeding.
const MAX_TOOL_DESCRIBE_RESPONSES: usize = 256;

/// Does this capsule manifest declare a `[subscribe]` to the tool-describe
/// request topic — i.e. is the capsule a tool-describe RESPONDER?
///
/// Section-aware on purpose. The fan-out REQUESTER (prompt-builder itself)
/// declares `tool.v1.request.describe` under `[publish]`, so a plain substring
/// match would miscount the requester as a responder. Only a `[subscribe]`
/// entry makes a capsule answer the fan-out. Hand-rolled rather than pulling a
/// TOML parser into the wasm guest: the manifests are our own well-formed TOML
/// and we only need one table-membership test.
fn manifest_subscribes_to_describe(manifest: &str) -> bool {
    const TOPIC: &str = "tool.v1.request.describe";
    let mut in_subscribe = false;
    for line in manifest.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // A table header enters `[subscribe]`; any other header (including a
        // `[subscribe.x]` subtable or a `[[array]]`) leaves it.
        if line.starts_with('[') {
            in_subscribe = line == "[subscribe]";
            continue;
        }
        if in_subscribe
            && let Some(key) = line.split('=').next()
            && key.trim().trim_matches('"') == TOPIC
        {
            return true;
        }
    }
    false
}

/// Count installed capsules that are tool-describe responders, given each
/// manifest's contents. Pure, so the fan-out completion target is unit-testable
/// without the VFS.
fn count_tool_responders<S: AsRef<str>>(manifests: &[S]) -> usize {
    manifests
        .iter()
        .filter(|m| manifest_subscribes_to_describe(m.as_ref()))
        .count()
}

/// Derive the EXPECTED number of tool-describe responders from the installed
/// capsule manifests in the per-principal VFS.
///
/// Returns `None` when the count cannot be established — the capsule dir is
/// unreadable, or no manifest subscribes to the describe topic. The caller then
/// falls back to the quiet-rounds heuristic rather than treating "couldn't tell"
/// (or a parse that found nothing) as an authoritative "no tools", which would
/// otherwise let collection conclude — and cache — empty.
fn expected_tool_responder_count() -> Option<usize> {
    let entries = astrid_sdk::fs::read_dir(INSTALLED_CAPSULES_DIR).ok()?;
    let mut manifests: Vec<String> = Vec::new();
    for entry in entries {
        let name = entry.file_name().to_string();
        let path = format!("{INSTALLED_CAPSULES_DIR}/{name}/Capsule.toml");
        if let Ok(content) = astrid_sdk::fs::read_to_string(&path) {
            manifests.push(content);
        }
    }
    let count = count_tool_responders(&manifests);
    (count > 0).then_some(count)
}

/// Whether a finished collection should be written to the cache.
///
/// Cache ONLY a collection known to be complete: every expected responder
/// answered (`expected == Some(n)` and `responders >= n`). An incomplete fan-out
/// MUST NOT be cached — caching it would pin the partial tool set until the next
/// `capsules_loaded` invalidation; leaving it uncached lets the next prompt
/// retry. When the expected count is unknown (`None`) the heuristic path is
/// best-effort: cache whatever non-empty set it gathered.
fn should_cache_collection(expected: Option<usize>, responders: usize, tools: usize) -> bool {
    if tools == 0 {
        return false;
    }
    match expected {
        Some(n) => responders >= n,
        None => true,
    }
}

/// Collect tool schemas from all capsules via IPC fan-out.
///
/// Checks `__tool_schema_cache` in KV first. On cache miss, subscribes to
/// `tool.v1.response.describe.*` and (re)publishes `tool.v1.request.describe`.
/// Tool-providing capsules respond on `tool.v1.response.describe.<source_id>`
/// (the suffix is the literal `self`; the real responder id rides in the
/// kernel-stamped `Message.source_id`). Responses are deduplicated by tool name.
///
/// A single fan-out publish reaches only a SUBSET of the tool responders (a
/// kernel delivery race), so the request is re-fired and the union of distinct
/// responders accumulated until every EXPECTED responder (derived from the
/// installed manifests at load time, [`expected_tool_responder_count`]) has
/// answered — or the overall ceiling elapses. Only a complete collection is
/// cached (see [`should_cache_collection`]); an incomplete one is returned but
/// left uncached so the next prompt retries and self-heals.
///
/// The pre-#752 implementation used `hooks::trigger`, removed from the host ABI;
/// this IPC fan-out replaces it, honouring the same `{ "tools": [...] }`
/// envelope (from SDK macro `tool_describe` and `astrid_bridge.mjs`).
fn collect_tool_schemas() -> Vec<serde_json::Value> {
    // Check KV cache first.
    if let Ok(cached) = kv::get_json::<Vec<serde_json::Value>>(TOOL_SCHEMA_CACHE_KEY)
        && !cached.is_empty()
    {
        log::debug(format!("Returning {} cached tool schemas", cached.len()));
        return cached;
    }

    // How many responders SHOULD answer, known from the installed manifests at
    // load time. `None` => couldn't tell; fall back to the quiet-rounds heuristic.
    let expected = expected_tool_responder_count();

    // Subscribe BEFORE publishing so we don't miss fast responders.
    let sub = match ipc::subscribe("tool.v1.response.describe.*") {
        Ok(s) => s,
        Err(e) => {
            log::error(format!(
                "Failed to subscribe to tool.v1.response.describe.*: {e}"
            ));
            return Vec::new();
        }
    };

    // Collect across one or more fan-out rounds. Each round (re)publishes the
    // request and drains responses for a short window; re-firing lets a
    // responder that a previous fan-out missed answer a later one. Responders
    // are deduped by `source_id`, tools by name. Termination: once every
    // expected responder has answered we are complete and stop; without an
    // expected count we stop after TOOL_DESCRIBE_STABLE_ROUNDS rounds add no new
    // responders. Either way the overall ceiling bounds the wait. Monotonic
    // clock via `astrid_sdk::time` — `std::time::Instant::now()` panics on
    // `wasm32-unknown-unknown`.
    let mut all_tools: Vec<serde_json::Value> = Vec::new();
    let mut seen_tools = std::collections::HashSet::<String>::new();
    let mut responders = std::collections::HashSet::<String>::new();
    let start = astrid_sdk::time::monotonic();
    let overall = std::time::Duration::from_millis(TOOL_DESCRIBE_FANOUT_TIMEOUT_MS);
    let round_dur = std::time::Duration::from_millis(TOOL_DESCRIBE_ROUND_MS);
    let mut quiet_rounds = 0u32;

    while astrid_sdk::time::monotonic().saturating_sub(start) < overall
        && all_tools.len() < MAX_TOOL_DESCRIBE_RESPONSES
    {
        match expected {
            // Complete: every expected responder has answered.
            Some(n) if responders.len() >= n => break,
            // No completion target — stop once responders go quiet.
            None if quiet_rounds >= TOOL_DESCRIBE_STABLE_ROUNDS => break,
            _ => {}
        }

        if let Err(e) = ipc::publish("tool.v1.request.describe", "{}") {
            log::error(format!("Failed to publish tool.v1.request.describe: {e}"));
            break;
        }

        let responders_before = responders.len();
        let round_start = astrid_sdk::time::monotonic();
        loop {
            let in_round = astrid_sdk::time::monotonic().saturating_sub(round_start);
            if in_round >= round_dur || all_tools.len() >= MAX_TOOL_DESCRIBE_RESPONSES {
                break;
            }
            let remaining_ms = round_dur.saturating_sub(in_round).as_millis();
            let remaining = u64::try_from(remaining_ms).unwrap_or(u64::MAX);
            match sub.recv(remaining) {
                Ok(result) if !result.messages.is_empty() => {
                    for msg in &result.messages {
                        if !msg.source_id.is_empty() {
                            responders.insert(msg.source_id.clone());
                        }
                        if let Some(tools) = extract_tools_from_response(&msg.payload) {
                            for tool in tools {
                                match tool.get("name").and_then(|n| n.as_str()) {
                                    Some(name) => {
                                        if seen_tools.insert(name.to_string()) {
                                            all_tools.push(tool);
                                        }
                                    }
                                    None => all_tools.push(tool),
                                }
                            }
                        }
                    }
                }
                // Round window elapsed (recv timeout) or an empty poll — this
                // round is drained; re-fire on the next loop iteration.
                _ => break,
            }
        }

        if responders.len() == responders_before {
            quiet_rounds = quiet_rounds.saturating_add(1);
        } else {
            quiet_rounds = 0;
        }
    }

    let cache = should_cache_collection(expected, responders.len(), all_tools.len());
    log::info(format!(
        "Collected {} tool schemas from {} responder(s) (expected {}, cached={cache}) \
         via tool.v1.request.describe fan-out",
        all_tools.len(),
        responders.len(),
        expected.map_or_else(|| "?".to_string(), |n| n.to_string()),
    ));

    if cache && let Err(e) = kv::set_json(TOOL_SCHEMA_CACHE_KEY, &all_tools) {
        log::warn(format!("Failed to cache tool schemas in KV: {e}"));
    }

    all_tools
}

/// Extract the `tools` array from a `tool.v1.response.describe.*` payload.
///
/// Honours both the direct envelope (`{ "tools": [...] }`, emitted by the
/// SDK macro `tool_describe` and `astrid_bridge.mjs`) and the wrapped
/// `{ "data": { "tools": [...] } }` envelope used by some Custom payload
/// publishers.
fn extract_tools_from_response(payload: &str) -> Option<Vec<serde_json::Value>> {
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    let tools = value
        .get("tools")
        .or_else(|| value.get("data").and_then(|d| d.get("tools")))
        .and_then(|t| t.as_array())?;
    Some(tools.clone())
}

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

    // Collect tool schemas (cached after first call).
    let tools = collect_tool_schemas();

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

    /// Invalidates the cached tool schemas when the capsule set changes.
    ///
    /// The kernel broadcasts `astrid.v1.capsules_loaded` after (un)loading
    /// capsules. Under the pooled-interceptor model there is no run loop to
    /// poll, so a dedicated interceptor drops `__tool_schema_cache` here — the
    /// next `handle_assemble` re-collects a fresh tool set instead of serving a
    /// stale (KV-persisted, restart-surviving) list, so newly installed tool
    /// capsules reach the LLM on the next prompt without a manual cache clear.
    #[astrid::interceptor("on_capsules_loaded")]
    pub(crate) fn on_capsules_loaded(&self, _payload: serde_json::Value) -> Result<(), SysError> {
        invalidate_tool_cache();
        Ok(())
    }
}

#[cfg(test)]
mod tests;
