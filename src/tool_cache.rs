//! Tool-schema cache for the prompt builder.
//!
//! `collect_tool_schemas()` fans out a `tool.v1.request.describe` to every
//! tool-providing capsule and caches the merged result in per-principal KV.
//! The cache is keyed on the kernel's capsule-set epoch
//! ([`astrid_sdk::runtime::capsule_set_epoch`]) so it self-heals: when a
//! capsule is installed / removed / upgraded the epoch moves, the stored
//! entry reads as a miss, and the next assemble re-describes. This replaces
//! the former `astrid.v1.capsules_loaded` interceptor, whose broadcast
//! carried no principal and so only ever cleared the default principal's
//! cache — gateway-minted principals never saw runtime-installed tools
//! (astrid#982).

use astrid_sdk::prelude::*;
use serde::{Deserialize, Serialize};

/// KV key for cached tool schemas. First call populates it via IPC fan-out;
/// subsequent calls read directly from KV until the capsule-set epoch moves.
const TOOL_SCHEMA_CACHE_KEY: &str = "__tool_schema_cache";

/// Timeout (ms) for fanning out the tool-describe request and collecting
/// responses from every tool-providing capsule.
const TOOL_DESCRIBE_FANOUT_TIMEOUT_MS: u64 = 2000;

/// Maximum number of tool-describe responses to collect before proceeding.
const MAX_TOOL_DESCRIBE_RESPONSES: usize = 256;

/// A cached tool list tagged with the capsule-set epoch it was described
/// under. Generic over the tool item type so the staleness logic
/// ([`cached_tools_if_fresh`]) is testable with cheap stand-ins.
#[derive(Serialize, Deserialize)]
pub(crate) struct CachedTools<T> {
    pub epoch: astrid_sdk::runtime::CapsuleSetEpoch,
    pub tools: Vec<T>,
}

/// Serve the cache only when it was described under the current epoch AND is
/// non-empty (a stale epoch — the #982 failure — and an empty list both read
/// as a miss, forcing a fresh describe). Pure; the IPC fan-out / KV stay in
/// `collect_tool_schemas`.
pub(crate) fn cached_tools_if_fresh<T>(
    cached: Option<CachedTools<T>>,
    current_epoch: astrid_sdk::runtime::CapsuleSetEpoch,
) -> Option<Vec<T>> {
    cached
        .filter(|c| c.epoch == current_epoch && !c.tools.is_empty())
        .map(|c| c.tools)
}

/// Collect tool schemas from all capsules via IPC fan-out.
///
/// Reads `__tool_schema_cache` from KV first and serves it only when it was
/// described under the current capsule-set epoch (see [`cached_tools_if_fresh`]).
/// On a miss it subscribes to `tool.v1.response.describe.*`, publishes a
/// `tool.v1.request.describe` event, collects responses within
/// `TOOL_DESCRIBE_FANOUT_TIMEOUT_MS`, deduplicates by tool name, and rewrites
/// the cache tagged with the current epoch.
///
/// The pre-#752 implementation used `hooks::trigger`, which has been removed
/// from the host ABI surface. This IPC-based fan-out replaces it; the same
/// `{ "tools": [...] }` envelope (from SDK macro `tool_describe` and
/// `astrid_bridge.mjs`) is honoured.
pub(crate) fn collect_tool_schemas() -> Vec<serde_json::Value> {
    let current_epoch = astrid_sdk::runtime::capsule_set_epoch();

    // Check the KV cache first. An older bare-`Vec` cache entry (pre-#982)
    // fails to deserialize into `CachedTools` → `.ok()` yields `None` → a
    // miss → the entry is rewritten in the new shape: a clean migration.
    let cached = kv::get_json::<CachedTools<serde_json::Value>>(TOOL_SCHEMA_CACHE_KEY).ok();
    if let Some(tools) = cached_tools_if_fresh(cached, current_epoch) {
        log::debug(format!(
            "Returning {} cached tool schemas (epoch {:?})",
            tools.len(),
            current_epoch
        ));
        return tools;
    }

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

    // Fire the fan-out request. Empty payload — every responder publishes its
    // own tool schema set onto `tool.v1.response.describe.<source_id>`.
    if let Err(e) = ipc::publish("tool.v1.request.describe", "{}") {
        log::error(format!("Failed to publish tool.v1.request.describe: {e}"));
        return Vec::new();
    }

    // Collect responses until we time out or hit the cap. Monotonic
    // clock via `astrid_sdk::time` — `std::time::Instant::now()`
    // panics on `wasm32-unknown-unknown`.
    let mut all_tools: Vec<serde_json::Value> = Vec::new();
    let start = astrid_sdk::time::monotonic();
    let timeout_dur = std::time::Duration::from_millis(TOOL_DESCRIBE_FANOUT_TIMEOUT_MS);

    while astrid_sdk::time::monotonic().saturating_sub(start) < timeout_dur
        && all_tools.len() < MAX_TOOL_DESCRIBE_RESPONSES
    {
        let elapsed = astrid_sdk::time::monotonic().saturating_sub(start);
        let remaining_ms = timeout_dur.saturating_sub(elapsed).as_millis();
        if remaining_ms == 0 {
            break;
        }
        let timeout = u64::try_from(remaining_ms).unwrap_or(u64::MAX);

        match sub.recv(timeout) {
            Ok(result) => {
                if result.messages.is_empty() {
                    break;
                }
                for msg in &result.messages {
                    if let Some(tools) = extract_tools_from_response(&msg.payload) {
                        all_tools.extend(tools);
                    }
                }
            }
            _ => break,
        }
    }

    // Deduplicate by tool name (first occurrence wins).
    let mut seen = std::collections::HashSet::new();
    all_tools.retain(|tool| {
        if let Some(name) = tool.get("name").and_then(|n| n.as_str()) {
            seen.insert(name.to_string())
        } else {
            true
        }
    });

    log::info(format!(
        "Collected {} tool schemas via tool.v1.request.describe fan-out",
        all_tools.len()
    ));

    // Cache the result tagged with the epoch it was described under. The next
    // assemble serves it until a capsule (un)install moves the epoch.
    let entry = CachedTools {
        epoch: current_epoch,
        tools: all_tools,
    };
    if let Err(e) = kv::set_json(TOOL_SCHEMA_CACHE_KEY, &entry) {
        log::warn(format!("Failed to cache tool schemas in KV: {e}"));
    }

    entry.tools
}

/// Extract the `tools` array from a `tool.v1.response.describe.*` payload.
///
/// Honours both the direct envelope (`{ "tools": [...] }`, emitted by the
/// SDK macro `tool_describe` and `astrid_bridge.mjs`) and the wrapped
/// `{ "data": { "tools": [...] } }` envelope used by some Custom payload
/// publishers.
pub(crate) fn extract_tools_from_response(payload: &str) -> Option<Vec<serde_json::Value>> {
    let value: serde_json::Value = serde_json::from_str(payload).ok()?;
    let tools = value
        .get("tools")
        .or_else(|| value.get("data").and_then(|d| d.get("tools")))
        .and_then(|t| t.as_array())?;
    Some(tools.clone())
}
