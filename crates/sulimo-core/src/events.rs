//! Shared agentic event contract for Súlimo GUI & CLI harnesses (@C:BRIDGE).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ToolStatus {
    Running,
    Success,
    Denied,
}

#[derive(Clone, Debug)]
pub enum AgentBlock {
    Thinking {
        elapsed_ms: u64,
        summary: String,
        body: String,
    },
    ToolUse {
        id: u32,
        tool: String,
        args: String,
        status: ToolStatus,
        diff: Option<String>,
    },
    ResponseChunk(String),
}

#[derive(Clone, Debug)]
pub struct TelemetrySnapshot {
    /// Pinned host arena allocation in megabytes (Hard cap: 5200 MB).
    pub pinned_mb: usize,
    /// Q8_0 quantized circular KV cache consumption in megabytes.
    pub kv_q8_mb: usize,
    /// Active hardware compute phase ("OCL-FP16-SIMD8" | "CPU-AVX2-P" | "RC6-SLEEP").
    pub compute_phase: &'static str,
    /// Real-time decode throughput (tokens/second).
    pub tok_s: f32,
    /// Package power consumption in Watts.
    pub pkg_watts: f32,
}

#[derive(Clone, Debug)]
pub enum EngineEvent {
    Block(AgentBlock),
    Telemetry(TelemetrySnapshot),
    Done,
    Error(String),
}
