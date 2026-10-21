//! Background engine thread coordination via lock-free flume channels (§4.5, §7 M5).
//!
//! Separates heavy inference execution from the GTK main loop, streaming tokens and
//! live telemetry asynchronously without blocking UI rendering.

use crate::power::{read_system_power, PowerReading};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use sulimo_core::loader::get_process_memory_stats;
use sulimo_engine::engine::{Engine, GenerationStats};
use sulimo_engine::sampler::SamplingParams;

/// Commands sent from the GTK UI to the background inference engine.
#[derive(Debug)]
pub enum Cmd {
    /// Load a GGUF model from the specified file path into the static arena.
    Load {
        path: PathBuf,
        requested_ctx: Option<usize>,
    },
    /// Unload the currently loaded model and free arena allocations.
    Unload,
    /// Generate a completion from a prompt string with the given sampling parameters.
    Generate {
        prompt: String,
        params: SamplingParams,
    },
    /// Stop ongoing generation immediately (cancel latency <= 1 token).
    Stop,
}

/// Execution phase for telemetry display.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionPhase {
    Idle,
    PrefillGpu,
    DecodeCpu,
}

impl ExecutionPhase {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Idle => "Idle",
            Self::PrefillGpu => "Prefill-GPU",
            Self::DecodeCpu => "Decode-CPU",
        }
    }
}

/// Events sent from the engine thread to the GTK UI.
#[derive(Debug, Clone)]
pub enum Event {
    /// Progress notification while loading model into arena.
    Loading(String),
    /// Model has been successfully loaded and pre-flight verified.
    Loaded {
        model_name: String,
        planned_bytes: usize,
        ctx_len: usize,
    },
    /// Model has been unloaded.
    Unloaded,
    /// A single decoded token string.
    Token(String),
    /// Periodic system telemetry (RAM, speed, power, phase).
    Telemetry {
        phase: ExecutionPhase,
        ram_bytes: usize,
        max_budget_bytes: usize,
        tok_per_sec: f64,
        power: PowerReading,
    },
    /// Generation completed successfully with detailed execution metrics.
    Done(GenerationStats),
    /// Error encountered during loading or generation.
    Error(String),
}

/// Spawns the background engine worker thread.
pub fn spawn_engine_worker() -> (flume::Sender<Cmd>, flume::Receiver<Event>) {
    let (cmd_tx, cmd_rx) = flume::unbounded::<Cmd>();
    let (event_tx, event_rx) = flume::unbounded::<Event>();

    let cancel_flag = Arc::new(AtomicBool::new(false));
    let cancel_clone = Arc::clone(&cancel_flag);

    thread::Builder::new()
        .name("sulimo-engine".to_string())
        .spawn(move || {
            worker_loop(cmd_rx, event_tx, cancel_clone);
        })
        .expect("failed to spawn engine worker thread");

    (cmd_tx, event_rx)
}

fn worker_loop(
    cmd_rx: flume::Receiver<Cmd>,
    event_tx: flume::Sender<Event>,
    cancel_flag: Arc<AtomicBool>,
) {
    let mut engine: Option<Engine> = None;
    const MAX_BUDGET: usize = 5_200 * 1024 * 1024;

    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            Cmd::Load {
                path,
                requested_ctx,
            } => {
                let file_name = path
                    .file_name()
                    .map(|f| f.to_string_lossy().to_string())
                    .unwrap_or_else(|| "model.gguf".to_string());

                let _ = event_tx.send(Event::Loading(format!("Loading {file_name}...")));

                match Engine::load(&path, requested_ctx, None) {
                    Ok(eng) => {
                        let planned_bytes = eng.model.plan.total_planned_bytes;
                        let ctx_len = eng.model.plan.context_length;
                        engine = Some(eng);

                        let (rss, _) = get_process_memory_stats();
                        let _ = event_tx.send(Event::Loaded {
                            model_name: file_name,
                            planned_bytes,
                            ctx_len,
                        });
                        let _ = event_tx.send(Event::Telemetry {
                            phase: ExecutionPhase::Idle,
                            ram_bytes: rss as usize,
                            max_budget_bytes: MAX_BUDGET,
                            tok_per_sec: 0.0,
                            power: read_system_power(),
                        });
                    }
                    Err(e) => {
                        let _ = event_tx.send(Event::Error(format!("Failed to load model: {e}")));
                    }
                }
            }
            Cmd::Unload => {
                engine = None;
                let _ = event_tx.send(Event::Unloaded);
                let (rss, _) = get_process_memory_stats();
                let _ = event_tx.send(Event::Telemetry {
                    phase: ExecutionPhase::Idle,
                    ram_bytes: rss as usize,
                    max_budget_bytes: MAX_BUDGET,
                    tok_per_sec: 0.0,
                    power: read_system_power(),
                });
            }
            Cmd::Stop => {
                cancel_flag.store(true, Ordering::Relaxed);
            }
            Cmd::Generate { prompt, params } => {
                cancel_flag.store(false, Ordering::Relaxed);

                let Some(ref mut eng) = engine else {
                    let _ = event_tx.send(Event::Error(
                        "No model loaded. Please load a GGUF model first.".into(),
                    ));
                    continue;
                };

                let phase = if eng.has_gpu_backend() && prompt.len() >= 32 {
                    ExecutionPhase::PrefillGpu
                } else {
                    ExecutionPhase::DecodeCpu
                };

                let (rss, _) = get_process_memory_stats();
                let _ = event_tx.send(Event::Telemetry {
                    phase,
                    ram_bytes: rss as usize,
                    max_budget_bytes: MAX_BUDGET,
                    tok_per_sec: 0.0,
                    power: read_system_power(),
                });

                let event_tx_clone = event_tx.clone();
                let res =
                    eng.generate_with_callback(&prompt, &params, &cancel_flag, move |token| {
                        let _ = event_tx_clone.send(Event::Token(token.to_string()));
                    });

                match res {
                    Ok(stats) => {
                        let (final_rss, _) = get_process_memory_stats();
                        let _ = event_tx.send(Event::Telemetry {
                            phase: ExecutionPhase::Idle,
                            ram_bytes: final_rss as usize,
                            max_budget_bytes: MAX_BUDGET,
                            tok_per_sec: stats.decode_tok_per_sec,
                            power: read_system_power(),
                        });
                        let _ = event_tx.send(Event::Done(stats));
                    }
                    Err(sulimo_engine::engine::EngineError::Cancelled) => {
                        let _ = event_tx
                            .send(Event::Error("Generation cancelled by user.".to_string()));
                    }
                    Err(e) => {
                        let _ = event_tx.send(Event::Error(format!("Generation error: {e}")));
                    }
                }
            }
        }
    }
}
