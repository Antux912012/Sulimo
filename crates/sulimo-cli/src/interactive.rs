//! Interactive Antigravity-style CLI REPL interface for Súlimo.
//!
//! Provides an interactive terminal experience closely modeled on `antigravity-cli` (`agy`):
//! - Styled rounded box-drawing frames and neon/slate palette
//! - Categorized slash commands (`/help`, `/clear`, `/reset`, `/stats`, `/model`, `/system`, `/temp`, `/tokens`, `/bench`, `/probe`, `/exit`)
//! - Multi-turn conversational context with automatic repacking
//! - Live streaming tokens and per-turn telemetry execution cards
//! - Hardware tier telemetry (OpenCL FP16 SIMD8 prefill, AVX2 physical core decode, iGPU RC6 deep sleep)

use anyhow::{bail, Context, Result};
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use sulimo_core::loader::get_process_memory_stats;
use sulimo_core::MAX_RAM_BUDGET_BYTES;
use sulimo_engine::bench::read_rc6_residency_ms;
use sulimo_engine::{CapabilityReport, Engine, SamplingParams, SchedulerConfig};

pub struct InteractiveSession {
    engine: Option<Engine>,
    model_path: Option<PathBuf>,
    system_prompt: Option<String>,
    params: SamplingParams,
    turns: Vec<(String, String)>,
    cancel_flag: AtomicBool,
    requested_ctx: Option<usize>,
    threads: Option<usize>,
    temp_model: Option<PathBuf>,
}

impl InteractiveSession {
    pub fn new(
        model_path: Option<&Path>,
        requested_ctx: Option<usize>,
        threads: Option<usize>,
    ) -> Self {
        Self {
            engine: None,
            model_path: model_path.map(PathBuf::from),
            system_prompt: Some(
                "You are Súlimo (Arda's Breath), a fast, knowledgeable, and helpful local AI assistant."
                    .to_string(),
            ),
            params: SamplingParams {
                temperature: 0.7,
                top_p: 0.9,
                top_k: 40,
                repetition_penalty: 1.1,
                greedy: false,
                max_tokens: 256,
                seed: 42,
                ignore_eos: false,
            },
            turns: Vec::new(),
            cancel_flag: AtomicBool::new(false),
            requested_ctx,
            threads,
            temp_model: None,
        }
    }

    pub fn run(&mut self) -> Result<()> {
        self.print_welcome();

        if let Some(ref path) = self.model_path.clone() {
            println!("\x1b[38;5;141m[•] Loading model:\x1b[0m {}", path.display());
            if let Err(e) = self.load_model(path) {
                eprintln!("\x1b[1;31m[!] Failed to load model: {e}\x1b[0m");
                eprintln!("    Type \x1b[1;36m/model <path>\x1b[0m or \x1b[1;36m/model synthetic\x1b[0m to load a model.");
            }
        } else {
            let local_models = find_local_gguf_models();
            if !local_models.is_empty() {
                println!("\x1b[38;5;49m[•] Found local GGUF model(s) in workspace:\x1b[0m");
                for (idx, p) in local_models.iter().enumerate() {
                    println!("    \x1b[1m[{}]\x1b[0m {}", idx + 1, p.display());
                }
                println!("    Type \x1b[1;36m/model <path>\x1b[0m to load, or \x1b[1;36m/model synthetic\x1b[0m for demo.");
            } else {
                println!("\x1b[38;5;214m[•] No model loaded.\x1b[0m Type \x1b[1;36m/model <path>\x1b[0m or \x1b[1;36m/model synthetic\x1b[0m to begin.");
            }
        }

        println!("\n  \x1b[38;5;244mType \x1b[1;38;5;51m/help\x1b[0;38;5;244m for slash commands, \x1b[1;38;5;51m/exit\x1b[0;38;5;244m to quit.\x1b[0m\n");

        let stdin = io::stdin();
        let mut reader = stdin.lock();

        loop {
            print!("\x1b[1;38;5;51m❯\x1b[0m ");
            io::stdout().flush().context("flush stdout")?;

            let mut input = String::new();
            if reader.read_line(&mut input)? == 0 {
                // EOF (Ctrl+D)
                println!("\n\x1b[38;5;244mExiting Súlimo.\x1b[0m");
                break;
            }

            let trimmed = input.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed.starts_with('/') {
                if !self.handle_slash_command(trimmed)? {
                    break;
                }
                continue;
            }

            self.handle_user_prompt(trimmed)?;
        }

        if let Some(ref tmp) = self.temp_model {
            let _ = std::fs::remove_file(tmp);
        }

        Ok(())
    }

    fn print_welcome(&self) {
        println!("\x1b[1;38;5;51m");
        println!("  ╭────────────────────────────────────────────────────────────────────────╮");
        println!("  │   ____        _ _                 (Arda's Breath v0.2.0)               │");
        println!("  │  / ___| _   _| (_)_ __ ___   ___  Antigravity-Style Local CLI          │");
        println!(
            "  │  \\___ \\| | | | | | '_ ` _ \\ / _ \\ Gen8–10 Legacy Tier Inference        │"
        );
        println!("  │   ___) | |_| | | | | | | | | (_) | Hybrid iGPU OpenCL FP16 + AVX2 CPU  │");
        println!("  │  |____/ \\__,_|_|_|_| |_| |_|\\___/  5.2 GB Unified RAM Budget Ceiling   │");
        println!(
            "  ╰────────────────────────────────────────────────────────────────────────╯\x1b[0m"
        );

        if let Ok(report) = CapabilityReport::probe() {
            let cpu = &report.cpu.model_name;
            let cores = report.cpu.physical_cores.len();
            let gpu = match &report.ocl {
                sulimo_ocl::OclProbeResult::Available(d) => {
                    format!("{} ({} EUs)", d.device_name, d.compute_units)
                }
                sulimo_ocl::OclProbeResult::Unavailable { .. } => "None (CPU Only)".to_string(),
            };
            println!(
                "  \x1b[1;38;5;141m[Host]\x1b[0m {} ({} physical cores)  │  \x1b[1;38;5;141m[GPU]\x1b[0m {}",
                cpu, cores, gpu
            );
            println!(
                "  \x1b[1;38;5;141m[Tier]\x1b[0m {}  │  \x1b[1;38;5;141m[RAM Budget]\x1b[0m 5.20 GiB Unified Ceiling",
                report.selected_tier
            );
        }
        println!();
    }

    fn handle_slash_command(&mut self, cmd_line: &str) -> Result<bool> {
        let parts: Vec<&str> = cmd_line.split_whitespace().collect();
        let cmd = parts[0];
        let arg = if parts.len() > 1 {
            cmd_line[cmd.len()..].trim()
        } else {
            ""
        };

        match cmd {
            "/help" | "/?" => {
                self.print_help();
            }
            "/clear" => {
                print!("\x1b[2J\x1b[H");
                io::stdout().flush().unwrap();
            }
            "/reset" => {
                self.turns.clear();
                println!("\x1b[38;5;49m[✓] Conversation turns and context reset.\x1b[0m");
            }
            "/stats" => {
                self.print_stats();
            }
            "/model" => {
                if arg.is_empty() {
                    if let Some(ref eng) = self.engine {
                        println!("\x1b[1;38;5;51mActive Model Information:\x1b[0m");
                        println!(
                            "  File:           {}",
                            self.model_path
                                .as_deref()
                                .unwrap_or(Path::new("unknown"))
                                .display()
                        );
                        println!("  Architecture:   {}", eng.model.config.arch.as_str());
                        println!("  Context Length: {} tokens", eng.model.plan.context_length);
                        println!(
                            "  Planned Memory: {:.2} GiB",
                            eng.model.plan.total_planned_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                        );
                        println!(
                            "  GPU Prefill:    {}",
                            if eng.has_gpu_backend() {
                                "\x1b[38;5;49mActive (OpenCL FP16 SIMD8)\x1b[0m"
                            } else {
                                "\x1b[38;5;214mInactive (CPU AVX2 fallback)\x1b[0m"
                            }
                        );
                    } else {
                        println!("\x1b[38;5;214m[!] No model loaded.\x1b[0m Usage: /model <path/to/model.gguf> or /model synthetic");
                    }
                } else if arg == "synthetic" {
                    println!(
                        "\x1b[38;5;141m[•] Generating and loading synthetic benchmark model (context 4096)...\x1b[0m"
                    );
                    let p = sulimo_engine::bench::create_synthetic_benchmark_model(4096);
                    self.temp_model = Some(p.clone());
                    self.load_model(&p)?;
                } else {
                    let p = Path::new(arg);
                    self.load_model(p)?;
                }
            }
            "/system" => {
                if arg.is_empty() {
                    println!(
                        "\x1b[1;38;5;51mSystem Prompt:\x1b[0m\n  {}",
                        self.system_prompt.as_deref().unwrap_or("None")
                    );
                } else {
                    self.system_prompt = Some(arg.to_string());
                    println!("\x1b[38;5;49m[✓] System prompt updated.\x1b[0m");
                }
            }
            "/temp" => {
                if arg.is_empty() {
                    println!("Temperature: {:.2}", self.params.temperature);
                } else if let Ok(val) = arg.parse::<f32>() {
                    self.params.temperature = val.clamp(0.0, 2.0);
                    self.params.greedy = self.params.temperature == 0.0;
                    println!(
                        "\x1b[38;5;49m[✓] Temperature set to {:.2}\x1b[0m",
                        self.params.temperature
                    );
                } else {
                    println!("Invalid temperature. Expected float (e.g. 0.7)");
                }
            }
            "/top_p" => {
                if arg.is_empty() {
                    println!("Top-P: {:.2}", self.params.top_p);
                } else if let Ok(val) = arg.parse::<f32>() {
                    self.params.top_p = val.clamp(0.0, 1.0);
                    println!(
                        "\x1b[38;5;49m[✓] Top-P set to {:.2}\x1b[0m",
                        self.params.top_p
                    );
                } else {
                    println!("Invalid top_p. Expected float (e.g. 0.9)");
                }
            }
            "/tokens" | "/max_tokens" => {
                if arg.is_empty() {
                    println!("Max Tokens: {}", self.params.max_tokens);
                } else if let Ok(val) = arg.parse::<usize>() {
                    self.params.max_tokens = val.max(1);
                    println!(
                        "\x1b[38;5;49m[✓] Max tokens set to {}\x1b[0m",
                        self.params.max_tokens
                    );
                } else {
                    println!("Invalid tokens. Expected positive integer (e.g. 256)");
                }
            }
            "/bench" => {
                println!(
                    "\x1b[38;5;141m[•] Running hardware throughput & TTFT benchmark...\x1b[0m"
                );
                let target_path = if let Some(ref p) = self.model_path {
                    p.clone()
                } else {
                    let tmp = sulimo_engine::bench::create_synthetic_benchmark_model(4096);
                    self.temp_model = Some(tmp.clone());
                    tmp
                };
                let report = sulimo_engine::bench::run_benchmark_suite(
                    &target_path,
                    Some(2048),
                    self.threads,
                    false,
                )
                .context("bench failed")?;
                println!("\n  \x1b[1mPrefill Throughput:\x1b[0m");
                for p in &report.prefill {
                    println!(
                        "    N={:<5} -> \x1b[1;38;5;49m{:>8.1} tok/s\x1b[0m  (TTFT: {:>5.1} ms, {})",
                        p.prompt_tokens, p.prefill_tok_per_sec, p.ttft_ms, p.backend
                    );
                }
                println!(
                    "  \x1b[1mDecode Throughput:\x1b[0m  \x1b[1;38;5;49m{:.1} tok/s\x1b[0m (CPU AVX2 pinned)",
                    report.decode.decode_tok_per_sec
                );
                if let Some(rc6) = report.decode.rc6_residency_pct {
                    println!("  \x1b[1miGPU RC6 Sleep:\x1b[0m     {:.1}%", rc6);
                }
                println!(
                    "  \x1b[1mPeak Resident RAM:\x1b[0m  {:.2} GiB (Ceiling: 5.20 GiB)\n",
                    report.memory.peak_rss_gib
                );
            }
            "/probe" => {
                let report = CapabilityReport::probe()?;
                println!("\n{}", serde_json::to_string_pretty(&report)?);
            }
            "/exit" | "/quit" => {
                println!("\x1b[38;5;244mExiting Súlimo.\x1b[0m");
                return Ok(false);
            }
            other => {
                println!(
                    "\x1b[1;31m[!] Unknown command: {other}\x1b[0m. Type \x1b[1;38;5;51m/help\x1b[0m for available commands."
                );
            }
        }

        Ok(true)
    }

    fn print_help(&self) {
        println!("\n\x1b[1;38;5;51m╭── Súlimo Antigravity CLI Slash Commands ──────────────────────────────╮\x1b[0m");
        println!("\x1b[1;38;5;141m  Session & Context:\x1b[0m");
        println!("    \x1b[1;36m/clear\x1b[0m                 Clear terminal screen");
        println!(
            "    \x1b[1;36m/reset\x1b[0m                 Clear multi-turn conversation memory"
        );
        println!(
            "    \x1b[1;36m/system [text]\x1b[0m         Inspect or set system instruction prompt"
        );
        println!("\x1b[1;38;5;141m  Model & Parameters:\x1b[0m");
        println!("    \x1b[1;36m/model [path]\x1b[0m          Inspect current model or load GGUF (/model synthetic for demo)");
        println!("    \x1b[1;36m/temp [val]\x1b[0m            Set sampling temperature (0.0 = greedy, default 0.7)");
        println!("    \x1b[1;36m/top_p [val]\x1b[0m           Set nucleus top-P sampling probability (default 0.9)");
        println!(
            "    \x1b[1;36m/tokens [N]\x1b[0m            Set maximum generated tokens per response"
        );
        println!("\x1b[1;38;5;141m  Hardware & Telemetry:\x1b[0m");
        println!("    \x1b[1;36m/stats\x1b[0m                 Display live telemetry, RAM vs 5.2GB budget & iGPU RC6");
        println!("    \x1b[1;36m/bench\x1b[0m                 Run instant hardware throughput & TTFT benchmark");
        println!("    \x1b[1;36m/probe\x1b[0m                 Run full system capability & execution tier probe");
        println!("\x1b[1;38;5;141m  General:\x1b[0m");
        println!("    \x1b[1;36m/help\x1b[0m, \x1b[1;36m/?\x1b[0m              Show this command reference");
        println!("    \x1b[1;36m/exit\x1b[0m, \x1b[1;36m/quit\x1b[0m           Exit the interactive session");
        println!("\x1b[1;38;5;51m╰───────────────────────────────────────────────────────────────────────╯\x1b[0m\n");
    }

    fn print_stats(&self) {
        let (rss_bytes, hwm_bytes) = get_process_memory_stats();
        let rss_gib = rss_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let hwm_gib = hwm_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
        let rc6 = read_rc6_residency_ms();

        println!("\n\x1b[1;38;5;51m╭── Telemetry & Hardware Status ────────────────────────────────────────╮\x1b[0m");
        println!(
            "  Resident Memory (RSS):    {:.2} GiB ({} bytes)",
            rss_gib, rss_bytes
        );
        println!(
            "  Peak Memory (VmHWM):      {:.2} GiB ({} bytes)",
            hwm_gib, hwm_bytes
        );
        println!("  Unified RAM Budget:       5.20 GiB (5,452,595,200 bytes)");
        let budget_pct = (hwm_bytes as f64 / MAX_RAM_BUDGET_BYTES as f64) * 100.0;
        let budget_status = if hwm_bytes <= MAX_RAM_BUDGET_BYTES {
            "\x1b[38;5;49mPASS (Within budget)\x1b[0m"
        } else {
            "\x1b[1;31mEXCEEDED\x1b[0m"
        };
        println!(
            "  Budget Utilization:       {:.1}% ({})",
            budget_pct, budget_status
        );
        if let Some(rc6_val) = rc6 {
            println!("  iGPU RC6 Deep Sleep:      {} ms", rc6_val);
        }
        if let Some(ref eng) = self.engine {
            let backend_str = if eng.has_gpu_backend() {
                "Prefill: GPU OpenCL FP16 SIMD8  │  Decode: CPU AVX2 Physical Pinned"
            } else {
                "Prefill: CPU AVX2  │  Decode: CPU AVX2 Physical Pinned"
            };
            println!("  Active Pipeline:          {}", backend_str);
            println!(
                "  Context Window:           {} tokens",
                eng.model.plan.context_length
            );
        } else {
            println!("  Active Pipeline:          None (No model loaded)");
        }
        println!("  Conversation Turns:       {}", self.turns.len());
        println!(
            "  Sampling Configuration:   temp={:.2}, top_p={:.2}, max_tokens={}",
            self.params.temperature, self.params.top_p, self.params.max_tokens
        );
        println!("\x1b[1;38;5;51m╰───────────────────────────────────────────────────────────────────────╯\x1b[0m\n");
    }

    fn load_model(&mut self, path: &Path) -> Result<()> {
        if !path.exists() {
            bail!("Model file not found: {}", path.display());
        }

        let config = SchedulerConfig {
            gpu_threshold: 8,
            ubatch_size: 512,
            opencl_enabled: true,
        };

        let t0 = Instant::now();
        let eng = Engine::load_with_config(path, self.requested_ctx, self.threads, config)?;
        let elapsed = t0.elapsed().as_secs_f64();

        println!(
            "\x1b[38;5;49m[✓] Model loaded in {:.2}s: {} (Context: {}, Planned: {:.2} GiB)\x1b[0m",
            elapsed,
            path.file_name().unwrap_or_default().to_string_lossy(),
            eng.model.plan.context_length,
            eng.model.plan.total_planned_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        self.model_path = Some(path.to_path_buf());
        self.engine = Some(eng);
        self.turns.clear();

        Ok(())
    }

    fn handle_user_prompt(&mut self, user_input: &str) -> Result<()> {
        if self.engine.is_none() {
            println!(
                "\x1b[38;5;214m[!] No model loaded.\x1b[0m Type \x1b[1;36m/model <path>\x1b[0m or \x1b[1;36m/model synthetic\x1b[0m to load a model first."
            );
            return Ok(());
        }

        let prompt = self.build_prompt(user_input);

        let Some(ref mut engine) = self.engine else {
            return Ok(());
        };

        print!("\n\x1b[1;38;5;141m● Súlimo:\x1b[0m ");
        io::stdout().flush().context("flush stdout")?;

        self.cancel_flag.store(false, Ordering::Relaxed);
        let rc6_before = read_rc6_residency_ms();
        let wall_start = Instant::now();

        let mut response_text = String::new();

        let stats =
            engine.generate_with_callback(&prompt, &self.params, &self.cancel_flag, |token| {
                print!("{token}");
                let _ = io::stdout().flush();
                response_text.push_str(token);
            });

        let wall_elapsed_ms = wall_start.elapsed().as_millis() as u64;
        let rc6_after = read_rc6_residency_ms();
        println!();

        match stats {
            Ok(gen_stats) => {
                let rc6_str = match (rc6_before, rc6_after) {
                    (Some(b), Some(a)) if wall_elapsed_ms > 0 => {
                        let pct = ((a.saturating_sub(b) as f64) / wall_elapsed_ms as f64) * 100.0;
                        format!("{:.0}% (GPU asleep)", pct.min(100.0))
                    }
                    _ => "n/a".to_string(),
                };

                let (rss_bytes, _) = get_process_memory_stats();
                let rss_gib = rss_bytes as f64 / (1024.0 * 1024.0 * 1024.0);

                println!("\x1b[38;5;244m╭────────────────────────────────────────────────────────────────────────╮\x1b[0m");
                println!(
                    "\x1b[38;5;244m│\x1b[0m  \x1b[1;38;5;49m⚡ {:.1} tok/s\x1b[0m  │  TTFT: {:.1}ms  │  Tokens: {}  │  RAM: {:.2} GiB  │  RC6: {}  \x1b[38;5;244m│\x1b[0m",
                    gen_stats.decode_tok_per_sec,
                    gen_stats.ttft_ms,
                    gen_stats.completion_tokens,
                    rss_gib,
                    rc6_str
                );
                println!("\x1b[38;5;244m╰────────────────────────────────────────────────────────────────────────╯\x1b[0m\n");

                self.turns.push((user_input.to_string(), response_text));
            }
            Err(sulimo_engine::EngineError::Cancelled) => {
                println!("\x1b[38;5;214m[Generation cancelled by user]\x1b[0m\n");
            }
            Err(e) => {
                println!("\x1b[1;31m[Generation error: {e}]\x1b[0m\n");
            }
        }

        Ok(())
    }

    fn build_prompt(&self, current_user_msg: &str) -> String {
        if self.turns.is_empty() {
            if let Some(ref sys) = self.system_prompt {
                format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{current_user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            } else {
                format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n{current_user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                )
            }
        } else {
            let mut prompt = String::new();
            if let Some(ref sys) = self.system_prompt {
                prompt.push_str(&format!(
                    "<|start_header_id|>system<|end_header_id|>\n\n{sys}<|eot_id|>"
                ));
            }
            for (u, a) in &self.turns {
                prompt.push_str(&format!(
                    "<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|>"
                ));
            }
            prompt.push_str(&format!(
                "<|start_header_id|>user<|end_header_id|>\n\n{current_user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            ));
            prompt
        }
    }
}

pub fn find_local_gguf_models() -> Vec<PathBuf> {
    let mut models = Vec::new();
    if let Ok(entries) = std::fs::read_dir(".") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                models.push(path);
            }
        }
    }
    models
}
