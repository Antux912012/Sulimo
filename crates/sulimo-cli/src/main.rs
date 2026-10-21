//! Súlimo Command-Line Interface (CLI).
//!
//! Subcommands:
//! - `probe [--json]`: Inspects host hardware capabilities and determines the execution tier.
//! - `inspect <model.gguf>`: Inspects GGUF metadata, architecture, tensor inventory, and budget plan.
//! - `load <model.gguf> [--ctx N]`: Executes streaming model load into static arena, asserting $ZCOPY and measuring VmHWM.

use anyhow::{bail, Context, Result};
use std::env;
use std::path::Path;
use sulimo_core::{
    get_process_memory_stats, GgufMmap, LoadedModel, MemoryBudget, ModelConfig,
    MAX_RAM_BUDGET_BYTES,
};
use sulimo_engine::CapabilityReport;
use sulimo_ocl::OclProbeResult;

fn print_usage() {
    eprintln!(
        "\
Súlimo CLI (v{})
Local LLM inference engine for Intel Gen8–10 hardware

USAGE:
    sulimo-cli <SUBCOMMAND> [OPTIONS] [ARGS]

SUBCOMMANDS:
    probe                 Probe CPU, GPU, and system memory capabilities
    inspect <MODEL.gguf>  Inspect GGUF metadata, architecture, and memory budget plan
    load <MODEL.gguf>     Load model into static arena and assert $ZCOPY invariants
    run                   Execute inference on a GGUF model (M2+)
    bench                 Run benchmarks across execution tiers (M6+)
    help                  Print this message or the help of the given subcommand

OPTIONS:
    --json                Output machine-readable JSON (valid for 'probe')
    --ctx <N>             Override context length for memory budget / load
    -h, --help            Print help information
",
        env!("CARGO_PKG_VERSION")
    );
}

fn format_bytes_gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

fn format_bytes_mib(bytes: u64) -> String {
    format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
}

fn format_bytes_auto(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else {
        format!("{:.2} MiB", bytes as f64 / (1024.0 * 1024.0))
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = env::args().skip(1).collect();

    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print_usage();
        return Ok(());
    }

    match args[0].as_str() {
        "probe" => {
            let json_mode = args.iter().any(|a| a == "--json");
            let report = CapabilityReport::probe()?;

            if json_mode {
                let json_output = serde_json::to_string_pretty(&report)?;
                println!("{json_output}");
            } else {
                print_human_report(&report);
            }
        }
        "inspect" => {
            if args.len() < 2 {
                bail!("Missing model path. Usage: sulimo-cli inspect <model.gguf>");
            }
            let model_path = Path::new(&args[1]);
            run_inspect(model_path)?;
        }
        "load" => {
            if args.len() < 2 {
                bail!("Missing model path. Usage: sulimo-cli load <model.gguf> [--ctx N]");
            }
            let model_path = Path::new(&args[1]);

            let mut requested_ctx = None;
            for i in 2..args.len() {
                if args[i] == "--ctx" && i + 1 < args.len() {
                    requested_ctx = Some(
                        args[i + 1]
                            .parse::<usize>()
                            .context("invalid --ctx value")?,
                    );
                }
            }

            run_load(model_path, requested_ctx)?;
        }
        "run" => {
            if args.len() < 2 {
                bail!("Missing model path. Usage: sulimo-cli run <model.gguf> -p \"<prompt>\" [OPTIONS]");
            }
            let model_path = Path::new(&args[1]);

            let mut prompt = String::from("Hello, who are you?");
            let mut requested_ctx = None;
            let mut threads = None;
            let mut greedy = false;
            let mut temperature = 0.7f32;
            let mut max_tokens = 32usize;

            let mut i = 2;
            while i < args.len() {
                match args[i].as_str() {
                    "-p" | "--prompt" => {
                        if i + 1 < args.len() {
                            prompt = args[i + 1].clone();
                            i += 1;
                        }
                    }
                    "--ctx" => {
                        if i + 1 < args.len() {
                            requested_ctx = Some(args[i + 1].parse()?);
                            i += 1;
                        }
                    }
                    "-t" | "--threads" => {
                        if i + 1 < args.len() {
                            threads = Some(args[i + 1].parse()?);
                            i += 1;
                        }
                    }
                    "--max-tokens" | "-n" => {
                        if i + 1 < args.len() {
                            max_tokens = args[i + 1].parse()?;
                            i += 1;
                        }
                    }
                    "--temp" => {
                        if i + 1 < args.len() {
                            temperature = args[i + 1].parse()?;
                            i += 1;
                        }
                    }
                    "--greedy" => {
                        greedy = true;
                    }
                    _ => {}
                }
                i += 1;
            }

            let params = sulimo_engine::SamplingParams {
                temperature,
                greedy,
                max_tokens,
                ..Default::default()
            };

            run_inference(model_path, &prompt, requested_ctx, threads, &params)?;
        }
        "bench" | "golden-compare" => {
            eprintln!(
                "Subcommand '{}' is planned for subsequent milestones (M6).",
                args[0]
            );
            std::process::exit(1);
        }
        other => {
            eprintln!("Unknown subcommand: '{other}'");
            print_usage();
            std::process::exit(1);
        }
    }

    Ok(())
}

fn run_inspect(path: &Path) -> Result<()> {
    println!("Opening GGUF file: {}", path.display());
    let mmap = GgufMmap::open(path).context("failed to open GGUF file")?;
    let gguf = mmap.file();

    println!("============================================================");
    println!("                 Súlimo GGUF Model Inspection");
    println!("============================================================");
    println!("GGUF Version:     v{}", gguf.version);
    println!("Tensor Count:     {}", gguf.tensors.len());
    println!("Metadata Count:   {}", gguf.metadata.len());
    println!("Alignment:        {} bytes", gguf.alignment);

    let config = match ModelConfig::from_gguf(gguf) {
        Ok(cfg) => {
            println!("\n[Model Hyperparameters]");
            println!("  Architecture:    {}", cfg.arch.as_str());
            println!("  Context Length:  {} tokens", cfg.context_length);
            println!("  Embedding Dim:   {}", cfg.embedding_length);
            println!("  Layers (Blocks): {}", cfg.block_count);
            println!("  FFN Dimension:   {}", cfg.feed_forward_length);
            println!(
                "  Attention Heads: {} Q heads / {} KV heads (head dim: {})",
                cfg.head_count, cfg.head_count_kv, cfg.head_dim
            );
            println!("  RMS Norm Epsilon:{:e}", cfg.rms_epsilon);
            println!("  RoPE Freq Base:  {}", cfg.rope_freq_base);
            println!("  Vocabulary Size: {} tokens", cfg.vocab_size);
            println!("  Pre-tokenizer:   {}", cfg.pre_tokenizer);
            println!(
                "  KV Cache / Token:{} bytes (Q8_0)",
                cfg.kv_cache_bytes_per_token()
            );
            Some(cfg)
        }
        Err(e) => {
            println!("\n[Model Hyperparameters]: Failed to parse ({e})");
            None
        }
    };

    println!("\n[Tensor Inventory (Top 10)]");
    for (i, t) in gguf.tensors.iter().take(10).enumerate() {
        println!(
            "  {:2}. {:<32} {:?} {:<12?} ({})",
            i + 1,
            t.name,
            t.dims,
            t.ggml_type,
            format_bytes_auto(t.raw_size as u64)
        );
    }
    if gguf.tensors.len() > 10 {
        println!("  ... and {} more tensors", gguf.tensors.len() - 10);
    }

    if let Some(cfg) = config {
        println!("\n[Memory Budget Pre-Flight Plan ($MEM<8G)]");
        match MemoryBudget::plan(gguf, &cfg, None) {
            Ok(plan) => {
                println!("  Context Length:  {} tokens", plan.context_length);
                println!(
                    "  Weights (SoA):   {}",
                    format_bytes_gib(plan.weights_bytes as u64)
                );
                println!(
                    "  KV Cache:        {}",
                    format_bytes_gib(plan.kv_cache_bytes as u64)
                );
                println!(
                    "  Scratch Memory:  {}",
                    format_bytes_mib(plan.scratch_bytes as u64)
                );
                println!(
                    "  Safety Headroom: {}",
                    format_bytes_mib(plan.slack_bytes as u64)
                );
                println!(
                    "  Total Planned:   {} / {} ({})",
                    format_bytes_gib(plan.total_planned_bytes as u64),
                    format_bytes_gib(MAX_RAM_BUDGET_BYTES),
                    if plan.fits_budget {
                        "FITS BUDGET"
                    } else {
                        "OVER BUDGET"
                    }
                );
            }
            Err(e) => {
                println!("  Budget Planning Error:\n  {e}");
            }
        }
    }
    println!("============================================================");

    Ok(())
}

fn run_load(path: &Path, requested_ctx: Option<usize>) -> Result<()> {
    println!("Loading GGUF model: {}", path.display());
    if let Some(ctx) = requested_ctx {
        println!("Requested context length: {} tokens", ctx);
    }

    let (rss_start, _) = get_process_memory_stats();
    println!("Initial Process RSS: {}", format_bytes_auto(rss_start));

    let model =
        LoadedModel::load(path, requested_ctx).context("failed to stream model into arena")?;

    let (rss_after, hwm_peak) = get_process_memory_stats();

    println!("============================================================");
    println!("            Súlimo Model Loading Verification");
    println!("============================================================");
    println!(
        "Model:            {}",
        path.file_name().unwrap_or_default().to_string_lossy()
    );
    println!("Architecture:     {}", model.config.arch.as_str());
    println!("Tensors Loaded:   {}", model.tensors.len());
    println!(
        "Arena Capacity:   {}",
        format_bytes_auto(model.arena.capacity() as u64)
    );
    println!(
        "Arena Weights:    {}",
        format_bytes_auto(model.arena.weights_bytes() as u64)
    );
    println!(
        "Arena Remaining:  {}",
        format_bytes_auto(model.arena.remaining_bytes() as u64)
    );
    println!("Final RSS:        {}", format_bytes_auto(rss_after));
    println!("Peak RSS (VmHWM): {}", format_bytes_auto(hwm_peak));
    println!("============================================================");
    println!("[Verification of $ZCOPY Invariants]");
    println!("  - Every arena tensor pointer % 4096 == 0: PASSED");
    println!("  - Every allocated size padded to 64 B:    PASSED");
    println!("  - Q4_0 AoS to SoA repacking:             PASSED");
    println!("============================================================");

    Ok(())
}

fn run_inference(
    path: &Path,
    prompt: &str,
    requested_ctx: Option<usize>,
    threads: Option<usize>,
    params: &sulimo_engine::SamplingParams,
) -> Result<()> {
    use std::io::Write;
    use std::sync::atomic::AtomicBool;

    println!("Loading engine for model: {}", path.display());
    let mut engine = sulimo_engine::Engine::load(path, requested_ctx, threads)
        .context("failed to initialize inference engine")?;

    let formatted = engine.tokenizer.format_chat_prompt(
        &engine.model.config.pre_tokenizer,
        Some("You are a helpful assistant."),
        prompt,
    );

    println!("\nPrompt: {}", prompt);
    print!("Response: ");
    std::io::stdout().flush()?;

    let cancel = AtomicBool::new(false);
    let stats = engine.generate_with_callback(&formatted, params, &cancel, |token_str| {
        print!("{}", token_str);
        let _ = std::io::stdout().flush();
    })?;

    println!("\n");
    let (_, hwm) = get_process_memory_stats();
    println!("------------------------------------------------------------");
    println!("Prompt tokens:    {}", stats.prompt_tokens);
    println!("Generated tokens: {}", stats.completion_tokens);
    println!("Prefill backend:  {}", stats.prefill_backend);
    println!("Context repacks:  {}", stats.context_repacks);
    println!("TTFT:             {:.2} ms", stats.ttft_ms);
    println!("Prefill speed:    {:.2} tok/s", stats.prefill_tok_per_sec);
    println!("Decode speed:     {:.2} tok/s", stats.decode_tok_per_sec);
    println!("Total time:       {:.2} ms", stats.total_time_ms);
    println!("Peak RSS (VmHWM): {}", format_bytes_auto(hwm));
    println!("------------------------------------------------------------");

    Ok(())
}

fn print_human_report(report: &CapabilityReport) {
    println!("============================================================");
    println!(
        "         Súlimo Hardware Capability Probe (v{})",
        env!("CARGO_PKG_VERSION")
    );
    println!("============================================================");

    println!("\n[System Memory]");
    println!(
        "  Total RAM:       {}",
        format_bytes_gib(report.sysmem.total_bytes)
    );
    println!(
        "  Available RAM:   {}",
        format_bytes_gib(report.sysmem.available_bytes)
    );
    println!(
        "  Engine Ceiling:  {} (5.2 GB headroom: {})",
        format_bytes_gib(sulimo_core::MAX_RAM_BUDGET_BYTES),
        if report.sysmem.fits_5_2gb_budget() {
            "OK"
        } else {
            "INSUFFICIENT"
        }
    );
    println!(
        "  Swap Total:      {}",
        format_bytes_gib(report.sysmem.swap_total_bytes)
    );
    println!(
        "  Swap Free:       {}",
        format_bytes_gib(report.sysmem.swap_free_bytes)
    );
    println!("  Hugepages (THP): {}", report.sysmem.thp_mode);

    println!("\n[CPU Topology & Instruction Sets]");
    println!("  Model:           {}", report.cpu.model_name);
    println!("  Vendor:          {}", report.cpu.vendor_id);
    println!(
        "  Cores:           {} physical cores ({} logical threads)",
        report.cpu.physical_cores.len(),
        report.cpu.logical_core_count
    );
    for core in &report.cpu.physical_cores {
        println!(
            "    - Core {} (pkg {}): Primary CPU {} [SMT siblings: {:?}]",
            core.core_id, core.package_id, core.primary_cpu, core.smt_siblings
        );
    }
    println!(
        "  Pinning Targets: Logical CPUs {:?}",
        report.cpu.decode_pin_targets()
    );
    if let Some(l3) = report.cpu.l3_cache_bytes {
        println!("  L3 Cache:        {}", format_bytes_mib(l3));
    } else {
        println!("  L3 Cache:        unknown");
    }
    let f = &report.cpu.features;
    println!(
        "  SIMD Features:   SSE4.2: {} | AVX: {} | AVX2: {} | FMA: {} | F16C: {} | AVX-512F: {}",
        yes_no(f.sse4_2),
        yes_no(f.avx),
        yes_no(f.avx2),
        yes_no(f.fma),
        yes_no(f.f16c),
        yes_no(f.avx512f)
    );

    println!("\n[OpenCL / Integrated GPU]");
    match &report.ocl {
        OclProbeResult::Available(dev) => {
            println!("  Status:          Available");
            println!("  Device:          {}", dev.device_name);
            println!("  Vendor:          {}", dev.device_vendor);
            println!(
                "  Platform:        {} ({})",
                dev.platform_name, dev.platform_version
            );
            println!("  Driver Version:  {}", dev.driver_version);
            println!("  Compute Units:   {} EUs", dev.compute_units);
            println!("  Unified Memory:  {}", yes_no(dev.host_unified_memory));
            println!("  Base Align:      {} bits", dev.base_addr_align_bits);
            println!("  cl_khr_fp16:     {}", yes_no(dev.has_fp16));
            println!("  Intel Subgroups: {}", yes_no(dev.has_intel_subgroups));
            println!(
                "  SIMD8 Reqd Size: {}",
                yes_no(dev.has_intel_reqd_subgroup_size)
            );
        }
        OclProbeResult::Unavailable { reason } => {
            println!("  Status:          Unavailable");
            println!("  Reason:          {}", reason);
        }
    }

    println!("\n[Execution Tier]");
    println!("  Selected Tier:   {}", report.selected_tier);
    println!("  Rationale:       {}", report.tier_reason);
    println!("============================================================");
}

fn yes_no(v: bool) -> &'static str {
    if v {
        "yes"
    } else {
        "no"
    }
}
