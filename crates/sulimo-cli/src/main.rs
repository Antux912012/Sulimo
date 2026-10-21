//! Súlimo Command-Line Interface (CLI).
//!
//! Subcommands:
//! - `probe [--json]`: Inspects host hardware capabilities and determines the execution tier.

use anyhow::Result;
use std::env;
use sulimo_engine::CapabilityReport;
use sulimo_ocl::OclProbeResult;

fn print_usage() {
    eprintln!(
        "\
Súlimo CLI (v{})
Local LLM inference engine for Intel Gen8–10 hardware

USAGE:
    sulimo-cli <SUBCOMMAND> [OPTIONS]

SUBCOMMANDS:
    probe       Probe CPU, GPU, and system memory capabilities
    run         Execute inference on a GGUF model (M2+)
    bench       Run benchmarks across execution tiers (M6+)
    help        Print this message or the help of the given subcommand

OPTIONS:
    --json      Output machine-readable JSON (valid for 'probe')
    -h, --help  Print help information
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
        "run" | "bench" | "golden-compare" => {
            eprintln!(
                "Subcommand '{}' is planned for subsequent milestones (M1–M6).",
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
