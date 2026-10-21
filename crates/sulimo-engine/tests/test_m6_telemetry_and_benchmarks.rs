//! Milestone M6 Acceptance Tests: Telemetry, benchmarks & threshold calibration (§7 M6, @C:ENG !TEST).
//!
//! Verifies:
//! 1. `run_benchmark_suite` runs against synthetic model up to context 4096.
//! 2. Prefill benchmarks (128, 512, 2048) record throughput, TTFT, and backend.
//! 3. Decode benchmark generates 128 tokens, measures decode tok/s and samples RC6 residency.
//! 4. Peak RSS (VmHWM) is strictly within 5.2 GB limit ($MEM<8G).
//! 5. Threshold calibration sweeps N in {1, 2, 4, 8, 16, 32} and picks an optimal crossover T.
//! 6. JSON serialization and deserialization adheres to schema.

use sulimo_engine::bench::{
    create_synthetic_benchmark_model, read_rc6_residency_ms, run_benchmark_suite, BenchmarkReport,
};

#[test]
fn test_m6_rc6_residency_read_safe() {
    // Reading sysfs RC6 should never panic even if unprivileged or non-existent
    let val = read_rc6_residency_ms();
    if let Some(ms) = val {
        println!("Detected live i915 RC6 residency: {ms} ms");
    } else {
        println!("RC6 residency unavailable in this environment (expected if container/virtualized/no i915)");
    }
}

#[test]
fn test_m6_benchmark_suite_execution_and_schema() {
    let model_path = create_synthetic_benchmark_model(4096);

    let report: BenchmarkReport =
        run_benchmark_suite(&model_path, Some(4096), Some(2), true).expect("run benchmark suite");

    let _ = std::fs::remove_file(&model_path);

    // 1. Host Hardware Info
    assert!(!report.host.cpu_model.is_empty());
    assert!(report.host.physical_cores >= 1);
    assert!(report.host.logical_cores >= 1);

    // 2. Prefill Benchmark Points: 128, 512, 2048
    assert_eq!(
        report.prefill.len(),
        3,
        "Prefill points must cover 128, 512, 2048"
    );
    assert_eq!(report.prefill[0].prompt_tokens, 128);
    assert_eq!(report.prefill[1].prompt_tokens, 512);
    assert_eq!(report.prefill[2].prompt_tokens, 2048);

    for p in &report.prefill {
        assert!(p.prefill_tok_per_sec > 0.0, "Throughput must be > 0");
        assert!(p.ttft_ms > 0.0, "TTFT must be > 0");
        assert!(!p.backend.is_empty());
    }

    // 3. Decode Benchmark: 128 tokens
    assert_eq!(report.decode.tokens_generated, 128);
    assert!(report.decode.decode_tok_per_sec > 0.0);

    // 4. Memory Footprint
    assert!(report.memory.within_5_2gb_budget);
    assert!(
        report.memory.peak_rss_bytes <= 5_200 * 1024 * 1024,
        "VmHWM must not breach 5.2 GB budget ceiling"
    );

    // 5. Threshold Calibration Sweep
    let sweep = report
        .threshold_calibration
        .as_ref()
        .expect("sweep must be populated when calibrate_t=true");
    assert_eq!(sweep.len(), 6, "Sweep must test N in [1, 2, 4, 8, 16, 32]");
    assert_eq!(sweep[0].n, 1);
    assert_eq!(sweep[5].n, 32);

    for pt in sweep {
        assert!(pt.cpu_prefill_ms > 0.0);
        assert!(!pt.faster_backend.is_empty());
    }
    assert!(report.recommended_threshold_t >= 1);

    // 6. JSON Serialization Round-trip
    let json_str = serde_json::to_string(&report).expect("serialize benchmark report to JSON");
    assert!(!json_str.is_empty());
    let deserialized: BenchmarkReport =
        serde_json::from_str(&json_str).expect("deserialize benchmark report from JSON");
    assert_eq!(deserialized.context_length, report.context_length);
    assert_eq!(deserialized.prefill.len(), report.prefill.len());
}
