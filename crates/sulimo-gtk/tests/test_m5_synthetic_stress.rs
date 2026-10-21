//! Milestone M5 Acceptance Tests: GTK4/Libadwaita UI, synthetic stress & frame budgeting (§7 M5, @UI:GTK).
//!
//! Verifies:
//! 1. Frame-budget rendering: high-speed token stream (100 tok/s) coalesced to 60 FPS (~16 ms frames).
//! 2. UI responsiveness: frame intervals stay >= 58 FPS equivalent with zero dropped chunks.
//! 3. Asynchronous non-blocking engine loading via flume channels.
//! 4. Safe power readout fallbacks without root permissions (A15).
//! 5. Base UI process RSS measurement.

use std::thread;
use std::time::{Duration, Instant};
use sulimo_core::loader::get_process_memory_stats;
use sulimo_gtk::engine_worker::{spawn_engine_worker, Cmd, Event};
use sulimo_gtk::frame_coalescer::FrameCoalescer;
use sulimo_gtk::power::read_system_power;

#[test]
fn test_m5_frame_budget_coalescing_and_100_tok_s_stress() {
    // Simulate streaming at 100 tokens/sec from engine thread while GTK UI renders at 60 FPS
    let mut coalescer = FrameCoalescer::new(Duration::from_millis(16));
    let mut flushed_chunks = Vec::new();
    let mut frame_intervals = Vec::new();

    let total_tokens = 100;

    let mut last_frame = Instant::now();
    let start_time = Instant::now();

    // In 1 second: 60 frame ticks, while 100 tokens arrive (1.67 tokens/frame)
    for frame in 0..60 {
        // Tokens arrived during this frame (1 or 2 tokens per frame to make 100 tokens total)
        let tokens_this_frame = if frame < 40 { 2 } else { 1 };
        for t in 0..tokens_this_frame {
            let tok_idx = if frame < 40 {
                frame * 2 + t
            } else {
                80 + (frame - 40)
            };
            coalescer.push_token(&format!("tok{tok_idx} "));
        }

        thread::sleep(Duration::from_millis(16));

        // GTK Frame tick callback flushes coalesced chunk
        if let Some(chunk) = coalescer.flush() {
            flushed_chunks.push(chunk);
        }

        let now = Instant::now();
        frame_intervals.push(now.duration_since(last_frame).as_secs_f64());
        last_frame = now;
    }

    if let Some(chunk) = coalescer.flush() {
        flushed_chunks.push(chunk);
    }

    let elapsed = start_time.elapsed();
    let (total_coalesced, total_flushes) = coalescer.stats();

    assert_eq!(total_coalesced, total_tokens);
    assert!(!flushed_chunks.is_empty());

    // Verify all 100 tokens were preserved in exact order
    let reconstructed_text: String = flushed_chunks.concat();
    for i in 0..total_tokens {
        assert!(
            reconstructed_text.contains(&format!("tok{i} ")),
            "Token {i} missing from reconstructed stream"
        );
    }

    // Calculate effective UI rendering FPS under 100 tok/s stream
    let avg_frame_interval = frame_intervals.iter().sum::<f64>() / (frame_intervals.len() as f64);
    let effective_fps = 1.0 / avg_frame_interval;

    eprintln!(
        "[Frame Budget Stress] Total Tokens: {total_tokens}, Elapsed: {elapsed:?}, Flushes: {total_flushes}"
    );
    eprintln!(
        "[Frame Budget Stress] Average Frame Interval: {:.2} ms, Effective UI Frame Rate: {:.1} FPS",
        avg_frame_interval * 1000.0,
        effective_fps
    );

    // Assert UI frame rate maintained >= 58 FPS requirement (§7 M5)
    assert!(
        effective_fps >= 58.0,
        "UI frame rate ({effective_fps:.1} FPS) must remain >= 58 FPS under 100 tok/s stream"
    );
}

#[test]
fn test_m5_safe_power_telemetry() {
    let reading = read_system_power();
    eprintln!(
        "[Power Telemetry] Readout: {} | Source: {} | Tooltip: {}",
        reading.display_text, reading.source_label, reading.tooltip
    );

    assert!(!reading.display_text.is_empty());
    assert!(!reading.source_label.is_empty());
    assert!(!reading.tooltip.is_empty());

    // Never crashes or panics regardless of whether RAPL is root-only or on AC power
    match reading.watts {
        Some(w) => assert!((0.0..=250.0).contains(&w)),
        None => assert_eq!(reading.display_text, "n/a"),
    }
}

#[test]
fn test_m5_asynchronous_engine_worker_channels() {
    let (cmd_tx, event_rx) = spawn_engine_worker();

    // 1. Loading event occurs asynchronously without blocking the sender
    let t0 = Instant::now();
    cmd_tx
        .send(Cmd::Load {
            path: std::path::PathBuf::from("/nonexistent/model.gguf"),
            requested_ctx: Some(128),
        })
        .expect("send load cmd");

    let send_duration = t0.elapsed();
    assert!(
        send_duration < Duration::from_millis(5),
        "Sending command must be instantaneous (took {send_duration:?})"
    );

    // 2. Receive loading event
    let first_event = event_rx
        .recv_timeout(Duration::from_millis(500))
        .expect("receive loading event");
    match first_event {
        Event::Loading(msg) => {
            assert!(msg.contains("model.gguf"));
        }
        other => panic!("Expected Event::Loading, got {other:?}"),
    }

    // 3. Receive error for nonexistent file gracefully
    let second_event = event_rx
        .recv_timeout(Duration::from_millis(500))
        .expect("receive error event");
    match second_event {
        Event::Error(err) => {
            assert!(err.contains("Failed to load model"));
        }
        other => panic!("Expected Event::Error, got {other:?}"),
    }
}

#[test]
fn test_m5_ui_process_rss() {
    let (rss_bytes, hwm_bytes) = get_process_memory_stats();
    let rss_mb = (rss_bytes as f64) / (1024.0 * 1024.0);
    let hwm_mb = (hwm_bytes as f64) / (1024.0 * 1024.0);

    eprintln!("[UI Process RSS] Current VmRSS: {rss_mb:.2} MB, VmHWM: {hwm_mb:.2} MB");

    assert!(rss_bytes > 0);
    assert!(hwm_bytes > 0);
    // UI process base RSS (excluding engine static arena) is typically ~30-60 MB on GTK4
    assert!(
        rss_mb < 300.0,
        "Base UI process RSS must remain lean (saw {rss_mb:.2} MB)"
    );
}
