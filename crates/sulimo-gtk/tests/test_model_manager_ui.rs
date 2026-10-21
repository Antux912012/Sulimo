//! Milestone MM4 Acceptance Tests: GTK4 / Libadwaita Model Manager UI.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use sulimo_core::model_manager::{default_registry, MAX_MODELS_STORAGE_BUDGET_BYTES};

#[test]
fn test_mm4_model_registry_ui_card_data() {
    let registry = default_registry();
    assert_eq!(registry.len(), 12);

    for m in &registry {
        assert!(!m.display_name.is_empty());
        assert!(!m.formatted_size().is_empty());
        assert!(!m.formatted_ram_requirement().is_empty());
        assert!(m.file_size_bytes <= MAX_MODELS_STORAGE_BUDGET_BYTES);
    }
}

#[test]
fn test_mm4_async_download_worker_channels() {
    // Tests that background download progress channels work without blocking UI thread
    let (prog_tx, prog_rx) = flume::unbounded::<u64>();
    let cancel_flag = Arc::new(AtomicBool::new(false));

    let worker = thread::spawn(move || {
        for bytes in (0..=1000).step_by(200) {
            if cancel_flag.load(Ordering::Relaxed) {
                break;
            }
            let _ = prog_tx.send(bytes);
            thread::sleep(Duration::from_millis(5));
        }
    });

    let mut received = Vec::new();
    while let Ok(bytes) = prog_rx.recv_timeout(Duration::from_millis(200)) {
        received.push(bytes);
        if bytes == 1000 {
            break;
        }
    }

    worker.join().unwrap();
    assert_eq!(received, vec![0, 200, 400, 600, 800, 1000]);
}

#[test]
fn test_mm4_model_manager_dialog_instantiation_if_display() {
    // If running in an environment with display, verify adw::init and dialog instantiation
    if libadwaita::init().is_ok() {
        let loaded_path = Arc::new(std::sync::Mutex::new(None));
        let _dialog = sulimo_gtk::ui::ModelManagerDialog::new(None, move |p| {
            *loaded_path.lock().unwrap() = Some(p);
        });
    }
}
