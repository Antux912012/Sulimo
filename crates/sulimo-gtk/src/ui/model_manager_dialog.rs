//! Model manager dialog for browsing, downloading, and deleting local AI models.

use adw::prelude::*;
use adw::{ActionRow, PreferencesDialog, PreferencesGroup};
use gtk4::prelude::*;
use gtk4::{glib, Box as GtkBox, Button, Label, Orientation, ProgressBar};
use libadwaita as adw;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use sulimo_core::model_manager::{
    default_registry, format_bytes, DownloadError, DownloadProgress, DownloadState,
    ModelDownloader, ModelMetadata, ModelStore, MAX_MODELS_STORAGE_BUDGET_BYTES,
};

/// Libadwaita PreferencesDialog presenting model cards, storage gauge, and streaming downloads.
pub struct ModelManagerDialog {
    pub dialog: PreferencesDialog,
}

impl ModelManagerDialog {
    pub fn new<F>(parent: Option<&gtk4::Window>, on_load_model: F) -> Self
    where
        F: Fn(std::path::PathBuf) + Clone + 'static,
    {
        let dialog = PreferencesDialog::builder()
            .title("Model Manager")
            .search_enabled(true)
            .build();

        let page = adw::PreferencesPage::new();
        page.set_title("Models");
        page.set_icon_name(Some("folder-download-symbolic"));

        let store = Rc::new(RefCell::new(
            ModelStore::new(None).expect("Failed to initialize model store"),
        ));

        // 1. Dedicated Storage Pool Group
        let storage_group = PreferencesGroup::new();
        storage_group.set_title("Dedicated Model Storage Pool");

        let storage_row = ActionRow::new();
        let storage_progress = ProgressBar::builder()
            .valign(gtk4::Align::Center)
            .hexpand(true)
            .build();

        update_storage_gauge(&storage_row, &storage_progress, &store.borrow());

        storage_row.add_suffix(&storage_progress);
        storage_group.add(&storage_row);
        page.add(&storage_group);

        // 2. Curated Models Group
        let models_group = PreferencesGroup::new();
        models_group.set_title("Curated Local AI Models");
        models_group.set_description(Some(
            "Hardware Envelope: 8 GB RAM | 5.20 GiB Unified RAM Budget Ceiling",
        ));

        let registry = default_registry();

        for meta in registry {
            let row = create_model_row(
                &meta,
                store.clone(),
                &storage_row,
                &storage_progress,
                on_load_model.clone(),
            );
            models_group.add(&row);
        }

        page.add(&models_group);
        dialog.add(&page);

        if let Some(win) = parent {
            dialog.present(Some(win));
        }

        Self { dialog }
    }
}

fn create_model_row<F>(
    meta: &ModelMetadata,
    store: Rc<RefCell<ModelStore>>,
    storage_row: &ActionRow,
    storage_progress: &ProgressBar,
    on_load_model: F,
) -> ActionRow
where
    F: Fn(std::path::PathBuf) + Clone + 'static,
{
    let arch_badge = match meta.architecture {
        sulimo_core::model_config::ModelArch::Qwen2 => "Qwen2",
        sulimo_core::model_config::ModelArch::Gemma2 => "Gemma2",
        sulimo_core::model_config::ModelArch::LFM => "LFM",
        sulimo_core::model_config::ModelArch::MiniCPM3 => "MiniCPM3",
        sulimo_core::model_config::ModelArch::Llama => "Llama",
    };

    let title = format!("{}  [{}]", meta.display_name, arch_badge);
    let subtitle = format!(
        "{} • Size: {} • 4K RAM: {} • Quant: {}\n{}",
        meta.family,
        meta.formatted_size(),
        meta.formatted_ram_requirement(),
        meta.quantization,
        meta.description
    );

    let row = ActionRow::builder().title(title).subtitle(subtitle).build();

    let action_box = GtkBox::new(Orientation::Horizontal, 8);
    action_box.set_valign(gtk4::Align::Center);

    update_row_actions(
        &row,
        &action_box,
        meta,
        store,
        storage_row,
        storage_progress,
        on_load_model,
    );

    row.add_suffix(&action_box);
    row
}

fn update_row_actions<F>(
    row: &ActionRow,
    action_box: &GtkBox,
    meta: &ModelMetadata,
    store: Rc<RefCell<ModelStore>>,
    storage_row: &ActionRow,
    storage_progress: &ProgressBar,
    on_load_model: F,
) where
    F: Fn(std::path::PathBuf) + Clone + 'static,
{
    // Clear existing children
    while let Some(child) = action_box.first_child() {
        action_box.remove(&child);
    }

    let is_installed = store.borrow().is_model_installed(&meta.id);

    if is_installed {
        let load_btn = Button::builder()
            .label("Load Model")
            .css_classes(["suggested-action"])
            .build();

        let delete_btn = Button::builder()
            .icon_name("user-trash-symbolic")
            .tooltip_text("Delete model file from disk")
            .css_classes(["destructive-action"])
            .build();

        let meta_id = meta.id.clone();
        let store_clone = store.clone();
        let on_load = on_load_model.clone();
        load_btn.connect_clicked(move |_| {
            if let Some(path) = store_clone.borrow().get_model_path(&meta_id) {
                on_load(path);
            }
        });

        let row_weak = row.downgrade();
        let action_box_weak = action_box.downgrade();
        let meta_clone = meta.clone();
        let store_del = store;
        let storage_row_clone = storage_row.clone();
        let storage_prog_clone = storage_progress.clone();
        let on_load_del = on_load_model;

        delete_btn.connect_clicked(move |_| {
            let _ = store_del.borrow().delete_model(&meta_clone.id);
            update_storage_gauge(&storage_row_clone, &storage_prog_clone, &store_del.borrow());

            if let (Some(r), Some(ab)) = (row_weak.upgrade(), action_box_weak.upgrade()) {
                update_row_actions(
                    &r,
                    &ab,
                    &meta_clone,
                    store_del.clone(),
                    &storage_row_clone,
                    &storage_prog_clone,
                    on_load_del.clone(),
                );
            }
        });

        action_box.append(&load_btn);
        action_box.append(&delete_btn);
    } else {
        let download_btn = Button::builder()
            .label(format!("Download ({})", meta.formatted_size()))
            .icon_name("folder-download-symbolic")
            .build();

        let row_weak = row.downgrade();
        let action_box_weak = action_box.downgrade();
        let meta_clone = meta.clone();
        let store_dl = store;
        let storage_row_clone = storage_row.clone();
        let storage_prog_clone = storage_progress.clone();
        let on_load_clone = on_load_model;

        download_btn.connect_clicked(move |_| {
            let Some(ab) = action_box_weak.upgrade() else {
                return;
            };

            // Switch action_box to in-progress download widget
            while let Some(child) = ab.first_child() {
                ab.remove(&child);
            }

            let dl_progress = ProgressBar::builder()
                .fraction(0.0)
                .valign(gtk4::Align::Center)
                .hexpand(true)
                .build();
            dl_progress.set_size_request(160, -1);

            let status_label = Label::builder()
                .label("Starting...")
                .css_classes(["dim-label"])
                .build();

            let cancel_btn = Button::builder()
                .icon_name("process-stop-symbolic")
                .tooltip_text("Cancel download")
                .build();

            ab.append(&dl_progress);
            ab.append(&status_label);
            ab.append(&cancel_btn);

            let cancel_flag = Arc::new(AtomicBool::new(false));
            let cancel_flag_btn = cancel_flag.clone();
            cancel_btn.connect_clicked(move |_| {
                cancel_flag_btn.store(true, Ordering::SeqCst);
            });

            let (prog_tx, prog_rx) = flume::unbounded::<DownloadProgress>();
            let (done_tx, done_rx) = flume::bounded::<Result<std::path::PathBuf, DownloadError>>(1);

            let meta_thread = meta_clone.clone();
            let store_thread = store_dl.borrow().clone();

            thread::spawn(move || {
                let downloader = ModelDownloader::new(store_thread);
                let p_tx = prog_tx.clone();
                let res = downloader.download(&meta_thread, cancel_flag, move |p| {
                    let _ = p_tx.send(p);
                });
                let _ = done_tx.send(res);
            });

            // UI event listener on GTK main loop
            let row_ev = row_weak.clone();
            let ab_ev = action_box_weak.clone();
            let meta_ev = meta_clone.clone();
            let store_ev = store_dl.clone();
            let s_row_ev = storage_row_clone.clone();
            let s_prog_ev = storage_prog_clone.clone();
            let on_load_ev = on_load_clone.clone();

            glib::spawn_future_local(async move {
                while let Ok(progress) = prog_rx.recv_async().await {
                    dl_progress.set_fraction(progress.percent as f64);
                    match progress.state {
                        DownloadState::Downloading => {
                            let speed = if progress.speed_bps >= 1_000_000.0 {
                                format!("{:.1} MB/s", progress.speed_bps / 1_000_000.0)
                            } else {
                                format!("{:.0} KB/s", progress.speed_bps / 1_000.0)
                            };
                            let eta = match progress.eta_seconds {
                                Some(s) => format!(" • {s}s left"),
                                None => String::new(),
                            };
                            status_label.set_text(&format!("{speed}{eta}"));
                        }
                        DownloadState::VerifyingChecksum => {
                            status_label.set_text("Verifying SHA-256...");
                        }
                        _ => {}
                    }
                }

                if let Ok(res) = done_rx.recv_async().await {
                    if let (Some(r), Some(ab_up)) = (row_ev.upgrade(), ab_ev.upgrade()) {
                        update_storage_gauge(&s_row_ev, &s_prog_ev, &store_ev.borrow());

                        update_row_actions(
                            &r,
                            &ab_up,
                            &meta_ev,
                            store_ev,
                            &s_row_ev,
                            &s_prog_ev,
                            on_load_ev.clone(),
                        );

                        if let Ok(path) = res {
                            on_load_ev(path);
                        }
                    }
                }
            });
        });

        action_box.append(&download_btn);
    }
}

fn update_storage_gauge(
    storage_row: &ActionRow,
    storage_progress: &ProgressBar,
    store: &sulimo_core::model_manager::ModelStore,
) {
    let used = store.total_storage_used_bytes().unwrap_or(0);
    let frac = (used as f64 / MAX_MODELS_STORAGE_BUDGET_BYTES as f64).clamp(0.0, 1.0);
    storage_progress.set_fraction(frac);
    storage_row.set_title(&format!(
        "SSD Storage Pool: {} / {} used",
        format_bytes(used),
        format_bytes(MAX_MODELS_STORAGE_BUDGET_BYTES)
    ));
    storage_row.set_subtitle(&format!(
        "{} free dedicated storage for models (8 GB Host RAM / 256 GB SSD)",
        format_bytes(MAX_MODELS_STORAGE_BUDGET_BYTES.saturating_sub(used))
    ));
}
