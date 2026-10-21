//! Súlimo main application window with Libadwaita chrome and asynchronous event handling (§7 M5).

use crate::engine_worker::{spawn_engine_worker, Cmd, Event, ExecutionPhase};
use crate::power::read_system_power;
use crate::ui::chat_view::ChatView;
use crate::ui::telemetry_bar::TelemetryBar;
use adw::prelude::*;
use adw::{
    AboutDialog, Application, ApplicationWindow, HeaderBar, Toast, ToastOverlay, ToolbarView,
    WindowTitle,
};
use gtk4::gdk::Key;
use gtk4::prelude::*;
use gtk4::{gio, glib};
use gtk4::{Button, EventControllerKey, FileDialog, FileFilter};
use libadwaita as adw;
use std::rc::Rc;
use std::time::Duration;
use sulimo_engine::sampler::SamplingParams;

pub struct SulimoWindow {
    pub window: ApplicationWindow,
}

impl SulimoWindow {
    pub fn new(app: &Application) -> Self {
        let (cmd_tx, event_rx) = spawn_engine_worker();

        let window = ApplicationWindow::builder()
            .application(app)
            .title("Súlimo")
            .default_width(900)
            .default_height(650)
            .build();

        let toast_overlay = ToastOverlay::new();
        let toolbar_view = ToolbarView::new();

        // 1. Header Bar
        let header_bar = HeaderBar::new();
        let window_title = WindowTitle::new("Súlimo", "Arda's Breath — Gen8–10 Legacy Tier");
        header_bar.set_title_widget(Some(&window_title));

        // Open Model Button
        let open_model_btn = Button::builder()
            .icon_name("document-open-symbolic")
            .tooltip_text("Open local GGUF file")
            .build();

        // Manage Models Button
        let manage_models_btn = Button::builder()
            .icon_name("folder-download-symbolic")
            .tooltip_text("Manage & Download Models (30 GB SSD Pool)")
            .build();

        // New Chat Button
        let new_chat_btn = Button::builder()
            .icon_name("document-new-symbolic")
            .tooltip_text("New conversation (clear context)")
            .build();

        // About Button
        let about_btn = Button::builder()
            .icon_name("help-about-symbolic")
            .tooltip_text("About Súlimo")
            .build();

        header_bar.pack_start(&manage_models_btn);
        header_bar.pack_start(&open_model_btn);
        header_bar.pack_start(&new_chat_btn);
        header_bar.pack_end(&about_btn);

        toolbar_view.add_top_bar(&header_bar);

        // 2. Chat View
        let chat_view = Rc::new(ChatView::new());
        toolbar_view.set_content(Some(&chat_view.container));

        // 3. Arda Telemetry Bar
        let telemetry_bar = Rc::new(TelemetryBar::new());
        toolbar_view.add_bottom_bar(&telemetry_bar.container);

        toast_overlay.set_child(Some(&toolbar_view));
        window.set_content(Some(&toast_overlay));

        // 4. Initial Telemetry Readout
        telemetry_bar.update(
            ExecutionPhase::Idle,
            0,
            5_200 * 1024 * 1024,
            0.0,
            &read_system_power(),
        );

        // 5. Wire Model Open FileDialog
        let window_weak = window.downgrade();
        let cmd_tx_open = cmd_tx.clone();
        let toast_overlay_open = toast_overlay.clone();

        open_model_btn.connect_clicked(move |_| {
            let Some(win) = window_weak.upgrade() else {
                return;
            };

            let filter = FileFilter::new();
            filter.set_name(Some("GGUF Models (*.gguf)"));
            filter.add_pattern("*.gguf");

            let filters = gio::ListStore::new::<FileFilter>();
            filters.append(&filter);

            let dialog = FileDialog::builder()
                .title("Select GGUF Model File")
                .modal(true)
                .filters(&filters)
                .build();

            let cmd_tx = cmd_tx_open.clone();
            let toast_overlay = toast_overlay_open.clone();

            dialog.open(Some(&win), gio::Cancellable::NONE, move |res| {
                if let Ok(file) = res {
                    if let Some(path) = file.path() {
                        let toast = Toast::new(&format!("Loading {}...", path.display()));
                        toast.set_timeout(3);
                        toast_overlay.add_toast(toast);

                        let _ = cmd_tx.send(Cmd::Load {
                            path,
                            requested_ctx: None,
                        });
                    }
                }
            });
        });

        // 6. Wire Manage Models Dialog
        let window_weak_models = window.downgrade();
        let cmd_tx_models = cmd_tx.clone();
        let toast_overlay_models = toast_overlay.clone();

        manage_models_btn.connect_clicked(move |_| {
            let Some(win) = window_weak_models.upgrade() else {
                return;
            };

            let cmd_tx = cmd_tx_models.clone();
            let toast_overlay = toast_overlay_models.clone();

            crate::ui::ModelManagerDialog::new(Some(win.upcast_ref()), move |path| {
                let toast = Toast::new(&format!("Loading {}...", path.display()));
                toast.set_timeout(3);
                toast_overlay.add_toast(toast);

                let _ = cmd_tx.send(Cmd::Load {
                    path,
                    requested_ctx: None,
                });
            });
        });

        // 7. Wire New Chat Button
        let chat_view_clear = Rc::clone(&chat_view);
        let toast_overlay_clear = toast_overlay.clone();
        new_chat_btn.connect_clicked(move |_| {
            chat_view_clear.clear();
            let toast = Toast::new("Conversation cleared.");
            toast.set_timeout(2);
            toast_overlay_clear.add_toast(toast);
        });

        // 7. Wire About Dialog
        let window_weak_about = window.downgrade();
        about_btn.connect_clicked(move |_| {
            let Some(win) = window_weak_about.upgrade() else {
                return;
            };

            let about = AboutDialog::builder()
                .application_name("Súlimo")
                .application_icon("io.github.antux912012.Sulimo")
                .version(env!("CARGO_PKG_VERSION"))
                .developer_name("Arda-Tech / Antonio")
                .license_type(gtk4::License::Gpl30)
                .website("https://antux912012.github.io/arda-tech/")
                .comments("Local, offline LLM inference engine with asymmetric iGPU/CPU battery scheduling designed for Intel 5th–10th Gen hardware.")
                .developers(vec!["Antonio <antux912012@gmail.com>"])
                .build();

            about.present(Some(&win));
        });

        // 8. Wire Prompt Submission & Stop Button
        let cmd_tx_send = cmd_tx.clone();
        let chat_view_send = Rc::clone(&chat_view);
        let submit_prompt = Rc::new(move || {
            let buffer = &chat_view_send.input_buffer;
            let start = buffer.start_iter();
            let end = buffer.end_iter();
            let text = buffer.text(&start, &end, false).trim().to_string();

            if text.is_empty() {
                return;
            }

            buffer.set_text("");
            chat_view_send.add_user_message(&text);
            chat_view_send.start_assistant_message();
            chat_view_send.set_generating(true);

            let params = SamplingParams {
                max_tokens: 512,
                temperature: 0.7,
                top_p: 0.9,
                ..Default::default()
            };

            let _ = cmd_tx_send.send(Cmd::Generate {
                prompt: text,
                params,
            });
        });

        let submit_prompt_btn = Rc::clone(&submit_prompt);
        chat_view.send_button.connect_clicked(move |_| {
            submit_prompt_btn();
        });

        let cmd_tx_stop = cmd_tx.clone();
        chat_view.stop_button.connect_clicked(move |_| {
            let _ = cmd_tx_stop.send(Cmd::Stop);
        });

        // Ctrl+Enter keyboard shortcut on input view
        let submit_key = Rc::clone(&submit_prompt);
        let key_controller = EventControllerKey::new();
        key_controller.connect_key_pressed(move |_, keyval, _, state| {
            if (keyval == Key::Return || keyval == Key::KP_Enter)
                && state.contains(gtk4::gdk::ModifierType::CONTROL_MASK)
            {
                submit_key();
                glib::Propagation::Stop
            } else {
                glib::Propagation::Proceed
            }
        });
        chat_view.input_text_view.add_controller(key_controller);

        // 9. Frame-Budget Coalescer Tick Callback (~16 ms)
        let chat_view_tick = Rc::clone(&chat_view);
        glib::timeout_add_local(Duration::from_millis(16), move || {
            chat_view_tick.flush_coalescer();
            glib::ControlFlow::Continue
        });

        // 10. Asynchronous Event Consumer Loop on GTK Main Context
        let chat_view_ev = Rc::clone(&chat_view);
        let telemetry_bar_ev = Rc::clone(&telemetry_bar);
        let window_title_ev = window_title.clone();
        let toast_overlay_ev = toast_overlay.clone();

        glib::spawn_future_local(async move {
            while let Ok(event) = event_rx.recv_async().await {
                match event {
                    Event::Loading(msg) => {
                        window_title_ev.set_subtitle(&msg);
                    }
                    Event::Loaded {
                        model_name,
                        planned_bytes,
                        ctx_len,
                    } => {
                        let planned_gb = (planned_bytes as f64) / (1024.0 * 1024.0 * 1024.0);
                        window_title_ev.set_subtitle(&format!("{model_name} ({ctx_len} ctx)"));

                        let toast = Toast::new(&format!(
                            "Ready: {model_name} loaded ({planned_gb:.2} GB arena memory)"
                        ));
                        toast.set_timeout(3);
                        toast_overlay_ev.add_toast(toast);
                    }
                    Event::Unloaded => {
                        window_title_ev.set_subtitle("No model loaded");
                        let toast = Toast::new("Model unloaded.");
                        toast.set_timeout(2);
                        toast_overlay_ev.add_toast(toast);
                    }
                    Event::Token(tok) => {
                        chat_view_ev.stream_token(&tok);
                    }
                    Event::Telemetry {
                        phase,
                        ram_bytes,
                        max_budget_bytes,
                        tok_per_sec,
                        power,
                    } => {
                        telemetry_bar_ev.update(
                            phase,
                            ram_bytes,
                            max_budget_bytes,
                            tok_per_sec,
                            &power,
                        );
                    }
                    Event::Done(stats) => {
                        chat_view_ev.flush_coalescer();
                        chat_view_ev.set_generating(false);

                        let toast = Toast::new(&format!(
                            "Generated {} tokens at {:.1} tok/s (prefill: {} @ {:.1} tok/s)",
                            stats.completion_tokens,
                            stats.decode_tok_per_sec,
                            stats.prefill_backend,
                            stats.prefill_tok_per_sec
                        ));
                        toast.set_timeout(3);
                        toast_overlay_ev.add_toast(toast);
                    }
                    Event::Error(err) => {
                        chat_view_ev.flush_coalescer();
                        chat_view_ev.set_generating(false);

                        let toast = Toast::new(&err);
                        toast.set_timeout(5);
                        toast_overlay_ev.add_toast(toast);
                    }
                }
            }
        });

        // 11. Auto-load first installed model if available in local store
        if let Ok(store) = sulimo_core::model_manager::ModelStore::new(None) {
            if let Ok(installed) = store.list_installed_models() {
                if let Some(first) = installed.first() {
                    let _ = cmd_tx.send(Cmd::Load {
                        path: first.path.clone(),
                        requested_ctx: None,
                    });
                }
            }
        }

        Self { window }
    }

    pub fn present(&self) {
        self.window.present();
    }
}
