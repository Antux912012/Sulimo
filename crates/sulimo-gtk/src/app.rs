//! Súlimo application setup and CSS styling provider (§7 M5).

use crate::ui::SulimoWindow;
use adw::Application;
use gtk4::gdk::Display;
use gtk4::prelude::*;
use gtk4::CssProvider;
use libadwaita as adw;

const APP_ID: &str = "io.github.antux912012.Sulimo";

const APP_CSS: &str = r#"
.arda-telemetry-bar {
    border-top: 1px solid alpha(@borders, 0.4);
    background-color: alpha(@window_bg_color, 0.85);
}

.phase-idle {
    background-color: alpha(@window_fg_color, 0.1);
    color: @window_fg_color;
    border-radius: 9999px;
    padding: 2px 10px;
    font-weight: 600;
    font-size: 0.8em;
}

.phase-prefill {
    background-color: alpha(#3584e4, 0.2);
    color: #3584e4;
    border: 1px solid #3584e4;
    border-radius: 9999px;
    padding: 2px 10px;
    font-weight: 600;
    font-size: 0.8em;
}

.phase-decode {
    background-color: alpha(#2ec27e, 0.2);
    color: #2ec27e;
    border: 1px solid #2ec27e;
    border-radius: 9999px;
    padding: 2px 10px;
    font-weight: 600;
    font-size: 0.8em;
}

.user-bubble {
    background-color: alpha(@accent_color, 0.18);
    border: 1px solid alpha(@accent_color, 0.4);
    border-radius: 16px;
    border-bottom-right-radius: 4px;
    padding: 12px 18px;
    max-width: 600px;
}

.assistant-bubble {
    background-color: alpha(@card_bg_color, 0.95);
    border: 1px solid alpha(@borders, 0.5);
    border-radius: 16px;
    border-bottom-left-radius: 4px;
    padding: 14px 20px;
    max-width: 680px;
}

.chat-input-bar {
    background-color: alpha(@window_bg_color, 0.9);
}

.chat-input-field {
    border-radius: 14px;
    font-size: 1.05em;
}
"#;

pub fn create_app() -> Application {
    let app = Application::builder().application_id(APP_ID).build();

    app.connect_startup(|_| {
        load_css();
    });

    app.connect_activate(|app| {
        let win = SulimoWindow::new(app);
        win.present();
    });

    app
}

fn load_css() {
    let provider = CssProvider::new();
    provider.load_from_string(APP_CSS);

    if let Some(display) = Display::default() {
        gtk4::style_context_add_provider_for_display(
            &display,
            &provider,
            gtk4::STYLE_PROVIDER_PRIORITY_APPLICATION,
        );
    }
}
