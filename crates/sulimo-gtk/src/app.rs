//! Súlimo application setup and CSS styling provider (§7 M5).

use crate::ui::SulimoWindow;
use adw::Application;
use gtk4::gdk::Display;
use gtk4::prelude::*;
use gtk4::CssProvider;
use libadwaita as adw;

const APP_ID: &str = "io.github.antux912012.Sulimo";

const APP_CSS: &str = r#"
/* $NORD Palette Definitions */
@define-color nord0 #2E3440;
@define-color nord1 #3B4252;
@define-color nord2 #434C5E;
@define-color nord3 #4C566A;
@define-color nord4 #D8DEE9;
@define-color nord5 #E5E9F0;
@define-color nord6 #ECEFF4;
@define-color nord7 #8FBCBB;
@define-color nord8 #88C0D0;
@define-color nord9 #81A1C1;
@define-color nord10 #5E81AC;

window {
    background-color: @nord0;
    color: @nord4;
}

.arda-telemetry-bar {
    border-top: 1px solid @nord3;
    background-color: @nord1;
    color: @nord4;
}

.phase-idle {
    background-color: alpha(@nord3, 0.4);
    color: @nord5;
    border-radius: 9999px;
    padding: 2px 10px;
    font-weight: 600;
    font-size: 0.8em;
}

.phase-prefill {
    background-color: alpha(@nord8, 0.2);
    color: @nord8;
    border: 1px solid @nord8;
    border-radius: 9999px;
    padding: 2px 10px;
    font-weight: 600;
    font-size: 0.8em;
}

.phase-decode {
    background-color: alpha(@nord9, 0.2);
    color: @nord9;
    border: 1px solid @nord9;
    border-radius: 9999px;
    padding: 2px 10px;
    font-weight: 600;
    font-size: 0.8em;
}

.user-bubble {
    background-color: @nord2;
    border: 1px solid @nord3;
    color: @nord6;
    border-radius: 16px;
    border-bottom-right-radius: 4px;
    padding: 12px 18px;
    max-width: 600px;
}

.assistant-bubble {
    background-color: @nord1;
    border: 1px solid @nord3;
    color: @nord4;
    border-radius: 16px;
    border-bottom-left-radius: 4px;
    padding: 14px 20px;
    max-width: 680px;
}

.chat-input-bar {
    background-color: @nord0;
    border-top: 1px solid @nord2;
}

.chat-input-field {
    background-color: @nord1;
    color: @nord6;
    border: 1px solid @nord3;
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
