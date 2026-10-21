//! Súlimo GTK4 + Libadwaita client application entry point (@UI:GTK, §7 M5).

use gtk4::gio::prelude::*;
use gtk4::glib;
use sulimo_gtk::create_app;

fn main() -> glib::ExitCode {
    let app = create_app();
    app.run()
}
