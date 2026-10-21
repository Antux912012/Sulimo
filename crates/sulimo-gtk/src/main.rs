//! Súlimo GTK4 + Libadwaita client application entry point (@UI:GTK).
//!
//! Full UI implementation is delivered in Milestone M5 per SULIMO_GEN8_10_BUILD_PLAN.md.

fn main() {
    println!(
        "Súlimo GTK4/Libadwaita desktop client v{} (Milestone M5).",
        env!("CARGO_PKG_VERSION")
    );
    println!("For headless capability probe, run: sulimo-cli probe [--json]");
}
