//! Arda Telemetry Bar tracking Unified RAM, execution phase, tok/s, and power (§7 M5, A14, A15).

use crate::engine_worker::ExecutionPhase;
use crate::power::PowerReading;
use gtk4::prelude::*;
use gtk4::{Align, Box as GtkBox, Label, LevelBar, LevelBarMode, Orientation};

pub struct TelemetryBar {
    pub container: GtkBox,
    ram_level_bar: LevelBar,
    ram_label: Label,
    phase_badge: Label,
    speed_label: Label,
    power_label: Label,
}

impl TelemetryBar {
    pub fn new() -> Self {
        let container = GtkBox::new(Orientation::Horizontal, 12);
        container.set_margin_start(16);
        container.set_margin_end(16);
        container.set_margin_top(6);
        container.set_margin_bottom(6);
        container.add_css_class("arda-telemetry-bar");

        // 1. Unified RAM Level Bar
        let ram_box = GtkBox::new(Orientation::Horizontal, 8);
        let ram_icon = Label::new(Some("RAM:"));
        ram_icon.add_css_class("dim-label");

        let ram_level_bar = LevelBar::new();
        ram_level_bar.set_mode(LevelBarMode::Continuous);
        ram_level_bar.set_min_value(0.0);
        ram_level_bar.set_max_value(5200.0); // 5.2 GB in MB
        ram_level_bar.set_value(0.0);
        ram_level_bar.set_size_request(120, 10);
        ram_level_bar.set_valign(Align::Center);

        // Add offset marks at 80% (4160 MB) and 95% (4940 MB)
        ram_level_bar.add_offset_value("low", 4160.0);
        ram_level_bar.add_offset_value("high", 4940.0);

        let ram_label = Label::new(Some("0.00 / 5.20 GB"));
        ram_label.add_css_class("caption");
        ram_label.set_tooltip_text(Some(
            "Unified Engine RAM Budget (5.2 GB hard ceiling, $MEM<8G)",
        ));

        ram_box.append(&ram_icon);
        ram_box.append(&ram_level_bar);
        ram_box.append(&ram_label);

        // 2. Execution Phase Indicator Badge
        let phase_badge = Label::new(Some("Idle"));
        phase_badge.add_css_class("badge");
        phase_badge.add_css_class("phase-idle");
        phase_badge.set_tooltip_text(Some("Asymmetric Phase: Prefill (OpenCL iGPU FP16) vs Decode (CPU AVX2+FMA, iGPU in RC6 deep sleep)"));

        // 3. Tok/s speed readout
        let speed_label = Label::new(Some("0.0 tok/s"));
        speed_label.add_css_class("caption");
        speed_label.add_css_class("dim-label");

        // 4. Power consumption readout
        let power_label = Label::new(Some("Power: n/a"));
        power_label.add_css_class("caption");
        power_label.set_tooltip_text(Some(
            "Power metrics: RAPL package energy or battery discharge rate",
        ));

        container.append(&ram_box);
        container.append(&phase_badge);

        // Push right elements
        let spacer = GtkBox::new(Orientation::Horizontal, 0);
        spacer.set_hexpand(true);
        container.append(&spacer);

        container.append(&speed_label);
        container.append(&power_label);

        Self {
            container,
            ram_level_bar,
            ram_label,
            phase_badge,
            speed_label,
            power_label,
        }
    }

    /// Updates telemetry widgets with fresh data.
    pub fn update(
        &self,
        phase: ExecutionPhase,
        ram_bytes: usize,
        max_budget_bytes: usize,
        tok_per_sec: f64,
        power: &PowerReading,
    ) {
        // Update RAM
        let ram_mb = (ram_bytes as f64) / (1024.0 * 1024.0);
        let max_mb = (max_budget_bytes as f64) / (1024.0 * 1024.0);
        let ram_gb = ram_mb / 1024.0;
        let max_gb = max_mb / 1024.0;
        let pct = if max_mb > 0.0 {
            (ram_mb / max_mb) * 100.0
        } else {
            0.0
        };

        self.ram_level_bar.set_value(ram_mb);
        self.ram_label
            .set_text(&format!("{ram_gb:.2} / {max_gb:.2} GB ({pct:.0}%)"));

        // Update Phase badge
        self.phase_badge.remove_css_class("phase-idle");
        self.phase_badge.remove_css_class("phase-prefill");
        self.phase_badge.remove_css_class("phase-decode");

        match phase {
            ExecutionPhase::Idle => {
                self.phase_badge.set_text("Idle");
                self.phase_badge.add_css_class("phase-idle");
            }
            ExecutionPhase::PrefillGpu => {
                self.phase_badge.set_text("Prefill-GPU");
                self.phase_badge.add_css_class("phase-prefill");
            }
            ExecutionPhase::DecodeCpu => {
                self.phase_badge.set_text("Decode-CPU");
                self.phase_badge.add_css_class("phase-decode");
            }
        }

        // Update speed
        if tok_per_sec > 0.0 {
            self.speed_label
                .set_text(&format!("{tok_per_sec:.1} tok/s"));
        } else {
            self.speed_label.set_text("— tok/s");
        }

        // Update power
        self.power_label.set_text(&format!(
            "Power: {} ({})",
            power.display_text, power.source_label
        ));
        self.power_label.set_tooltip_text(Some(&power.tooltip));
    }
}

impl Default for TelemetryBar {
    fn default() -> Self {
        Self::new()
    }
}
