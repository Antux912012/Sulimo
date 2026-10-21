//! System power telemetry readout per A15 (§7 M5, $BATT).
//!
//! Tries Intel RAPL `energy_uj` first (package power).
//! Falls back to `/sys/class/power_supply/BAT*/power_now` (system draw on battery).
//! Otherwise displays "n/a" with a clear explanatory tooltip. Never requires root.

use std::fs;
use std::path::Path;
use std::sync::Mutex;
use std::time::Instant;

struct RaplState {
    last_energy_uj: u64,
    last_sample_time: Instant,
}

static RAPL_TRACKER: Mutex<Option<RaplState>> = Mutex::new(None);

/// Result of reading system / package power.
#[derive(Debug, Clone)]
pub struct PowerReading {
    pub display_text: String,
    pub source_label: String,
    pub tooltip: String,
    pub watts: Option<f64>,
}

/// Reads current system power draw, prioritizing RAPL package energy if accessible.
pub fn read_system_power() -> PowerReading {
    // 1. Try Intel RAPL energy counter
    if let Some(watts) = try_read_rapl_watts() {
        return PowerReading {
            display_text: format!("{watts:.1} W"),
            source_label: "pkg (RAPL)".to_string(),
            tooltip: format!(
                "Intel CPU/iGPU package power consumption via RAPL sysfs: {watts:.2} W"
            ),
            watts: Some(watts),
        };
    }

    // 2. Try Battery power_now or (current_now * voltage_now)
    if let Some(watts) = try_read_battery_watts() {
        return PowerReading {
            display_text: format!("{watts:.1} W"),
            source_label: "system (battery)".to_string(),
            tooltip: format!("Total host battery discharge power draw: {watts:.2} W"),
            watts: Some(watts),
        };
    }

    // 3. Fallback: n/a
    PowerReading {
        display_text: "n/a".to_string(),
        source_label: "n/a".to_string(),
        tooltip: "Power metrics unavailable (RAPL is restricted to root, or host is connected to AC mains without active battery discharge).".to_string(),
        watts: None,
    }
}

fn try_read_rapl_watts() -> Option<f64> {
    let rapl_path = Path::new("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj");
    let content = fs::read_to_string(rapl_path).ok()?;
    let energy_uj: u64 = content.trim().parse().ok()?;

    let mut tracker = RAPL_TRACKER.lock().ok()?;
    let now = Instant::now();

    if let Some(prev) = tracker.as_mut() {
        let elapsed_sec = now.duration_since(prev.last_sample_time).as_secs_f64();
        if elapsed_sec >= 0.2 {
            let delta_uj = energy_uj.wrapping_sub(prev.last_energy_uj);
            let watts = (delta_uj as f64) / (elapsed_sec * 1_000_000.0);
            prev.last_energy_uj = energy_uj;
            prev.last_sample_time = now;
            if (0.0..=200.0).contains(&watts) {
                return Some(watts);
            }
        }
    } else {
        *tracker = Some(RaplState {
            last_energy_uj: energy_uj,
            last_sample_time: now,
        });
    }

    None
}

fn try_read_battery_watts() -> Option<f64> {
    let base = Path::new("/sys/class/power_supply");
    let entries = fs::read_dir(base).ok()?;

    for entry in entries.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with("BAT") {
            let path = entry.path();

            // Check power_now in microwatts
            if let Ok(content) = fs::read_to_string(path.join("power_now")) {
                if let Ok(val) = content.trim().parse::<u64>() {
                    if val > 0 {
                        let watts = (val as f64) / 1_000_000.0;
                        return Some(watts);
                    }
                }
            }

            // Check current_now (microamperes) and voltage_now (microvolts)
            let curr = fs::read_to_string(path.join("current_now"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok());
            let volt = fs::read_to_string(path.join("voltage_now"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok());

            if let (Some(c), Some(v)) = (curr, volt) {
                if c > 0 && v > 0 {
                    let u_watts = (c as f64) * (v as f64) / 1_000_000.0;
                    let watts = u_watts / 1_000_000.0;
                    return Some(watts);
                }
            }
        }
    }

    None
}
