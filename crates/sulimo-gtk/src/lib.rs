//! Súlimo GTK4 + Libadwaita client application library (@UI:GTK, §7 M5).

pub mod app;
pub mod engine_worker;
pub mod frame_coalescer;
pub mod power;
pub mod ui;

pub use app::create_app;
pub use engine_worker::{Cmd, Event, ExecutionPhase};
pub use frame_coalescer::FrameCoalescer;
pub use power::{read_system_power, PowerReading};
pub use ui::SulimoWindow;
