//! CLI model management commands: list, search, download, remove, verify.

use anyhow::{bail, Result};
use serde::Serialize;
use std::io::{self, BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use sulimo_core::model_manager::{
    default_registry, find_model_by_id, format_bytes, DownloadError, DownloadProgress,
    DownloadState, ModelDownloader, ModelMetadata, ModelStore, MAX_MODELS_STORAGE_BUDGET_BYTES,
};

static CLI_CANCEL_FLAG: AtomicBool = AtomicBool::new(false);

extern "C" fn sigint_handler(_: libc::c_int) {
    CLI_CANCEL_FLAG.store(true, Ordering::SeqCst);
}

/// Dispatches subcommands under `sulimo-cli models [SUBCOMMAND]`.
pub fn handle_models_command(args: &[String]) -> Result<()> {
    let store = ModelStore::new(None)?;

    if args.is_empty() {
        return print_models_table(&store, false);
    }

    match args[0].as_str() {
        "list" | "ls" => {
            let json_mode = args.iter().any(|a| a == "--json");
            print_models_table(&store, json_mode)
        }
        "search" | "find" => {
            if args.len() < 2 {
                bail!("Missing search query. Usage: sulimo-cli models search <query>");
            }
            search_models(&store, &args[1])
        }
        "download" | "pull" | "get" => {
            if args.len() < 2 {
                bail!("Missing model ID. Usage: sulimo-cli models download <id>");
            }
            download_model_cli(&store, &args[1])?;
            Ok(())
        }
        "remove" | "rm" | "delete" => {
            if args.len() < 2 {
                bail!("Missing model ID. Usage: sulimo-cli models remove <id>");
            }
            remove_model_cli(&store, &args[1])
        }
        "verify" | "check" => {
            if args.len() < 2 {
                bail!("Missing model ID. Usage: sulimo-cli models verify <id>");
            }
            verify_model_cli(&store, &args[1])
        }
        "-h" | "--help" | "help" => {
            print_models_help();
            Ok(())
        }
        other => {
            // If user typed 'sulimo-cli models qwen2.5-1.5b-instruct', assume download or info
            if find_model_by_id(other).is_some() || store.is_model_installed(other) {
                download_model_cli(&store, other)?;
                Ok(())
            } else {
                eprintln!("\x1b[1;31mUnknown models subcommand:\x1b[0m '{other}'");
                print_models_help();
                std::process::exit(1);
            }
        }
    }
}

pub fn print_models_help() {
    println!(
        "\
\x1b[1mSúlimo Model Management CLI\x1b[0m

\x1b[1mUSAGE:\x1b[0m
    sulimo-cli models [SUBCOMMAND]

\x1b[1mSUBCOMMANDS:\x1b[0m
    list, ls [--json]     List all curated models, download status, and storage quota
    search <query>        Search curated models by keyword, architecture, or family
    download, pull <id>   Download and verify a curated model with resumable streaming
    remove, rm <id>       Remove an installed model and reclaim SSD storage
    verify <id>           Verify cryptographic SHA-256 integrity of an installed model
    help                  Print this help message
"
    );
}

/// Prints a formatted table of curated and installed models.
pub fn print_models_table(store: &ModelStore, json_mode: bool) -> Result<()> {
    let registry = default_registry();
    let installed = store.list_installed_models()?;
    let used_bytes = store.total_storage_used_bytes()?;
    let remaining_bytes = store.remaining_quota_bytes()?;

    if json_mode {
        #[derive(Serialize)]
        struct ModelsReport<'a> {
            storage_budget_bytes: u64,
            storage_used_bytes: u64,
            storage_remaining_bytes: u64,
            models_dir: String,
            installed: &'a [sulimo_core::model_manager::InstalledModel],
            registry: &'a [ModelMetadata],
        }

        let report = ModelsReport {
            storage_budget_bytes: MAX_MODELS_STORAGE_BUDGET_BYTES,
            storage_used_bytes: used_bytes,
            storage_remaining_bytes: remaining_bytes,
            models_dir: store.models_dir().display().to_string(),
            installed: &installed,
            registry: &registry,
        };

        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }

    println!("\x1b[38;5;141m========================================================================================================================\x1b[0m");
    println!(
        "\x1b[1m                                             Súlimo Curated Model Registry\x1b[0m"
    );
    println!("                                Host Envelope: 8 GB RAM | 30.0 GB Dedicated SSD Storage Pool");
    println!("\x1b[38;5;141m========================================================================================================================\x1b[0m");
    println!(
        " \x1b[1m{:<25} {:<10} {:<8} {:<11} {:<11} {:<15}\x1b[0m",
        "ID", "Arch", "Quant", "File Size", "4K RAM", "Status"
    );
    println!("------------------------------------------------------------------------------------------------------------------------");

    for m in &registry {
        let is_inst = store.is_model_installed(&m.id) || store.is_model_installed(&m.filename);
        let status_str = if is_inst {
            "\x1b[38;5;49m● Installed\x1b[0m"
        } else {
            "\x1b[38;5;244m○ Available\x1b[0m"
        };

        let arch_str = match m.architecture {
            sulimo_core::model_config::ModelArch::Qwen2 => "Qwen2",
            sulimo_core::model_config::ModelArch::Gemma2 => "Gemma2",
            sulimo_core::model_config::ModelArch::LFM => "LFM",
            sulimo_core::model_config::ModelArch::MiniCPM3 => "MiniCPM3",
            sulimo_core::model_config::ModelArch::Llama => "Llama",
        };

        println!(
            " {:<25} {:<10} {:<8} {:<11} {:<11} {}",
            m.id,
            arch_str,
            m.quantization,
            m.formatted_size(),
            m.formatted_ram_requirement(),
            status_str
        );
    }

    // Check for custom uncurated models in models directory
    let custom_models: Vec<_> = installed.iter().filter(|i| !i.is_registered).collect();
    if !custom_models.is_empty() {
        println!("------------------------------------------------------------------------------------------------------------------------");
        println!(" \x1b[1mCustom Models Installed in Store:\x1b[0m");
        for c in custom_models {
            println!("   • {} ({})", c.filename, format_bytes(c.size_bytes));
        }
    }

    println!("------------------------------------------------------------------------------------------------------------------------");
    println!(
        " \x1b[1mStorage Pool:\x1b[0m {} / {} used ({} available on SSD)",
        format_bytes(used_bytes),
        format_bytes(MAX_MODELS_STORAGE_BUDGET_BYTES),
        format_bytes(remaining_bytes)
    );
    println!(" \x1b[1mRAM Envelope:\x1b[0m 5.20 GiB Unified Memory Ceiling ($MEM<8G)");
    println!(
        " \x1b[1mStore Path:\x1b[0m   {}",
        store.models_dir().display()
    );
    println!("\x1b[38;5;141m========================================================================================================================\x1b[0m");
    println!("  Type \x1b[1;36msulimo-cli models download <id>\x1b[0m to install, or \x1b[1;36msulimo-cli run <id>\x1b[0m to chat.");

    Ok(())
}

/// Searches curated models by keyword.
pub fn search_models(store: &ModelStore, query: &str) -> Result<()> {
    let q = query.to_lowercase();
    let registry = default_registry();
    let matches: Vec<_> = registry
        .iter()
        .filter(|m| {
            m.id.to_lowercase().contains(&q)
                || m.display_name.to_lowercase().contains(&q)
                || m.family.to_lowercase().contains(&q)
                || m.quantization.to_lowercase().contains(&q)
                || m.description.to_lowercase().contains(&q)
        })
        .collect();

    if matches.is_empty() {
        println!("\x1b[38;5;214m[!] No models found matching query:\x1b[0m '{query}'");
        println!("    Type \x1b[1;36msulimo-cli models list\x1b[0m to view all available models.");
        return Ok(());
    }

    println!(
        "\x1b[1mFound {} model(s) matching '{}':\x1b[0m",
        matches.len(),
        query
    );
    println!("----------------------------------------------------------------------------------------------------");
    for m in matches {
        let is_inst = store.is_model_installed(&m.id) || store.is_model_installed(&m.filename);
        let status_str = if is_inst {
            "\x1b[38;5;49m● Installed\x1b[0m"
        } else {
            "\x1b[38;5;244m○ Available\x1b[0m"
        };

        println!(
            " • \x1b[1;36m{:<24}\x1b[0m [{}]  Size: {:<8}  RAM: {:<9}  {}",
            m.id,
            m.display_name,
            m.formatted_size(),
            m.formatted_ram_requirement(),
            status_str
        );
        println!("   \x1b[38;5;244m{}\x1b[0m", m.description);
    }
    println!("----------------------------------------------------------------------------------------------------");

    Ok(())
}

/// Downloads a model with a real-time progress bar and Ctrl-C trap.
pub fn download_model_cli(store: &ModelStore, model_id_or_name: &str) -> Result<PathBuf> {
    let meta = match find_model_by_id(model_id_or_name) {
        Some(m) => m,
        None => {
            // Check if matches filename
            let reg = default_registry();
            match reg
                .iter()
                .find(|m| m.filename.eq_ignore_ascii_case(model_id_or_name))
                .cloned()
            {
                Some(m) => m,
                None => {
                    bail!(
                        "Model '{}' not found in curated registry. Type 'sulimo-cli models list' to view available models.",
                        model_id_or_name
                    );
                }
            }
        }
    };

    if store.is_model_installed(&meta.id) {
        if let Some(path) = store.get_model_path(&meta.id) {
            println!(
                "\x1b[38;5;49m[✓] Model '{}' is already installed:\x1b[0m {}",
                meta.id,
                path.display()
            );
            return Ok(path);
        }
    }

    println!(
        "\x1b[1;38;5;51m[•] Preparing download for:\x1b[0m \x1b[1m{}\x1b[0m ({})",
        meta.display_name,
        meta.formatted_size()
    );
    println!("    Endpoint:     {}", meta.download_url);
    println!(
        "    Storage Path: {}/{}",
        store.models_dir().display(),
        meta.filename
    );
    println!("    SHA-256:      {}", meta.sha256);

    // Setup cooperative cancellation via SIGINT (Ctrl-C)
    CLI_CANCEL_FLAG.store(false, Ordering::SeqCst);
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let cancel_clone = cancel_flag.clone();

    unsafe {
        libc::signal(
            libc::SIGINT,
            sigint_handler as *const () as libc::sighandler_t,
        );
    }

    // Monitor thread synchronizing CLI_CANCEL_FLAG to cancel_flag
    let monitor_running = Arc::new(AtomicBool::new(true));
    let m_run_clone = monitor_running.clone();
    let _monitor_thread = std::thread::spawn(move || {
        while m_run_clone.load(Ordering::Relaxed) {
            if CLI_CANCEL_FLAG.load(Ordering::Relaxed) {
                cancel_flag.store(true, Ordering::SeqCst);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
    });

    let downloader = ModelDownloader::new(store.clone());

    let res = downloader.download(&meta, cancel_clone, |p| {
        render_cli_progress_bar(&p);
    });

    // Cleanup monitor thread and restore default SIGINT
    monitor_running.store(false, Ordering::Relaxed);
    unsafe {
        libc::signal(libc::SIGINT, libc::SIG_DFL);
    }

    println!(); // Line break after progress bar

    match res {
        Ok(path) => {
            println!(
                "\x1b[1;38;5;49m[✓] Successfully downloaded and verified '{}'!\x1b[0m",
                meta.id
            );
            println!("    Installed path: {}", path.display());
            println!(
                "    You can run this model now with: \x1b[1;36msulimo-cli run {}\x1b[0m",
                meta.id
            );
            Ok(path)
        }
        Err(DownloadError::Cancelled) => {
            eprintln!(
                "\n\x1b[38;5;214m[!] Download cancelled by user.\x1b[0m Partial progress is saved for resumption."
            );
            bail!("Download cancelled.");
        }
        Err(DownloadError::ChecksumMismatch {
            model_id,
            expected,
            actual,
        }) => {
            eprintln!("\n\x1b[1;31m[✗] Cryptographic checksum mismatch for '{model_id}'!\x1b[0m");
            eprintln!("    Expected SHA-256: {expected}");
            eprintln!("    Actual SHA-256:   {actual}");
            eprintln!("    File was rejected and not installed.");
            bail!("Checksum verification failed.");
        }
        Err(DownloadError::StorageQuotaExceeded {
            current_used,
            needed,
            max_quota,
        }) => {
            eprintln!("\n\x1b[1;31m[✗] Súlimo dedicated model storage quota exceeded!\x1b[0m");
            eprintln!("    Current storage used: {}", format_bytes(current_used));
            eprintln!("    Required additional:  {}", format_bytes(needed));
            eprintln!("    Max pool budget:      {}", format_bytes(max_quota));
            eprintln!("    Use 'sulimo-cli models remove <id>' to remove unused models.");
            bail!("Storage quota exceeded.");
        }
        Err(DownloadError::InsufficientDiskSpace {
            available,
            required,
        }) => {
            eprintln!("\n\x1b[1;31m[✗] Insufficient filesystem disk space!\x1b[0m");
            eprintln!("    Available on disk:  {}", format_bytes(available));
            eprintln!(
                "    Required (with 500 MiB safety headroom): {}",
                format_bytes(required)
            );
            bail!("Insufficient disk space.");
        }
        Err(e) => {
            eprintln!("\n\x1b[1;31m[✗] Download failed: {e}\x1b[0m");
            bail!(e);
        }
    }
}

/// Removes an installed model from the store.
pub fn remove_model_cli(store: &ModelStore, model_id_or_name: &str) -> Result<()> {
    if !store.is_model_installed(model_id_or_name) {
        println!(
            "\x1b[38;5;214m[!] Model '{}' is not installed.\x1b[0m",
            model_id_or_name
        );
        return Ok(());
    }

    let path = store.get_model_path(model_id_or_name);
    let size = path
        .as_ref()
        .and_then(|p| std::fs::metadata(p).ok())
        .map(|m| m.len())
        .unwrap_or(0);

    print!(
        "Are you sure you want to delete model '{}' ({} reclaimed)? [y/N]: ",
        model_id_or_name,
        format_bytes(size)
    );
    io::stdout().flush()?;

    let mut line = String::new();
    let stdin = io::stdin();
    stdin.lock().read_line(&mut line)?;

    if line.trim().eq_ignore_ascii_case("y") || line.trim().eq_ignore_ascii_case("yes") {
        store.delete_model(model_id_or_name)?;
        println!(
            "\x1b[38;5;49m[✓] Model '{}' successfully removed. Reclaimed {}.\x1b[0m",
            model_id_or_name,
            format_bytes(size)
        );
    } else {
        println!("Deletion aborted.");
    }

    Ok(())
}

/// Verifies SHA-256 integrity of an installed model.
pub fn verify_model_cli(store: &ModelStore, model_id_or_name: &str) -> Result<()> {
    if !store.is_model_installed(model_id_or_name) {
        bail!("Model '{}' is not installed locally.", model_id_or_name);
    }

    let path = store.get_model_path(model_id_or_name).unwrap();
    println!(
        "\x1b[38;5;141m[•] Computing SHA-256 for:\x1b[0m {}",
        path.display()
    );

    let verified = store.verify_installed_model(model_id_or_name)?;
    if verified {
        println!(
            "\x1b[1;38;5;49m[✓] SHA-256 verification passed!\x1b[0m Integrity verified bit-exact."
        );
    } else {
        println!("\x1b[1;31m[✗] SHA-256 verification failed!\x1b[0m File appears corrupt.");
        println!(
            "    Run 'sulimo-cli models download {}' to re-fetch the verified model.",
            model_id_or_name
        );
    }

    Ok(())
}

/// Resolves a model path from a direct path or registry ID, prompting to download if missing.
pub fn resolve_or_prompt_download(model_target: &str) -> Result<PathBuf> {
    // 1. Direct existing filesystem path
    let direct = Path::new(model_target);
    if direct.is_file() {
        return Ok(direct.to_path_buf());
    }

    let store = ModelStore::new(None)?;

    // 2. Check if already installed in ModelStore
    if let Some(path) = store.get_model_path(model_target) {
        return Ok(path);
    }

    // 3. Check if target is a known curated registry model
    if let Some(meta) = find_model_by_id(model_target) {
        println!(
            "\x1b[38;5;214m[•] Model '{}' is not installed locally.\x1b[0m",
            meta.id
        );
        println!(
            "    Display Name: \x1b[1m{}\x1b[0m ({})",
            meta.display_name,
            meta.formatted_size()
        );
        println!(
            "    Architecture: {:?} (Quant: {})",
            meta.architecture, meta.quantization
        );
        println!(
            "    RAM Envelope: {} (Unified 5.20 GiB Ceiling)",
            meta.formatted_ram_requirement()
        );
        println!("    Store Quota:  30.0 GB dedicated pool");
        print!("\nWould you like to download and install it now? [Y/n]: ");
        io::stdout().flush()?;

        let mut line = String::new();
        let stdin = io::stdin();
        stdin.lock().read_line(&mut line)?;
        let trimmed = line.trim();

        if trimmed.is_empty()
            || trimmed.eq_ignore_ascii_case("y")
            || trimmed.eq_ignore_ascii_case("yes")
        {
            return download_model_cli(&store, &meta.id);
        } else {
            bail!("Model download cancelled by user.");
        }
    }

    // 4. Fallback: target not found
    bail!(
        "Model '{}' was not found on disk or in the curated registry.\nType 'sulimo-cli models list' to view available models.",
        model_target
    );
}

fn render_cli_progress_bar(p: &DownloadProgress) {
    match p.state {
        DownloadState::Pending => {
            print!("\r\x1b[38;5;244m[•] Initializing download stream...\x1b[0m\x1b[K");
            let _ = io::stdout().flush();
        }
        DownloadState::Downloading => {
            let bar_width = 28;
            let filled = ((p.percent * bar_width as f32).round() as usize).min(bar_width);
            let empty = bar_width.saturating_sub(filled);

            let bar = format!("{}{}", "█".repeat(filled), "░".repeat(empty));

            let speed_str = if p.speed_bps >= 1_000_000.0 {
                format!("{:.1} MB/s", p.speed_bps / 1_000_000.0)
            } else if p.speed_bps >= 1_000.0 {
                format!("{:.1} KB/s", p.speed_bps / 1_000.0)
            } else {
                format!("{:.0} B/s", p.speed_bps)
            };

            let eta_str = match p.eta_seconds {
                Some(s) if s >= 3600 => format!("ETA: {}h {}m", s / 3600, (s % 3600) / 60),
                Some(s) if s >= 60 => format!("ETA: {}m {}s", s / 60, s % 60),
                Some(s) => format!("ETA: {s}s"),
                None => "ETA: --".to_string(),
            };

            print!(
                "\r\x1b[38;5;51m[{bar}]\x1b[0m \x1b[1m{:5.1}%\x1b[0m ({:>8} / {:>8})  \x1b[38;5;49m{:>9}\x1b[0m  \x1b[38;5;244m{:>9}\x1b[0m\x1b[K",
                p.percent * 100.0,
                format_bytes(p.downloaded_bytes),
                format_bytes(p.total_bytes),
                speed_str,
                eta_str
            );
            let _ = io::stdout().flush();
        }
        DownloadState::VerifyingChecksum => {
            print!("\r\x1b[38;5;141m[•] Verifying SHA-256 integrity digest...\x1b[0m\x1b[K");
            let _ = io::stdout().flush();
        }
        DownloadState::Completed => {
            print!(
                "\r\x1b[38;5;49m[✓] Download and cryptographic verification complete!\x1b[0m\x1b[K"
            );
            let _ = io::stdout().flush();
        }
        DownloadState::Cancelled => {
            print!("\r\x1b[38;5;214m[!] Download cancelled.\x1b[0m\x1b[K");
            let _ = io::stdout().flush();
        }
        DownloadState::Failed => {
            print!("\r\x1b[38;5;196m[✗] Download failed.\x1b[0m\x1b[K");
            let _ = io::stdout().flush();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    struct TempStoreDir(PathBuf);
    impl TempStoreDir {
        fn new(name: &str) -> Self {
            let path =
                std::env::temp_dir().join(format!("sulimo_cli_test_{name}_{}", std::process::id()));
            let _ = fs::remove_dir_all(&path);
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }
    }
    impl Drop for TempStoreDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn test_print_models_table_human_and_json() {
        let temp = TempStoreDir::new("models_table");
        let store = ModelStore::new(Some(temp.0.clone())).unwrap();

        // Human table rendering succeeds without error
        assert!(print_models_table(&store, false).is_ok());

        // JSON table rendering succeeds without error
        assert!(print_models_table(&store, true).is_ok());
    }

    #[test]
    fn test_search_models() {
        let temp = TempStoreDir::new("models_search");
        let store = ModelStore::new(Some(temp.0.clone())).unwrap();

        assert!(search_models(&store, "qwen").is_ok());
        assert!(search_models(&store, "liquid").is_ok());
        assert!(search_models(&store, "gemma").is_ok());
        assert!(search_models(&store, "minicpm").is_ok());
        assert!(search_models(&store, "llama").is_ok());
        assert!(search_models(&store, "nonexistent_query_xyz").is_ok());
    }

    #[test]
    fn test_verify_uninstalled_model_fails() {
        let temp = TempStoreDir::new("models_verify_uninstalled");
        let store = ModelStore::new(Some(temp.0.clone())).unwrap();

        assert!(verify_model_cli(&store, "nonexistent-model").is_err());
    }

    #[test]
    fn test_resolve_or_prompt_download_existing_file() {
        let temp = TempStoreDir::new("resolve_file");
        let file_path = temp.0.join("model.gguf");
        fs::write(&file_path, b"test").unwrap();

        let resolved = resolve_or_prompt_download(file_path.to_str().unwrap()).unwrap();
        assert_eq!(resolved, file_path);
    }

    #[test]
    fn test_resolve_or_prompt_download_unknown_fails() {
        assert!(resolve_or_prompt_download("completely_unknown_model_12345").is_err());
    }

    #[test]
    fn test_handle_models_subcommands_dispatch() {
        let list_args = vec!["list".to_string()];
        assert!(handle_models_command(&list_args).is_ok());

        let search_args = vec!["search".to_string(), "qwen".to_string()];
        assert!(handle_models_command(&search_args).is_ok());

        let help_args = vec!["help".to_string()];
        assert!(handle_models_command(&help_args).is_ok());
    }
}
