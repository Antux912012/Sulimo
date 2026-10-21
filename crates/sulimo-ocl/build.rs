fn main() {
    if let Ok(home) = std::env::var("HOME") {
        let driver_lib = std::path::Path::new(&home).join(".local/share/sulimo/drivers/lib");
        if driver_lib.exists() {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", driver_lib.display());
        }
    }
}
