#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

#[cfg(not(target_arch = "wasm32"))]
pub fn app_storage_file(file_name: &str) -> PathBuf {
    let mut path = native_storage_dir();
    path.push(file_name);
    path
}

#[cfg(not(target_arch = "wasm32"))]
pub fn ensure_parent_dir(path: &Path) -> Result<(), String> {
    let Some(parent) = path.parent() else {
        return Ok(());
    };

    std::fs::create_dir_all(parent)
        .map_err(|error| format!("Failed to create {}: {}", parent.display(), error))
}

#[cfg(all(not(target_arch = "wasm32"), target_os = "android"))]
fn native_storage_dir() -> PathBuf {
    let mut path = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    path.push("sdf_modeler");
    path
}

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
fn native_storage_dir() -> PathBuf {
    let mut path = std::env::current_exe()
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    path.pop();
    path
}
