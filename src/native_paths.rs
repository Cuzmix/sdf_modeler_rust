#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};

pub const DEFAULT_BUNDLED_ENVIRONMENT_HDRI_PATH: &str =
    "assets/environments/glasshouse_interior_4k.exr";

#[cfg(not(target_arch = "wasm32"))]
pub fn app_storage_file(file_name: &str) -> PathBuf {
    let mut path = native_storage_dir();
    path.push(file_name);
    path
}

#[cfg(not(target_arch = "wasm32"))]
pub fn resolve_bundled_asset_path(asset_path: &str) -> Option<PathBuf> {
    resolve_bundled_asset_path_from_roots(
        Path::new(asset_path),
        executable_dir().as_deref(),
        Some(Path::new(env!("CARGO_MANIFEST_DIR"))),
    )
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

#[cfg(not(target_arch = "wasm32"))]
fn executable_dir() -> Option<PathBuf> {
    let mut path = std::env::current_exe().ok()?;
    path.pop();
    Some(path)
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_bundled_asset_path_from_roots(
    asset_path: &Path,
    executable_dir: Option<&Path>,
    manifest_dir: Option<&Path>,
) -> Option<PathBuf> {
    if asset_path.as_os_str().is_empty() {
        return None;
    }

    if asset_path.is_absolute() {
        return Some(asset_path.to_path_buf());
    }

    if let Some(executable_dir) = executable_dir {
        let candidate = executable_dir.join(asset_path);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    if let Some(manifest_dir) = manifest_dir {
        let candidate = manifest_dir.join(asset_path);
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
fn native_storage_dir() -> PathBuf {
    let mut path = std::env::current_exe()
        .unwrap_or_else(|_| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    path.pop();
    path
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(label: &str) -> PathBuf {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "sdf_modeler_native_paths_{label}_{}_{}",
            std::process::id(),
            timestamp
        ))
    }

    #[test]
    fn resolve_bundled_asset_path_prefers_executable_directory() {
        let root = unique_temp_dir("prefer_exe");
        let executable_dir = root.join("exe");
        let manifest_dir = root.join("manifest");
        let relative_path = Path::new("assets/environments/test.exr");
        fs::create_dir_all(executable_dir.join("assets/environments"))
            .expect("create executable asset directory");
        fs::create_dir_all(manifest_dir.join("assets/environments"))
            .expect("create manifest asset directory");
        fs::write(executable_dir.join(relative_path), b"exe").expect("write executable asset");
        fs::write(manifest_dir.join(relative_path), b"manifest").expect("write manifest asset");

        let resolved = resolve_bundled_asset_path_from_roots(
            relative_path,
            Some(&executable_dir),
            Some(&manifest_dir),
        );

        assert_eq!(resolved, Some(executable_dir.join(relative_path)));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn resolve_bundled_asset_path_falls_back_to_manifest_directory() {
        let root = unique_temp_dir("prefer_manifest");
        let executable_dir = root.join("exe");
        let manifest_dir = root.join("manifest");
        let relative_path = Path::new("assets/environments/test.exr");
        fs::create_dir_all(manifest_dir.join("assets/environments"))
            .expect("create manifest asset directory");
        fs::write(manifest_dir.join(relative_path), b"manifest").expect("write manifest asset");

        let resolved = resolve_bundled_asset_path_from_roots(
            relative_path,
            Some(&executable_dir),
            Some(&manifest_dir),
        );

        assert_eq!(resolved, Some(manifest_dir.join(relative_path)));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn resolve_bundled_asset_path_passes_through_absolute_paths() {
        let absolute_path = std::env::temp_dir().join("glasshouse_interior_4k.exr");
        let resolved = resolve_bundled_asset_path_from_roots(&absolute_path, None, None);
        assert_eq!(resolved, Some(absolute_path));
    }
}
