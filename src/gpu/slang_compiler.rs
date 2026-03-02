use std::path::PathBuf;
use std::process::Command;

/// Compiles Slang shader source → WGSL using the `slangc` command-line compiler.
///
/// Looks for `slangc` in:
///   1. `tools/slang/bin/slangc` relative to the project root (dev builds)
///   2. `tools/slang/bin/slangc` relative to the executable
///   3. The system PATH
pub struct SlangCompiler {
    slangc_path: PathBuf,
}

impl SlangCompiler {
    pub fn new() -> Option<Self> {
        // Try project root (dev builds via CARGO_MANIFEST_DIR)
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let dev_path = manifest.join("tools/slang/bin/slangc");
        if dev_path.exists() {
            log::info!("Found slangc at {}", dev_path.display());
            return Some(Self { slangc_path: dev_path });
        }

        // Try next to the executable (release builds)
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let tools_path = dir.join("tools/slang/bin/slangc");
                if tools_path.exists() {
                    log::info!("Found slangc at {}", tools_path.display());
                    return Some(Self { slangc_path: tools_path });
                }
            }
        }

        // Try system PATH
        if Command::new("slangc").arg("--version").output().is_ok() {
            log::info!("Using slangc from system PATH");
            return Some(Self {
                slangc_path: PathBuf::from("slangc"),
            });
        }

        log::warn!("slangc not found — Slang compilation unavailable");
        None
    }

    /// Compile Slang source to WGSL.
    /// Writes source to a temp file, invokes `slangc -target wgsl`, reads output file.
    pub fn compile_to_wgsl(&self, source: &str) -> Result<String, String> {
        let tmp_dir = std::env::temp_dir();
        let input_path = tmp_dir.join("sdf_modeler_slang_input.slang");
        let output_path = tmp_dir.join("sdf_modeler_slang_output.wgsl");

        std::fs::write(&input_path, source)
            .map_err(|e| format!("Failed to write temp Slang file: {e}"))?;

        let output = Command::new(&self.slangc_path)
            .arg(&input_path)
            .arg("-target")
            .arg("wgsl")
            .arg("-o")
            .arg(&output_path)
            .output()
            .map_err(|e| format!("Failed to run slangc: {e}"))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let _ = std::fs::remove_file(&input_path);
            return Err(format!(
                "slangc failed (exit {}):\n{}\n{}",
                output.status,
                stderr.trim(),
                stdout.trim(),
            ));
        }

        let wgsl = std::fs::read_to_string(&output_path)
            .map_err(|e| format!("Failed to read slangc output: {e}"))?;

        let _ = std::fs::remove_file(&input_path);
        let _ = std::fs::remove_file(&output_path);

        log::info!("Slang → WGSL: {} bytes", wgsl.len());
        Ok(wgsl)
    }
}
