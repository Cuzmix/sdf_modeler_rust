#![cfg_attr(all(target_os = "windows", not(debug_assertions)), windows_subsystem = "windows")]

fn main() -> Result<(), String> {
    sdf_modeler_lib::run_native()
}
