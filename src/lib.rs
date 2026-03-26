mod app;
mod compat;
mod desktop_dialogs;
mod export;
pub mod expression;
mod gizmo;
mod gpu;
mod graph;
mod io;
pub mod keymap;
mod material_preset;
mod mesh_import;
mod native_paths;
mod native_wgpu;
mod sculpt;
mod settings;
mod viewport;

pub fn run_native() -> Result<(), String> {
    let _ = env_logger::builder().is_test(false).try_init();

    let settings = settings::Settings::load();
    app::slint_frontend::run_slint_host(settings)
}

pub fn run_slint_native() -> Result<(), String> {
    run_native()
}
