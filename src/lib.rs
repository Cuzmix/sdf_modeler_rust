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
mod platform;
mod sculpt;
mod settings;
mod viewport;

pub fn run_native() -> Result<(), String> {
    platform::run_desktop()
}

pub fn run_slint_native() -> Result<(), String> {
    run_native()
}

#[cfg(target_os = "android")]
#[unsafe(no_mangle)]
pub fn android_main(app: slint::android::AndroidApp) {
    platform::run_android(app);
}
