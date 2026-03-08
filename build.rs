fn main() {
    if std::env::var("CARGO_FEATURE_SLINT_UI").is_ok() {
        println!("cargo:rerun-if-changed=src/ui_slint/app_shell.slint");
        slint_build::compile("src/ui_slint/app_shell.slint")
            .expect("failed to compile Slint UI");
    }
}
