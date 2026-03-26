fn main() {
    slint_build::compile("src/app/slint_ui/slint_host_window.slint")
        .expect("failed to compile Slint UI");
}
