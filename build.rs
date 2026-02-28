fn main() {
    #[cfg(feature = "slint_ui")]
    slint_build::compile("ui/main.slint").unwrap();
}
