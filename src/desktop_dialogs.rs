use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(
    all(not(target_arch = "wasm32"), not(target_os = "android")),
    allow(dead_code)
)]
pub enum FileDialogSelection {
    Selected(PathBuf),
    Cancelled,
    Unsupported,
}

#[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
fn dialog_selection(path: Option<PathBuf>) -> FileDialogSelection {
    match path {
        Some(path) => FileDialogSelection::Selected(path),
        None => FileDialogSelection::Cancelled,
    }
}

#[cfg(any(target_arch = "wasm32", target_os = "android"))]
fn unsupported_dialog() -> FileDialogSelection {
    FileDialogSelection::Unsupported
}

pub fn save_project_dialog() -> FileDialogSelection {
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
    {
        dialog_selection(
            rfd::FileDialog::new()
                .set_title("Save Project")
                .add_filter("SDF Project", &["sdf", "json"])
                .save_file(),
        )
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    {
        unsupported_dialog()
    }
}

pub fn open_project_dialog() -> FileDialogSelection {
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
    {
        dialog_selection(
            rfd::FileDialog::new()
                .set_title("Open Project")
                .add_filter("SDF Project", &["sdf", "json"])
                .pick_file(),
        )
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    {
        unsupported_dialog()
    }
}

pub fn save_node_preset_dialog(default_name: &str) -> FileDialogSelection {
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
    {
        dialog_selection(
            rfd::FileDialog::new()
                .set_title("Save Node Preset")
                .set_file_name(default_name)
                .add_filter("SDF Node Preset", &["sdfpreset"])
                .save_file(),
        )
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    {
        let _ = default_name;
        unsupported_dialog()
    }
}

pub fn load_node_preset_dialog() -> FileDialogSelection {
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
    {
        dialog_selection(
            rfd::FileDialog::new()
                .set_title("Load Node Preset")
                .add_filter("SDF Node Preset", &["sdfpreset"])
                .pick_file(),
        )
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    {
        unsupported_dialog()
    }
}

pub fn reference_image_dialog() -> FileDialogSelection {
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
    {
        dialog_selection(
            rfd::FileDialog::new()
                .set_title("Add Reference Image")
                .add_filter("Images", &["png", "jpg", "jpeg"])
                .pick_file(),
        )
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    {
        unsupported_dialog()
    }
}

pub fn environment_hdri_dialog() -> FileDialogSelection {
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
    {
        dialog_selection(
            rfd::FileDialog::new()
                .set_title("Import HDR / EXR Environment")
                .add_filter("HDR Environment", &["hdr", "exr"])
                .pick_file(),
        )
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    {
        unsupported_dialog()
    }
}

pub fn screenshot_dialog() -> FileDialogSelection {
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
    {
        dialog_selection(
            rfd::FileDialog::new()
                .set_title("Save Screenshot")
                .add_filter("PNG Image", &["png"])
                .save_file(),
        )
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    {
        unsupported_dialog()
    }
}

pub fn mesh_export_dialog() -> FileDialogSelection {
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
    {
        dialog_selection(
            rfd::FileDialog::new()
                .set_title("Export Mesh")
                .add_filter("Wavefront OBJ", &["obj"])
                .add_filter("STL Binary", &["stl"])
                .add_filter("Stanford PLY", &["ply"])
                .add_filter("glTF Binary", &["glb"])
                .add_filter("USD ASCII", &["usda"])
                .save_file(),
        )
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    {
        unsupported_dialog()
    }
}

pub fn mesh_import_dialog() -> FileDialogSelection {
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
    {
        dialog_selection(
            rfd::FileDialog::new()
                .set_title("Import Mesh")
                .add_filter("Wavefront OBJ", &["obj"])
                .add_filter("STL Binary", &["stl"])
                .add_filter("All Mesh Files", &["obj", "stl"])
                .pick_file(),
        )
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    {
        unsupported_dialog()
    }
}

pub fn settings_export_dialog(file_name: &str) -> FileDialogSelection {
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
    {
        dialog_selection(
            rfd::FileDialog::new()
                .set_title("Export Settings")
                .add_filter("JSON", &["json"])
                .set_file_name(file_name)
                .save_file(),
        )
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    {
        let _ = file_name;
        unsupported_dialog()
    }
}

pub fn settings_import_dialog() -> FileDialogSelection {
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
    {
        dialog_selection(
            rfd::FileDialog::new()
                .set_title("Import Settings")
                .add_filter("JSON", &["json"])
                .pick_file(),
        )
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    {
        unsupported_dialog()
    }
}

pub fn keybindings_export_dialog() -> FileDialogSelection {
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
    {
        dialog_selection(
            rfd::FileDialog::new()
                .set_title("Export Keybindings")
                .add_filter("JSON", &["json"])
                .set_file_name("keybindings.json")
                .save_file(),
        )
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    {
        unsupported_dialog()
    }
}

pub fn keybindings_import_dialog() -> FileDialogSelection {
    #[cfg(all(not(target_arch = "wasm32"), not(target_os = "android")))]
    {
        dialog_selection(
            rfd::FileDialog::new()
                .set_title("Import Keybindings")
                .add_filter("JSON", &["json"])
                .pick_file(),
        )
    }

    #[cfg(any(target_arch = "wasm32", target_os = "android"))]
    {
        unsupported_dialog()
    }
}
