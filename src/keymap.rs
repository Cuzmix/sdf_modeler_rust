use std::collections::HashMap;
use std::fmt;

use eframe::egui;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Key combination (key + modifiers)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
pub struct KeyCombo {
    pub key: SerializableKey,
    pub ctrl: bool,
    pub shift: bool,
    pub alt: bool,
}

impl KeyCombo {
    pub const fn new(key: SerializableKey) -> Self {
        Self {
            key,
            ctrl: false,
            shift: false,
            alt: false,
        }
    }

    pub const fn ctrl(key: SerializableKey) -> Self {
        Self {
            key,
            ctrl: true,
            shift: false,
            alt: false,
        }
    }

    pub const fn shift(key: SerializableKey) -> Self {
        Self {
            key,
            ctrl: false,
            shift: true,
            alt: false,
        }
    }

    pub const fn alt(key: SerializableKey) -> Self {
        Self {
            key,
            ctrl: false,
            shift: false,
            alt: true,
        }
    }

    pub const fn ctrl_shift(key: SerializableKey) -> Self {
        Self {
            key,
            ctrl: true,
            shift: true,
            alt: false,
        }
    }

    pub fn matches_egui(&self, modifiers: &egui::Modifiers) -> bool {
        self.ctrl == modifiers.ctrl && self.shift == modifiers.shift && self.alt == modifiers.alt
    }

    pub fn egui_key(&self) -> egui::Key {
        self.key.to_egui()
    }

    pub fn egui_modifiers(&self) -> egui::Modifiers {
        egui::Modifiers {
            alt: self.alt,
            ctrl: self.ctrl,
            shift: self.shift,
            mac_cmd: false,
            command: self.ctrl,
        }
    }
}

impl fmt::Display for KeyCombo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut parts = Vec::new();
        if self.ctrl {
            parts.push("Ctrl");
        }
        if self.shift {
            parts.push("Shift");
        }
        if self.alt {
            parts.push("Alt");
        }
        parts.push(self.key.label());
        write!(f, "{}", parts.join("+"))
    }
}

// ---------------------------------------------------------------------------
// Serializable key enum (wraps egui::Key for serde support)
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SerializableKey {
    A,
    B,
    C,
    D,
    E,
    F,
    G,
    H,
    I,
    J,
    K,
    L,
    M,
    N,
    O,
    P,
    Q,
    R,
    S,
    T,
    U,
    V,
    W,
    X,
    Y,
    Z,
    Num0,
    Num1,
    Num2,
    Num3,
    Num4,
    Num5,
    Num6,
    Num7,
    Num8,
    Num9,
    F1,
    F2,
    F3,
    F4,
    F5,
    F6,
    F7,
    F8,
    F9,
    F10,
    F11,
    F12,
    Space,
    Enter,
    Escape,
    Tab,
    Delete,
    Home,
    End,
    ArrowUp,
    ArrowDown,
    ArrowLeft,
    ArrowRight,
    OpenBracket,
    CloseBracket,
    Slash,
}

impl SerializableKey {
    /// Try to convert an egui::Key back to a SerializableKey. Returns None for unsupported keys.
    pub fn from_egui(key: egui::Key) -> Option<Self> {
        match key {
            egui::Key::A => Some(Self::A),
            egui::Key::B => Some(Self::B),
            egui::Key::C => Some(Self::C),
            egui::Key::D => Some(Self::D),
            egui::Key::E => Some(Self::E),
            egui::Key::F => Some(Self::F),
            egui::Key::G => Some(Self::G),
            egui::Key::H => Some(Self::H),
            egui::Key::I => Some(Self::I),
            egui::Key::J => Some(Self::J),
            egui::Key::K => Some(Self::K),
            egui::Key::L => Some(Self::L),
            egui::Key::M => Some(Self::M),
            egui::Key::N => Some(Self::N),
            egui::Key::O => Some(Self::O),
            egui::Key::P => Some(Self::P),
            egui::Key::Q => Some(Self::Q),
            egui::Key::R => Some(Self::R),
            egui::Key::S => Some(Self::S),
            egui::Key::T => Some(Self::T),
            egui::Key::U => Some(Self::U),
            egui::Key::V => Some(Self::V),
            egui::Key::W => Some(Self::W),
            egui::Key::X => Some(Self::X),
            egui::Key::Y => Some(Self::Y),
            egui::Key::Z => Some(Self::Z),
            egui::Key::Num0 => Some(Self::Num0),
            egui::Key::Num1 => Some(Self::Num1),
            egui::Key::Num2 => Some(Self::Num2),
            egui::Key::Num3 => Some(Self::Num3),
            egui::Key::Num4 => Some(Self::Num4),
            egui::Key::Num5 => Some(Self::Num5),
            egui::Key::Num6 => Some(Self::Num6),
            egui::Key::Num7 => Some(Self::Num7),
            egui::Key::Num8 => Some(Self::Num8),
            egui::Key::Num9 => Some(Self::Num9),
            egui::Key::F1 => Some(Self::F1),
            egui::Key::F2 => Some(Self::F2),
            egui::Key::F3 => Some(Self::F3),
            egui::Key::F4 => Some(Self::F4),
            egui::Key::F5 => Some(Self::F5),
            egui::Key::F6 => Some(Self::F6),
            egui::Key::F7 => Some(Self::F7),
            egui::Key::F8 => Some(Self::F8),
            egui::Key::F9 => Some(Self::F9),
            egui::Key::F10 => Some(Self::F10),
            egui::Key::F11 => Some(Self::F11),
            egui::Key::F12 => Some(Self::F12),
            egui::Key::Space => Some(Self::Space),
            egui::Key::Enter => Some(Self::Enter),
            egui::Key::Escape => Some(Self::Escape),
            egui::Key::Tab => Some(Self::Tab),
            egui::Key::Delete => Some(Self::Delete),
            egui::Key::Home => Some(Self::Home),
            egui::Key::End => Some(Self::End),
            egui::Key::ArrowUp => Some(Self::ArrowUp),
            egui::Key::ArrowDown => Some(Self::ArrowDown),
            egui::Key::ArrowLeft => Some(Self::ArrowLeft),
            egui::Key::ArrowRight => Some(Self::ArrowRight),
            egui::Key::OpenBracket => Some(Self::OpenBracket),
            egui::Key::CloseBracket => Some(Self::CloseBracket),
            egui::Key::Slash => Some(Self::Slash),
            _ => None,
        }
    }

    pub fn to_egui(self) -> egui::Key {
        match self {
            Self::A => egui::Key::A,
            Self::B => egui::Key::B,
            Self::C => egui::Key::C,
            Self::D => egui::Key::D,
            Self::E => egui::Key::E,
            Self::F => egui::Key::F,
            Self::G => egui::Key::G,
            Self::H => egui::Key::H,
            Self::I => egui::Key::I,
            Self::J => egui::Key::J,
            Self::K => egui::Key::K,
            Self::L => egui::Key::L,
            Self::M => egui::Key::M,
            Self::N => egui::Key::N,
            Self::O => egui::Key::O,
            Self::P => egui::Key::P,
            Self::Q => egui::Key::Q,
            Self::R => egui::Key::R,
            Self::S => egui::Key::S,
            Self::T => egui::Key::T,
            Self::U => egui::Key::U,
            Self::V => egui::Key::V,
            Self::W => egui::Key::W,
            Self::X => egui::Key::X,
            Self::Y => egui::Key::Y,
            Self::Z => egui::Key::Z,
            Self::Num0 => egui::Key::Num0,
            Self::Num1 => egui::Key::Num1,
            Self::Num2 => egui::Key::Num2,
            Self::Num3 => egui::Key::Num3,
            Self::Num4 => egui::Key::Num4,
            Self::Num5 => egui::Key::Num5,
            Self::Num6 => egui::Key::Num6,
            Self::Num7 => egui::Key::Num7,
            Self::Num8 => egui::Key::Num8,
            Self::Num9 => egui::Key::Num9,
            Self::F1 => egui::Key::F1,
            Self::F2 => egui::Key::F2,
            Self::F3 => egui::Key::F3,
            Self::F4 => egui::Key::F4,
            Self::F5 => egui::Key::F5,
            Self::F6 => egui::Key::F6,
            Self::F7 => egui::Key::F7,
            Self::F8 => egui::Key::F8,
            Self::F9 => egui::Key::F9,
            Self::F10 => egui::Key::F10,
            Self::F11 => egui::Key::F11,
            Self::F12 => egui::Key::F12,
            Self::Space => egui::Key::Space,
            Self::Enter => egui::Key::Enter,
            Self::Escape => egui::Key::Escape,
            Self::Tab => egui::Key::Tab,
            Self::Delete => egui::Key::Delete,
            Self::Home => egui::Key::Home,
            Self::End => egui::Key::End,
            Self::ArrowUp => egui::Key::ArrowUp,
            Self::ArrowDown => egui::Key::ArrowDown,
            Self::ArrowLeft => egui::Key::ArrowLeft,
            Self::ArrowRight => egui::Key::ArrowRight,
            Self::OpenBracket => egui::Key::OpenBracket,
            Self::CloseBracket => egui::Key::CloseBracket,
            Self::Slash => egui::Key::Slash,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::A => "A",
            Self::B => "B",
            Self::C => "C",
            Self::D => "D",
            Self::E => "E",
            Self::F => "F",
            Self::G => "G",
            Self::H => "H",
            Self::I => "I",
            Self::J => "J",
            Self::K => "K",
            Self::L => "L",
            Self::M => "M",
            Self::N => "N",
            Self::O => "O",
            Self::P => "P",
            Self::Q => "Q",
            Self::R => "R",
            Self::S => "S",
            Self::T => "T",
            Self::U => "U",
            Self::V => "V",
            Self::W => "W",
            Self::X => "X",
            Self::Y => "Y",
            Self::Z => "Z",
            Self::Num0 => "0",
            Self::Num1 => "1",
            Self::Num2 => "2",
            Self::Num3 => "3",
            Self::Num4 => "4",
            Self::Num5 => "5",
            Self::Num6 => "6",
            Self::Num7 => "7",
            Self::Num8 => "8",
            Self::Num9 => "9",
            Self::F1 => "F1",
            Self::F2 => "F2",
            Self::F3 => "F3",
            Self::F4 => "F4",
            Self::F5 => "F5",
            Self::F6 => "F6",
            Self::F7 => "F7",
            Self::F8 => "F8",
            Self::F9 => "F9",
            Self::F10 => "F10",
            Self::F11 => "F11",
            Self::F12 => "F12",
            Self::Space => "Space",
            Self::Enter => "Enter",
            Self::Escape => "Esc",
            Self::Tab => "Tab",
            Self::Delete => "Del",
            Self::Home => "Home",
            Self::End => "End",
            Self::ArrowUp => "Up",
            Self::ArrowDown => "Down",
            Self::ArrowLeft => "Left",
            Self::ArrowRight => "Right",
            Self::OpenBracket => "[",
            Self::CloseBracket => "]",
            Self::Slash => "/",
        }
    }
}

// ---------------------------------------------------------------------------
// Action binding — keyboard-triggerable actions
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ActionBinding {
    // General
    ToggleHelp,
    ToggleDebug,
    NewScene,
    OpenProject,
    SaveProject,
    Undo,
    Redo,
    Copy,
    Paste,
    Duplicate,
    DeleteSelected,
    CopyProperties,
    PasteProperties,
    TakeScreenshot,
    ShowExportDialog,
    ToggleCommandPalette,

    // Camera
    CameraFront,
    CameraTop,
    CameraRight,
    CameraBack,
    CameraLeft,
    CameraBottom,
    ToggleOrtho,
    FocusSelected,
    FrameAll,

    // Gizmo / Tools
    GizmoTranslate,
    GizmoRotate,
    GizmoScale,
    ToggleGizmoSpace,
    ResetPivot,
    EnterSculptMode,
    QuickSculptMode,

    // Viewport
    ToggleIsolation,
    CycleShadingMode,
    ToggleTurntable,
    ShowQuickToolbar,
    ToggleReferenceImages,
    ToggleMeasurementTool,

    // Sculpt brush modes
    SculptBrushAdd,
    SculptBrushCarve,
    SculptBrushSmooth,
    SculptBrushFlatten,
    SculptBrushInflate,
    SculptBrushGrab,
    SculptBrushShrink,
    SculptBrushGrow,

    // Sculpt symmetry
    SculptSymmetryX,
    SculptSymmetryY,
    SculptSymmetryZ,

    // Sculpt escape
    ExitSculptMode,
}

impl ActionBinding {
    /// All action bindings for iteration.
    pub const ALL: &'static [Self] = &[
        Self::ToggleHelp,
        Self::ToggleDebug,
        Self::NewScene,
        Self::OpenProject,
        Self::SaveProject,
        Self::Undo,
        Self::Redo,
        Self::Copy,
        Self::Paste,
        Self::Duplicate,
        Self::DeleteSelected,
        Self::CopyProperties,
        Self::PasteProperties,
        Self::TakeScreenshot,
        Self::ShowExportDialog,
        Self::ToggleCommandPalette,
        Self::CameraFront,
        Self::CameraTop,
        Self::CameraRight,
        Self::CameraBack,
        Self::CameraLeft,
        Self::CameraBottom,
        Self::ToggleOrtho,
        Self::FocusSelected,
        Self::FrameAll,
        Self::GizmoTranslate,
        Self::GizmoRotate,
        Self::GizmoScale,
        Self::ToggleGizmoSpace,
        Self::ResetPivot,
        Self::EnterSculptMode,
        Self::QuickSculptMode,
        Self::ToggleIsolation,
        Self::CycleShadingMode,
        Self::ToggleTurntable,
        Self::ShowQuickToolbar,
        Self::ToggleReferenceImages,
        Self::ToggleMeasurementTool,
        Self::SculptBrushAdd,
        Self::SculptBrushCarve,
        Self::SculptBrushSmooth,
        Self::SculptBrushFlatten,
        Self::SculptBrushInflate,
        Self::SculptBrushGrab,
        Self::SculptBrushShrink,
        Self::SculptBrushGrow,
        Self::SculptSymmetryX,
        Self::SculptSymmetryY,
        Self::SculptSymmetryZ,
        Self::ExitSculptMode,
    ];

    /// Human-readable label for this action.
    pub fn label(self) -> &'static str {
        match self {
            Self::ToggleHelp => "Toggle Help",
            Self::ToggleDebug => "Toggle Profiler",
            Self::NewScene => "New Scene",
            Self::OpenProject => "Open Project",
            Self::SaveProject => "Save Project",
            Self::Undo => "Undo",
            Self::Redo => "Redo",
            Self::Copy => "Copy",
            Self::Paste => "Paste",
            Self::Duplicate => "Duplicate",
            Self::DeleteSelected => "Delete Selected",
            Self::CopyProperties => "Copy Properties",
            Self::PasteProperties => "Paste Properties",
            Self::TakeScreenshot => "Take Screenshot",
            Self::ShowExportDialog => "Export Mesh",
            Self::ToggleCommandPalette => "Command Palette",
            Self::CameraFront => "Camera: Front",
            Self::CameraTop => "Camera: Top",
            Self::CameraRight => "Camera: Right",
            Self::CameraBack => "Camera: Back",
            Self::CameraLeft => "Camera: Left",
            Self::CameraBottom => "Camera: Bottom",
            Self::ToggleOrtho => "Toggle Orthographic",
            Self::FocusSelected => "Focus Selected",
            Self::FrameAll => "Frame All",
            Self::GizmoTranslate => "Gizmo: Translate",
            Self::GizmoRotate => "Gizmo: Rotate",
            Self::GizmoScale => "Gizmo: Scale",
            Self::ToggleGizmoSpace => "Toggle Gizmo Space",
            Self::ResetPivot => "Reset Pivot",
            Self::EnterSculptMode => "Enter Sculpt Mode",
            Self::QuickSculptMode => "Quick Sculpt Mode",
            Self::ToggleIsolation => "Toggle Isolation",
            Self::CycleShadingMode => "Cycle Shading Mode",
            Self::ToggleTurntable => "Toggle Turntable",
            Self::ShowQuickToolbar => "Quick Primitives",
            Self::ToggleReferenceImages => "Toggle Reference Images",
            Self::ToggleMeasurementTool => "Toggle Measurement Tool",
            Self::SculptBrushAdd => "Brush: Add",
            Self::SculptBrushCarve => "Brush: Carve",
            Self::SculptBrushSmooth => "Brush: Smooth",
            Self::SculptBrushFlatten => "Brush: Flatten",
            Self::SculptBrushInflate => "Brush: Inflate",
            Self::SculptBrushGrab => "Brush: Grab",
            Self::SculptBrushShrink => "Brush: Shrink",
            Self::SculptBrushGrow => "Brush: Grow",
            Self::SculptSymmetryX => "Symmetry: X",
            Self::SculptSymmetryY => "Symmetry: Y",
            Self::SculptSymmetryZ => "Symmetry: Z",
            Self::ExitSculptMode => "Exit Sculpt Mode",
        }
    }

    /// Whether this action is only active during sculpt mode.
    pub fn is_sculpt_only(self) -> bool {
        matches!(
            self,
            Self::SculptBrushAdd
                | Self::SculptBrushCarve
                | Self::SculptBrushSmooth
                | Self::SculptBrushFlatten
                | Self::SculptBrushInflate
                | Self::SculptBrushGrab
                | Self::SculptBrushShrink
                | Self::SculptBrushGrow
                | Self::SculptSymmetryX
                | Self::SculptSymmetryY
                | Self::SculptSymmetryZ
                | Self::ExitSculptMode
        )
    }

    /// Category label for grouping in the UI.
    pub fn category(self) -> &'static str {
        match self {
            Self::ToggleHelp
            | Self::ToggleDebug
            | Self::NewScene
            | Self::OpenProject
            | Self::SaveProject
            | Self::Undo
            | Self::Redo
            | Self::Copy
            | Self::Paste
            | Self::Duplicate
            | Self::DeleteSelected
            | Self::CopyProperties
            | Self::PasteProperties
            | Self::TakeScreenshot
            | Self::ShowExportDialog
            | Self::ToggleCommandPalette => "General",

            Self::CameraFront
            | Self::CameraTop
            | Self::CameraRight
            | Self::CameraBack
            | Self::CameraLeft
            | Self::CameraBottom
            | Self::ToggleOrtho
            | Self::FocusSelected
            | Self::FrameAll => "Camera",

            Self::GizmoTranslate
            | Self::GizmoRotate
            | Self::GizmoScale
            | Self::ToggleGizmoSpace
            | Self::ResetPivot
            | Self::EnterSculptMode
            | Self::QuickSculptMode => "Tools",

            Self::ToggleIsolation
            | Self::CycleShadingMode
            | Self::ToggleTurntable
            | Self::ShowQuickToolbar
            | Self::ToggleReferenceImages
            | Self::ToggleMeasurementTool => "Viewport",

            Self::SculptBrushAdd
            | Self::SculptBrushCarve
            | Self::SculptBrushSmooth
            | Self::SculptBrushFlatten
            | Self::SculptBrushInflate
            | Self::SculptBrushGrab
            | Self::SculptBrushShrink
            | Self::SculptBrushGrow
            | Self::SculptSymmetryX
            | Self::SculptSymmetryY
            | Self::SculptSymmetryZ
            | Self::ExitSculptMode => "Sculpt",
        }
    }
}

// ---------------------------------------------------------------------------
// Keymap configuration
// ---------------------------------------------------------------------------

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct KeymapConfig {
    bindings: HashMap<ActionBinding, KeyCombo>,
}

impl Default for KeymapConfig {
    fn default() -> Self {
        use SerializableKey as K;

        let mut bindings = HashMap::new();
        let mut bind = |action: ActionBinding, combo: KeyCombo| {
            bindings.insert(action, combo);
        };

        // General
        bind(ActionBinding::ToggleHelp, KeyCombo::new(K::F1));
        bind(ActionBinding::ToggleDebug, KeyCombo::new(K::F4));
        bind(ActionBinding::OpenProject, KeyCombo::ctrl(K::O));
        bind(ActionBinding::SaveProject, KeyCombo::ctrl(K::S));
        bind(ActionBinding::Undo, KeyCombo::ctrl(K::Z));
        bind(ActionBinding::Redo, KeyCombo::ctrl(K::Y));
        bind(ActionBinding::Copy, KeyCombo::ctrl(K::C));
        bind(ActionBinding::Paste, KeyCombo::ctrl(K::V));
        bind(ActionBinding::Duplicate, KeyCombo::ctrl(K::D));
        bind(ActionBinding::DeleteSelected, KeyCombo::new(K::Delete));
        bind(ActionBinding::CopyProperties, KeyCombo::ctrl_shift(K::C));
        bind(ActionBinding::PasteProperties, KeyCombo::ctrl_shift(K::V));
        bind(ActionBinding::TakeScreenshot, KeyCombo::ctrl(K::P));
        bind(ActionBinding::ShowExportDialog, KeyCombo::ctrl(K::E));
        bind(ActionBinding::ToggleCommandPalette, KeyCombo::ctrl(K::K));

        // Camera
        bind(ActionBinding::CameraFront, KeyCombo::new(K::F5));
        bind(ActionBinding::CameraTop, KeyCombo::new(K::F6));
        bind(ActionBinding::CameraRight, KeyCombo::new(K::F7));
        bind(ActionBinding::CameraBack, KeyCombo::new(K::F8));
        bind(ActionBinding::CameraLeft, KeyCombo::new(K::F9));
        bind(ActionBinding::CameraBottom, KeyCombo::new(K::F10));
        bind(ActionBinding::ToggleOrtho, KeyCombo::new(K::O));
        bind(ActionBinding::FocusSelected, KeyCombo::new(K::F));
        bind(ActionBinding::FrameAll, KeyCombo::new(K::Home));

        // Gizmo / Tools
        bind(ActionBinding::GizmoTranslate, KeyCombo::new(K::W));
        bind(ActionBinding::GizmoRotate, KeyCombo::new(K::E));
        bind(ActionBinding::GizmoScale, KeyCombo::new(K::R));
        bind(ActionBinding::ToggleGizmoSpace, KeyCombo::new(K::G));
        bind(ActionBinding::ResetPivot, KeyCombo::alt(K::C));
        bind(ActionBinding::EnterSculptMode, KeyCombo::ctrl(K::R));
        bind(ActionBinding::QuickSculptMode, KeyCombo::new(K::S));

        // Viewport
        bind(ActionBinding::ToggleIsolation, KeyCombo::new(K::Slash));
        bind(ActionBinding::CycleShadingMode, KeyCombo::new(K::Z));
        bind(ActionBinding::ToggleTurntable, KeyCombo::new(K::Space));
        bind(ActionBinding::ShowQuickToolbar, KeyCombo::shift(K::A));
        bind(ActionBinding::ToggleReferenceImages, KeyCombo::alt(K::R));
        bind(ActionBinding::ToggleMeasurementTool, KeyCombo::new(K::M));

        // Sculpt brush modes
        bind(ActionBinding::SculptBrushAdd, KeyCombo::new(K::Num1));
        bind(ActionBinding::SculptBrushCarve, KeyCombo::new(K::Num2));
        bind(ActionBinding::SculptBrushSmooth, KeyCombo::new(K::Num3));
        bind(ActionBinding::SculptBrushFlatten, KeyCombo::new(K::Num4));
        bind(ActionBinding::SculptBrushInflate, KeyCombo::new(K::Num5));
        bind(ActionBinding::SculptBrushGrab, KeyCombo::new(K::Num6));
        bind(
            ActionBinding::SculptBrushShrink,
            KeyCombo::new(K::OpenBracket),
        );
        bind(
            ActionBinding::SculptBrushGrow,
            KeyCombo::new(K::CloseBracket),
        );

        // Sculpt symmetry
        bind(ActionBinding::SculptSymmetryX, KeyCombo::new(K::X));
        bind(ActionBinding::SculptSymmetryY, KeyCombo::new(K::Y));
        bind(ActionBinding::SculptSymmetryZ, KeyCombo::new(K::Z));

        // Sculpt escape
        bind(ActionBinding::ExitSculptMode, KeyCombo::new(K::Escape));

        Self { bindings }
    }
}

impl KeymapConfig {
    /// Get the key combo for an action, if bound.
    pub fn get_binding(&self, action: ActionBinding) -> Option<&KeyCombo> {
        self.bindings.get(&action)
    }

    /// Set or update the key combo for an action.
    pub fn set_binding(&mut self, action: ActionBinding, combo: KeyCombo) {
        self.bindings.insert(action, combo);
    }

    /// Remove a binding for an action.
    pub fn remove_binding(&mut self, action: ActionBinding) {
        self.bindings.remove(&action);
    }

    /// Reset all bindings to defaults.
    pub fn reset_to_defaults(&mut self) {
        *self = Self::default();
    }

    /// Get a formatted shortcut string for an action (e.g., "Ctrl+S").
    /// Returns None if the action is unbound.
    pub fn format_shortcut(&self, action: ActionBinding) -> Option<String> {
        self.bindings.get(&action).map(|combo| combo.to_string())
    }

    /// Find which action (if any) is bound to the given key combo.
    pub fn find_action_for_combo(&self, combo: &KeyCombo) -> Option<ActionBinding> {
        self.bindings
            .iter()
            .find(|(_, c)| *c == combo)
            .map(|(action, _)| *action)
    }

    /// Check if a key combo conflicts with an existing binding (excluding the given action).
    pub fn find_conflict(&self, combo: &KeyCombo, exclude: ActionBinding) -> Option<ActionBinding> {
        self.bindings
            .iter()
            .find(|(action, c)| **action != exclude && *c == combo)
            .map(|(action, _)| *action)
    }

    /// Iterate over all bindings.
    pub fn bindings(&self) -> &HashMap<ActionBinding, KeyCombo> {
        &self.bindings
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_keymap_contains_expected_bindings() {
        let keymap = KeymapConfig::default();
        // Verify a selection of key bindings match the hardcoded shortcuts
        assert_eq!(
            keymap.get_binding(ActionBinding::SaveProject),
            Some(&KeyCombo::ctrl(SerializableKey::S))
        );
        assert_eq!(
            keymap.get_binding(ActionBinding::Undo),
            Some(&KeyCombo::ctrl(SerializableKey::Z))
        );
        assert_eq!(
            keymap.get_binding(ActionBinding::DeleteSelected),
            Some(&KeyCombo::new(SerializableKey::Delete))
        );
        assert_eq!(
            keymap.get_binding(ActionBinding::CameraFront),
            Some(&KeyCombo::new(SerializableKey::F5))
        );
        assert_eq!(
            keymap.get_binding(ActionBinding::GizmoTranslate),
            Some(&KeyCombo::new(SerializableKey::W))
        );
        assert_eq!(
            keymap.get_binding(ActionBinding::ShowQuickToolbar),
            Some(&KeyCombo::shift(SerializableKey::A))
        );
        assert_eq!(
            keymap.get_binding(ActionBinding::ToggleReferenceImages),
            Some(&KeyCombo::alt(SerializableKey::R))
        );
        assert_eq!(
            keymap.get_binding(ActionBinding::ToggleMeasurementTool),
            Some(&KeyCombo::new(SerializableKey::M))
        );
    }

    #[test]
    fn set_binding_updates_keymap() {
        let mut keymap = KeymapConfig::default();
        let new_combo = KeyCombo::ctrl_shift(SerializableKey::S);
        keymap.set_binding(ActionBinding::SaveProject, new_combo.clone());
        assert_eq!(
            keymap.get_binding(ActionBinding::SaveProject),
            Some(&new_combo)
        );
    }

    #[test]
    fn reset_restores_defaults() {
        let mut keymap = KeymapConfig::default();
        let original = keymap.get_binding(ActionBinding::SaveProject).cloned();
        keymap.set_binding(
            ActionBinding::SaveProject,
            KeyCombo::new(SerializableKey::F12),
        );
        assert_ne!(
            keymap.get_binding(ActionBinding::SaveProject),
            original.as_ref()
        );

        keymap.reset_to_defaults();
        assert_eq!(
            keymap.get_binding(ActionBinding::SaveProject),
            original.as_ref()
        );
    }

    #[test]
    fn format_shortcut_produces_readable_strings() {
        let keymap = KeymapConfig::default();
        assert_eq!(
            keymap.format_shortcut(ActionBinding::SaveProject),
            Some("Ctrl+S".to_string())
        );
        assert_eq!(
            keymap.format_shortcut(ActionBinding::CopyProperties),
            Some("Ctrl+Shift+C".to_string())
        );
        assert_eq!(
            keymap.format_shortcut(ActionBinding::CameraFront),
            Some("F5".to_string())
        );
        assert_eq!(
            keymap.format_shortcut(ActionBinding::ResetPivot),
            Some("Alt+C".to_string())
        );
        assert_eq!(
            keymap.format_shortcut(ActionBinding::ShowQuickToolbar),
            Some("Shift+A".to_string())
        );
        assert_eq!(
            keymap.format_shortcut(ActionBinding::ToggleReferenceImages),
            Some("Alt+R".to_string())
        );
        assert_eq!(
            keymap.format_shortcut(ActionBinding::ToggleMeasurementTool),
            Some("M".to_string())
        );
    }

    #[test]
    fn find_conflict_detects_duplicate_bindings() {
        let keymap = KeymapConfig::default();
        // Ctrl+S is bound to SaveProject by default
        let conflict = keymap.find_conflict(
            &KeyCombo::ctrl(SerializableKey::S),
            ActionBinding::OpenProject,
        );
        assert_eq!(conflict, Some(ActionBinding::SaveProject));

        // No conflict when checking against the same action
        let no_conflict = keymap.find_conflict(
            &KeyCombo::ctrl(SerializableKey::S),
            ActionBinding::SaveProject,
        );
        assert_eq!(no_conflict, None);
    }

    #[test]
    fn serialization_roundtrip() {
        let keymap = KeymapConfig::default();
        let json = serde_json::to_string_pretty(&keymap).expect("serialize");
        let deserialized: KeymapConfig = serde_json::from_str(&json).expect("deserialize");

        // Verify a few bindings survived the roundtrip
        assert_eq!(
            deserialized.get_binding(ActionBinding::SaveProject),
            keymap.get_binding(ActionBinding::SaveProject)
        );
        assert_eq!(
            deserialized.get_binding(ActionBinding::ToggleHelp),
            keymap.get_binding(ActionBinding::ToggleHelp)
        );
        assert_eq!(
            deserialized.get_binding(ActionBinding::SculptBrushAdd),
            keymap.get_binding(ActionBinding::SculptBrushAdd)
        );
    }

    #[test]
    fn remove_binding_works() {
        let mut keymap = KeymapConfig::default();
        assert!(keymap.get_binding(ActionBinding::SaveProject).is_some());
        keymap.remove_binding(ActionBinding::SaveProject);
        assert!(keymap.get_binding(ActionBinding::SaveProject).is_none());
        assert_eq!(keymap.format_shortcut(ActionBinding::SaveProject), None);
    }

    #[test]
    fn all_actions_have_default_binding() {
        let keymap = KeymapConfig::default();
        // NewScene intentionally has no default keybinding
        for action in ActionBinding::ALL {
            if *action == ActionBinding::NewScene {
                continue;
            }
            assert!(
                keymap.get_binding(*action).is_some(),
                "ActionBinding::{:?} has no default binding",
                action
            );
        }
    }

    #[test]
    fn action_binding_labels_are_nonempty() {
        for action in ActionBinding::ALL {
            assert!(!action.label().is_empty(), "{:?} has empty label", action);
        }
    }

    #[test]
    fn key_combo_display_format() {
        assert_eq!(KeyCombo::new(SerializableKey::F1).to_string(), "F1");
        assert_eq!(KeyCombo::ctrl(SerializableKey::S).to_string(), "Ctrl+S");
        assert_eq!(
            KeyCombo::ctrl_shift(SerializableKey::C).to_string(),
            "Ctrl+Shift+C"
        );
        assert_eq!(KeyCombo::alt(SerializableKey::C).to_string(), "Alt+C");
        assert_eq!(KeyCombo::shift(SerializableKey::A).to_string(), "Shift+A");
    }
}
