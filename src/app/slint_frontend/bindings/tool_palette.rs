use std::rc::Rc;

use slint::VecModel;

use crate::app::frontend_models::{ShellSnapshot, ToolPaletteEntry, ToolPaletteKind};
use crate::app::slint_frontend::{ToolPaletteAction, ToolPaletteButtonView, ToolPaletteState};

pub(super) fn build_tool_palette_state(snapshot: &ShellSnapshot) -> ToolPaletteState {
    ToolPaletteState {
        visible: snapshot.tool_palette.visible,
        select_tool: tool_button_view(&snapshot.tool_palette.select_tool),
        brush_tools: Rc::new(VecModel::from(
            snapshot
                .tool_palette
                .brush_tools
                .iter()
                .map(tool_button_view)
                .collect::<Vec<_>>(),
        ))
        .into(),
    }
}

fn tool_button_view(entry: &ToolPaletteEntry) -> ToolPaletteButtonView {
    ToolPaletteButtonView {
        label: entry.label.clone().into(),
        active: entry.active,
        action: tool_palette_action(entry.kind),
    }
}

fn tool_palette_action(kind: ToolPaletteKind) -> ToolPaletteAction {
    match kind {
        ToolPaletteKind::Select => ToolPaletteAction::Select,
        ToolPaletteKind::Brush(crate::sculpt::BrushMode::Add) => ToolPaletteAction::BrushAdd,
        ToolPaletteKind::Brush(crate::sculpt::BrushMode::Carve) => ToolPaletteAction::BrushCarve,
        ToolPaletteKind::Brush(crate::sculpt::BrushMode::Smooth) => ToolPaletteAction::BrushSmooth,
        ToolPaletteKind::Brush(crate::sculpt::BrushMode::Flatten) => {
            ToolPaletteAction::BrushFlatten
        }
        ToolPaletteKind::Brush(crate::sculpt::BrushMode::Inflate) => {
            ToolPaletteAction::BrushInflate
        }
        ToolPaletteKind::Brush(crate::sculpt::BrushMode::Grab) => ToolPaletteAction::BrushGrab,
    }
}
