use crate::app::frontend_models::ShellSnapshot;
use crate::app::slint_frontend::OverlayLayoutViewState;

pub(super) fn build_overlay_layout_state(snapshot: &ShellSnapshot) -> OverlayLayoutViewState {
    OverlayLayoutViewState {
        safe_area_left: snapshot.overlay_layout.safe_area_left,
        safe_area_top: snapshot.overlay_layout.safe_area_top,
        safe_area_right: snapshot.overlay_layout.safe_area_right,
        safe_area_bottom: snapshot.overlay_layout.safe_area_bottom,
        virtual_keyboard_x: snapshot.overlay_layout.virtual_keyboard_x,
        virtual_keyboard_y: snapshot.overlay_layout.virtual_keyboard_y,
        virtual_keyboard_width: snapshot.overlay_layout.virtual_keyboard_width,
        virtual_keyboard_height: snapshot.overlay_layout.virtual_keyboard_height,
        usable_x: snapshot.overlay_layout.usable_x,
        usable_y: snapshot.overlay_layout.usable_y,
        usable_width: snapshot.overlay_layout.usable_width,
        usable_height: snapshot.overlay_layout.usable_height,
    }
}
