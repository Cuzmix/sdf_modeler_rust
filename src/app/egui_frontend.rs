use std::sync::atomic::Ordering;

use eframe::egui;

use crate::compat::Instant;
use crate::ui::dock::{SceneTreeContext, SdfTabViewer, ViewportContext};

use super::actions::ActionSink;
use super::backend_frame::UiFrameFeedback;
use super::{BakeStatus, SdfApp, Toast};

impl SdfApp {
    pub(super) fn draw_egui_frontend(
        &mut self,
        ctx: &egui::Context,
        now_seconds: f64,
        action_sink: &mut ActionSink,
    ) -> UiFrameFeedback {
        self.show_menu_bar(ctx, action_sink);
        self.show_status_bar(ctx);
        crate::ui::help::draw(ctx, &mut self.ui.show_help, &self.settings.keymap);
        crate::ui::profiler::draw(
            ctx,
            self.ui.show_debug,
            &self.perf.timings,
            &self.doc.scene,
            &self.gpu,
            &self.settings,
            &self.doc.camera,
        );
        crate::ui::toasts::draw(ctx, &mut self.ui.toasts);
        crate::ui::quick_toolbar::draw(ctx, &mut self.ui.show_quick_toolbar, action_sink);

        crate::ui::sculpt_convert_dialog::draw(
            ctx,
            &mut self.ui.sculpt_convert_dialog,
            action_sink,
            self.settings.max_sculpt_resolution,
        );

        crate::ui::import_dialog::draw(
            ctx,
            &mut self.ui.import_dialog,
            action_sink,
            self.settings.max_sculpt_resolution,
        );

        if let crate::ui::export_dialog::ExportDialogResult::Export = crate::ui::export_dialog::draw(
            ctx,
            &mut self.ui.show_export_dialog,
            &mut self.settings,
            &self.async_state.export_status,
        ) {
            self.settings.save();
            self.start_export(ctx);
        }

        crate::ui::export_progress::draw_export(ctx, &self.async_state.export_status);
        crate::ui::export_progress::draw_import(ctx, &self.async_state.import_status);

        crate::ui::settings_window::draw(
            ctx,
            &mut self.ui.show_settings,
            &mut self.settings,
            &mut self.ui.show_debug,
            self.initial_vsync,
            action_sink,
            &mut self.ui.rebinding_action,
        );

        let ui_start = Instant::now();
        let bake_progress = match &self.async_state.bake_status {
            BakeStatus::InProgress {
                progress, total, ..
            } => Some((progress.load(Ordering::Relaxed), *total)),
            BakeStatus::Idle => None,
        };

        let mut pending_pick = None;
        let mut sculpt_ctrl_held = false;
        let mut sculpt_shift_held = false;
        let mut sculpt_pressure: f32 = 0.0;
        let mut is_hover_pick = false;
        let mut gizmo_drag_active = false;
        let sculpt_count = self.gpu.sculpt_tex_indices.len();
        let isolation_label: Option<String> =
            self.ui.isolation_state.as_ref().and_then(|isolation| {
                self.doc
                    .scene
                    .nodes
                    .get(&isolation.isolated_node)
                    .map(|node| node.name.clone())
            });
        let solo_label: Option<String> = self
            .doc
            .soloed_light
            .and_then(|id| self.doc.scene.nodes.get(&id).map(|node| node.name.clone()));
        let fps_info = if self.settings.show_fps_overlay {
            Some((self.perf.timings.avg_fps, self.perf.timings.avg_frame_ms))
        } else {
            None
        };

        {
            let (active_light_ids, total_light_count) =
                crate::gpu::buffers::identify_active_lights(&self.doc.scene, self.doc.camera.eye());
            self.ui.active_light_ids = active_light_ids;
            self.ui.total_light_count = total_light_count;

            if total_light_count > crate::graph::scene::MAX_SCENE_LIGHTS {
                if self.ui.last_light_warning_count != Some(total_light_count) {
                    self.ui.last_light_warning_count = Some(total_light_count);
                    self.ui.toasts.push(Toast {
                        message: format!(
                            "Scene has {} lights - only the {} nearest to camera are active.",
                            total_light_count,
                            crate::graph::scene::MAX_SCENE_LIGHTS,
                        ),
                        is_error: false,
                        created: Instant::now(),
                        duration: std::time::Duration::from_secs(5),
                    });
                }
            } else {
                self.ui.last_light_warning_count = None;
            }
        }

        let selection_behavior = self.settings.selection_behavior;
        let dock_style = self
            .settings
            .dock_style
            .to_egui_dock_style(ctx.style().as_ref());
        let mut tab_viewer = SdfTabViewer {
            camera: &mut self.doc.camera,
            scene: &mut self.doc.scene,
            node_graph_state: &mut self.ui.node_graph_state,
            light_graph_state: &mut self.ui.light_graph_state,
            active_tool: &self.doc.active_tool,
            sculpt_state: &mut self.doc.sculpt_state,
            settings: &mut self.settings,
            time: now_seconds as f32,
            bake_progress,
            viewport: ViewportContext {
                gizmo_state: &mut self.gizmo.state,
                gizmo_mode: &self.gizmo.mode,
                gizmo_space: &self.gizmo.space,
                gizmo_visible: self.gizmo.gizmo_visible,
                pivot_offset: &mut self.gizmo.pivot_offset,
                pending_pick: &mut pending_pick,
                sculpt_count,
                fps_info,
                sculpt_ctrl_held: &mut sculpt_ctrl_held,
                sculpt_shift_held: &mut sculpt_shift_held,
                sculpt_pressure: &mut sculpt_pressure,
                last_sculpt_hit: self.async_state.last_sculpt_hit,
                isolation_label: isolation_label.clone(),
                turntable_active: self.ui.turntable_active,
                is_hover_pick: &mut is_hover_pick,
                gizmo_drag_active: &mut gizmo_drag_active,
                hover_world_pos: self.async_state.hover_world_pos,
                cursor_over_geometry: self.async_state.cursor_over_geometry,
                sculpt_brush_adjust: &mut self.ui.sculpt_brush_adjust,
                soloed_light: self.doc.soloed_light,
                solo_label: solo_label.clone(),
                show_distance_readout: &mut self.ui.show_distance_readout,
                selection_behavior,
                measurement_mode: &mut self.ui.measurement_mode,
                measurement_points: &mut self.ui.measurement_points,
            },
            scene_tree: SceneTreeContext {
                renaming_node: &mut self.ui.renaming_node,
                rename_buf: &mut self.ui.rename_buf,
                drag_state: &mut self.ui.scene_tree_drag,
                search_filter: &mut self.ui.scene_tree_search,
            },
            actions: action_sink,
            history: &self.doc.history,
            active_light_ids: &self.ui.active_light_ids,
            material_library: &mut self.material_library,
            reference_images: &mut self.ui.reference_images,
            multi_transform_edit: &mut self.ui.multi_transform_edit,
            timings: &self.perf.timings,
        };

        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                egui_dock::DockArea::new(&mut self.ui.dock_state)
                    .style(dock_style)
                    .show_inside(ui, &mut tab_viewer);
            });

        crate::ui::command_palette::draw(
            ctx,
            &mut self.ui.command_palette_open,
            &mut self.ui.command_palette_query,
            &mut self.ui.command_palette_selected,
            &self.doc.scene,
            &self.settings.keymap,
            action_sink,
        );

        self.perf.timings.ui_draw_s = ui_start.elapsed().as_secs_f64();

        UiFrameFeedback {
            pending_pick,
            sculpt_ctrl_held,
            sculpt_shift_held,
            sculpt_pressure,
            is_hover_pick,
            gizmo_drag_active,
        }
    }
}
