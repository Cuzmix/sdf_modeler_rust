use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;

use crate::app::actions::{Action, ActionSink};
use crate::gpu::camera::{Camera, CameraUniform};
use crate::gpu::picking::PendingPick;
use crate::graph::scene::{CsgOp, ModifierKind, NodeData, NodeId, Scene, SdfPrimitive, TransformKind};
use crate::sculpt::{ActiveTool, BrushMode, SculptState};
use crate::settings::SnapConfig;
use crate::ui::gizmo::{self, GizmoMode, GizmoSpace, GizmoState};

use super::ViewportResources;

// ---------------------------------------------------------------------------
// Paint callback
// ---------------------------------------------------------------------------

struct ViewportCallback {
    /// Camera uniform with viewport set to render dimensions (offscreen texture size).
    render_uniform: CameraUniform,
    /// Display viewport in physical pixels: [x, y, width, height].
    display_viewport: [f32; 4],
    /// Render scale factor (0.25 - 1.0).
    render_scale: f32,
    /// Outline color (RGB).
    outline_color: [f32; 3],
    /// Outline width in pixels.
    outline_width: f32,
}

impl egui_wgpu::CallbackTrait for ViewportCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _encoder: &mut wgpu::CommandEncoder,
        callback_resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let resources = callback_resources
            .get_mut::<ViewportResources>()
            .unwrap();

        let display_w = self.display_viewport[2] as u32;
        let display_h = self.display_viewport[3] as u32;
        let render_w = ((display_w as f32) * self.render_scale).max(1.0) as u32;
        let render_h = ((display_h as f32) * self.render_scale).max(1.0) as u32;

        // Ensure offscreen texture + blit bind group are the right size
        resources.ensure_offscreen_texture(device, render_w, render_h);

        // Write camera uniform (viewport = render dimensions for the SDF shader)
        queue.write_buffer(
            &resources.camera_buffer,
            0,
            bytemuck::bytes_of(&self.render_uniform),
        );

        // Write blit params (display viewport + outline settings for the blit shader)
        let blit_data: [f32; 8] = [
            self.display_viewport[0], self.display_viewport[1],
            self.display_viewport[2], self.display_viewport[3],
            self.outline_color[0], self.outline_color[1], self.outline_color[2],
            self.outline_width,
        ];
        queue.write_buffer(
            &resources.blit_params_buffer,
            0,
            bytemuck::cast_slice(&blit_data),
        );

        // Render SDF scene to offscreen texture
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Offscreen Encoder"),
        });
        {
            let view = resources.offscreen_view.as_ref().unwrap();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Offscreen SDF Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // Use composite render pipeline when enabled, otherwise direct render
            if resources.use_composite {
                if let Some(ref comp) = resources.composite {
                    pass.set_pipeline(&comp.render_pipeline);
                    pass.set_bind_group(0, &resources.camera_bind_group, &[]);
                    pass.set_bind_group(1, &resources.scene_bind_group, &[]);
                    pass.set_bind_group(2, &comp.render_bg, &[]);
                }
            } else {
                pass.set_pipeline(&resources.pipeline);
                pass.set_bind_group(0, &resources.camera_bind_group, &[]);
                pass.set_bind_group(1, &resources.scene_bind_group, &[]);
                pass.set_bind_group(2, &resources.voxel_tex_bind_group, &[]);
            }
            pass.draw(0..3, 0..1);
        }

        vec![encoder.finish()]
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        callback_resources: &egui_wgpu::CallbackResources,
    ) {
        let resources = callback_resources.get::<ViewportResources>().unwrap();
        if let Some(ref blit_bg) = resources.blit_bind_group {
            render_pass.set_pipeline(&resources.blit_pipeline);
            render_pass.set_bind_group(0, blit_bg, &[]);
            render_pass.draw(0..3, 0..1);
        }
    }
}

fn brush_cursor_color(mode: &BrushMode) -> egui::Color32 {
    match mode {
        BrushMode::Add => egui::Color32::from_rgba_premultiplied(100, 200, 100, 160),
        BrushMode::Carve => egui::Color32::from_rgba_premultiplied(200, 100, 100, 160),
        BrushMode::Smooth => egui::Color32::from_rgba_premultiplied(100, 150, 255, 160),
        BrushMode::Flatten => egui::Color32::from_rgba_premultiplied(255, 200, 80, 160),
        BrushMode::Inflate => egui::Color32::from_rgba_premultiplied(200, 140, 255, 160),
        BrushMode::Grab => egui::Color32::from_rgba_premultiplied(255, 160, 60, 160),
    }
}

/// Compute the effective brush mode given modifier key state.
/// Shift → Smooth (overrides everything), Ctrl → invert (Add↔Carve, Inflate→Carve).
fn effective_brush_mode(base: &BrushMode, ctrl: bool, shift: bool) -> BrushMode {
    if shift {
        return BrushMode::Smooth;
    }
    if ctrl {
        return match base {
            BrushMode::Add => BrushMode::Carve,
            BrushMode::Carve => BrushMode::Add,
            BrushMode::Inflate => BrushMode::Carve,
            other => other.clone(),
        };
    }
    base.clone()
}

/// Draw a semi-transparent symmetry plane overlay at the mirror axis.
fn draw_symmetry_plane(painter: &egui::Painter, camera: &Camera, rect: egui::Rect, axis: u8) {
    let aspect = rect.width() / rect.height();
    let view_proj = camera.projection_matrix(aspect) * camera.view_matrix();

    let extent = 5.0_f32;
    let corners: [glam::Vec3; 4] = match axis {
        0 => [
            glam::Vec3::new(0.0, -extent, -extent),
            glam::Vec3::new(0.0, -extent, extent),
            glam::Vec3::new(0.0, extent, extent),
            glam::Vec3::new(0.0, extent, -extent),
        ],
        1 => [
            glam::Vec3::new(-extent, 0.0, -extent),
            glam::Vec3::new(extent, 0.0, -extent),
            glam::Vec3::new(extent, 0.0, extent),
            glam::Vec3::new(-extent, 0.0, extent),
        ],
        _ => [
            glam::Vec3::new(-extent, -extent, 0.0),
            glam::Vec3::new(extent, -extent, 0.0),
            glam::Vec3::new(extent, extent, 0.0),
            glam::Vec3::new(-extent, extent, 0.0),
        ],
    };

    let screen_pts: Vec<egui::Pos2> = corners
        .iter()
        .filter_map(|&c| gizmo::world_to_screen(c, &view_proj, rect))
        .collect();

    if screen_pts.len() < 3 {
        return;
    }

    let (fill, border) = match axis {
        0 => (
            egui::Color32::from_rgba_premultiplied(255, 50, 50, 20),
            egui::Color32::from_rgba_premultiplied(255, 80, 80, 80),
        ),
        1 => (
            egui::Color32::from_rgba_premultiplied(50, 255, 50, 20),
            egui::Color32::from_rgba_premultiplied(80, 255, 80, 80),
        ),
        _ => (
            egui::Color32::from_rgba_premultiplied(50, 50, 255, 20),
            egui::Color32::from_rgba_premultiplied(80, 80, 255, 80),
        ),
    };

    painter.add(egui::Shape::convex_polygon(
        screen_pts,
        fill,
        egui::Stroke::new(1.0, border),
    ));
}

pub struct ViewportOutput {
    pub pending_pick: Option<PendingPick>,
    /// Modifier keys at time of sculpt drag (Ctrl = invert, Shift = smooth).
    pub sculpt_ctrl_held: bool,
    pub sculpt_shift_held: bool,
}

pub fn draw(
    ui: &mut egui::Ui,
    camera: &mut Camera,
    scene: &mut Scene,
    selected: &mut Option<NodeId>,
    gizmo_state: &mut GizmoState,
    gizmo_mode: &GizmoMode,
    gizmo_space: &GizmoSpace,
    pivot_offset: &mut glam::Vec3,
    sculpt_state: &SculptState,
    active_tool: &ActiveTool,
    time: f32,
    render_config: &crate::settings::RenderConfig,
    sculpt_count: usize,
    fps_info: Option<(f64, f64)>, // (fps, frame_ms)
    actions: &mut ActionSink,
    snap_config: &SnapConfig,
    isolation_label: Option<&str>,
) -> ViewportOutput {
    let mut output = ViewportOutput {
        pending_pick: None,
        sculpt_ctrl_held: false,
        sculpt_shift_held: false,
    };
    let rect = ui.available_rect_before_wrap();
    let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());

    // --- Paint the SDF viewport (WGPU callback) ---
    let pixels_per_point = ui.ctx().pixels_per_point();
    let viewport = [
        rect.min.x * pixels_per_point,
        rect.min.y * pixels_per_point,
        rect.width() * pixels_per_point,
        rect.height() * pixels_per_point,
    ];
    // Interaction detection
    let multi_touch_active = ui.input(|i| i.multi_touch()).is_some();
    let sculpt_active = sculpt_state.is_active();
    let camera_dragging = if sculpt_active {
        response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
    } else {
        response.dragged_by(egui::PointerButton::Primary)
            || response.dragged_by(egui::PointerButton::Secondary)
            || response.dragged_by(egui::PointerButton::Middle)
    };
    let sculpt_brushing = sculpt_active
        && response.dragged_by(egui::PointerButton::Primary);
    let multi_sculpt_reduce = render_config.auto_reduce_steps && sculpt_count >= 2;
    let is_interacting = camera_dragging || sculpt_brushing;
    // Fast quality mode: half steps + skip AO/shadows
    let quality_mode = if (is_interacting && render_config.sculpt_fast_mode) || multi_sculpt_reduce {
        1.0
    } else {
        0.0
    };

    // Resolution scaling: reduced resolution during interaction
    let render_scale = if is_interacting {
        render_config.interaction_render_scale.clamp(0.25, 1.0)
    } else {
        render_config.rest_render_scale.clamp(0.25, 1.0)
    };

    let scene_bounds = scene.compute_bounds();

    // Render uniform uses the RENDER viewport (offscreen texture dimensions)
    let render_w = (viewport[2] * render_scale).max(1.0);
    let render_h = (viewport[3] * render_scale).max(1.0);
    let render_viewport = [0.0, 0.0, render_w, render_h];
    let selected_idx = (*selected)
        .and_then(|id| {
            let order = scene.visible_topo_order();
            order.iter().position(|&nid| nid == id)
        })
        .map(|i| i as f32)
        .unwrap_or(-1.0);
    let shading_mode_val = render_config.shading_mode.gpu_value();
    let render_uniform = camera.to_uniform(render_viewport, time, quality_mode, render_config.show_grid, scene_bounds, selected_idx, shading_mode_val);

    ui.painter().add(egui_wgpu::Callback::new_paint_callback(
        rect,
        ViewportCallback {
            render_uniform,
            display_viewport: viewport,
            render_scale,
            outline_color: render_config.outline_color,
            outline_width: render_config.outline_thickness,
        },
    ));

    // --- Symmetry plane overlay ---
    if let Some(axis) = sculpt_state.symmetry_axis() {
        draw_symmetry_plane(ui.painter(), camera, rect, axis);
    }

    // --- Node labels overlay ---
    if render_config.show_node_labels {
        draw_node_labels(ui.painter(), camera, scene, *selected, rect);
    }

    // --- Gizmo overlay (drawn on top of WGPU content) ---

    let gizmo_consumed = if sculpt_active {
        false // Gizmo is disabled during sculpt mode
    } else {
        gizmo::draw_and_interact(
            ui.painter(),
            &response,
            camera,
            scene,
            *selected,
            gizmo_state,
            gizmo_mode,
            gizmo_space,
            pivot_offset,
            rect,
            snap_config,
        )
    };

    // --- Interaction priority: sculpt > gizmo > pick > orbit ---

    if sculpt_active {
        // Sculpt mode: drag applies brush continuously via pick
        if response.dragged_by(egui::PointerButton::Primary) && !multi_touch_active {
            if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                if rect.contains(pos) {
                    let mouse_px = [
                        (pos.x - rect.min.x) * pixels_per_point,
                        (pos.y - rect.min.y) * pixels_per_point,
                    ];
                    output.pending_pick = Some(PendingPick {
                        mouse_pos: mouse_px,
                        camera_uniform: camera.to_uniform(viewport, time, 0.0, false, scene_bounds, -1.0, 0.0),
                    });
                    // Capture modifier keys for Ctrl-invert / Shift-smooth
                    let modifiers = ui.input(|i| i.modifiers);
                    output.sculpt_ctrl_held = modifiers.ctrl;
                    output.sculpt_shift_held = modifiers.shift;
                }
            }
        }

        // Enhanced brush cursor preview
        if let SculptState::Active {
            ref brush_mode,
            brush_radius,
            brush_strength,
            symmetry_axis,
            ..
        } = sculpt_state
        {
            if let Some(hover_pos) = response.hover_pos() {
                let screen_radius = brush_radius / camera.distance * rect.height() * 0.5;
                // Reflect Ctrl/Shift modifier overrides in cursor color
                let modifiers = ui.input(|i| i.modifiers);
                let effective_mode = effective_brush_mode(brush_mode, modifiers.ctrl, modifiers.shift);
                let mode_color = brush_cursor_color(&effective_mode);

                // Outer ring: brush extent (color-coded by mode)
                ui.painter().circle_stroke(
                    hover_pos,
                    screen_radius,
                    egui::Stroke::new(1.5, mode_color),
                );

                // Inner fill: strength indicator with mode tint
                let strength_alpha = (brush_strength / 0.5 * 60.0).clamp(8.0, 60.0) as u8;
                ui.painter().circle_filled(
                    hover_pos,
                    screen_radius * 0.6,
                    egui::Color32::from_rgba_premultiplied(
                        mode_color.r(), mode_color.g(), mode_color.b(), strength_alpha,
                    ),
                );

                // Crosshair center dot
                ui.painter().circle_filled(
                    hover_pos,
                    2.0,
                    egui::Color32::from_rgba_premultiplied(255, 255, 255, 160),
                );

                // Symmetry mirror cursor
                if let Some(axis) = symmetry_axis {
                    let mirror_color = match axis {
                        0 => egui::Color32::from_rgba_premultiplied(255, 100, 100, 100),
                        1 => egui::Color32::from_rgba_premultiplied(100, 255, 100, 100),
                        _ => egui::Color32::from_rgba_premultiplied(100, 100, 255, 100),
                    };
                    // Mirror the hover position through the symmetry plane in screen space
                    // Project the origin and the mirrored point to get screen-space mirror
                    let aspect = rect.width() / rect.height();
                    let vp = camera.projection_matrix(aspect) * camera.view_matrix();
                    let origin = gizmo::world_to_screen(glam::Vec3::ZERO, &vp, rect);
                    if let Some(origin_screen) = origin {
                        // Mirror hover_pos around the axis line through origin
                        let mirror_pos = match axis {
                            0 => {
                                // X symmetry: mirror horizontally around origin.x
                                egui::pos2(
                                    2.0 * origin_screen.x - hover_pos.x,
                                    hover_pos.y,
                                )
                            }
                            1 => {
                                // Y symmetry: mirror vertically around origin.y
                                egui::pos2(
                                    hover_pos.x,
                                    2.0 * origin_screen.y - hover_pos.y,
                                )
                            }
                            _ => {
                                // Z symmetry: approximate mirror horizontally
                                egui::pos2(
                                    2.0 * origin_screen.x - hover_pos.x,
                                    hover_pos.y,
                                )
                            }
                        };
                        if rect.contains(mirror_pos) {
                            ui.painter().circle_stroke(
                                mirror_pos,
                                screen_radius,
                                egui::Stroke::new(1.0, mirror_color),
                            );
                            ui.painter().circle_filled(
                                mirror_pos,
                                2.0,
                                mirror_color,
                            );
                        }
                    }
                }

                // Brush HUD text near cursor
                let mode_name = match brush_mode {
                    BrushMode::Add => "Add",
                    BrushMode::Carve => "Carve",
                    BrushMode::Smooth => "Smooth",
                    BrushMode::Flatten => "Flatten",
                    BrushMode::Inflate => "Inflate",
                    BrushMode::Grab => "Grab",
                };
                let hud_text = format!("{} R:{:.2} S:{:.3}", mode_name, brush_radius, brush_strength);
                let text_pos = hover_pos + egui::vec2(screen_radius + 8.0, screen_radius + 4.0);
                let font = egui::FontId::monospace(10.0);
                ui.painter().text(
                    text_pos + egui::vec2(1.0, 1.0),
                    egui::Align2::LEFT_TOP,
                    &hud_text,
                    font.clone(),
                    egui::Color32::from_black_alpha(180),
                );
                ui.painter().text(
                    text_pos,
                    egui::Align2::LEFT_TOP,
                    &hud_text,
                    font,
                    egui::Color32::from_white_alpha(200),
                );
            }
        }

        // Right-click still orbits in sculpt mode, secondary drag pans
        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            camera.pan(delta.x, delta.y);
        }
        if response.dragged_by(egui::PointerButton::Middle) {
            let delta = response.drag_delta();
            let modifiers = ui.input(|i| i.modifiers);
            if modifiers.ctrl && modifiers.alt {
                let sign = if render_config.invert_roll { -1.0 } else { 1.0 };
                camera.roll_by(sign * delta.x, render_config.roll_sensitivity);
            } else {
                camera.orbit(delta.x, delta.y);
                if render_config.clamp_orbit_pitch {
                    camera.clamp_pitch();
                }
            }
        }
    } else if !gizmo_consumed {
        // Normal mode: click to pick
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let mouse_px = [
                    (pos.x - rect.min.x) * pixels_per_point,
                    (pos.y - rect.min.y) * pixels_per_point,
                ];
                let pick_uniform = camera.to_uniform(viewport, time, 0.0, false, scene_bounds, -1.0, 0.0);
                output.pending_pick = Some(PendingPick {
                    mouse_pos: mouse_px,
                    camera_uniform: pick_uniform,
                });
            }
        }

        if response.dragged_by(egui::PointerButton::Primary) && !multi_touch_active {
            let delta = response.drag_delta();
            let modifiers = ui.input(|i| i.modifiers);
            if modifiers.ctrl && modifiers.alt {
                let sign = if render_config.invert_roll { -1.0 } else { 1.0 };
                camera.roll_by(sign * delta.x, render_config.roll_sensitivity);
            } else {
                camera.orbit(delta.x, delta.y);
                if render_config.clamp_orbit_pitch {
                    camera.clamp_pitch();
                }
            }
        }

        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            camera.pan(delta.x, delta.y);
        }
    }

    if response.hovered() {
        let scroll = ui.input(|i| i.smooth_scroll_delta.y);
        if scroll != 0.0 {
            camera.zoom(scroll);
        }
    }

    // --- Multi-touch: pinch-to-zoom + two-finger pan ---
    if let Some(touch) = ui.input(|i| i.multi_touch()) {
        if touch.zoom_delta != 1.0 {
            let zoom_amount = (touch.zoom_delta - 1.0) * render_config.touch_zoom_sensitivity;
            camera.zoom(zoom_amount);
        }
        if touch.translation_delta != egui::Vec2::ZERO {
            let sign = if render_config.invert_touch_pan { -1.0 } else { 1.0 };
            camera.pan(sign * touch.translation_delta.x, sign * touch.translation_delta.y);
        }
        if touch.rotation_delta != 0.0 {
            let sign = if render_config.invert_roll { -1.0 } else { 1.0 };
            camera.roll += sign * touch.rotation_delta;
        }
    }

    // --- FPS counter overlay (top-left of viewport) ---
    if let Some((fps, frame_ms)) = fps_info {
        let text = format!("{:.0} FPS ({:.1} ms)", fps, frame_ms);
        let font = egui::FontId::monospace(11.0);
        let pos = rect.min + egui::vec2(6.0, 4.0);
        // Shadow for readability
        ui.painter().text(
            pos + egui::vec2(1.0, 1.0),
            egui::Align2::LEFT_TOP,
            &text,
            font.clone(),
            egui::Color32::from_black_alpha(180),
        );
        let color = if fps >= 55.0 {
            egui::Color32::from_rgb(120, 220, 120) // green
        } else if fps >= 30.0 {
            egui::Color32::from_rgb(220, 200, 80) // yellow
        } else {
            egui::Color32::from_rgb(220, 100, 100) // red
        };
        ui.painter().text(
            pos,
            egui::Align2::LEFT_TOP,
            &text,
            font,
            color,
        );
    }

    // --- Isolation mode indicator ---
    if let Some(label) = isolation_label {
        let text = format!("ISOLATED: {}", label);
        let font = egui::FontId::proportional(13.0);
        let pos = egui::pos2(rect.center().x, rect.min.y + 8.0);
        ui.painter().text(
            pos + egui::vec2(1.0, 1.0),
            egui::Align2::CENTER_TOP,
            &text,
            font.clone(),
            egui::Color32::from_black_alpha(180),
        );
        ui.painter().text(
            pos,
            egui::Align2::CENTER_TOP,
            &text,
            font,
            egui::Color32::from_rgb(255, 180, 60),
        );
    }

    // --- Floating toolbar overlay ---
    let toolbar_id = ui.id().with("viewport_toolbar");
    egui::Area::new(toolbar_id)
        .order(egui::Order::Foreground)
        .fixed_pos(rect.min + egui::vec2(8.0, 28.0)) // Below FPS counter
        .show(ui.ctx(), |ui| {
            egui::Frame::none()
                .fill(egui::Color32::from_rgba_premultiplied(30, 30, 38, 200))
                .rounding(6.0)
                .inner_margin(6.0)
                .show(ui, |ui| {
                    ui.set_min_width(32.0);

                    // Tool strip
                    ui.horizontal(|ui| {
                        let sel_style = |active: bool| {
                            if active {
                                egui::RichText::new("").color(egui::Color32::from_rgb(255, 200, 60))
                            } else {
                                egui::RichText::new("").color(egui::Color32::from_gray(180))
                            }
                        };
                        let _ = sel_style; // suppress unused

                        let select_active = *active_tool == ActiveTool::Select;
                        let sculpt_tool_active = *active_tool == ActiveTool::Sculpt;

                        if ui.selectable_label(select_active, "Select").clicked() && !select_active {
                            actions.push(Action::SetTool(ActiveTool::Select));
                        }
                        if ui.selectable_label(sculpt_tool_active, "Sculpt").clicked() && !sculpt_tool_active {
                            actions.push(Action::SetTool(ActiveTool::Sculpt));
                        }
                    });

                    // Shape/Boolean/Modifier buttons (hidden when sculpt tool active)
                    if *active_tool != ActiveTool::Sculpt {
                        ui.separator();

                        // + Shape menu
                        ui.menu_button("+ Shape", |ui| {
                            for prim in SdfPrimitive::ALL {
                                if ui.button(prim.base_name()).clicked() {
                                    actions.push(Action::CreatePrimitive(prim.clone()));
                                    ui.close_menu();
                                }
                            }
                        });

                        // Boolean menu
                        ui.menu_button("Boolean", |ui| {
                            for op in CsgOp::ALL {
                                if ui.button(op.base_name()).clicked() {
                                    let tops = scene.top_level_nodes();
                                    if tops.len() >= 2 {
                                        let left = Some(tops[tops.len() - 2]);
                                        let right = Some(tops[tops.len() - 1]);
                                        actions.push(Action::CreateOperation { op: op.clone(), left, right });
                                    }
                                    ui.close_menu();
                                }
                            }
                        });

                        // + Modifier menu (needs selection)
                        let has_selection = selected.is_some();
                        ui.add_enabled_ui(has_selection, |ui| {
                            ui.menu_button("+ Modifier", |ui| {
                                if let Some(sel_id) = *selected {
                                    ui.label("Deform");
                                    for kind in &[ModifierKind::Twist, ModifierKind::Bend, ModifierKind::Taper] {
                                        if ui.button(kind.base_name()).clicked() {
                                            actions.push(Action::InsertModifierAbove { target: sel_id, kind: kind.clone() });
                                            ui.close_menu();
                                        }
                                    }
                                    ui.separator();
                                    ui.label("Shape");
                                    for kind in &[ModifierKind::Round, ModifierKind::Onion, ModifierKind::Elongate] {
                                        if ui.button(kind.base_name()).clicked() {
                                            actions.push(Action::InsertModifierAbove { target: sel_id, kind: kind.clone() });
                                            ui.close_menu();
                                        }
                                    }
                                    ui.separator();
                                    ui.label("Repeat");
                                    for kind in &[ModifierKind::Mirror, ModifierKind::Repeat, ModifierKind::FiniteRepeat] {
                                        if ui.button(kind.base_name()).clicked() {
                                            actions.push(Action::InsertModifierAbove { target: sel_id, kind: kind.clone() });
                                            ui.close_menu();
                                        }
                                    }
                                    ui.separator();
                                    ui.label("Transform");
                                    for kind in TransformKind::ALL {
                                        if ui.button(kind.base_name()).clicked() {
                                            actions.push(Action::InsertTransformAbove { target: sel_id, kind: kind.clone() });
                                            ui.close_menu();
                                        }
                                    }
                                }
                            });
                        });

                        ui.separator();

                        // Delete button (needs selection)
                        ui.add_enabled_ui(has_selection, |ui| {
                            if ui.button("Delete").clicked() {
                                if let Some(sel_id) = *selected {
                                    actions.push(Action::DeleteNode(sel_id));
                                }
                            }
                        });
                    }

                    // Shading mode buttons
                    ui.separator();
                    ui.horizontal(|ui| {
                        let mode = &render_config.shading_mode;
                        let mode_label = mode.label();
                        if ui.selectable_label(false, format!("Shading: {}", mode_label))
                            .on_hover_text("Click to cycle (Z key)")
                            .clicked()
                        {
                            actions.push(Action::CycleShadingMode);
                        }
                    });
                });
        });

    // --- Orientation Gizmo (top-right corner, interactive) ---
    {
        let gizmo_size = 60.0_f32;
        let gizmo_center = egui::pos2(
            rect.right() - gizmo_size * 0.6,
            rect.top() + gizmo_size * 0.6 + 4.0,
        );
        let arm_len = gizmo_size * 0.35;
        let hit_radius = 12.0_f32;

        let view = camera.view_matrix();
        let project_axis = |axis: glam::Vec3| -> egui::Vec2 {
            let v = view.transform_vector3(axis);
            egui::vec2(v.x, -v.y) * arm_len
        };

        // Axis definitions: positive and negative directions with their view actions
        let axes: [(glam::Vec3, egui::Color32, &str, Action, Action); 3] = [
            (glam::Vec3::X, egui::Color32::from_rgb(220, 60, 60), "X", Action::CameraRight, Action::CameraLeft),
            (glam::Vec3::Y, egui::Color32::from_rgb(60, 200, 60), "Y", Action::CameraTop, Action::CameraBottom),
            (glam::Vec3::Z, egui::Color32::from_rgb(60, 100, 220), "Z", Action::CameraFront, Action::CameraBack),
        ];

        let mut sorted_axes: Vec<_> = axes.iter().map(|(axis, color, label, pos_action, neg_action)| {
            let v = view.transform_vector3(*axis);
            (v.z, *axis, *color, *label, pos_action, neg_action)
        }).collect();
        sorted_axes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        ui.painter().circle_filled(
            gizmo_center,
            arm_len + 8.0,
            egui::Color32::from_rgba_premultiplied(20, 20, 25, 180),
        );

        // Check for clicks on axis endpoints
        let click_pos = if response.clicked() {
            ui.input(|i| i.pointer.interact_pos())
        } else {
            None
        };
        let hover_pos = ui.input(|i| i.pointer.hover_pos());

        for (depth, axis, color, label, pos_action, neg_action) in &sorted_axes {
            let pos_end = gizmo_center + project_axis(*axis);
            let neg_end = gizmo_center + project_axis(-*axis);

            // Check hover for highlighting
            let pos_hovered = hover_pos.map_or(false, |p| p.distance(pos_end) < hit_radius);
            let neg_hovered = hover_pos.map_or(false, |p| p.distance(neg_end) < hit_radius);

            let pos_alpha = if *depth > 0.0 { 255u8 } else { 80u8 };
            let neg_alpha = if *depth < 0.0 { 255u8 } else { 80u8 };

            let pos_color = if pos_hovered {
                egui::Color32::WHITE
            } else {
                egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), pos_alpha)
            };
            let neg_color = if neg_hovered {
                egui::Color32::from_rgba_unmultiplied(200, 200, 200, 180)
            } else {
                egui::Color32::from_rgba_unmultiplied(color.r(), color.g(), color.b(), neg_alpha)
            };

            let pos_thickness = if pos_hovered { 3.0 } else if *depth > 0.0 { 2.0 } else { 1.0 };

            ui.painter().line_segment(
                [gizmo_center, pos_end],
                egui::Stroke::new(pos_thickness, pos_color),
            );
            if *depth > -0.3 {
                ui.painter().text(
                    pos_end,
                    egui::Align2::CENTER_CENTER,
                    label,
                    egui::FontId::proportional(if pos_hovered { 12.0 } else { 10.0 }),
                    pos_color,
                );
            }
            // Small dot for negative axis endpoint (clickable)
            let neg_radius = if neg_hovered { 4.0 } else { 3.0 };
            ui.painter().circle_filled(neg_end, neg_radius, neg_color);

            // Handle clicks on axis endpoints
            if let Some(cp) = click_pos {
                if cp.distance(pos_end) < hit_radius {
                    actions.push(match pos_action {
                        Action::CameraFront => Action::CameraFront,
                        Action::CameraTop => Action::CameraTop,
                        Action::CameraRight => Action::CameraRight,
                        Action::CameraBack => Action::CameraBack,
                        Action::CameraLeft => Action::CameraLeft,
                        Action::CameraBottom => Action::CameraBottom,
                        _ => Action::CameraFront,
                    });
                } else if cp.distance(neg_end) < hit_radius {
                    actions.push(match neg_action {
                        Action::CameraFront => Action::CameraFront,
                        Action::CameraTop => Action::CameraTop,
                        Action::CameraRight => Action::CameraRight,
                        Action::CameraBack => Action::CameraBack,
                        Action::CameraLeft => Action::CameraLeft,
                        Action::CameraBottom => Action::CameraBottom,
                        _ => Action::CameraFront,
                    });
                }
            }
        }

        // Ortho/Persp label below the gizmo
        let label_pos = egui::pos2(gizmo_center.x, gizmo_center.y + arm_len + 14.0);
        let proj_label = if camera.orthographic { "Ortho" } else { "Persp" };
        ui.painter().text(
            label_pos,
            egui::Align2::CENTER_CENTER,
            proj_label,
            egui::FontId::proportional(9.0),
            egui::Color32::from_rgb(160, 160, 170),
        );
    }

    output
}

fn draw_node_labels(
    painter: &egui::Painter,
    camera: &Camera,
    scene: &Scene,
    selected: Option<NodeId>,
    rect: egui::Rect,
) {
    let view = camera.view_matrix();
    let aspect = rect.width() / rect.height().max(1.0);
    let proj = camera.projection_matrix(aspect);
    let vp = proj * view;
    let cam_pos = camera.eye();

    for (&id, node) in &scene.nodes {
        // Only show labels for geometry nodes with position
        let pos = match &node.data {
            NodeData::Primitive { position, .. } => *position,
            NodeData::Sculpt { position, .. } => *position,
            _ => continue,
        };

        // Skip hidden nodes
        if scene.is_hidden(id) {
            continue;
        }

        let dist = (pos - cam_pos).length();

        // Project to screen
        let Some(screen_pos) = gizmo::world_to_screen(pos, &vp, rect) else { continue };

        // Skip if outside viewport
        if !rect.contains(screen_pos) {
            continue;
        }

        // Font size scales with distance
        let font_size: f32 = (14.0_f32 / (dist * 0.3 + 1.0)).clamp(8.0, 14.0);
        // Alpha fades with distance
        let alpha: f32 = ((10.0_f32 - dist) / 8.0).clamp(0.3, 0.9);

        let is_sel = selected == Some(id);
        let base_color = if is_sel {
            egui::Color32::from_rgb(255, 200, 60)
        } else {
            egui::Color32::from_rgb(200, 200, 210)
        };
        let color = base_color.gamma_multiply(alpha);
        let shadow_color = egui::Color32::from_rgba_premultiplied(0, 0, 0, (alpha * 180.0) as u8);

        let label_pos = egui::pos2(screen_pos.x, screen_pos.y - font_size - 4.0);
        let font = egui::FontId::proportional(font_size);

        // Text shadow
        painter.text(
            label_pos + egui::vec2(1.0, 1.0),
            egui::Align2::CENTER_CENTER,
            &node.name,
            font.clone(),
            shadow_color,
        );
        // Text
        painter.text(
            label_pos,
            egui::Align2::CENTER_CENTER,
            &node.name,
            font,
            color,
        );
    }
}
