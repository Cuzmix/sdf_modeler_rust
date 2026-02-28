use eframe::egui;
use eframe::egui_wgpu;
use eframe::wgpu;

use crate::gpu::camera::{Camera, CameraUniform};
use crate::gpu::picking::PendingPick;
use crate::graph::scene::{CsgOp, ModifierKind, NodeId, Scene, SdfPrimitive, TransformKind};
use crate::sculpt::{ActiveTool, SculptState};
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

        // Write blit params (display viewport for the blit shader)
        queue.write_buffer(
            &resources.blit_params_buffer,
            0,
            bytemuck::cast_slice(&self.display_viewport),
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

const BRUSH_CURSOR_COLOR: egui::Color32 = egui::Color32::from_rgba_premultiplied(200, 200, 200, 128);

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
    pub created_node: Option<NodeId>,
    pub tool_switch: Option<ActiveTool>,
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
) -> ViewportOutput {
    let mut output = ViewportOutput {
        pending_pick: None,
        created_node: None,
        tool_switch: None,
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
    let render_uniform = camera.to_uniform(render_viewport, time, quality_mode, render_config.show_grid, scene_bounds, selected_idx);

    ui.painter().add(egui_wgpu::Callback::new_paint_callback(
        rect,
        ViewportCallback {
            render_uniform,
            display_viewport: viewport,
            render_scale,
        },
    ));

    // --- Symmetry plane overlay ---
    if let Some(axis) = sculpt_state.symmetry_axis() {
        draw_symmetry_plane(ui.painter(), camera, rect, axis);
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
        )
    };

    // --- Interaction priority: sculpt > gizmo > pick > orbit ---

    if sculpt_active {
        // Sculpt mode: drag applies brush continuously via pick
        if response.dragged_by(egui::PointerButton::Primary) {
            if let Some(pos) = ui.input(|i| i.pointer.hover_pos()) {
                if rect.contains(pos) {
                    let mouse_px = [
                        (pos.x - rect.min.x) * pixels_per_point,
                        (pos.y - rect.min.y) * pixels_per_point,
                    ];
                    output.pending_pick = Some(PendingPick {
                        mouse_pos: mouse_px,
                        camera_uniform: camera.to_uniform(viewport, time, 0.0, false, scene_bounds, -1.0),
                    });
                }
            }
        }

        // Enhanced brush cursor preview
        if let SculptState::Active {
            brush_radius,
            brush_strength,
            symmetry_axis,
            ..
        } = sculpt_state
        {
            if let Some(hover_pos) = response.hover_pos() {
                let screen_radius = brush_radius / camera.distance * rect.height() * 0.5;

                // Outer ring: brush extent
                ui.painter().circle_stroke(
                    hover_pos,
                    screen_radius,
                    egui::Stroke::new(1.5, BRUSH_CURSOR_COLOR),
                );

                // Inner fill: strength indicator (opacity proportional to strength)
                let strength_alpha = (brush_strength / 0.5 * 60.0).clamp(8.0, 60.0) as u8;
                ui.painter().circle_filled(
                    hover_pos,
                    screen_radius * 0.6,
                    egui::Color32::from_rgba_premultiplied(200, 200, 200, strength_alpha),
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
            }
        }

        // Right-click still orbits in sculpt mode, secondary drag pans
        if response.dragged_by(egui::PointerButton::Secondary) {
            let delta = response.drag_delta();
            camera.pan(delta.x, delta.y);
        }
        if response.dragged_by(egui::PointerButton::Middle) {
            let delta = response.drag_delta();
            camera.orbit(delta.x, delta.y);
        }
    } else if !gizmo_consumed {
        // Normal mode: click to pick
        if response.clicked() {
            if let Some(pos) = response.interact_pointer_pos() {
                let mouse_px = [
                    (pos.x - rect.min.x) * pixels_per_point,
                    (pos.y - rect.min.y) * pixels_per_point,
                ];
                let pick_uniform = camera.to_uniform(viewport, time, 0.0, false, scene_bounds, -1.0);
                output.pending_pick = Some(PendingPick {
                    mouse_pos: mouse_px,
                    camera_uniform: pick_uniform,
                });
            }
        }

        if response.dragged_by(egui::PointerButton::Primary) {
            let delta = response.drag_delta();
            camera.orbit(delta.x, delta.y);
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
                            output.tool_switch = Some(ActiveTool::Select);
                        }
                        if ui.selectable_label(sculpt_tool_active, "Sculpt").clicked() && !sculpt_tool_active {
                            output.tool_switch = Some(ActiveTool::Sculpt);
                        }
                    });

                    // Shape/Boolean/Modifier buttons (hidden when sculpt tool active)
                    if *active_tool != ActiveTool::Sculpt {
                        ui.separator();

                        // + Shape menu
                        ui.menu_button("+ Shape", |ui| {
                            for prim in SdfPrimitive::ALL {
                                if ui.button(prim.base_name()).clicked() {
                                    let id = scene.create_primitive(prim.clone());
                                    output.created_node = Some(id);
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
                                        let id = scene.create_operation(op.clone(), left, right);
                                        output.created_node = Some(id);
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
                                            scene.insert_modifier_above(sel_id, kind.clone());
                                            ui.close_menu();
                                        }
                                    }
                                    ui.separator();
                                    ui.label("Shape");
                                    for kind in &[ModifierKind::Round, ModifierKind::Onion, ModifierKind::Elongate] {
                                        if ui.button(kind.base_name()).clicked() {
                                            scene.insert_modifier_above(sel_id, kind.clone());
                                            ui.close_menu();
                                        }
                                    }
                                    ui.separator();
                                    ui.label("Repeat");
                                    for kind in &[ModifierKind::Mirror, ModifierKind::Repeat, ModifierKind::FiniteRepeat] {
                                        if ui.button(kind.base_name()).clicked() {
                                            scene.insert_modifier_above(sel_id, kind.clone());
                                            ui.close_menu();
                                        }
                                    }
                                    ui.separator();
                                    ui.label("Transform");
                                    for kind in TransformKind::ALL {
                                        if ui.button(kind.base_name()).clicked() {
                                            scene.insert_transform_above(sel_id, kind.clone());
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
                                    scene.remove_node(sel_id);
                                    *selected = None;
                                }
                            }
                        });
                    }
                });
        });

    output
}
