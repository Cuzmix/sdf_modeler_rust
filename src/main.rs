use std::cell::RefCell;
use std::rc::Rc;

mod gpu;

use gpu::scene::{
    GizmoAxis, GraphLayoutData, NodeData, NodeId, PickResult, Scene, SdfPrimitive, SdfTransform,
};

slint::include_modules!();

/// Clone-based undo/redo stack. Snapshots are full Scene clones.
struct UndoStack {
    history: Vec<Scene>,
    redo: Vec<Scene>,
    max_size: usize,
}

impl UndoStack {
    fn new() -> Self {
        Self {
            history: Vec::new(),
            redo: Vec::new(),
            max_size: 50,
        }
    }

    /// Push a snapshot before a mutation. Clears redo stack.
    fn push(&mut self, scene: Scene) {
        if self.history.len() >= self.max_size {
            self.history.remove(0);
        }
        self.history.push(scene);
        self.redo.clear();
    }

    /// Undo: save current state to redo, pop from history.
    fn undo(&mut self, current: &Scene) -> Option<Scene> {
        let prev = self.history.pop()?;
        self.redo.push(current.clone());
        Some(prev)
    }

    /// Redo: save current state to history, pop from redo.
    fn redo(&mut self, current: &Scene) -> Option<Scene> {
        let next = self.redo.pop()?;
        self.history.push(current.clone());
        Some(next)
    }
}

/// Tracks mouse drag state, dirty flag, click detection, and gizmo interaction.
struct InputState {
    dragging: bool,
    drag_button: i32,
    last_x: f32,
    last_y: f32,
    down_x: f32,
    down_y: f32,
    needs_redraw: bool,
    scene_dirty: bool,
    last_vp_w: u32,
    last_vp_h: u32,
    // FPS tracking
    frame_count: u32,
    fps_accum_start: std::time::Instant,
    // Gizmo drag state
    gizmo_axis: Option<GizmoAxis>,
    gizmo_node_start: glam::Vec3,
    gizmo_drag_origin: glam::Vec3,
    // Graph panel interaction state
    graph_panning: bool,
    graph_pan_x: f32,
    graph_pan_y: f32,
    graph_down_x: f32,
    graph_down_y: f32,
    graph_last_x: f32,
    graph_last_y: f32,
    graph_dragging: bool,
    // Wire drag state
    graph_wire_dragging: bool,
    graph_wire_source: (f32, f32),
    graph_wire_source_node_idx: Option<usize>,
    graph_wire_from_output: bool,
    // Codegen fast/slow path
    last_structure_key: u64,
}

impl InputState {
    fn new() -> Self {
        Self {
            dragging: false,
            drag_button: -1,
            last_x: 0.0,
            last_y: 0.0,
            down_x: 0.0,
            down_y: 0.0,
            needs_redraw: true,
            scene_dirty: true,
            last_vp_w: 0,
            last_vp_h: 0,
            frame_count: 0,
            fps_accum_start: std::time::Instant::now(),
            gizmo_axis: None,
            gizmo_node_start: glam::Vec3::ZERO,
            gizmo_drag_origin: glam::Vec3::ZERO,
            graph_panning: false,
            graph_pan_x: 0.0,
            graph_pan_y: 0.0,
            graph_down_x: 0.0,
            graph_down_y: 0.0,
            graph_last_x: 0.0,
            graph_last_y: 0.0,
            graph_dragging: false,
            graph_wire_dragging: false,
            graph_wire_source: (0.0, 0.0),
            graph_wire_source_node_idx: None,
            graph_wire_from_output: true,
            last_structure_key: 0,
        }
    }
}

/// Generate a world-space ray from a pixel position.
fn screen_to_ray(
    x: f32,
    y: f32,
    vp_w: u32,
    vp_h: u32,
    camera: &gpu::camera::OrbitCamera,
) -> (glam::Vec3, glam::Vec3) {
    let aspect = vp_w as f32 / vp_h.max(1) as f32;
    let view = camera.view_matrix();
    let proj = camera.projection_matrix(aspect);
    let inv_vp = (proj * view).inverse();

    let ndc_x = (x / vp_w as f32) * 2.0 - 1.0;
    let ndc_y = 1.0 - (y / vp_h as f32) * 2.0;

    let near = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 0.0, 1.0);
    let far = inv_vp * glam::Vec4::new(ndc_x, ndc_y, 1.0, 1.0);

    let origin = near.truncate() / near.w;
    let target = far.truncate() / far.w;
    let dir = (target - origin).normalize();

    (origin, dir)
}

/// Find closest point on an axis line to a camera ray (line-line closest point).
fn closest_point_on_axis(
    axis_origin: glam::Vec3,
    axis_dir: glam::Vec3,
    ray_origin: glam::Vec3,
    ray_dir: glam::Vec3,
) -> glam::Vec3 {
    let w = axis_origin - ray_origin;
    let a = axis_dir.dot(axis_dir);
    let b = axis_dir.dot(ray_dir);
    let c = ray_dir.dot(ray_dir);
    let d = axis_dir.dot(w);
    let e = ray_dir.dot(w);
    let denom = a * c - b * b;
    let t = if denom.abs() < 1e-6 {
        0.0
    } else {
        (b * e - c * d) / denom
    };
    axis_origin + axis_dir * t
}

// ── Graph hit-testing and helpers ──────────────────────────────

use gpu::scene::GraphNodeData;
use gpu::scene::GraphWireData;

enum PortHit {
    Output(usize),
    InputTop(usize),
    InputBottom(usize),
    InputSingle(usize),
}

fn hit_test_graph_node(canvas_x: f32, canvas_y: f32, nodes: &[GraphNodeData]) -> Option<usize> {
    for (i, gn) in nodes.iter().enumerate().rev() {
        if canvas_x >= gn.x
            && canvas_x <= gn.x + gn.width
            && canvas_y >= gn.y
            && canvas_y <= gn.y + gn.height
        {
            return Some(i);
        }
    }
    None
}

fn hit_test_port(canvas_x: f32, canvas_y: f32, nodes: &[GraphNodeData]) -> Option<PortHit> {
    let radius = 10.0; // slightly larger than visual port for easy clicking
    for (i, gn) in nodes.iter().enumerate().rev() {
        let dx = canvas_x - gn.out_port_x;
        let dy = canvas_y - gn.out_port_y;
        if dx * dx + dy * dy < radius * radius {
            return Some(PortHit::Output(i));
        }
        if gn.has_input_ports {
            let dx = canvas_x - gn.in_port_top_x;
            let dy = canvas_y - gn.in_port_top_y;
            if dx * dx + dy * dy < radius * radius {
                return Some(PortHit::InputTop(i));
            }
            let dx = canvas_x - gn.in_port_bot_x;
            let dy = canvas_y - gn.in_port_bot_y;
            if dx * dx + dy * dy < radius * radius {
                return Some(PortHit::InputBottom(i));
            }
        }
        if gn.has_single_input {
            let dx = canvas_x - gn.in_port_single_x;
            let dy = canvas_y - gn.in_port_single_y;
            if dx * dx + dy * dy < radius * radius {
                return Some(PortHit::InputSingle(i));
            }
        }
    }
    None
}

fn hit_test_wire(canvas_x: f32, canvas_y: f32, wires: &[GraphWireData]) -> Option<usize> {
    let threshold = 6.0;
    for (i, w) in wires.iter().enumerate() {
        if point_near_bezier(canvas_x, canvas_y, w.start, w.end, threshold) {
            return Some(i);
        }
    }
    None
}

fn point_near_bezier(px: f32, py: f32, start: (f32, f32), end: (f32, f32), threshold: f32) -> bool {
    let (x1, y1) = start;
    let (x2, y2) = end;
    let dx = (x2 - x1).abs();
    let cx1 = x1 + dx / 3.0;
    let cx2 = x2 - dx / 3.0;

    for i in 0..=20 {
        let t = i as f32 / 20.0;
        let it = 1.0 - t;
        let bx =
            it * it * it * x1 + 3.0 * it * it * t * cx1 + 3.0 * it * t * t * cx2 + t * t * t * x2;
        let by =
            it * it * it * y1 + 3.0 * it * it * t * y1 + 3.0 * it * t * t * y2 + t * t * t * y2;
        let dist = ((px - bx) * (px - bx) + (py - by) * (py - by)).sqrt();
        if dist < threshold {
            return true;
        }
    }
    false
}

fn build_preview_wire_svg(x1: f32, y1: f32, x2: f32, y2: f32) -> String {
    let dx = (x2 - x1).abs();
    let cx1 = x1 + dx / 3.0;
    let cx2 = x2 - dx / 3.0;
    format!(
        "M {:.1} {:.1} C {:.1} {:.1} {:.1} {:.1} {:.1} {:.1}",
        x1, y1, cx1, y1, cx2, y2, x2, y2
    )
}

fn build_wire_svg(start: (f32, f32), end: (f32, f32)) -> String {
    build_preview_wire_svg(start.0, start.1, end.0, end.1)
}

/// Push the selected node's properties to the Slint UI.
fn push_selection_to_ui(app: &MainWindow, scene: &Scene) {
    if let Some(node_id) = scene.selected {
        if let Some(node) = scene.get_node(node_id) {
            match &node.data {
                NodeData::Primitive(prim) => {
                    app.set_has_selection(true);
                    app.set_selected_name(slint::format!("{}", node.name));
                    app.set_selected_node_type(prim.primitive as i32);
                    app.set_prop_pos_x(prim.position.x);
                    app.set_prop_pos_y(prim.position.y);
                    app.set_prop_pos_z(prim.position.z);
                    app.set_prop_scale_x(prim.scale.x);
                    app.set_prop_scale_y(prim.scale.y);
                    app.set_prop_scale_z(prim.scale.z);
                    return;
                }
                NodeData::Transform(tr) => {
                    app.set_has_selection(true);
                    app.set_selected_name(slint::format!("{}", node.name));
                    app.set_selected_node_type(20 + tr.transform as i32);
                    app.set_prop_pos_x(tr.offset.x);
                    app.set_prop_pos_y(tr.offset.y);
                    app.set_prop_pos_z(tr.offset.z);
                    return;
                }
                NodeData::Operation(op) => {
                    app.set_has_selection(true);
                    app.set_selected_name(slint::format!("{}", node.name));
                    app.set_selected_node_type(10 + op.operation as i32);
                    return;
                }
            }
        }
    }
    app.set_has_selection(false);
}

/// Push the scene tree structure to the Slint UI.
fn push_tree_to_ui(app: &MainWindow, scene: &Scene, tree_node_ids: &mut Vec<NodeId>) {
    let items = scene.build_tree_items();
    tree_node_ids.clear();
    let slint_items: Vec<TreeItem> = items
        .iter()
        .enumerate()
        .map(|(idx, item)| {
            tree_node_ids.push(item.node_id);
            TreeItem {
                depth: item.depth,
                name: slint::format!("{}", item.name),
                node_type: item.node_type,
                is_selected: item.is_selected,
                item_index: idx as i32,
            }
        })
        .collect();
    let model = Rc::new(slint::VecModel::from(slint_items));
    app.set_tree_items(slint::ModelRc::from(model));
}

/// Push the node graph layout to the Slint UI, storing layout data for hit-testing.
fn push_graph_to_ui(
    app: &MainWindow,
    scene: &Scene,
    graph_node_ids: &mut Vec<NodeId>,
    graph_layout: &mut Option<GraphLayoutData>,
) {
    let layout = scene.build_graph_layout();
    graph_node_ids.clear();

    let slint_nodes: Vec<GraphNode> = layout
        .nodes
        .iter()
        .enumerate()
        .map(|(idx, gn)| {
            graph_node_ids.push(gn.node_id);
            GraphNode {
                x: gn.x,
                y: gn.y,
                width: gn.width,
                height: gn.height,
                name: slint::format!("{}", gn.name),
                node_type: gn.node_type,
                is_selected: gn.is_selected,
                item_index: idx as i32,
                has_input_ports: gn.has_input_ports,
                has_single_input: gn.has_single_input,
            }
        })
        .collect();

    let model = Rc::new(slint::VecModel::from(slint_nodes));
    app.set_graph_nodes(slint::ModelRc::from(model));
    app.set_graph_connections_svg(slint::format!("{}", layout.connections_svg));
    app.set_graph_canvas_width(layout.canvas_width);
    app.set_graph_canvas_height(layout.canvas_height);
    // Clear any highlight/preview wire when layout changes
    app.set_graph_highlight_wire_svg(slint::SharedString::default());
    app.set_graph_preview_wire_svg(slint::SharedString::default());

    *graph_layout = Some(layout);
}

fn main() {
    env_logger::init();

    // Request storage buffer support from wgpu
    let mut wgpu_settings = slint::wgpu_28::WGPUSettings::default();
    wgpu_settings
        .device_required_limits
        .max_storage_buffers_per_shader_stage = 4;
    wgpu_settings
        .device_required_limits
        .max_storage_buffer_binding_size = 1 << 20; // 1 MB
    slint::BackendSelector::new()
        .require_wgpu_28(slint::wgpu_28::WGPUConfiguration::Automatic(wgpu_settings))
        .select()
        .expect("Failed to select wgpu 28 backend");

    let app = MainWindow::new().expect("Failed to create MainWindow");
    let app_weak = app.as_weak();

    // Shared state
    let gpu_state: Rc<RefCell<Option<gpu::state::GpuState>>> = Rc::new(RefCell::new(None));
    let camera: Rc<RefCell<gpu::camera::OrbitCamera>> =
        Rc::new(RefCell::new(gpu::camera::OrbitCamera::new()));
    let input: Rc<RefCell<InputState>> = Rc::new(RefCell::new(InputState::new()));
    let scene: Rc<RefCell<Scene>> = Rc::new(RefCell::new(Scene::default_scene()));
    let tree_node_ids: Rc<RefCell<Vec<NodeId>>> = Rc::new(RefCell::new(Vec::new()));
    let graph_node_ids: Rc<RefCell<Vec<NodeId>>> = Rc::new(RefCell::new(Vec::new()));
    let graph_layout: Rc<RefCell<Option<GraphLayoutData>>> = Rc::new(RefCell::new(None));
    let undo_stack: Rc<RefCell<UndoStack>> = Rc::new(RefCell::new(UndoStack::new()));
    let start_time = std::time::Instant::now();

    // Push initial selection, tree, and graph
    push_selection_to_ui(&app, &scene.borrow());
    push_tree_to_ui(&app, &scene.borrow(), &mut tree_node_ids.borrow_mut());
    push_graph_to_ui(
        &app,
        &scene.borrow(),
        &mut graph_node_ids.borrow_mut(),
        &mut graph_layout.borrow_mut(),
    );

    // Grab device+queue from Slint's wgpu backend
    let gpu_for_notifier = gpu_state.clone();
    let scene_for_notifier = scene.clone();
    app.window()
        .set_rendering_notifier(move |state, graphics_api| match (state, graphics_api) {
            (
                slint::RenderingState::RenderingSetup,
                slint::GraphicsAPI::WGPU28 { device, queue, .. },
            ) => {
                log::info!("wgpu RenderingSetup — creating GpuState");
                let sc = scene_for_notifier.borrow();
                let gs = gpu::state::GpuState::new(device.clone(), queue.clone(), &sc);
                *gpu_for_notifier.borrow_mut() = Some(gs);
            }
            (slint::RenderingState::RenderingTeardown, _) => {
                log::info!("wgpu RenderingTeardown — dropping GpuState");
                *gpu_for_notifier.borrow_mut() = None;
            }
            _ => {}
        })
        .expect("Failed to set rendering notifier");

    // ── Add Primitive ───────────────────────────────────────────
    {
        let scene_add = scene.clone();
        let input_add = input.clone();
        let app_add = app.as_weak();
        let tree_ids_add = tree_node_ids.clone();
        let graph_ids_add = graph_node_ids.clone();
        let graph_layout_add = graph_layout.clone();
        let undo_add = undo_stack.clone();
        app.on_on_add_primitive(move |prim_type| {
            let prim = match prim_type {
                0 => SdfPrimitive::Sphere,
                1 => SdfPrimitive::Box,
                2 => SdfPrimitive::Cylinder,
                3 => SdfPrimitive::Torus,
                _ => SdfPrimitive::Sphere,
            };
            let mut s = scene_add.borrow_mut();
            undo_add.borrow_mut().push(s.clone());
            s.add_primitive(prim);
            let mut inp = input_add.borrow_mut();
            inp.scene_dirty = true;
            inp.needs_redraw = true;
            if let Some(app) = app_add.upgrade() {
                push_selection_to_ui(&app, &s);
                push_tree_to_ui(&app, &s, &mut tree_ids_add.borrow_mut());
                push_graph_to_ui(
                    &app,
                    &s,
                    &mut graph_ids_add.borrow_mut(),
                    &mut graph_layout_add.borrow_mut(),
                );
            }
        });
    }

    // ── Add Transform ────────────────────────────────────────────
    {
        let scene_tr = scene.clone();
        let input_tr = input.clone();
        let app_tr = app.as_weak();
        let tree_ids_tr = tree_node_ids.clone();
        let graph_ids_tr = graph_node_ids.clone();
        let graph_layout_tr = graph_layout.clone();
        let undo_tr = undo_stack.clone();
        app.on_on_add_transform(move |transform_type| {
            let transform = match transform_type {
                0 => SdfTransform::Translate,
                1 => SdfTransform::Rotate,
                2 => SdfTransform::Scale,
                _ => SdfTransform::Translate,
            };
            let mut s = scene_tr.borrow_mut();
            let Some(sel_id) = s.selected else { return };
            undo_tr.borrow_mut().push(s.clone());
            s.wrap_in_transform(sel_id, transform);
            let mut inp = input_tr.borrow_mut();
            inp.scene_dirty = true;
            inp.needs_redraw = true;
            if let Some(app) = app_tr.upgrade() {
                push_selection_to_ui(&app, &s);
                push_tree_to_ui(&app, &s, &mut tree_ids_tr.borrow_mut());
                push_graph_to_ui(
                    &app,
                    &s,
                    &mut graph_ids_tr.borrow_mut(),
                    &mut graph_layout_tr.borrow_mut(),
                );
            }
        });
    }

    // ── Delete Selected ─────────────────────────────────────────
    {
        let scene_del = scene.clone();
        let input_del = input.clone();
        let app_del = app.as_weak();
        let tree_ids_del = tree_node_ids.clone();
        let graph_ids_del = graph_node_ids.clone();
        let graph_layout_del = graph_layout.clone();
        let undo_del = undo_stack.clone();
        app.on_on_delete_selected(move || {
            let mut s = scene_del.borrow_mut();
            undo_del.borrow_mut().push(s.clone());
            s.remove_selected();
            let mut inp = input_del.borrow_mut();
            inp.scene_dirty = true;
            inp.needs_redraw = true;
            if let Some(app) = app_del.upgrade() {
                push_selection_to_ui(&app, &s);
                push_tree_to_ui(&app, &s, &mut tree_ids_del.borrow_mut());
                push_graph_to_ui(
                    &app,
                    &s,
                    &mut graph_ids_del.borrow_mut(),
                    &mut graph_layout_del.borrow_mut(),
                );
            }
        });
    }

    // ── Property Changed (from sliders) ─────────────────────────
    {
        let scene_prop = scene.clone();
        let input_prop = input.clone();
        let app_prop = app.as_weak();
        app.on_on_property_changed(move || {
            let Some(app) = app_prop.upgrade() else {
                return;
            };
            let mut s = scene_prop.borrow_mut();
            if let Some(node_id) = s.selected {
                if let Some(node) = s.get_node_mut(node_id) {
                    match &mut node.data {
                        NodeData::Primitive(ref mut prim) => {
                            prim.position.x = app.get_prop_pos_x();
                            prim.position.y = app.get_prop_pos_y();
                            prim.position.z = app.get_prop_pos_z();
                            prim.scale.x = app.get_prop_scale_x();
                            prim.scale.y = app.get_prop_scale_y();
                            prim.scale.z = app.get_prop_scale_z();
                        }
                        NodeData::Transform(ref mut tr) => {
                            tr.offset.x = app.get_prop_pos_x();
                            tr.offset.y = app.get_prop_pos_y();
                            tr.offset.z = app.get_prop_pos_z();
                        }
                        _ => {}
                    }
                }
            }
            let mut inp = input_prop.borrow_mut();
            inp.scene_dirty = true;
            inp.needs_redraw = true;
        });
    }

    // ── Pointer Events ──────────────────────────────────────────
    {
        let input_down = input.clone();
        let gpu_down = gpu_state.clone();
        let scene_down = scene.clone();
        let cam_down = camera.clone();
        let undo_down = undo_stack.clone();
        app.on_on_pointer_down(move |x, y, button| {
            let mut s = input_down.borrow_mut();
            s.dragging = false;
            s.drag_button = button;
            s.last_x = x;
            s.last_y = y;
            s.down_x = x;
            s.down_y = y;
            s.gizmo_axis = None;

            // On left-click, check if a gizmo axis was hit
            if button == 0 && s.last_vp_w > 0 && s.last_vp_h > 0 {
                let ndc_x = (x / s.last_vp_w as f32) * 2.0 - 1.0;
                let ndc_y = 1.0 - (y / s.last_vp_h as f32) * 2.0;
                let vp_w = s.last_vp_w;
                let vp_h = s.last_vp_h;
                drop(s);

                let gpu_ref = gpu_down.borrow();
                let pick = if let Some(gs) = gpu_ref.as_ref() {
                    gs.pick_at(ndc_x, ndc_y)
                } else {
                    PickResult::Background
                };
                drop(gpu_ref);

                if let PickResult::GizmoAxis(axis) = pick {
                    let sc = scene_down.borrow();
                    if let Some(pos) = sc.selected_primitive_position() {
                        // Snapshot before gizmo drag modifies the scene
                        undo_down.borrow_mut().push(sc.clone());
                        let cam = cam_down.borrow();
                        let (ro, rd) = screen_to_ray(x, y, vp_w, vp_h, &cam);
                        drop(cam);

                        let axis_dir = match axis {
                            GizmoAxis::X => glam::Vec3::X,
                            GizmoAxis::Y => glam::Vec3::Y,
                            GizmoAxis::Z => glam::Vec3::Z,
                        };

                        let drag_origin = closest_point_on_axis(pos, axis_dir, ro, rd);

                        let mut s = input_down.borrow_mut();
                        s.gizmo_axis = Some(axis);
                        s.gizmo_node_start = pos;
                        s.gizmo_drag_origin = drag_origin;
                        s.dragging = true; // immediately start drag
                    }
                }
            }
        });
    }
    {
        let input_move = input.clone();
        let cam_move = camera.clone();
        let scene_move = scene.clone();
        let app_move = app.as_weak();
        app.on_on_pointer_move(move |x, y| {
            let mut s = input_move.borrow_mut();
            let dx = x - s.last_x;
            let dy = y - s.last_y;
            s.last_x = x;
            s.last_y = y;

            // Detect drag (> 3px from down position)
            let total_dx = x - s.down_x;
            let total_dy = y - s.down_y;
            if total_dx * total_dx + total_dy * total_dy > 9.0 {
                s.dragging = true;
            }

            if !s.dragging {
                return;
            }

            // Gizmo drag mode
            if let Some(axis) = s.gizmo_axis {
                let vp_w = s.last_vp_w;
                let vp_h = s.last_vp_h;
                let node_start = s.gizmo_node_start;
                let drag_origin = s.gizmo_drag_origin;
                s.needs_redraw = true;
                s.scene_dirty = true;
                drop(s);

                let axis_dir = match axis {
                    GizmoAxis::X => glam::Vec3::X,
                    GizmoAxis::Y => glam::Vec3::Y,
                    GizmoAxis::Z => glam::Vec3::Z,
                };

                let cam = cam_move.borrow();
                let (ro, rd) = screen_to_ray(x, y, vp_w, vp_h, &cam);
                drop(cam);

                let current = closest_point_on_axis(node_start, axis_dir, ro, rd);
                let displacement = current - drag_origin;
                // Only take the component along the axis
                let delta = axis_dir * displacement.dot(axis_dir);
                let new_pos = node_start + delta;

                let mut sc = scene_move.borrow_mut();
                sc.set_selected_position(new_pos);
                drop(sc);

                if let Some(app) = app_move.upgrade() {
                    let sc = scene_move.borrow();
                    push_selection_to_ui(&app, &sc);
                }

                return;
            }

            // Camera control mode
            let mut cam = cam_move.borrow_mut();
            match s.drag_button {
                0 => cam.orbit(dx, dy),
                1 | 2 => cam.pan(dx, dy),
                _ => {}
            }
            s.needs_redraw = true;
        });
    }
    {
        let input_up = input.clone();
        let scene_up = scene.clone();
        let gpu_up = gpu_state.clone();
        let app_up = app.as_weak();
        let tree_ids_up = tree_node_ids.clone();
        let graph_ids_up = graph_node_ids.clone();
        let graph_layout_up = graph_layout.clone();
        app.on_on_pointer_up(move || {
            let s = input_up.borrow();
            let was_click = !s.dragging && s.drag_button == 0;
            let click_x = s.down_x;
            let click_y = s.down_y;
            let vp_w = s.last_vp_w;
            let vp_h = s.last_vp_h;
            drop(s);

            if was_click && vp_w > 0 && vp_h > 0 {
                // Pixel → NDC
                let ndc_x = (click_x / vp_w as f32) * 2.0 - 1.0;
                let ndc_y = 1.0 - (click_y / vp_h as f32) * 2.0;

                // GPU pick
                let gpu_ref = gpu_up.borrow();
                let pick_result = if let Some(gs) = gpu_ref.as_ref() {
                    gs.pick_at(ndc_x, ndc_y)
                } else {
                    PickResult::Background
                };
                drop(gpu_ref);

                let mut sc = scene_up.borrow_mut();
                match pick_result {
                    PickResult::Node(gpu_idx) => {
                        sc.selected = sc.node_id_from_gpu_index(gpu_idx);
                    }
                    _ => {
                        sc.selected = None;
                    }
                }

                let mut inp = input_up.borrow_mut();
                inp.scene_dirty = true;
                inp.needs_redraw = true;
                drop(inp);

                if let Some(app) = app_up.upgrade() {
                    push_selection_to_ui(&app, &sc);
                    push_tree_to_ui(&app, &sc, &mut tree_ids_up.borrow_mut());
                    push_graph_to_ui(
                        &app,
                        &sc,
                        &mut graph_ids_up.borrow_mut(),
                        &mut graph_layout_up.borrow_mut(),
                    );
                }
            }

            let mut s = input_up.borrow_mut();
            s.dragging = false;
            s.drag_button = -1;
            s.gizmo_axis = None;
        });
    }
    {
        let input_scroll = input.clone();
        let cam_scroll = camera.clone();
        app.on_on_scroll(move |delta| {
            cam_scroll.borrow_mut().zoom(delta);
            input_scroll.borrow_mut().needs_redraw = true;
        });
    }

    // ── Tree Item Clicked ──────────────────────────────────────
    {
        let scene_tree = scene.clone();
        let input_tree = input.clone();
        let tree_ids_tree = tree_node_ids.clone();
        let graph_ids_tree = graph_node_ids.clone();
        let graph_layout_tree = graph_layout.clone();
        let app_tree = app.as_weak();
        app.on_on_tree_item_clicked(move |item_index| {
            let node_id = {
                let ids = tree_ids_tree.borrow();
                ids.get(item_index as usize).copied()
            };
            if let Some(node_id) = node_id {
                let mut s = scene_tree.borrow_mut();
                s.selected = Some(node_id);
                let mut inp = input_tree.borrow_mut();
                inp.scene_dirty = true;
                inp.needs_redraw = true;
                drop(inp);
                if let Some(app) = app_tree.upgrade() {
                    push_selection_to_ui(&app, &s);
                    push_tree_to_ui(&app, &s, &mut tree_ids_tree.borrow_mut());
                    push_graph_to_ui(
                        &app,
                        &s,
                        &mut graph_ids_tree.borrow_mut(),
                        &mut graph_layout_tree.borrow_mut(),
                    );
                }
            }
        });
    }

    // ── Graph Pointer Events ─────────────────────────────────
    {
        let scene_gp = scene.clone();
        let input_gp = input.clone();
        let graph_ids_gp = graph_node_ids.clone();
        let graph_layout_gp = graph_layout.clone();
        let tree_ids_gp = tree_node_ids.clone();
        let app_gp = app.as_weak();
        let undo_gp = undo_stack.clone();
        app.on_on_graph_pointer_down(move |button, x, y| {
            let mut inp = input_gp.borrow_mut();
            inp.graph_down_x = x;
            inp.graph_down_y = y;
            inp.graph_last_x = x;
            inp.graph_last_y = y;
            inp.graph_dragging = false;
            inp.graph_panning = false;
            inp.graph_wire_dragging = false;

            let pan_x = inp.graph_pan_x;
            let pan_y = inp.graph_pan_y;
            let canvas_x = x + pan_x;
            let canvas_y = y + pan_y;

            // Hit-test ports first (for wire dragging — Step 4)
            let layout_ref = graph_layout_gp.borrow();
            if let Some(layout) = layout_ref.as_ref() {
                if button == 0 {
                    // Check port hits
                    if let Some(port_hit) = hit_test_port(canvas_x, canvas_y, &layout.nodes) {
                        match port_hit {
                            PortHit::Output(idx) => {
                                let gn = &layout.nodes[idx];
                                inp.graph_wire_dragging = true;
                                inp.graph_wire_source = (gn.out_port_x, gn.out_port_y);
                                inp.graph_wire_source_node_idx = Some(idx);
                                inp.graph_wire_from_output = true;
                                return;
                            }
                            PortHit::InputTop(idx) | PortHit::InputBottom(idx) => {
                                let gn = &layout.nodes[idx];
                                let is_top = matches!(port_hit, PortHit::InputTop(_));
                                let port_y = if is_top {
                                    gn.in_port_top_y
                                } else {
                                    gn.in_port_bot_y
                                };
                                inp.graph_wire_dragging = true;
                                inp.graph_wire_source = (gn.in_port_top_x, port_y);
                                inp.graph_wire_source_node_idx = Some(idx);
                                inp.graph_wire_from_output = false;
                                return;
                            }
                            PortHit::InputSingle(idx) => {
                                let gn = &layout.nodes[idx];
                                inp.graph_wire_dragging = true;
                                inp.graph_wire_source =
                                    (gn.in_port_single_x, gn.in_port_single_y);
                                inp.graph_wire_source_node_idx = Some(idx);
                                inp.graph_wire_from_output = false;
                                return;
                            }
                        }
                    }

                    // Hit-test nodes (reverse order for z-order)
                    if let Some(node_idx) = hit_test_graph_node(canvas_x, canvas_y, &layout.nodes) {
                        drop(layout_ref);
                        drop(inp);
                        let node_id = {
                            let ids = graph_ids_gp.borrow();
                            ids.get(node_idx).copied()
                        };
                        if let Some(node_id) = node_id {
                            let mut s = scene_gp.borrow_mut();
                            s.selected = Some(node_id);
                            let mut inp2 = input_gp.borrow_mut();
                            inp2.scene_dirty = true;
                            inp2.needs_redraw = true;
                            drop(inp2);
                            if let Some(app) = app_gp.upgrade() {
                                push_selection_to_ui(&app, &s);
                                push_tree_to_ui(&app, &s, &mut tree_ids_gp.borrow_mut());
                                push_graph_to_ui(
                                    &app,
                                    &s,
                                    &mut graph_ids_gp.borrow_mut(),
                                    &mut graph_layout_gp.borrow_mut(),
                                );
                            }
                        }
                        return;
                    }

                    // Hit-test wires (for disconnect — Step 5)
                    if let Some(wire_idx) = hit_test_wire(canvas_x, canvas_y, &layout.wires) {
                        drop(layout_ref);
                        drop(inp);
                        let wire_info = {
                            let lr = graph_layout_gp.borrow();
                            lr.as_ref().map(|l| {
                                let w = &l.wires[wire_idx];
                                (w.parent_node_id, w.is_left_child)
                            })
                        };
                        if let Some((parent_id, is_left)) = wire_info {
                            let mut s = scene_gp.borrow_mut();
                            undo_gp.borrow_mut().push(s.clone());
                            s.disconnect(parent_id, is_left);
                            let mut inp2 = input_gp.borrow_mut();
                            inp2.scene_dirty = true;
                            inp2.needs_redraw = true;
                            drop(inp2);
                            if let Some(app) = app_gp.upgrade() {
                                push_selection_to_ui(&app, &s);
                                push_tree_to_ui(&app, &s, &mut tree_ids_gp.borrow_mut());
                                push_graph_to_ui(
                                    &app,
                                    &s,
                                    &mut graph_ids_gp.borrow_mut(),
                                    &mut graph_layout_gp.borrow_mut(),
                                );
                            }
                        }
                        return;
                    }
                }
            }
            // No hit — will pan on drag
        });
    }
    {
        let input_gm = input.clone();
        let graph_layout_gm = graph_layout.clone();
        let app_gm = app.as_weak();
        app.on_on_graph_pointer_move(move |x, y| {
            let mut inp = input_gm.borrow_mut();

            let move_dx = x - inp.graph_last_x;
            let move_dy = y - inp.graph_last_y;
            inp.graph_last_x = x;
            inp.graph_last_y = y;

            // Detect drag threshold
            let total_dx = x - inp.graph_down_x;
            let total_dy = y - inp.graph_down_y;
            if total_dx * total_dx + total_dy * total_dy > 9.0 {
                inp.graph_dragging = true;
            }

            if inp.graph_wire_dragging && inp.graph_dragging {
                // Wire drag: update preview wire
                let (sx, sy) = inp.graph_wire_source;
                let pan_x = inp.graph_pan_x;
                let pan_y = inp.graph_pan_y;
                let canvas_x = x + pan_x;
                let canvas_y = y + pan_y;
                drop(inp);

                let svg = if sx < canvas_x {
                    build_preview_wire_svg(sx, sy, canvas_x, canvas_y)
                } else {
                    build_preview_wire_svg(canvas_x, canvas_y, sx, sy)
                };
                if let Some(app) = app_gm.upgrade() {
                    app.set_graph_preview_wire_svg(slint::format!("{}", svg));
                }
                return;
            }

            if inp.graph_dragging && !inp.graph_wire_dragging {
                // Pan mode: move canvas by mouse delta
                inp.graph_panning = true;
                inp.graph_pan_x -= move_dx;
                inp.graph_pan_y -= move_dy;
                let pan_x = inp.graph_pan_x;
                let pan_y = inp.graph_pan_y;
                drop(inp);

                if let Some(app) = app_gm.upgrade() {
                    app.set_graph_pan_x(pan_x);
                    app.set_graph_pan_y(pan_y);
                }
                return;
            }

            // Not dragging yet — do wire hover highlight
            if !inp.graph_dragging {
                let pan_x = inp.graph_pan_x;
                let pan_y = inp.graph_pan_y;
                let canvas_x = x + pan_x;
                let canvas_y = y + pan_y;
                drop(inp);

                let layout_ref = graph_layout_gm.borrow();
                if let Some(layout) = layout_ref.as_ref() {
                    if let Some(wire_idx) = hit_test_wire(canvas_x, canvas_y, &layout.wires) {
                        let w = &layout.wires[wire_idx];
                        let svg = build_wire_svg(w.start, w.end);
                        drop(layout_ref);
                        if let Some(app) = app_gm.upgrade() {
                            app.set_graph_highlight_wire_svg(slint::format!("{}", svg));
                        }
                    } else {
                        drop(layout_ref);
                        if let Some(app) = app_gm.upgrade() {
                            app.set_graph_highlight_wire_svg(slint::SharedString::default());
                        }
                    }
                }
            }
        });
    }
    {
        let input_gu = input.clone();
        let scene_gu = scene.clone();
        let graph_ids_gu = graph_node_ids.clone();
        let graph_layout_gu = graph_layout.clone();
        let tree_ids_gu = tree_node_ids.clone();
        let app_gu = app.as_weak();
        let undo_gu = undo_stack.clone();
        app.on_on_graph_pointer_up(move || {
            let mut inp = input_gu.borrow_mut();
            let was_wire_drag = inp.graph_wire_dragging && inp.graph_dragging;

            if was_wire_drag {
                let pan_x = inp.graph_pan_x;
                let pan_y = inp.graph_pan_y;
                let canvas_x = inp.graph_last_x + pan_x;
                let canvas_y = inp.graph_last_y + pan_y;
                let source_idx = inp.graph_wire_source_node_idx;
                let from_output = inp.graph_wire_from_output;
                inp.graph_wire_dragging = false;
                inp.graph_dragging = false;
                inp.graph_panning = false;
                drop(inp);

                // Clear preview wire
                if let Some(app) = app_gu.upgrade() {
                    app.set_graph_preview_wire_svg(slint::SharedString::default());
                }

                // Hit-test the drop position against ports
                let layout_ref = graph_layout_gu.borrow();
                if let (Some(layout), Some(src_idx)) = (layout_ref.as_ref(), source_idx) {
                    if let Some(port_hit) = hit_test_port(canvas_x, canvas_y, &layout.nodes) {
                        let graph_ids = graph_ids_gu.borrow();
                        let source_node_id = graph_ids.get(src_idx).copied();

                        // Determine target and slot type
                        enum DropTarget {
                            OpSlot(usize, bool),      // (node_idx, is_left)
                            TransformInput(usize),    // node_idx
                        }
                        let drop_target = match port_hit {
                            PortHit::InputTop(idx) => DropTarget::OpSlot(idx, true),
                            PortHit::InputBottom(idx) => DropTarget::OpSlot(idx, false),
                            PortHit::InputSingle(idx) => DropTarget::TransformInput(idx),
                            PortHit::Output(_) => {
                                // Dropped on an output port — invalid for connection
                                return;
                            }
                        };

                        let target_idx = match &drop_target {
                            DropTarget::OpSlot(idx, _) => *idx,
                            DropTarget::TransformInput(idx) => *idx,
                        };
                        let target_node_id = graph_ids.get(target_idx).copied();
                        drop(graph_ids);
                        drop(layout_ref);

                        if let (Some(src_id), Some(tgt_id)) = (source_node_id, target_node_id) {
                            if from_output {
                                // Dragged from output to input: rewire
                                let mut s = scene_gu.borrow_mut();
                                undo_gu.borrow_mut().push(s.clone());
                                let ok = match &drop_target {
                                    DropTarget::OpSlot(_, is_left) => {
                                        s.rewire(tgt_id, *is_left, src_id).is_ok()
                                    }
                                    DropTarget::TransformInput(_) => {
                                        s.rewire_transform(tgt_id, src_id).is_ok()
                                    }
                                };
                                if ok {
                                    let mut inp2 = input_gu.borrow_mut();
                                    inp2.scene_dirty = true;
                                    inp2.needs_redraw = true;
                                    drop(inp2);
                                    if let Some(app) = app_gu.upgrade() {
                                        push_selection_to_ui(&app, &s);
                                        push_tree_to_ui(&app, &s, &mut tree_ids_gu.borrow_mut());
                                        push_graph_to_ui(
                                            &app,
                                            &s,
                                            &mut graph_ids_gu.borrow_mut(),
                                            &mut graph_layout_gu.borrow_mut(),
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                return;
            }

            // If we didn't drag (it was a click on the background), deselect
            if !inp.graph_dragging {
                drop(inp);
                let mut s = scene_gu.borrow_mut();
                s.selected = None;
                let mut inp2 = input_gu.borrow_mut();
                inp2.scene_dirty = true;
                inp2.needs_redraw = true;
                drop(inp2);
                if let Some(app) = app_gu.upgrade() {
                    push_selection_to_ui(&app, &s);
                    push_tree_to_ui(&app, &s, &mut tree_ids_gu.borrow_mut());
                    push_graph_to_ui(
                        &app,
                        &s,
                        &mut graph_ids_gu.borrow_mut(),
                        &mut graph_layout_gu.borrow_mut(),
                    );
                }
                return;
            }

            inp.graph_panning = false;
            inp.graph_dragging = false;
            inp.graph_wire_dragging = false;
        });
    }

    // ── Undo / Redo ──────────────────────────────────────────────
    {
        let scene_undo = scene.clone();
        let input_undo = input.clone();
        let undo_undo = undo_stack.clone();
        let tree_ids_undo = tree_node_ids.clone();
        let graph_ids_undo = graph_node_ids.clone();
        let graph_layout_undo = graph_layout.clone();
        let app_undo = app.as_weak();
        app.on_on_undo(move || {
            let mut s = scene_undo.borrow_mut();
            let restored = undo_undo.borrow_mut().undo(&s);
            if let Some(prev) = restored {
                *s = prev;
                let mut inp = input_undo.borrow_mut();
                inp.scene_dirty = true;
                inp.needs_redraw = true;
                drop(inp);
                if let Some(app) = app_undo.upgrade() {
                    push_selection_to_ui(&app, &s);
                    push_tree_to_ui(&app, &s, &mut tree_ids_undo.borrow_mut());
                    push_graph_to_ui(
                        &app,
                        &s,
                        &mut graph_ids_undo.borrow_mut(),
                        &mut graph_layout_undo.borrow_mut(),
                    );
                }
            }
        });
    }
    {
        let scene_redo = scene.clone();
        let input_redo = input.clone();
        let undo_redo = undo_stack.clone();
        let tree_ids_redo = tree_node_ids.clone();
        let graph_ids_redo = graph_node_ids.clone();
        let graph_layout_redo = graph_layout.clone();
        let app_redo = app.as_weak();
        app.on_on_redo(move || {
            let mut s = scene_redo.borrow_mut();
            let restored = undo_redo.borrow_mut().redo(&s);
            if let Some(next) = restored {
                *s = next;
                let mut inp = input_redo.borrow_mut();
                inp.scene_dirty = true;
                inp.needs_redraw = true;
                drop(inp);
                if let Some(app) = app_redo.upgrade() {
                    push_selection_to_ui(&app, &s);
                    push_tree_to_ui(&app, &s, &mut tree_ids_redo.borrow_mut());
                    push_graph_to_ui(
                        &app,
                        &s,
                        &mut graph_ids_redo.borrow_mut(),
                        &mut graph_layout_redo.borrow_mut(),
                    );
                }
            }
        });
    }

    // ── Render Loop (~60 fps timer) ─────────────────────────────
    let gpu_for_timer = gpu_state.clone();
    let cam_for_timer = camera.clone();
    let input_for_timer = input.clone();
    let scene_for_timer = scene.clone();
    let timer = slint::Timer::default();
    timer.start(
        slint::TimerMode::Repeated,
        std::time::Duration::from_millis(16),
        move || {
            let Some(app) = app_weak.upgrade() else {
                return;
            };

            let vp_w = app.get_viewport_width() as u32;
            let vp_h = app.get_viewport_height() as u32;

            if vp_w == 0 || vp_h == 0 {
                return;
            }

            let mut inp = input_for_timer.borrow_mut();

            // Track viewport size for click-to-select
            if vp_w != inp.last_vp_w || vp_h != inp.last_vp_h {
                inp.last_vp_w = vp_w;
                inp.last_vp_h = vp_h;
                inp.needs_redraw = true;
            }

            if !inp.needs_redraw {
                return;
            }
            inp.needs_redraw = false;
            let scene_dirty = inp.scene_dirty;
            inp.scene_dirty = false;
            let last_key = inp.last_structure_key;
            drop(inp);

            let elapsed = start_time.elapsed().as_secs_f32();
            let uniforms = cam_for_timer.borrow().uniforms(vp_w, vp_h, elapsed);

            let mut gpu_ref = gpu_for_timer.borrow_mut();
            if let Some(gs) = gpu_ref.as_mut() {
                gs.update_camera(&uniforms);

                if scene_dirty {
                    let mut sc = scene_for_timer.borrow_mut();

                    // SLOW PATH: topology changed → regenerate shader + rebuild pipelines
                    let new_key = gpu::codegen::structure_key(&sc);
                    if new_key != last_key {
                        let shader_src = gpu::codegen::compose_shader(&sc);
                        gs.rebuild_pipelines(&shader_src);
                        input_for_timer.borrow_mut().last_structure_key = new_key;
                        log::debug!("Slow path: pipeline rebuilt (key {last_key} → {new_key})");
                    }

                    // FAST PATH (always): update storage buffer parameters
                    let (gpu_nodes, gpu_info) = sc.flatten_for_gpu();
                    gs.update_scene(&gpu_nodes, &gpu_info);
                }

                if let Some(image) = gs.render_frame(vp_w, vp_h) {
                    app.set_viewport_texture(image);
                }
            }
            drop(gpu_ref);

            // Update FPS counter every ~500ms
            let mut inp = input_for_timer.borrow_mut();
            inp.frame_count += 1;
            let fps_elapsed = inp.fps_accum_start.elapsed().as_secs_f32();
            if fps_elapsed >= 0.5 {
                let fps = inp.frame_count as f32 / fps_elapsed;
                app.set_fps_text(slint::format!("{:.0} fps", fps));
                inp.frame_count = 0;
                inp.fps_accum_start = std::time::Instant::now();
            }
        },
    );

    app.run().expect("Failed to run application");
}
