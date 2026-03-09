use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use eframe::egui_wgpu::RenderState;
use egui_dock::DockState;
use glam::Vec3;

use crate::compat::Instant;
use crate::gpu::camera::Camera;
use crate::gpu::picking::PendingPick;
use crate::graph::history::History;
use crate::graph::scene::{NodeId, Scene};
use crate::mesh_import::TriMesh;
use crate::sculpt::{ActiveTool, SculptState};
use crate::sculpt_history::SculptHistory;
use crate::ui::dock::Tab;
use crate::ui::gizmo::{GizmoMode, GizmoSpace, GizmoState};
use crate::ui::node_graph::NodeGraphState;
use crate::ui::reference_image::ReferenceImageManager;

use super::{BakeStatus, ExportStatus, FrameTimings, ImportStatus, PickState, Toast};

// ---------------------------------------------------------------------------
// Core document state: the "model" being edited.
// ---------------------------------------------------------------------------

pub struct DocumentState {
    pub scene: Scene,
    pub camera: Camera,
    pub history: History,
    pub active_tool: ActiveTool,
    pub sculpt_state: SculptState,
    pub sculpt_history: SculptHistory,
    pub clipboard_node: Option<NodeId>,
    /// When set, only this Light node contributes to scene lighting (Solo mode).
    pub soloed_light: Option<NodeId>,
}

// ---------------------------------------------------------------------------
// Gizmo interaction state.
// ---------------------------------------------------------------------------

pub struct GizmoContext {
    pub state: GizmoState,
    pub mode: GizmoMode,
    pub space: GizmoSpace,
    pub pivot_offset: Vec3,
    pub last_selection: Option<NodeId>,
    pub gizmo_visible: bool,
}

// ---------------------------------------------------------------------------
// GPU synchronization state — tracks what needs rebuilding / uploading.
// ---------------------------------------------------------------------------

pub struct GpuSyncState {
    pub render_state: RenderState,
    pub current_structure_key: u64,
    pub buffer_dirty: bool,
    pub last_data_fingerprint: u64,
    pub voxel_gpu_offsets: HashMap<NodeId, u32>,
    pub sculpt_tex_indices: HashMap<NodeId, usize>,
}

// ---------------------------------------------------------------------------
// Async task tracking (bake, export, sculpt pick).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct SculptRuntimeCache {
    pub node_id: NodeId,
    pub structure_key: u64,
    pub material_id: i32,
    pub position: Vec3,
    pub rotation: Vec3,
    pub gpu_offset: Option<u32>,
    pub grid_resolution: u32,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}
pub struct AsyncState {
    pub bake_status: BakeStatus,
    pub export_status: ExportStatus,
    pub import_status: ImportStatus,
    pub pick_state: PickState,
    pub pending_pick: Option<PendingPick>,
    pub last_sculpt_hit: Option<Vec3>,
    pub lazy_brush_pos: Option<Vec3>,
    /// Modifier keys captured at the time of sculpt drag (for Ctrl-invert / Shift-smooth).
    pub sculpt_ctrl_held: bool,
    pub sculpt_shift_held: bool,
    /// Pen pressure (0.0-1.0) during sculpt drag. 0.0 = no pressure data.
    pub sculpt_pressure: f32,
    /// World position from the latest hover pick (for 3D brush preview).
    /// Independent of drag state — persists while hovering, cleared when cursor leaves.
    pub hover_world_pos: Option<Vec3>,
    /// Whether the cursor is currently over geometry (from last hover pick).
    /// Used to decide: LMB on empty space → orbit instead of sculpt.
    pub cursor_over_geometry: bool,
    /// Whether a sculpt drag is actively in progress (LMB held on geometry).
    pub sculpt_dragging: bool,
    /// Per-stroke cached node metadata to avoid repeated topo/transform lookups.
    pub sculpt_runtime_cache: Option<SculptRuntimeCache>,
}

// ---------------------------------------------------------------------------
// UI-only state: dialog visibility, rename editing, toasts, dock layout.
// ---------------------------------------------------------------------------

/// Saved state when entering isolation mode.
pub struct IsolationState {
    pub pre_hidden: HashSet<NodeId>,
    pub isolated_node: NodeId,
}

/// Clipboard for property copy/paste.
#[derive(Clone)]
pub struct PropertyClipboard {
    pub color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub emissive: [f32; 3],
    pub emissive_intensity: f32,
    pub fresnel: f32,
}

/// State for the "Convert to Sculpt" dialog shown by Ctrl+R.
pub struct SculptConvertDialog {
    pub target: NodeId,
    pub mode: crate::app::actions::SculptConvertMode,
    pub resolution: u32,
}

impl SculptConvertDialog {
    pub fn new(target: NodeId) -> Self {
        Self {
            target,
            mode: crate::app::actions::SculptConvertMode::BakeActiveNode,
            resolution: 64,
        }
    }
}

/// State for the import mesh dialog shown after picking a file.
pub struct ImportDialog {
    /// The loaded triangle mesh (shared with the voxelize thread when committed).
    pub mesh: Arc<TriMesh>,
    /// Display filename (e.g. "bunny.obj").
    pub filename: String,
    /// User-chosen voxel resolution.
    pub resolution: u32,
    /// Auto-calculated resolution suggestion based on mesh stats.
    pub auto_resolution: u32,
    /// Whether the user is using the auto-calculated resolution.
    pub use_auto: bool,
    /// Mesh vertex count (cached for display).
    pub vertex_count: usize,
    /// Mesh triangle count (cached for display).
    pub triangle_count: usize,
    /// Mesh bounding box dimensions (cached for display).
    pub bounds_size: Vec3,
}

impl ImportDialog {
    pub fn new(mesh: TriMesh, filename: String, max_resolution: u32) -> Self {
        let vertex_count = mesh.vertices.len();
        let triangle_count = mesh.triangles.len();

        // Compute mesh bounds for display
        let mut mesh_min = Vec3::splat(f32::MAX);
        let mut mesh_max = Vec3::splat(f32::MIN);
        for v in &mesh.vertices {
            mesh_min = mesh_min.min(*v);
            mesh_max = mesh_max.max(*v);
        }
        let bounds_size = mesh_max - mesh_min;

        // Auto-calculate resolution: scale with cube root of triangle count.
        // ~2.5x multiplier gives reasonable results:
        //   1K tris → 25 → clamped to 32
        //   8K tris → 50
        //  64K tris → 100
        // 500K tris → 198
        let auto_resolution = ((triangle_count as f32).cbrt() * 2.5)
            .round()
            .clamp(32.0, max_resolution as f32) as u32;

        Self {
            mesh: Arc::new(mesh),
            filename,
            resolution: auto_resolution,
            auto_resolution,
            use_auto: true,
            vertex_count,
            triangle_count,
            bounds_size,
        }
    }
}

pub struct UiState {
    pub dock_state: DockState<Tab>,
    pub node_graph_state: NodeGraphState,
    pub light_graph_state: NodeGraphState,
    pub show_debug: bool,
    pub show_help: bool,
    pub show_export_dialog: bool,
    pub show_settings: bool,
    pub renaming_node: Option<NodeId>,
    pub rename_buf: String,
    pub scene_tree_drag: Option<NodeId>,
    pub scene_tree_search: String,
    pub isolation_state: Option<IsolationState>,
    pub toasts: Vec<Toast>,
    pub turntable_active: bool,
    pub property_clipboard: Option<PropertyClipboard>,
    pub command_palette_open: bool,
    pub command_palette_query: String,
    pub command_palette_selected: usize,
    /// Open "Convert to Sculpt" dialog state (None = hidden).
    pub sculpt_convert_dialog: Option<SculptConvertDialog>,
    /// Open "Import Mesh" settings dialog state (None = hidden).
    pub import_dialog: Option<ImportDialog>,
    /// Quick Primitives floating toolbar (Shift+A).
    pub show_quick_toolbar: bool,
    /// Keybinding editor: which action is currently waiting for a key press (None = not rebinding).
    pub rebinding_action: Option<crate::keymap::ActionBinding>,
    /// Set of Light NodeIds currently active on GPU (nearest MAX_SCENE_LIGHTS to camera).
    pub active_light_ids: HashSet<NodeId>,
    /// Total number of visible lights in the scene (for >8 warning).
    pub total_light_count: usize,
    /// The last total light count that triggered a toast warning.
    /// Used to avoid spamming the same warning repeatedly.
    pub last_light_warning_count: Option<usize>,
    /// Startup crash recovery modal visibility.
    pub show_recovery_dialog: bool,
    /// Recovery modal context built from autosave.meta.
    pub recovery_summary: String,
    /// Reference images used as modeling guides in the viewport.
    pub reference_images: ReferenceImageManager,
    /// Show a cursor-relative SDF distance readout in the viewport.
    pub show_distance_readout: bool,
    /// Interactive two-point measurement mode.
    pub measurement_mode: bool,
    /// Measurement points in world space (0, 1, or 2 points).
    pub measurement_points: Vec<Vec3>,
}

// ---------------------------------------------------------------------------
// Persistence state: file path, save tracking, auto-save timer.
// ---------------------------------------------------------------------------

pub struct PersistenceState {
    pub current_file_path: Option<PathBuf>,
    pub scene_dirty: bool,
    pub saved_fingerprint: u64,
    pub last_auto_save: Instant,
}

// ---------------------------------------------------------------------------
// Performance / profiling state.
// ---------------------------------------------------------------------------

pub struct PerfState {
    pub timings: FrameTimings,
    pub resolution_upgrade_pending: bool,
    pub composite_full_update_needed: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh_import::TriMesh;

    /// Create a simple test mesh (two triangles forming a 2x2x0 quad).
    fn test_quad_mesh(num_triangles: usize) -> TriMesh {
        let mut vertices = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(2.0, 2.0, 0.0),
            Vec3::new(0.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let mut triangles = vec![[0, 1, 2], [0, 2, 3]];
        // Pad with duplicate triangles to reach desired count
        while triangles.len() < num_triangles {
            let base = vertices.len() as u32;
            vertices.push(Vec3::new(0.0, 0.0, 0.0));
            vertices.push(Vec3::new(1.0, 0.0, 0.0));
            vertices.push(Vec3::new(0.0, 1.0, 0.0));
            triangles.push([base, base + 1, base + 2]);
        }
        TriMesh {
            vertices,
            triangles,
        }
    }

    #[test]
    fn import_dialog_auto_resolution_small_mesh() {
        let mesh = test_quad_mesh(100);
        let dialog = ImportDialog::new(mesh, "test.obj".into(), 320);
        // cbrt(100) * 2.5 = 4.64 * 2.5 = 11.6 → clamped to 32
        assert_eq!(dialog.auto_resolution, 32);
        assert_eq!(dialog.resolution, 32);
        assert!(dialog.use_auto);
    }

    #[test]
    fn import_dialog_auto_resolution_medium_mesh() {
        let mesh = test_quad_mesh(8000);
        let dialog = ImportDialog::new(mesh, "test.obj".into(), 320);
        // cbrt(8000) = 20, * 2.5 = 50
        assert_eq!(dialog.auto_resolution, 50);
    }

    #[test]
    fn import_dialog_auto_resolution_large_mesh() {
        let mesh = test_quad_mesh(64000);
        let dialog = ImportDialog::new(mesh, "test.obj".into(), 320);
        // cbrt(64000) = 40, * 2.5 = 100
        assert_eq!(dialog.auto_resolution, 100);
    }

    #[test]
    fn import_dialog_auto_resolution_clamped_to_max() {
        let mesh = test_quad_mesh(500_000);
        let dialog = ImportDialog::new(mesh, "test.obj".into(), 128);
        // cbrt(500000) * 2.5 ≈ 198, but max is 128
        assert_eq!(dialog.auto_resolution, 128);
    }

    #[test]
    fn import_dialog_preserves_mesh_stats() {
        let mesh = test_quad_mesh(2);
        let vert_count = mesh.vertices.len();
        let tri_count = mesh.triangles.len();
        let dialog = ImportDialog::new(mesh, "monkey.obj".into(), 320);
        assert_eq!(dialog.vertex_count, vert_count);
        assert_eq!(dialog.triangle_count, tri_count);
        assert_eq!(dialog.filename, "monkey.obj");
    }

    #[test]
    fn import_dialog_bounds_size_computed() {
        let mesh = TriMesh {
            vertices: vec![
                Vec3::new(-1.0, -2.0, -3.0),
                Vec3::new(4.0, 5.0, 6.0),
                Vec3::new(0.0, 0.0, 0.0),
            ],
            triangles: vec![[0, 1, 2]],
        };
        let dialog = ImportDialog::new(mesh, "test.obj".into(), 320);
        assert!((dialog.bounds_size.x - 5.0).abs() < f32::EPSILON);
        assert!((dialog.bounds_size.y - 7.0).abs() < f32::EPSILON);
        assert!((dialog.bounds_size.z - 9.0).abs() < f32::EPSILON);
    }

    #[test]
    fn import_dialog_defaults_to_auto() {
        let mesh = test_quad_mesh(1000);
        let dialog = ImportDialog::new(mesh, "test.obj".into(), 320);
        assert!(dialog.use_auto);
        assert_eq!(dialog.resolution, dialog.auto_resolution);
    }
}
