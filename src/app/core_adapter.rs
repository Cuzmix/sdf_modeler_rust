use std::collections::{HashMap, HashSet};

use crate::core::{AppCore, AppCoreInit, CoreAsyncState, CoreCommand, CoreSelection};
use crate::graph::history::History;
use crate::graph::scene::Scene;
use crate::sculpt::SculptState;

use super::actions::Action;
use super::SdfApp;

impl SdfApp {
    pub(super) fn try_process_action_via_core(&mut self, action: &Action) -> bool {
        let Some(core_command) = Self::to_core_command(action) else {
            return false;
        };

        let result = self.with_core(|core| core.apply_command(core_command));

        if result.buffer_dirty {
            self.gpu.buffer_dirty = true;
        }
        if result.needs_graph_rebuild {
            self.ui.node_graph_state.needs_initial_rebuild = true;
            self.ui.light_graph_state.needs_initial_rebuild = true;
        }
        if result.clear_isolation {
            self.ui.isolation_state = None;
        }
        if result.deactivate_sculpt {
            self.doc.sculpt_state = SculptState::Inactive;
        }
        if let Some(node_id) = result.pending_center_node {
            match action {
                Action::CreateLight(_) => {
                    self.ui.light_graph_state.pending_center_node = Some(node_id);
                }
                _ => {
                    self.ui.node_graph_state.pending_center_node = Some(node_id);
                }
            }
        }

        result.handled
    }

    fn to_core_command(action: &Action) -> Option<CoreCommand> {
        match action {
            Action::Select(id) => Some(CoreCommand::Select(*id)),
            Action::CreatePrimitive(kind) => Some(CoreCommand::CreatePrimitive(kind.clone())),
            Action::CreateOperation { op, left, right } => Some(CoreCommand::CreateOperation {
                op: op.clone(),
                left: *left,
                right: *right,
            }),
            Action::CreateTransform { input } => Some(CoreCommand::CreateTransform { input: *input }),
            Action::CreateModifier { kind, input } => Some(CoreCommand::CreateModifier {
                kind: kind.clone(),
                input: *input,
            }),
            Action::CreateLight(light_type) => Some(CoreCommand::CreateLight(light_type.clone())),
            Action::DeleteSelected => Some(CoreCommand::DeleteSelected),
            Action::DeleteNode(id) => Some(CoreCommand::DeleteNode(*id)),
            Action::Undo => Some(CoreCommand::Undo),
            Action::Redo => Some(CoreCommand::Redo),
            _ => None,
        }
    }

    fn with_core<R>(&mut self, f: impl FnOnce(&mut AppCore) -> R) -> R {
        let scene = std::mem::replace(&mut self.doc.scene, Self::empty_scene());
        let history = std::mem::replace(&mut self.doc.history, History::new());
        let selection = CoreSelection {
            primary: self.ui.node_graph_state.selected,
            set: self.ui.node_graph_state.selected_set.clone(),
        };
        let async_state = CoreAsyncState {
            bake_in_progress: !matches!(self.async_state.bake_status, super::BakeStatus::Idle),
            export_in_progress: !matches!(self.async_state.export_status, super::ExportStatus::Idle),
            import_in_progress: !matches!(self.async_state.import_status, super::ImportStatus::Idle),
            pick_in_progress: !matches!(self.async_state.pick_state, super::PickState::Idle),
        };

        let mut core = AppCore::from_init(AppCoreInit {
            scene,
            history,
            camera: std::mem::take(&mut self.doc.camera),
            selection,
            active_tool: self.doc.active_tool,
            sculpt_state: self.doc.sculpt_state.clone(),
            async_state,
            soloed_light: self.doc.soloed_light,
            show_debug: self.ui.show_debug,
            show_settings: self.ui.show_settings,
        });

        let output = f(&mut core);

        self.doc.scene = core.scene;
        self.doc.history = core.history;
        self.doc.camera = core.camera;
        self.doc.active_tool = core.active_tool;
        self.doc.sculpt_state = core.sculpt_state;
        self.doc.soloed_light = core.soloed_light;
        self.ui.show_debug = core.show_debug;
        self.ui.show_settings = core.show_settings;
        self.ui.node_graph_state.selected = core.selection.primary;
        self.ui.node_graph_state.selected_set = core.selection.set;

        output
    }

    fn empty_scene() -> Scene {
        Scene {
            nodes: HashMap::new(),
            next_id: 0,
            name_counters: HashMap::new(),
            hidden_nodes: HashSet::new(),
            light_masks: HashMap::new(),
        }
    }
}
