use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use slint::{Image, SharedPixelBuffer, Timer, TimerMode};

use crate::app::frontend_models::{
    build_shell_snapshot, ShellSnapshot, ShellSnapshotInputs, ViewportFrameImage,
};
use crate::app::runtime::{AppRenderContext, WakeHandle};
use crate::settings::Settings;

use super::actions::ActionSink;
use super::slint_bridge::{capture_frame_input, dispatch_event, empty_feedback, SlintUiEvent};
use super::SdfApp;

slint::slint! {
import { Button, HorizontalBox, VerticalBox } from "std-widgets.slint";

export component SlintHostWindow inherits Window {
    in property <string> window_title;
    title: root.window_title;

    in property <image> viewport_image;
    in property <string> scene_summary;
    in property <string> expert_summary;
    in property <string> inspector_title;
    in property <string> inspector_chips;
    in property <string> inspector_properties;
    in property <string> inspector_display;
    in property <string> history_summary;
    in property <string> reference_summary;
    in property <string> viewport_status;

    callback frame_all();
    callback undo();
    callback redo();
    callback select_previous_scene();
    callback select_next_scene();

    width: 1480px;
    height: 920px;

    VerticalBox {
        spacing: 8px;

        HorizontalBox {
            spacing: 8px;

            Button {
                text: "Frame All";
                clicked => { root.frame_all(); }
            }

            Button {
                text: "Undo";
                clicked => { root.undo(); }
            }

            Button {
                text: "Redo";
                clicked => { root.redo(); }
            }

            Button {
                text: "Previous Selection";
                clicked => { root.select_previous_scene(); }
            }

            Button {
                text: "Next Selection";
                clicked => { root.select_next_scene(); }
            }

            Text {
                text: root.viewport_status;
            }
        }

        Text {
            text: root.scene_summary;
            wrap: word-wrap;
        }

        Image {
            source: root.viewport_image;
            image-fit: contain;
            height: 540px;
        }

        Text {
            text: root.inspector_title;
        }

        Text {
            text: root.inspector_chips;
            wrap: word-wrap;
        }

        Text {
            text: root.inspector_properties;
            wrap: word-wrap;
        }

        Text {
            text: root.inspector_display;
            wrap: word-wrap;
        }

        Text {
            text: root.expert_summary;
            wrap: word-wrap;
        }

        Text {
            text: root.history_summary;
            wrap: word-wrap;
        }

        Text {
            text: root.reference_summary;
            wrap: word-wrap;
        }
    }
}
}

struct SlintHostState {
    app: SdfApp,
    frame_started_at: Instant,
    queued_actions: ActionSink,
    wake_flag: Arc<AtomicBool>,
    viewport_generation: u64,
    viewport_size: (u32, u32),
    last_snapshot: Option<ShellSnapshot>,
    last_window_title: String,
}

impl SlintHostState {
    fn new(app: SdfApp, wake_flag: Arc<AtomicBool>) -> Self {
        Self {
            app,
            frame_started_at: Instant::now(),
            queued_actions: Vec::new(),
            wake_flag,
            viewport_generation: 0,
            viewport_size: (960, 540),
            last_snapshot: None,
            last_window_title: "SDF Modeler".to_string(),
        }
    }

    fn queue_event(&mut self, event: SlintUiEvent) {
        dispatch_event(event, self.last_snapshot.as_ref(), &mut self.queued_actions);
    }

    fn tick(&mut self, window: &SlintHostWindow) {
        let frame_input = capture_frame_input(self.frame_started_at.elapsed().as_secs_f64());
        let camera_animating = self.app.run_backend_pre_ui(&frame_input);
        let actions = std::mem::take(&mut self.queued_actions);
        self.app.process_actions(actions);
        let commands =
            self.app
                .run_backend_post_ui(&frame_input, camera_animating, empty_feedback());

        if let Some(title) = commands.window_title {
            self.last_window_title = title;
            window.set_window_title(self.last_window_title.clone().into());
        }

        let snapshot = build_shell_snapshot(ShellSnapshotInputs {
            scene: &self.app.doc.scene,
            selection: &self.app.ui.selection,
            history: &self.app.doc.history,
            reference_images: &self.app.ui.reference_images,
            expert_panels: &self.app.ui.expert_panels,
            settings: &self.app.settings,
            interaction_mode: self.app.ui.primary_shell.interaction_mode,
            gizmo_mode: self.app.gizmo.mode.clone(),
            gizmo_space: self.app.gizmo.space.clone(),
        });

        let snapshot_changed = self.last_snapshot.as_ref() != Some(&snapshot);
        if snapshot_changed {
            apply_shell_snapshot(window, &snapshot);
            self.last_snapshot = Some(snapshot);
        }

        let should_refresh_viewport = commands.request_repaint
            || snapshot_changed
            || self.wake_flag.swap(false, Ordering::Relaxed)
            || self.viewport_generation == 0;
        if should_refresh_viewport {
            self.viewport_generation = self.viewport_generation.wrapping_add(1);
            let frame = self.app.viewport_frame_image(
                self.viewport_size.0,
                self.viewport_size.1,
                self.viewport_generation,
            );
            apply_viewport_frame(window, &frame);
        }
    }
}

pub(crate) fn run_slint_host(settings: Settings) -> Result<(), String> {
    let render_context = create_render_context()?;
    let wake_flag = Arc::new(AtomicBool::new(false));
    let wake = WakeHandle::new({
        let wake_flag = wake_flag.clone();
        move || {
            wake_flag.store(true, Ordering::Relaxed);
        }
    });

    let app = SdfApp::new_from_runtime(render_context, wake, settings);
    let window = SlintHostWindow::new().map_err(|error| error.to_string())?;
    window.set_window_title("SDF Modeler".into());

    let host = Rc::new(RefCell::new(SlintHostState::new(app, wake_flag)));

    install_callbacks(&window, &host);
    host.borrow_mut().tick(&window);

    let timer = Timer::default();
    let window_weak = window.as_weak();
    let host_for_timer = host.clone();
    timer.start(TimerMode::Repeated, Duration::from_millis(33), move || {
        let Some(window) = window_weak.upgrade() else {
            return;
        };
        host_for_timer.borrow_mut().tick(&window);
    });

    window.run().map_err(|error| error.to_string())?;
    drop(timer);
    host.borrow_mut().app.mark_clean_exit();

    Ok(())
}

fn create_render_context() -> Result<AppRenderContext, String> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .ok_or_else(|| "Failed to acquire a WGPU adapter for the Slint host".to_string())?;
    let descriptor = crate::native_wgpu::native_device_descriptor(&adapter);
    let (device, queue) = pollster::block_on(adapter.request_device(&descriptor, None))
        .map_err(|error| format!("Failed to create a WGPU device for the Slint host: {error}"))?;

    Ok(AppRenderContext::new(
        Arc::new(device),
        Arc::new(queue),
        Arc::new(adapter),
        wgpu::TextureFormat::Rgba8UnormSrgb,
    ))
}

fn install_callbacks(window: &SlintHostWindow, host: &Rc<RefCell<SlintHostState>>) {
    {
        let host = host.clone();
        window.on_frame_all(move || {
            host.borrow_mut().queue_event(SlintUiEvent::FrameAll);
        });
    }

    {
        let host = host.clone();
        window.on_undo(move || {
            host.borrow_mut().queue_event(SlintUiEvent::Undo);
        });
    }

    {
        let host = host.clone();
        window.on_redo(move || {
            host.borrow_mut().queue_event(SlintUiEvent::Redo);
        });
    }

    {
        let host = host.clone();
        window.on_select_previous_scene(move || {
            host.borrow_mut()
                .queue_event(SlintUiEvent::SelectPreviousSceneRow);
        });
    }

    {
        let host = host.clone();
        window.on_select_next_scene(move || {
            host.borrow_mut()
                .queue_event(SlintUiEvent::SelectNextSceneRow);
        });
    }
}

fn apply_shell_snapshot(window: &SlintHostWindow, snapshot: &ShellSnapshot) {
    window.set_scene_summary(
        join_lines(
            &snapshot
                .scene_panel
                .rows
                .iter()
                .map(scene_row_label)
                .collect::<Vec<_>>(),
        )
        .into(),
    );
    window.set_expert_summary(
        join_lines(
            &snapshot
                .utility
                .expert_panels
                .iter()
                .map(expert_panel_label)
                .collect::<Vec<_>>(),
        )
        .into(),
    );
    window.set_inspector_title(snapshot.inspector.title.clone().into());
    window.set_inspector_chips(join_lines(&snapshot.inspector.chips).into());
    window.set_inspector_properties(join_lines(&snapshot.inspector.property_lines).into());
    window.set_inspector_display(join_lines(&snapshot.inspector.display_lines).into());
    window.set_history_summary(join_lines(&snapshot.utility.history_lines).into());
    window.set_reference_summary(join_lines(&snapshot.utility.reference_lines).into());
    window.set_viewport_status(
        format!(
            "{} / {} / {}",
            snapshot.viewport_status.interaction_label,
            snapshot.viewport_status.transform_label,
            snapshot.viewport_status.space_label
        )
        .into(),
    );
}

fn apply_viewport_frame(window: &SlintHostWindow, frame: &ViewportFrameImage) {
    let pixels = SharedPixelBuffer::<slint::Rgba8Pixel>::clone_from_slice(
        frame.rgba8.as_slice(),
        frame.width,
        frame.height,
    );
    window.set_viewport_image(Image::from_rgba8(pixels));
}

fn scene_row_label(row: &crate::app::frontend_models::ScenePanelRow) -> String {
    let indent = "  ".repeat(row.depth);
    let visibility = if row.hidden { "[hidden] " } else { "" };
    let selection = if row.selected { "> " } else { "  " };
    format!("{selection}{indent}{visibility}{}", row.label)
}

fn expert_panel_label(entry: &crate::app::frontend_models::ExpertPanelEntry) -> String {
    let state = if entry.open { "open" } else { "closed" };
    format!("{} | {}", state, entry.label)
}

fn join_lines(lines: &[String]) -> String {
    lines.join("\n")
}
