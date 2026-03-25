use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use slint::{Timer, TimerMode};

use crate::app::frontend_models::{build_shell_snapshot, ShellSnapshot, ShellSnapshotInputs};
use crate::app::runtime::{AppRenderContext, WakeHandle};
use crate::settings::Settings;

use super::actions::ActionSink;
use super::backend_frame::FrameCommands;
use super::slint_bridge::{
    capture_frame_input, dispatch_event, SlintUiEvent, SlintViewportInputState,
};
use super::{BakeStatus, ExportStatus, ImportStatus, PickState, SdfApp};

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
    out property <float> viewport_width: viewport_frame.width / 1px;
    out property <float> viewport_height: viewport_frame.height / 1px;

    callback frame_all();
    callback undo();
    callback redo();
    callback select_previous_scene();
    callback select_next_scene();
    callback viewport_pointer_event(event: PointerEvent, x: float, y: float);
    callback viewport_scroll_event(event: PointerScrollEvent);
    callback viewport_double_clicked(x: float, y: float);

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

        Rectangle {
            background: #111217;
            height: 540px;

            viewport_frame := Image {
                width: parent.width;
                height: parent.height;
                source: root.viewport_image;
                image-fit: fill;
            }

            TouchArea {
                width: parent.width;
                height: parent.height;

                pointer-event(event) => {
                    root.viewport_pointer_event(event, self.mouse_x / 1px, self.mouse_y / 1px);
                }

                scroll-event(event) => {
                    root.viewport_scroll_event(event);
                    accept
                }

                double-clicked => {
                    root.viewport_double_clicked(self.mouse_x / 1px, self.mouse_y / 1px);
                }
            }
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

struct NativeWgpuContext {
    instance: wgpu::Instance,
    render_context: AppRenderContext,
}

struct SlintViewportTexture {
    view: wgpu::TextureView,
    size: (u32, u32),
}

struct TickOutcome {
    request_redraw: bool,
    needs_continuous_ticks: bool,
}

struct SlintHostState {
    app: SdfApp,
    frame_started_at: Instant,
    queued_actions: ActionSink,
    wake_flag: Arc<AtomicBool>,
    viewport_size: (u32, u32),
    viewport_input: SlintViewportInputState,
    viewport_texture: Option<SlintViewportTexture>,
    viewport_dirty: bool,
    continuous_tick_active: bool,
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
            viewport_size: (960, 540),
            viewport_input: SlintViewportInputState::default(),
            viewport_texture: None,
            viewport_dirty: true,
            continuous_tick_active: false,
            last_snapshot: None,
            last_window_title: "SDF Modeler".to_string(),
        }
    }

    fn queue_event(&mut self, event: SlintUiEvent) {
        dispatch_event(event, self.last_snapshot.as_ref(), &mut self.queued_actions);
    }

    fn tick(&mut self, window: &SlintHostWindow) -> TickOutcome {
        self.viewport_input.set_viewport_geometry(
            window.get_viewport_width(),
            window.get_viewport_height(),
            window.window().scale_factor(),
        );
        let next_viewport_size = (
            (window.get_viewport_width().max(1.0) * window.window().scale_factor()) as u32,
            (window.get_viewport_height().max(1.0) * window.window().scale_factor()) as u32,
        );
        let viewport_size_changed = next_viewport_size != self.viewport_size;
        self.viewport_size = next_viewport_size;
        if viewport_size_changed {
            self.viewport_dirty = true;
        }
        self.ensure_viewport_texture(window);

        let frame_now = self.frame_started_at.elapsed().as_secs_f64();
        let frame_input = capture_frame_input(frame_now, &self.viewport_input);
        let camera_animating = self.app.run_backend_pre_ui(&frame_input);
        let mut actions = std::mem::take(&mut self.queued_actions);
        let had_actions = !actions.is_empty();
        let viewport_feedback = self
            .app
            .run_viewport_interaction(&self.viewport_input.take_snapshot(frame_now), &mut actions);
        apply_brush_adjust_feedback(&mut self.app, &viewport_feedback);
        let had_interaction_actions = !actions.is_empty();
        self.app.process_actions(actions);
        let commands = self
            .app
            .run_backend_post_ui(&frame_input, camera_animating, viewport_feedback);

        if let Some(ref title) = commands.window_title {
            self.last_window_title = title.clone();
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

        let woke = self.wake_flag.swap(false, Ordering::Relaxed);
        self.viewport_dirty |= viewport_size_changed
            || had_actions
            || had_interaction_actions
            || commands.request_repaint
            || snapshot_changed
            || woke;

        TickOutcome {
            request_redraw: self.viewport_dirty || snapshot_changed || woke,
            needs_continuous_ticks: self.needs_continuous_ticks(camera_animating, &commands),
        }
    }

    fn render_viewport_if_needed(&mut self) {
        if !self.viewport_dirty {
            return;
        }
        let Some(texture) = self.viewport_texture.as_ref() else {
            return;
        };
        self.app.render_viewport_texture(
            &texture.view,
            self.viewport_size.0.max(1),
            self.viewport_size.1.max(1),
        );
        self.viewport_dirty = false;
    }

    fn release_viewport_texture(&mut self) {
        self.viewport_texture = None;
        self.viewport_dirty = true;
    }

    fn ensure_viewport_texture(&mut self, window: &SlintHostWindow) {
        let width = self.viewport_size.0.max(1);
        let height = self.viewport_size.1.max(1);
        if self
            .viewport_texture
            .as_ref()
            .is_some_and(|texture| texture.size == (width, height))
        {
            return;
        }

        let texture = self
            .app
            .gpu
            .render_context
            .device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Slint Viewport Texture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.app.gpu.render_context.target_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let image = match slint::Image::try_from(texture.clone()) {
            Ok(image) => image,
            Err(error) => {
                log::error!("Failed to import viewport texture into Slint: {error}");
                return;
            }
        };
        window.set_viewport_image(image.clone());
        self.viewport_texture = Some(SlintViewportTexture {
            view,
            size: (width, height),
        });
        self.viewport_dirty = true;
    }

    fn needs_continuous_ticks(
        &self,
        camera_animating: bool,
        commands: &FrameCommands,
    ) -> bool {
        camera_animating
            || commands.request_repaint
            || self.viewport_input.needs_continuous_ticks()
            || self.app.ui.turntable_active
            || self.app.async_state.sculpt_dragging
            || self.app.async_state.pending_pick.is_some()
            || matches!(self.app.async_state.pick_state, PickState::Pending { .. })
            || matches!(self.app.async_state.bake_status, BakeStatus::InProgress { .. })
            || matches!(self.app.async_state.export_status, ExportStatus::InProgress { .. })
            || matches!(self.app.async_state.import_status, ImportStatus::InProgress { .. })
    }
}

pub(crate) fn run_slint_host(settings: Settings) -> Result<(), String> {
    let native_wgpu = create_render_context()?;
    slint::BackendSelector::new()
        .require_wgpu_27(slint::wgpu_27::WGPUConfiguration::Manual {
            instance: native_wgpu.instance.clone(),
            adapter: native_wgpu.render_context.adapter.as_ref().clone(),
            device: native_wgpu.render_context.device.as_ref().clone(),
            queue: native_wgpu.render_context.queue.as_ref().clone(),
        })
        .select()
        .map_err(|error| error.to_string())?;

    let window = SlintHostWindow::new().map_err(|error| error.to_string())?;
    window.set_window_title("SDF Modeler".into());

    let wake_flag = Arc::new(AtomicBool::new(false));
    let wake = WakeHandle::new({
        let wake_flag = wake_flag.clone();
        let window_weak = window.as_weak();
        move || {
            wake_flag.store(true, Ordering::Relaxed);
            let _ = window_weak.upgrade_in_event_loop(|window| {
                window.window().request_redraw();
            });
        }
    });

    let app = SdfApp::new_from_runtime(native_wgpu.render_context, wake, settings);
    let host = Rc::new(RefCell::new(SlintHostState::new(app, wake_flag)));
    let active_timer = Rc::new(Timer::default());

    install_callbacks(&window, &host, &active_timer);
    install_rendering_notifier(&window, &host, &active_timer)?;
    drive_host_tick(&window, &host, &active_timer);

    window.run().map_err(|error| error.to_string())?;
    active_timer.stop();
    host.borrow_mut().app.mark_clean_exit();

    Ok(())
}

fn create_render_context() -> Result<NativeWgpuContext, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::DX12,
        ..Default::default()
    });
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|error| format!("Failed to acquire a WGPU adapter for the Slint host: {error}"))?;
    let descriptor = crate::native_wgpu::native_device_descriptor(&adapter);
    let (device, queue) = pollster::block_on(adapter.request_device(&descriptor))
        .map_err(|error| format!("Failed to create a WGPU device for the Slint host: {error}"))?;

    Ok(NativeWgpuContext {
        instance,
        render_context: AppRenderContext::new(
            Arc::new(device),
            Arc::new(queue),
            Arc::new(adapter),
            wgpu::TextureFormat::Rgba8UnormSrgb,
        ),
    })
}

fn install_callbacks(
    window: &SlintHostWindow,
    host: &Rc<RefCell<SlintHostState>>,
    active_timer: &Rc<Timer>,
) {
    {
        let host = host.clone();
        let window_weak = window.as_weak();
        let active_timer = active_timer.clone();
        window.on_frame_all(move || {
            host.borrow_mut().queue_event(SlintUiEvent::FrameAll);
            if let Some(window) = window_weak.upgrade() {
                drive_host_tick(&window, &host, &active_timer);
            }
        });
    }

    {
        let host = host.clone();
        let window_weak = window.as_weak();
        let active_timer = active_timer.clone();
        window.on_undo(move || {
            host.borrow_mut().queue_event(SlintUiEvent::Undo);
            if let Some(window) = window_weak.upgrade() {
                drive_host_tick(&window, &host, &active_timer);
            }
        });
    }

    {
        let host = host.clone();
        let window_weak = window.as_weak();
        let active_timer = active_timer.clone();
        window.on_redo(move || {
            host.borrow_mut().queue_event(SlintUiEvent::Redo);
            if let Some(window) = window_weak.upgrade() {
                drive_host_tick(&window, &host, &active_timer);
            }
        });
    }

    {
        let host = host.clone();
        let window_weak = window.as_weak();
        let active_timer = active_timer.clone();
        window.on_select_previous_scene(move || {
            host.borrow_mut()
                .queue_event(SlintUiEvent::SelectPreviousSceneRow);
            if let Some(window) = window_weak.upgrade() {
                drive_host_tick(&window, &host, &active_timer);
            }
        });
    }

    {
        let host = host.clone();
        let window_weak = window.as_weak();
        let active_timer = active_timer.clone();
        window.on_select_next_scene(move || {
            host.borrow_mut()
                .queue_event(SlintUiEvent::SelectNextSceneRow);
            if let Some(window) = window_weak.upgrade() {
                drive_host_tick(&window, &host, &active_timer);
            }
        });
    }

    {
        let host = host.clone();
        let window_weak = window.as_weak();
        let active_timer = active_timer.clone();
        window.on_viewport_pointer_event(move |event, x, y| {
            let modifiers = crate::keymap::KeyboardModifiers {
                ctrl: event.modifiers.control,
                shift: event.modifiers.shift,
                alt: event.modifiers.alt,
            };
            match format!("{:?}", event.kind).as_str() {
                "Down" => host.borrow_mut().viewport_input.handle_pointer_down(
                    x,
                    y,
                    pointer_button_to_code(&event.button),
                    modifiers,
                    event.is_touch,
                ),
                "Up" => host.borrow_mut().viewport_input.handle_pointer_up(
                    x,
                    y,
                    pointer_button_to_code(&event.button),
                    modifiers,
                    event.is_touch,
                ),
                "Cancel" => {
                    host.borrow_mut().viewport_input.handle_pointer_cancel();
                }
                "Move" => host.borrow_mut().viewport_input.handle_pointer_move(
                    x,
                    y,
                    modifiers,
                    event.is_touch,
                ),
                _ => {}
            }
            if let Some(window) = window_weak.upgrade() {
                drive_host_tick(&window, &host, &active_timer);
            }
        });
    }

    {
        let host = host.clone();
        let window_weak = window.as_weak();
        let active_timer = active_timer.clone();
        window.on_viewport_scroll_event(move |event| {
            host.borrow_mut().viewport_input.handle_scroll(
                event.delta_x,
                event.delta_y,
                crate::keymap::KeyboardModifiers {
                    ctrl: event.modifiers.control,
                    shift: event.modifiers.shift,
                    alt: event.modifiers.alt,
                },
            );
            if let Some(window) = window_weak.upgrade() {
                drive_host_tick(&window, &host, &active_timer);
            }
        });
    }

    {
        let host = host.clone();
        let window_weak = window.as_weak();
        let active_timer = active_timer.clone();
        window.on_viewport_double_clicked(move |x, y| {
            host.borrow_mut().viewport_input.handle_double_click(x, y);
            if let Some(window) = window_weak.upgrade() {
                drive_host_tick(&window, &host, &active_timer);
            }
        });
    }
}

fn install_rendering_notifier(
    window: &SlintHostWindow,
    host: &Rc<RefCell<SlintHostState>>,
    active_timer: &Rc<Timer>,
) -> Result<(), String> {
    let _window_weak = window.as_weak();
    let _host = host.clone();
    let _active_timer = active_timer.clone();
    window
        .window()
        .set_rendering_notifier(move |state, graphics_api| {
            let Some(window) = _window_weak.upgrade() else {
                return;
            };
            match state {
                slint::RenderingState::RenderingSetup => {
                    if !matches!(graphics_api, slint::GraphicsAPI::WGPU27 { .. }) {
                        log::error!("Slint did not initialize the WGPU 27 renderer");
                    }
                    _host.borrow_mut().viewport_dirty = true;
                }
                slint::RenderingState::BeforeRendering => {
                    if _host.borrow().wake_flag.load(Ordering::Relaxed) {
                        drive_host_tick(&window, &_host, &_active_timer);
                    }
                    _host.borrow_mut().render_viewport_if_needed();
                }
                slint::RenderingState::RenderingTeardown => {
                    _host.borrow_mut().release_viewport_texture();
                }
                slint::RenderingState::AfterRendering => {}
                _ => {}
            }
        })
        .map_err(|error| error.to_string())
}

fn drive_host_tick(
    window: &SlintHostWindow,
    host: &Rc<RefCell<SlintHostState>>,
    active_timer: &Rc<Timer>,
) {
    let outcome = host.borrow_mut().tick(window);
    sync_continuous_timer(window.as_weak(), host, active_timer, outcome.needs_continuous_ticks);
    if outcome.request_redraw {
        window.window().request_redraw();
    }
}

fn sync_continuous_timer(
    window_weak: slint::Weak<SlintHostWindow>,
    host: &Rc<RefCell<SlintHostState>>,
    active_timer: &Rc<Timer>,
    active: bool,
) {
    if host.borrow().continuous_tick_active == active {
        return;
    }
    host.borrow_mut().continuous_tick_active = active;
    if active {
        let window_weak = window_weak.clone();
        let host = host.clone();
        let active_timer = active_timer.clone();
        let timer_for_callback = active_timer.clone();
        active_timer.start(TimerMode::Repeated, Duration::from_millis(16), move || {
            let Some(window) = window_weak.upgrade() else {
                return;
            };
            drive_host_tick(&window, &host, &timer_for_callback);
        });
    } else {
        active_timer.stop();
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

fn pointer_button_to_code(button: &impl std::fmt::Debug) -> i32 {
    match format!("{button:?}").as_str() {
        "Left" => super::slint_bridge::POINTER_BUTTON_PRIMARY,
        "Right" => super::slint_bridge::POINTER_BUTTON_SECONDARY,
        "Middle" => super::slint_bridge::POINTER_BUTTON_MIDDLE,
        _ => super::slint_bridge::POINTER_BUTTON_OTHER,
    }
}

fn apply_brush_adjust_feedback(
    app: &mut SdfApp,
    feedback: &super::backend_frame::ViewportUiFeedback,
) {
    if (feedback.brush_radius_delta != 0.0 || feedback.brush_strength_delta != 0.0)
        && app.doc.sculpt_state.is_active()
    {
        let selected_mode = app.doc.sculpt_state.selected_brush();
        let profile = app.doc.sculpt_state.selected_profile_mut();
        profile.radius = (profile.radius + feedback.brush_radius_delta).clamp(0.05, 2.0);
        profile.strength += feedback.brush_strength_delta;
        profile.clamp_strength_for_mode(selected_mode);
    }
}
