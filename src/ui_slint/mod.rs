#[cfg(all(not(target_arch = "wasm32"), feature = "slint-ui", not(doc)))]
mod winit_host;

#[cfg(all(not(target_arch = "wasm32"), feature = "slint-ui", not(doc)))]
mod panels;

#[cfg(all(not(target_arch = "wasm32"), feature = "slint-ui", not(doc)))]
mod imp {
    use std::cell::RefCell;
    use std::error::Error;
    use std::process::{Child, Command};
    use std::rc::Rc;
    use std::time::{Duration, Instant};

    use crate::core::{AppCore, AppCoreInit, CoreAsyncState, CoreCommand, CoreSelection};
    use crate::gpu::camera::Camera;
    use crate::graph::history::History;
    use crate::graph::scene::{LightType, Scene, SdfPrimitive};
    use crate::sculpt::{ActiveTool, SculptState};
    use crate::settings::Settings;
    use crate::ui_slint::panels::build_panel_read_model;

    slint::include_modules!();

    #[derive(Default)]
    struct ViewportProcessController {
        child: Option<Child>,
    }

    impl ViewportProcessController {
        fn poll_running(&mut self) -> bool {
            let Some(child) = self.child.as_mut() else {
                return false;
            };

            match child.try_wait() {
                Ok(Some(_status)) => {
                    self.child = None;
                    false
                }
                Ok(None) => true,
                Err(_error) => {
                    self.child = None;
                    false
                }
            }
        }

        fn open(&mut self) -> Result<String, String> {
            if self.poll_running() {
                return Ok("Viewport window already running".to_string());
            }

            let executable_path = std::env::current_exe()
                .map_err(|error| format!("Failed to resolve current executable: {error}"))?;

            let child = Command::new(executable_path)
                .arg("--frontend")
                .arg("slint")
                .env("SDF_SLINT_HOST_MODE", "viewport")
                .spawn()
                .map_err(|error| format!("Failed to launch viewport window: {error}"))?;

            self.child = Some(child);
            Ok("Opened viewport window".to_string())
        }

        fn close(&mut self) -> Result<String, String> {
            let Some(mut child) = self.child.take() else {
                return Ok("Viewport window is not running".to_string());
            };

            match child.try_wait() {
                Ok(Some(_status)) => Ok("Viewport window already closed".to_string()),
                Ok(None) => {
                    child
                        .kill()
                        .map_err(|error| format!("Failed to stop viewport window: {error}"))?;
                    let _ = child.wait();
                    Ok("Closed viewport window".to_string())
                }
                Err(error) => Err(format!("Failed to query viewport window state: {error}")),
            }
        }
    }

    fn apply_snapshot(
        app: &AppShell,
        core: &AppCore,
        settings: &Settings,
        fps: f32,
        viewport_running: bool,
    ) {
        let snap = core.snapshot();
        let primary = snap
            .selected_primary
            .map(|id| id.to_string())
            .unwrap_or_else(|| "none".to_string());

        app.set_fpsText(format!("FPS: {:.1}", fps).into());
        app.set_selectionText(
            format!("Selection: {} ({} selected)", primary, snap.selected_count).into(),
        );
        app.set_sceneText(
            format!(
                "Nodes: {} (hidden: {}) | Tool: {}",
                snap.node_count, snap.hidden_node_count, snap.active_tool
            )
            .into(),
        );
        app.set_undoText(format!("Undo: {} / Redo: {}", snap.undo_count, snap.redo_count).into());
        app.set_debugEnabled(snap.show_debug);
        app.set_settingsOpen(snap.show_settings);
        app.set_viewportWindowRunning(viewport_running);
        app.set_viewportStatusText(
            if viewport_running {
                "Viewport window: running (external host)"
            } else {
                "Viewport window: stopped"
            }
            .into(),
        );

        let scene_tree_filter = app.get_sceneTreeFilterText().to_string();
        let panel_model = build_panel_read_model(core, settings, &scene_tree_filter);
        app.set_sceneTreePanelText(panel_model.scene_tree_text.into());
        app.set_historyPanelText(panel_model.history_text.into());
        app.set_lightsPanelText(panel_model.lights_text.into());
        app.set_renderSettingsPanelText(panel_model.render_settings_text.into());
        app.set_sceneStatsPanelText(panel_model.scene_stats_text.into());
        app.set_propertiesPanelText(panel_model.properties_text.into());
    }

    fn show_toast(
        app_weak: &slint::Weak<AppShell>,
        hide_timer: &Rc<slint::Timer>,
        message: impl Into<String>,
    ) {
        if let Some(app) = app_weak.upgrade() {
            app.set_toastText(message.into().into());
            app.set_toastVisible(true);

            let hide_target = app_weak.clone();
            hide_timer.start(
                slint::TimerMode::SingleShot,
                Duration::from_millis(2300),
                move || {
                    if let Some(app) = hide_target.upgrade() {
                        app.set_toastVisible(false);
                    }
                },
            );
        }
    }

    pub fn run(settings: &Settings) -> Result<(), Box<dyn Error>> {
        let host_mode = std::env::var("SDF_SLINT_HOST_MODE").ok();
        let use_viewport_host = host_mode
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case("viewport"));
        if use_viewport_host {
            return crate::ui_slint::winit_host::run(settings);
        }
        slint::BackendSelector::new().backend_name("winit".into()).select()?;

        let app = AppShell::new()?;
        let app_weak = app.as_weak();
        let shell_settings = settings.clone();

        let core = Rc::new(RefCell::new(AppCore::from_init(AppCoreInit {
            scene: Scene::new(),
            history: History::new(),
            camera: Camera::default(),
            selection: CoreSelection::default(),
            active_tool: ActiveTool::Select,
            sculpt_state: SculptState::Inactive,
            async_state: CoreAsyncState::default(),
            soloed_light: None,
            show_debug: false,
            show_settings: false,
        })));

        let started = Instant::now();
        let tick_count = Rc::new(RefCell::new(0_u64));
        let hide_timer = Rc::new(slint::Timer::default());
        let viewport_controller = Rc::new(RefCell::new(ViewportProcessController::default()));

        {
            let core_ref = core.borrow();
            let viewport_running = viewport_controller.borrow_mut().poll_running();
            apply_snapshot(&app, &core_ref, &shell_settings, 0.0, viewport_running);
        }

        let dispatch: Rc<dyn Fn(CoreCommand)> = {
            let core = core.clone();
            let app_weak = app_weak.clone();
            let hide_timer = hide_timer.clone();
            let tick_count = tick_count.clone();
            let shell_settings = shell_settings.clone();
            let viewport_controller = viewport_controller.clone();
            Rc::new(move |command: CoreCommand| {
                let mut core_mut = core.borrow_mut();
                let result = core_mut.apply_command(command);
                let ticks = (*tick_count.borrow()).max(1) as f32;
                let elapsed = started.elapsed().as_secs_f32().max(0.001);
                let fps = ticks / elapsed;
                if let Some(app) = app_weak.upgrade() {
                    let viewport_running = viewport_controller.borrow_mut().poll_running();
                    apply_snapshot(&app, &core_mut, &shell_settings, fps, viewport_running);
                }
                if let Some(message) = result.toast_message {
                    show_toast(&app_weak, &hide_timer, message);
                }
            })
        };

        {
            let dispatch = dispatch.clone();
            app.on_fileNew(move || dispatch(CoreCommand::NewScene));
        }
        {
            let app_weak = app_weak.clone();
            let hide_timer = hide_timer.clone();
            app.on_fileOpen(move || {
                show_toast(&app_weak, &hide_timer, "Open project is pending Slint shell integration");
            });
        }
        {
            let app_weak = app_weak.clone();
            let hide_timer = hide_timer.clone();
            app.on_fileSave(move || {
                show_toast(&app_weak, &hide_timer, "Save project is pending Slint shell integration");
            });
        }
        {
            let dispatch = dispatch.clone();
            app.on_addSphere(move || dispatch(CoreCommand::CreatePrimitive(SdfPrimitive::Sphere)));
        }
        {
            let dispatch = dispatch.clone();
            app.on_addPointLight(move || dispatch(CoreCommand::CreateLight(LightType::Point)));
        }
        {
            let dispatch = dispatch.clone();
            app.on_addSpotLight(move || dispatch(CoreCommand::CreateLight(LightType::Spot)));
        }
        {
            let dispatch = dispatch.clone();
            app.on_addDirectionalLight(move || {
                dispatch(CoreCommand::CreateLight(LightType::Directional))
            });
        }
        {
            let dispatch = dispatch.clone();
            app.on_addAmbientLight(move || dispatch(CoreCommand::CreateLight(LightType::Ambient)));
        }
        {
            let dispatch = dispatch.clone();
            app.on_soloSelectedLight(move || dispatch(CoreCommand::SoloSelectedLight));
        }
        {
            let dispatch = dispatch.clone();
            app.on_clearSoloLight(move || dispatch(CoreCommand::ClearSoloLight));
        }
        {
            let dispatch = dispatch.clone();
            app.on_deleteSelected(move || dispatch(CoreCommand::DeleteSelected));
        }
        {
            let dispatch = dispatch.clone();
            app.on_toggleSelectedVisibility(move || dispatch(CoreCommand::ToggleSelectedVisibility));
        }
        {
            let dispatch = dispatch.clone();
            app.on_focusSelected(move || dispatch(CoreCommand::FocusSelected));
        }
        {
            let dispatch = dispatch.clone();
            let app_weak = app_weak.clone();
            app.on_renameSelected(move || {
                if let Some(app) = app_weak.upgrade() {
                    let new_name = app.get_renameSelectedInput().to_string();
                    dispatch(CoreCommand::RenameSelected(new_name));
                }
            });
        }
        {
            let dispatch = dispatch.clone();
            app.on_editUndo(move || dispatch(CoreCommand::Undo));
        }
        {
            let dispatch = dispatch.clone();
            app.on_editRedo(move || dispatch(CoreCommand::Redo));
        }
        {
            let dispatch = dispatch.clone();
            app.on_toolSelect(move || dispatch(CoreCommand::SetActiveTool(ActiveTool::Select)));
        }
        {
            let dispatch = dispatch.clone();
            app.on_toolSculpt(move || dispatch(CoreCommand::SetActiveTool(ActiveTool::Sculpt)));
        }
        {
            let dispatch = dispatch.clone();
            app.on_toggleDebug(move || dispatch(CoreCommand::ToggleDebug));
        }
        {
            let dispatch = dispatch.clone();
            app.on_toggleSettings(move || dispatch(CoreCommand::ToggleSettings));
        }
        {
            let app_weak = app_weak.clone();
            let hide_timer = hide_timer.clone();
            let viewport_controller = viewport_controller.clone();
            app.on_openViewportWindow(move || {
                let message = viewport_controller.borrow_mut().open();
                match message {
                    Ok(message) => show_toast(&app_weak, &hide_timer, message),
                    Err(error_message) => show_toast(&app_weak, &hide_timer, error_message),
                }
            });
        }
        {
            let app_weak = app_weak.clone();
            let hide_timer = hide_timer.clone();
            let viewport_controller = viewport_controller.clone();
            app.on_closeViewportWindow(move || {
                let message = viewport_controller.borrow_mut().close();
                match message {
                    Ok(message) => show_toast(&app_weak, &hide_timer, message),
                    Err(error_message) => show_toast(&app_weak, &hide_timer, error_message),
                }
            });
        }

        let snapshot_timer = slint::Timer::default();
        {
            let core = core.clone();
            let app_weak = app_weak.clone();
            let tick_count = tick_count.clone();
            let shell_settings = shell_settings.clone();
            let viewport_controller = viewport_controller.clone();
            snapshot_timer.start(
                slint::TimerMode::Repeated,
                Duration::from_millis(100),
                move || {
                    *tick_count.borrow_mut() += 1;
                    let ticks = (*tick_count.borrow()) as f32;
                    let elapsed = started.elapsed().as_secs_f32().max(0.001);
                    let fps = ticks / elapsed;

                    if let Some(app) = app_weak.upgrade() {
                        let core_ref = core.borrow();
                        let viewport_running = viewport_controller.borrow_mut().poll_running();
                        apply_snapshot(&app, &core_ref, &shell_settings, fps, viewport_running);
                    }
                },
            );
        }

        show_toast(
            &app_weak,
            &hide_timer,
            format!(
                "Slint shell active (preferred frontend: {:?})",
                settings.preferred_frontend
            ),
        );

        app.run()?;
        if let Err(error) = viewport_controller.borrow_mut().close() {
            eprintln!("Failed to close viewport window on shell exit: {error}");
        }
        Ok(())
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "slint-ui", not(doc)))]
pub use imp::run as run_slint_shell;

#[cfg(not(all(not(target_arch = "wasm32"), feature = "slint-ui", not(doc))))]
pub fn run_slint_shell(_settings: &crate::settings::Settings) -> Result<(), Box<dyn std::error::Error>> {
    Err(std::io::Error::new(
        std::io::ErrorKind::Unsupported,
        "slint-ui feature is not enabled",
    )
    .into())
}

