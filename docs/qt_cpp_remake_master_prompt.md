# SDF Modeler Qt/C++ Remake Master Prompt And Build Plan

## 1) Project Intent

### Goal
Rebuild the current SDF Modeler as a native Qt desktop application using C++ and Qt Quick/QML, with a custom viewport renderer integrated into the Qt scene graph.

### Fixed Technology Decisions
- UI framework: Qt Quick + QML
- Language/runtime: C++ (full rewrite, no Rust core bridge)
- Viewport integration: custom Qt Quick render node path (`QSGRenderNode` + `QQuickWindow` render hooks)
- Platform target: cross-platform desktop from day one (Windows, macOS, Linux)
- Delivery strategy: MVP first, then feature parity phases

### Quality Bar (Non-Negotiable)
- No regressions in viewport responsiveness or sculpt interaction continuity.
- Keep architecture modular, toolkit boundaries explicit, and behavior testable.
- Preserve real-time editing feel under interactive workloads.
- Keep user-facing workflows simple; keep engine complexity behind typed interfaces.

### Non-Goals For MVP
- Mobile-first UX and Android/iOS packaging.
- Full advanced workspace/docking parity on first release.
- Every export/import format in phase 1 (OBJ required first).
- Experimental rendering modes not required for core workflow.

## 2) Master Implementation Prompt (Paste Into Your Other Project)

```md
You are implementing a production-grade Qt/C++ rewrite of an SDF modeling desktop app.

You must follow these fixed constraints:
- Qt Quick + QML for shell UI.
- C++ only for core/engine/runtime.
- Custom viewport renderer in Qt Quick via QSGRenderNode/QQuickWindow hooks.
- Cross-platform desktop support (Windows/macOS/Linux) from the start.
- MVP first, parity later.

Architectural rules:
- Structural state changes must go through Action dispatch + reducer path only.
- Keep toolkit-agnostic frame logic in core modules, isolated from QML types.
- Keep input translation in a dedicated bridge that outputs FrameInputSnapshot and ViewportInputSnapshot.
- Keep renderer lifecycle behind RendererBridge.
- Keep async operations behind AsyncJob interfaces; UI must never block.
- Keep settings versioned and backward compatible.

Build these subsystem boundaries:
- app/core: state, reducer, controllers, backend frame lifecycle
- ui bridge: qml callbacks/events -> typed actions/snapshots
- presenter: core state -> QML view models
- renderer: GPU device/context, pipelines, picking, viewport texture lifecycle
- graph/domain: scene graph, history, presented objects, sculpt voxel data
- platform services: dialogs, recent files, paths, settings storage, screenshot/export destinations

MVP capabilities:
- scene create/edit basics: primitives, select, transform, duplicate/delete, undo/redo
- viewport nav: orbit/pan/zoom, frame-all, focus-selected, ortho toggle
- basic render controls: shading mode cycle, shadows toggle, AO toggle, screenshot
- sculpt entry and key brushes with continuous drag responsiveness
- project load/save and OBJ export

Required improvements over legacy patterns:
- strict action dispatcher + reducer audit trail
- latency telemetry (pick latency, stroke samples/frame, GPU dispatch count/frame)
- deterministic interaction replay tests for gizmo/sculpt
- explicit renderer capability abstraction per platform
- shader/pipeline cache strategy to reduce hitching
- clear UI split between high-frequency sculpt controls and advanced low-frequency controls
- failure-safe async flows for picking/import/export/screenshot

Responsiveness guardrails (must pass):
- fast circular sculpt strokes remain continuous
- grab continues off-mesh until release
- toggling shadows/AO during sculpt does not introduce stepping
- no visible latency spike at stroke start or during continuous drag
- multi-select rotate drag does not snap back between frames

Required deliverables:
- complete module scaffold
- typed C++ interfaces and DTOs
- QML shell with presenter-driven state updates
- automated test suite + manual validation checklist
- CI for Windows/macOS/Linux build + test

Do not leave TODO stubs or partial placeholders. Implement complete vertical slices per milestone.
```

## 3) Architecture Blueprint

### 3.1 Layered Runtime Model

1. **QML UI Layer**
- Declarative shell composition (top bar, scene panel, inspector, utility panel, viewport host item).
- Emits domain-level intents, not direct engine mutations.

2. **Qt Frontend Host Layer**
- Owns `QQmlApplicationEngine`, root context wiring, tick/redraw scheduling, viewport surface lifecycle.
- Installs callback bindings and connects presenter outputs to QML properties/models.

3. **Bridge Layer**
- Converts Qt input/events to toolkit-neutral DTO snapshots.
- Converts QML command callbacks to typed `Action` dispatches.

4. **App/Core Layer**
- Owns state, reducers, controllers, backend frame lifecycle, and command processing.
- Produces immutable frame snapshots for presenter consumption.

5. **Domain + GPU Layer**
- Scene graph, history, sculpt voxel domain, renderer, picking, shaders/pipeline cache.

### 3.2 Mandatory Dependency Direction

- `qml/` depends on view-model contracts only.
- `frontend/` may depend on Qt + presenter + bridge.
- `bridge/` depends on Qt event types + DTO contracts only.
- `app/core/` must not depend on Qt types.
- `renderer/` must not depend on QML types.
- `domain/` must not depend on Qt types.
- `platform/` exposes interfaces consumed by app/core and frontend.

### 3.3 Suggested C++ Module Layout

```text
src/
  app/
    Action.h
    ActionDispatcher.h
    ActionReducer.h
    AppState.h
    BackendFrame.h
    Controllers/
    Presenter/
  bridge/
    FrameInputSnapshot.h
    ViewportInputSnapshot.h
    QtInputBridge.h
    QtActionBridge.h
  frontend/
    QtHostRuntime.h
    QtHostState.h
    QmlBindings/
    QmlCallbacks/
    ViewportHostItem.{h,cpp}
  domain/
    scene/
    history/
    sculpt/
    gizmo/
  renderer/
    RendererBridge.h
    RenderDevice.h
    PipelineCache.h
    PickService.h
    ViewportCompositor.h
  async/
    AsyncJob.h
    JobScheduler.h
  platform/
    PlatformService.h
    DesktopPlatformService.{h,cpp}
  settings/
    SettingsSchema.h
    SettingsStore.h
qml/
  MainWindow.qml
  panels/
  viewport/
tests/
  unit/
  integration/
  replay/
```

### 3.4 Rust To Qt/C++ Responsibility Mapping

| Current Rust Area | Qt/C++ Rewrite Target |
|---|---|
| `src/app/actions.rs` | `src/app/Action.h` + typed payload headers |
| `src/app/action_handler.rs` | `src/app/ActionReducer.{h,cpp}` |
| `src/app/backend_frame.rs` | `src/app/BackendFrame.{h,cpp}` |
| `src/app/frontend_models.rs` | `src/app/Presenter/FrontendPresenter.{h,cpp}` |
| `src/app/slint_bridge.rs` | `src/bridge/QtInputBridge.{h,cpp}` + `QtActionBridge.{h,cpp}` |
| `src/app/slint_frontend/*` | `src/frontend/QtHostRuntime.*`, `QmlBindings/*`, `QmlCallbacks/*` |
| `src/app/slint_ui/*.slint` | `qml/**/*.qml` components |
| `src/graph/*` | `src/domain/scene/*` + `src/domain/history/*` |
| `src/sculpt.rs` + `src/app/sculpting.rs` | `src/domain/sculpt/*` + `src/app/Controllers/SculptController.*` |
| `src/gpu/*` + `src/viewport/*` | `src/renderer/*` |
| `src/settings.rs` | `src/settings/SettingsSchema.*` + `SettingsStore.*` |
| `src/desktop_dialogs.rs` + path helpers | `src/platform/DesktopPlatformService.*` |

## 4) Key Interface Contracts (Decision-Complete)

### 4.1 `Action` Structural Mutation Contract

```cpp
// Action.h
enum class ActionType {
    NewScene,
    OpenProject,
    SaveProject,
    SelectNode,
    ToggleSelection,
    DeleteSelected,
    DuplicateSelected,
    Undo,
    Redo,
    SetInteractionMode,
    SetGizmoMode,
    SetBrushMode,
    EnterSculptMode,
    CommitSculptConvert,
    CreatePrimitive,
    CreateOperation,
    CreateModifier,
    CreateLight,
    RenameNode,
    ToggleVisibility,
    ToggleLock,
    RequestExportObj,
    ImportMesh,
    ToggleShadows,
    ToggleAo,
    CycleShadingMode,
    ToggleOrtho,
    FocusSelected,
    FrameAll,
    TakeScreenshot
};

struct ActionPayload {
    // Use std::variant of typed payload structs in implementation.
};

struct Action {
    ActionType type;
    ActionPayload payload;
    uint64_t sequenceId;
    uint64_t timestampNs;
};
```

Rules:
- Every structural mutation goes through `ActionDispatcher -> ActionReducer`.
- Reducer is deterministic and side-effect free.
- Side effects (file IO, exports, imports, async picks) are requested by reducer outputs and executed by effect handlers.
- Persist an action audit trail for debugging and replay.

### 4.2 Input Snapshot DTOs

```cpp
// FrameInputSnapshot.h
struct FrameInputSnapshot {
    bool undoRequested;
    bool redoRequested;
    bool frameAllRequested;
    bool focusSelectedRequested;
    bool viewportHovered;
    bool viewportFocused;
    bool stylusActive;
    bool touchActive;
    uint64_t frameNumber;
    double dtSeconds;
};

// ViewportInputSnapshot.h
enum class PointerPhase { Down, Move, Up, Cancel };
enum class PointerButton { Other, Primary, Secondary, Middle };

struct Modifiers {
    bool ctrl;
    bool shift;
    bool alt;
};

struct PointerEvent {
    PointerPhase phase;
    PointerButton button;
    float x;
    float y;
    Modifiers modifiers;
    bool isTouch;
};

struct ScrollEvent {
    float deltaX;
    float deltaY;
    Modifiers modifiers;
};

struct DoubleClickEvent {
    float x;
    float y;
};

struct ViewportInputSnapshot {
    std::optional<PointerEvent> pointer;
    std::optional<ScrollEvent> scroll;
    std::optional<DoubleClickEvent> doubleClick;
};
```

### 4.3 `BackendFrame` Lifecycle

```cpp
class BackendFrame {
public:
    virtual ~BackendFrame() = default;
    virtual void runPreUi(const FrameInputSnapshot& frameInput,
                          const ViewportInputSnapshot& viewportInput) = 0;
    virtual void runPostUi() = 0;
};
```

Rules:
- `runPreUi` executes interaction, camera, pick/sculpt orchestration, and presenter snapshot generation.
- `runPostUi` executes deferred effects, async completion handling, and frame-level housekeeping.

### 4.4 `FrontendPresenter` Contract

```cpp
class FrontendPresenter {
public:
    virtual ~FrontendPresenter() = default;
    virtual ShellSnapshot buildSnapshot(const AppState& state) const = 0;
};
```

Rules:
- Presenter is read-only over `AppState`.
- QML binds to presenter snapshot models only; no direct domain graph access.

### 4.5 `RendererBridge` Contract

```cpp
class RendererBridge {
public:
    virtual ~RendererBridge() = default;
    virtual void initialize(void* nativeWindowHandle) = 0;
    virtual void resize(uint32_t width, uint32_t height, float devicePixelRatio) = 0;
    virtual void renderFrame(const RenderFrameInput& input) = 0;
    virtual PickRequestId requestPick(const PickRay& ray) = 0;
    virtual std::optional<PickResult> pollPick(PickRequestId id) = 0;
    virtual ViewportTextureHandle currentTexture() const = 0;
};
```

Rules:
- Must support async picking without blocking UI/render thread.
- Must provide texture/frame output compatible with Qt Quick scene graph composition.
- Must expose capability flags for backend differences.

### 4.6 `AsyncJob` Contract

```cpp
enum class AsyncJobType { MeshImport, MeshExportObj, SculptPick, Screenshot, SettingsIo };

struct AsyncJobRequest { AsyncJobType type; /* typed payload */ };
struct AsyncJobResult { AsyncJobType type; bool success; std::string message; /* payload */ };

class AsyncJob {
public:
    virtual ~AsyncJob() = default;
    virtual AsyncJobId submit(const AsyncJobRequest& request) = 0;
    virtual std::optional<AsyncJobResult> poll(AsyncJobId id) = 0;
    virtual void cancel(AsyncJobId id) = 0;
};
```

Rules:
- UI thread never blocks on job completion.
- All completion results flow back as typed reducer/effect events.
- Job failures are surfaced as non-blocking user notifications.

### 4.7 Settings + Serialization Contract

```cpp
struct SettingsFileEnvelope {
    uint32_t schemaVersion;
    SettingsData data;
};
```

Rules:
- Version every settings file.
- Add migration path from `N -> N+1` for all schema changes.
- Preserve unknown fields where practical for forward compatibility.
- Keep recent-files list bounded and de-duplicated.

### 4.8 `PlatformService` Contract

```cpp
class PlatformService {
public:
    virtual ~PlatformService() = default;
    virtual std::optional<std::string> openProjectDialog() = 0;
    virtual std::optional<std::string> saveProjectDialog() = 0;
    virtual std::optional<std::string> openMeshDialog() = 0;
    virtual std::optional<std::string> saveObjDialog() = 0;
    virtual std::optional<std::string> saveScreenshotDialog() = 0;
    virtual std::string appSettingsPath() const = 0;
    virtual std::string appDataPath() const = 0;
};
```

## 5) Feature Parity Matrix

| Capability | Current Rust App | MVP (Phase 1) | Parity Phase 2 | Parity Phase 3 |
|---|---|---|---|---|
| Scene primitives + selection + transforms | Yes | Yes | - | - |
| Undo/redo history | Yes | Yes | - | - |
| Core viewport navigation | Yes | Yes | - | - |
| Shading cycle + shadows/AO toggles | Yes | Yes | - | - |
| Screenshot | Yes | Yes | - | - |
| Sculpt entry + key brushes | Yes | Yes | Improve polish | Advanced brush parity |
| Async pick + predictive fallback behavior | Yes | Yes | Optimize | Add telemetry auto-tuning |
| Off-mesh grab continuation | Yes | Yes | - | - |
| Voxel-aware interpolation + clamped deltas | Yes | Yes | - | - |
| Taubin smoothing behavior | Yes | Yes | - | - |
| OBJ export | Yes | Yes | - | - |
| STL/PLY/GLB/USDA export | Yes | No | Yes | - |
| Reference images tooling | Yes | Optional | Yes | - |
| Advanced workspace panels/docking | Yes | Basic | Improved | Full parity |
| Lighting presets, linking, cookies | Yes | Minimal | Yes | Full parity |
| Full import dialog and workflows | Yes | Basic | Yes | Full parity |

## 6) Improvement Opportunities (What To Change vs Preserve)

### Preserve
- Explicit action/reducer mutation model.
- Toolkit-neutral frame lifecycle.
- Async/non-blocking pick path and sculpt continuity guardrails.
- Separation of renderer, graph domain, and UI shell.

### Change/Improve
- Replace ad-hoc callback growth with typed dispatcher bus and centralized command routing.
- Add deterministic action log and replay harness for input/regression debugging.
- Add runtime telemetry overlay and trace export for latency diagnostics.
- Add renderer capability abstraction (`supportsTimestampQuery`, `supportsStorageImage`, `maxWorkgroupSize`, etc.).
- Add persistent shader/pipeline cache with invalidation keys per shader signature + device profile.
- Split shell UI into high-frequency sculpt controls and low-frequency advanced panels.
- Make async failure handling first-class with retry/timeout policy per job type.

## 7) Execution Phases With Done Criteria

### Phase 0: Foundation And Skeleton
- Set up CMake + Qt project structure and CI for Windows/macOS/Linux.
- Implement module scaffolding, action dispatcher, reducer skeleton, and typed DTOs.
- Implement minimal QML host window + viewport host item + render loop wiring.

Done criteria:
- App boots on all 3 platforms.
- Reducer unit tests and snapshot contract tests pass.
- Renderer presents a clear test frame in viewport panel.

### Phase 1 (MVP): End-To-End Core Workflow
- Implement scene basics: create primitive, select, transform, duplicate, delete, undo/redo.
- Implement viewport nav: orbit/pan/zoom, frame all, focus selected, ortho toggle.
- Implement render controls: shading cycle, shadows toggle, AO toggle, screenshot.
- Implement sculpt entry and key brush flows with responsive continuous drag behavior.
- Implement project save/load and OBJ export.

Done criteria:
- MVP workflows complete without blocking UI.
- Manual sculpt responsiveness checks pass.
- CI green on Windows/macOS/Linux for build + tests.

### Phase 2: Parity Expansion
- Expand export formats (STL/PLY/GLB/USDA).
- Add reference image workflows, richer inspector controls, and expanded render settings.
- Extend light workflows (linking, solo, cookies, presets).
- Harden async job orchestration and cancellation.

Done criteria:
- Feature parity for major daily workflows.
- Import/export and settings migration tests pass.

### Phase 3: Performance + Expert UX
- Pipeline/shader cache optimization and hitch reduction.
- Advanced workspace docking/panel persistence parity.
- Telemetry-driven tuning for sculpt/pick latency.
- Deterministic replay pack for gizmo + sculpt regressions.

Done criteria:
- Interaction latency and hitch budgets met.
- Replay suite stable across platforms.

## 8) Acceptance Criteria (Objective Gates)

### Functional
- Structural mutations occur only via action dispatch + reducer.
- QML never mutates domain state directly.
- Project save/load preserves scene + settings + recent files behavior.
- MVP tools operate end-to-end with no dead UI paths.

### Performance/Responsiveness
- No UI thread blocking during pick/import/export/screenshot tasks.
- Sculpt stroke continuity preserved during rapid drag.
- Off-mesh grab continuation preserved until release.
- Shadows/AO toggles do not introduce stepping artifacts during active sculpt.
- MVP interaction frame time target at 1080p: <= 16.7ms median on reference GPU.
- MVP contiguous sculpt input stall target: no stall > 50ms during continuous drag.
- MVP async pick latency target during active sculpt: <= 2 frames median.

### Reliability
- Reducer tests cover all MVP action families.
- Serialization roundtrip tests pass for current schema.
- Schema migration tests pass for at least previous version input.
- Integration tests validate bridge -> action dispatch -> state update path.

### Cross-Platform
- CI builds and runs automated tests on Windows/macOS/Linux.
- Platform services produce correct native paths/dialog behavior per OS.

## 9) Test Plan

### Automated
- **Build/CI**: compile and run tests on Windows/macOS/Linux.
- **Unit tests**:
  - reducer transitions
  - settings schema serialization + migration
  - presenter snapshot mapping
  - key input decoding and modifier mapping
- **Integration tests**:
  - QML callback -> action dispatch -> state transition
  - viewport input bridge -> interaction controller updates
  - async job completion -> effect event -> UI status update
- **Replay tests**:
  - deterministic gizmo rotation drag path
  - deterministic sculpt stroke path with known expected outputs

### Manual Non-Regression Scenarios
1. Fast circular Add/Carve stroke remains continuous without stepping.
2. Grab drag continues while pointer moves off geometry, until release.
3. Toggle Shadows/AO during sculpt without introducing stepping.
4. No latency spike at stroke start or during continuous drag.
5. Multi-select rotate drag does not snap back between frames.

## 10) Implementation Defaults And Assumptions

- Full C++ rewrite is intentional; no Rust-core interoperability layer is planned.
- Qt Quick/QML is the only shell UI system for this rewrite.
- Renderer integration is in-process via Qt scene graph, not a separate floating native renderer window.
- Cross-platform desktop parity starts at architecture level, not as a later retrofit.
- MVP ships smaller scope with strict stability/latency gates before feature expansion.
- This document is the source of truth for initial implementation decisions in the new Qt project.
