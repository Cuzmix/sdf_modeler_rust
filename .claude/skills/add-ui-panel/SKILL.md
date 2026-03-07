---
name: add-ui-panel
description: Add a new dockable UI panel, tab, inspector, or settings panel to the application. Trigger when creating new panels, tool windows, or any new dockable UI tab.
---

# Add UI Panel

Step-by-step guide to add a new dockable panel to the egui dock layout.

## Step 1: Create Panel Module (`src/ui/<panel_name>.rs`)

Create a new file with a `draw()` function:

```rust
use eframe::egui;
use crate::app::actions::ActionSink;
use crate::graph::scene::Scene;

pub fn draw(
    ui: &mut egui::Ui,
    scene: &mut Scene,          // If reading/writing scene data
    actions: &mut ActionSink,   // If pushing structural actions
) {
    // Panel content using egui widgets
    ui.heading("My Panel");

    if ui.button("Do Something").clicked() {
        actions.push(Action::MyAction);
    }
}
```

### Parameter conventions (follow existing panels):

| Need | Parameters | Example Panel |
|------|-----------|---------------|
| Read-only display | `&Scene` | `history_panel` |
| Data-level edits | `&mut Scene` + `&mut ActionSink` | `properties` |
| Settings | `&mut Settings` + `&mut ActionSink` | `render_settings` |
| Selection | `&mut Option<NodeId>` + `&mut HashSet<NodeId>` | `scene_tree` |
| Sculpt state | `&mut SculptState` | `brush_settings` |

**Key rule**: Data-level edits (sliders, colors) mutate `&mut Scene` directly for zero-latency. Structural changes (add/delete/reparent) push to `ActionSink`.

## Step 2: Module Declaration (`src/ui/mod.rs`)

Add the module declaration:

```rust
pub mod <panel_name>;
```

## Step 3: Dock Integration (`src/ui/dock.rs`)

### 3a. Tab Enum

Add variant to `Tab` enum:

```rust
pub enum Tab {
    // ... existing variants
    MyPanel,
}
```

### 3b. ALL Array

```rust
pub const ALL: &[Tab] = &[
    // ... existing entries
    Tab::MyPanel,
];
```

### 3c. label() Method

```rust
Tab::MyPanel => "My Panel",
```

### 3d. title() Method (in `SdfTabViewer`)

```rust
Tab::MyPanel => "My Panel".into(),
```

### 3e. ui() Dispatcher (in `SdfTabViewer`)

Add the match arm that calls your draw function. Import the module at the top of dock.rs if needed.

**Simple panel:**
```rust
Tab::MyPanel => {
    my_panel::draw(ui, self.scene, self.actions);
}
```

**Panel needing extra context:**
```rust
Tab::MyPanel => {
    crate::ui::my_panel::draw(
        ui,
        self.scene,
        &mut self.node_graph_state.selected,
        self.actions,
    );
}
```

### 3f. SdfTabViewer Fields (if needed)

If your panel needs state not already in `SdfTabViewer`, add a field to the viewer struct and thread it from `SdfApp`.

## Step 4: Default Layout (Optional)

Add the tab to a workspace preset in `create_dock_state()`, `create_dock_sculpting()`, or `create_dock_rendering()`.

## Step 5: If Adding New Actions

When your panel needs to trigger structural mutations:

### 5a. Action Variant (`src/app/actions.rs`)

```rust
pub enum Action {
    // ... existing variants
    MyAction(SomeData),
}
```

### 5b. Handler (`src/app/action_handler.rs`)

Add match arm in `process_actions()`:

```rust
Action::MyAction(data) => {
    // Mutate self.doc.scene, self.ui, etc.
    self.gpu.buffer_dirty = true; // If scene data changed
}
```

## Step 6: Verify

```bash
cargo check && cargo clippy -- -D warnings && cargo test && cargo build
```

## Checklist

- [ ] Panel module created at `src/ui/<panel_name>.rs` with `draw()` function
- [ ] Module declared in `src/ui/mod.rs`
- [ ] `Tab` enum variant added + ALL + label() + title() + ui() dispatcher
- [ ] (Optional) New Action variant + handler if structural mutations needed
- [ ] (Optional) Added to default dock layout
- [ ] All 4 verification steps pass
