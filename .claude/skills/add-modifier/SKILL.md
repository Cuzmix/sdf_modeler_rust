---
name: add-modifier
description: Add a new SDF modifier to the modeler. Trigger when creating modifiers like twist, bend, taper, round, onion, elongate, mirror, repeat, offset, noise, or any new domain deformation or distance modifier.
---

# Add SDF Modifier

Step-by-step guide to add a new SDF modifier. Modifiers come in two categories:

- **Point modifiers** (`is_point_modifier() = true`): Modify the point `p` before child SDF evaluation (domain deformations like twist, bend, mirror, repeat). These integrate into the transform chain.
- **Distance modifiers** (`is_point_modifier() = false`): Modify the distance after child SDF evaluation (like round, onion, offset). These wrap the child's result.

**Choose your category first — it determines which codegen path to use.**

## Step 1: Scene Graph Enum (`src/graph/scene.rs`)

Add the variant to `ModifierKind` enum and update **all 8 match arms**:

1. **Enum variant**: Add to `ModifierKind` (group with similar modifiers via comments)
2. **`ALL` array**: Add to the const array
3. **`base_name()`**: Display name (e.g., `"Revolve"`)
4. **`badge()`**: 3-4 char label in brackets (e.g., `"[Rev]"`)
5. **`default_value()`**: Initial `Vec3` parameters — `x` is typically the primary param
6. **`default_extra()`**: Secondary parameters (usually `Vec3::ZERO` unless needed)
7. **`gpu_type_id()`**: Next sequential float (currently max is `41.0` for Noise → use `42.0`)
8. **`is_point_modifier()`**: `true` for domain deformations, `false` for distance modifiers

## Step 2: WGSL Shader Function (`src/shaders/modifiers.wgsl`)

### Point Modifier

Add a function that transforms the point:

```wgsl
fn <name>_point(p: vec3f, <params>) -> vec3f {
    // Transform p and return new point
}
```

Parameters come from `nodes[idx].position.xyz` (maps to `value` field) and `nodes[idx].rotation.xyz` (maps to `extra` field).

### Distance Modifier

Distance modifiers are typically inline in codegen (no WGSL function needed). If the operation is complex enough to warrant a function, add it to `modifiers.wgsl`.

## Step 3: Codegen (`src/gpu/codegen.rs`)

### Point Modifier — `emit_transform_chain()`

Add a match arm in the `emit_transform_chain()` function:

```rust
ChainEntry::Modifier(ModifierKind::YourMod) => {
    lines.push(format!(
        "    let {new_var} = <name>_point({current_var}, nodes[{idx}].position.x);"
    ));
}
```

Parameters from the node buffer:
- `nodes[{idx}].position.xyz` → `value` field (primary params)
- `nodes[{idx}].rotation.xyz` → `extra` field (secondary params)
- `nodes[{idx}].position.x` → single scalar param

### Distance Modifier — `emit_node_wgsl()`

Add a match arm in the `NodeData::Modifier` block of `emit_node_wgsl()`:

```rust
ModifierKind::YourMod => {
    lines.push(format!(
        "    let n{i} = vec4f(<modified_distance>, n{ci}.y, n{ci}.z, n{ci}.w);"
    ));
}
```

The `n{ci}` variable is the child's result: `.x` = distance, `.y` = material ID, `.z` = blend, `.w` = extra.

Also update the `unreachable!()` arm in `emit_transform_chain()` to include your new distance modifier variant.

## Step 4: CPU SDF Evaluation (`src/graph/voxel.rs`)

Add match arms in `evaluate_sdf_tree()` in the `NodeData::Modifier` block:

### Point Modifier (pre-child transform)

```rust
ModifierKind::YourMod => {
    // Transform p before recursing into child
    let modified_p = /* ... */;
    evaluate_sdf_tree(scene, input_id, modified_p)
}
```

### Distance Modifier (post-child result)

```rust
ModifierKind::YourMod => {
    let child_dist = evaluate_sdf_tree(scene, input_id, p);
    // Modify distance
    child_dist + value.x
}
```

**Critical**: CPU and WGSL implementations must produce identical results for sculpt baking accuracy.

## Step 5: Verify

Tests auto-validate via `ModifierKind::ALL` iteration in naga validation tests.

Run the verification loop:
```bash
cargo check && cargo clippy -- -D warnings && cargo test && cargo build
```

## Checklist

- [ ] `ModifierKind` enum variant added
- [ ] All 8 match arms updated (ALL, base_name, badge, default_value, default_extra, gpu_type_id, is_point_modifier)
- [ ] Category chosen: point modifier or distance modifier
- [ ] WGSL function in `modifiers.wgsl` (point modifiers) or inline codegen (distance modifiers)
- [ ] Codegen match arm in correct function (`emit_transform_chain` or `emit_node_wgsl`)
- [ ] If distance modifier: added to `unreachable!()` pattern in `emit_transform_chain`
- [ ] CPU eval in `voxel.rs` matching WGSL exactly
- [ ] `gpu_type_id` is unique and sequential
- [ ] All 4 verification steps pass
