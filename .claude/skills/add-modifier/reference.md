# ModifierKind Reference — Current State

## Enum Definition (`src/graph/scene.rs`)

```rust
pub enum ModifierKind {
    // Domain deformations (point modifiers — modify p before child eval)
    Twist, Bend, Taper,
    // Unary modifiers (distance modifiers — modify distance after child eval)
    Round, Onion, Elongate,
    // Repetition (point modifiers)
    Mirror, Repeat, FiniteRepeat, RadialRepeat,
    // Distance offset (distance modifier)
    Offset,
    // Domain warp (point modifier via noise displacement)
    Noise,
}
```

## GPU Type IDs

| Variant | ID | Category |
|---------|-----|----------|
| Twist | 30.0 | Point |
| Bend | 31.0 | Point |
| Taper | 32.0 | Point |
| Round | 33.0 | Distance |
| Onion | 34.0 | Distance |
| Elongate | 35.0 | Point |
| Mirror | 36.0 | Point |
| Repeat | 37.0 | Point |
| FiniteRepeat | 38.0 | Point |
| RadialRepeat | 39.0 | Point |
| Offset | 40.0 | Distance |
| Noise | 41.0 | Point |

**Next available ID: 42.0**

## Codegen Patterns

### Point Modifier in `emit_transform_chain()`:

```rust
// Simple — single scalar param (value.x via position.x):
ChainEntry::Modifier(ModifierKind::Twist) => {
    lines.push(format!(
        "    let {new_var} = twist_point({current_var}, nodes[{idx}].position.x);"
    ));
}

// Vec3 param (value.xyz via position.xyz):
ChainEntry::Modifier(ModifierKind::Mirror) => {
    lines.push(format!(
        "    let {new_var} = mirror_point({current_var}, nodes[{idx}].position.xyz);"
    ));
}

// Two params (value + extra via position + rotation):
ChainEntry::Modifier(ModifierKind::FiniteRepeat) => {
    lines.push(format!(
        "    let {new_var} = finite_repeat_point({current_var}, nodes[{idx}].position.xyz, nodes[{idx}].rotation.xyz);"
    ));
}

// Noise — special (inline expression, no separate WGSL function for the modifier):
ChainEntry::Modifier(ModifierKind::Noise) => {
    lines.push(format!(
        "    let {new_var} = {current_var} + fbm_noise({current_var}, nodes[{idx}].position.x, nodes[{idx}].position.y, i32(nodes[{idx}].position.z));"
    ));
}
```

### Distance Modifier in `emit_node_wgsl()`:

```rust
// Round — subtract from distance:
ModifierKind::Round => {
    lines.push(format!(
        "    let n{i} = vec4f(n{ci}.x - nodes[{i}].position.x, n{ci}.y, n{ci}.z, n{ci}.w);"
    ));
}

// Onion — absolute value then subtract:
ModifierKind::Onion => {
    lines.push(format!(
        "    let n{i} = vec4f(abs(n{ci}.x) - nodes[{i}].position.x, n{ci}.y, n{ci}.z, n{ci}.w);"
    ));
}

// Offset — add to distance:
ModifierKind::Offset => {
    lines.push(format!(
        "    let n{i} = vec4f(n{ci}.x + nodes[{i}].position.x, n{ci}.y, n{ci}.z, n{ci}.w);"
    ));
}
```

### Unreachable pattern (distance modifiers not in chain):

```rust
ChainEntry::Modifier(ModifierKind::Round | ModifierKind::Onion | ModifierKind::Offset) => unreachable!(),
```

**When adding a new distance modifier, add its variant to this pattern.**

## WGSL Function Signatures (`src/shaders/modifiers.wgsl`)

```wgsl
// Point modifiers return transformed point:
fn twist_point(p: vec3f, k: f32) -> vec3f { ... }
fn bend_point(p: vec3f, k: f32) -> vec3f { ... }
fn taper_point(p: vec3f, k: f32) -> vec3f { ... }
fn elongate_point(p: vec3f, h: vec3f) -> vec3f { ... }
fn mirror_point(p: vec3f, axes: vec3f) -> vec3f { ... }
fn repeat_point(p: vec3f, period: vec3f) -> vec3f { ... }
fn finite_repeat_point(p: vec3f, period: vec3f, count: vec3f) -> vec3f { ... }
fn radial_repeat_point(p: vec3f, count: f32, axis: f32) -> vec3f { ... }
```

## Node Buffer Field Mapping

| Scene Field | GPU Access | Notes |
|-------------|-----------|-------|
| `value.x` | `nodes[idx].position.x` | Primary scalar param |
| `value.xyz` | `nodes[idx].position.xyz` | Primary vec3 param |
| `extra.xyz` | `nodes[idx].rotation.xyz` | Secondary params (e.g., repeat count) |
