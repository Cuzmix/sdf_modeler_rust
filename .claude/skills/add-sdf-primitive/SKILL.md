---
name: add-sdf-primitive
description: Add a new SDF primitive shape to the modeler. Trigger when creating new geometry types like sphere, box, cylinder, torus, or any new signed distance function primitive.
---

# Add SDF Primitive

Step-by-step guide to add a new SDF primitive shape. This touches 3 files + tests.

## Prerequisites

- Know the SDF distance function (see [iq's reference](https://iquilezles.org/articles/distfunctions/))
- Choose a unique name (check `SdfPrimitive::ALL` in `src/graph/scene.rs` for existing variants)

## Step 1: Scene Graph Enum (`src/graph/scene.rs`)

Add the variant to `SdfPrimitive` enum and update **all 9 match arms**:

1. **Enum variant**: Add to `SdfPrimitive` (after the last variant)
2. **`ALL` array**: Add to the const array
3. **`base_name()`**: Display name (e.g., `"Octahedron"`)
4. **`default_position()`**: Usually `Vec3::ZERO` (shared default, only override if needed)
5. **`default_scale()`**: Initial size vector — choose values that produce a ~1 unit shape
6. **`default_color()`**: Unique RGB — pick a color not already used by other primitives
7. **`gpu_type_id()`**: Next sequential float (currently max is `9.0` for Pyramid → use `10.0`)
8. **`scale_params()`**: UI slider labels + axis indices. Maps scale axes to named parameters:
   - `&[("Radius", 0)]` for uniform shapes
   - `&[("Width", 0), ("Height", 1)]` for 2-param shapes
   - Axis index maps to `s.x`, `s.y`, `s.z` in the WGSL function
9. **`sdf_function_name()`**: WGSL function name (e.g., `"sdf_octahedron"`)
10. **`badge()`**: 3-4 char label in brackets (e.g., `"[Oct]"`)

### Pattern (copy from existing primitive like Pyramid):

```rust
// In enum:
Octahedron,

// In ALL:
Self::Octahedron,

// In base_name():
Self::Octahedron => "Octahedron",

// In default_scale():
Self::Octahedron => Vec3::ONE,

// In default_color():
Self::Octahedron => Vec3::new(0.6, 0.4, 0.9),

// In gpu_type_id():
Self::Octahedron => 10.0,

// In scale_params():
Self::Octahedron => &[("Size", 0)],

// In sdf_function_name():
Self::Octahedron => "sdf_octahedron",

// In badge():
Self::Octahedron => "[Oct]",
```

## Step 2: WGSL Shader Function (`src/shaders/primitives.wgsl`)

Add the SDF function at the end of the file. **All primitives share the same signature:**

```wgsl
fn sdf_<name>(p: vec3f, s: vec3f) -> f32 {
    // p = local-space point (already transformed by parent chain)
    // s = scale vector (maps to scale_params axes)
    // Return: signed distance (negative = inside)
}
```

Rules:
- Follow iq's conventions for exact/bound distance
- Use `s.x`, `s.y`, `s.z` matching your `scale_params()` axis indices
- Keep it pure (no side effects, no globals)

## Step 3: CPU SDF Evaluation (`src/graph/voxel.rs`)

Add a match arm in `evaluate_sdf()` — the Rust equivalent of your WGSL function.

```rust
// In evaluate_sdf() match:
SdfPrimitive::Octahedron => {
    // Rust translation of the WGSL function using glam types
    // Vec3 instead of vec3f, .length() instead of length(), etc.
}
```

**Critical**: CPU and WGSL implementations MUST produce identical results. The CPU version is used for sculpt baking — any divergence causes visible seams.

## Step 4: Verify

No codegen changes needed — `codegen.rs` auto-dispatches via `kind.sdf_function_name()`.

Tests auto-validate: `SdfPrimitive::ALL` is iterated in naga validation tests, so your new primitive gets shader-validated automatically.

Run the verification loop:
```bash
cargo check && cargo clippy -- -D warnings && cargo test && cargo build
```

## Checklist

- [ ] `SdfPrimitive` enum variant added
- [ ] All 9 match arms updated (ALL, base_name, default_position, default_scale, default_color, gpu_type_id, scale_params, sdf_function_name, badge)
- [ ] WGSL function in `primitives.wgsl` with correct signature
- [ ] CPU eval in `voxel.rs` `evaluate_sdf()` matching WGSL exactly
- [ ] `gpu_type_id` is unique and sequential
- [ ] All 4 verification steps pass
