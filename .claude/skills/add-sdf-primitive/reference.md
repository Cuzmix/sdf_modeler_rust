# SdfPrimitive Reference — Current State

## Enum Definition (`src/graph/scene.rs`)

```rust
pub enum SdfPrimitive {
    Sphere, Box, Cylinder, Torus, Plane,
    Cone, Capsule, Ellipsoid, HexPrism, Pyramid,
}
```

## ALL Array

```rust
pub const ALL: &[Self] = &[
    Self::Sphere, Self::Box, Self::Cylinder, Self::Torus, Self::Plane,
    Self::Cone, Self::Capsule, Self::Ellipsoid, Self::HexPrism, Self::Pyramid,
];
```

## GPU Type IDs (must be unique, sequential)

| Variant | ID | WGSL Function |
|---------|-----|---------------|
| Sphere | 0.0 | `sdf_sphere` |
| Box | 1.0 | `sdf_box` |
| Cylinder | 2.0 | `sdf_cylinder` |
| Torus | 3.0 | `sdf_torus` |
| Plane | 4.0 | `sdf_plane` |
| Cone | 5.0 | `sdf_cone` |
| Capsule | 6.0 | `sdf_capsule` |
| Ellipsoid | 7.0 | `sdf_ellipsoid` |
| HexPrism | 8.0 | `sdf_hex_prism` |
| Pyramid | 9.0 | `sdf_pyramid` |

**Next available ID: 10.0**

## Scale Params Pattern

```rust
// 1 parameter (uniform):
Self::Sphere => &[("Radius", 0)],

// 2 parameters:
Self::Cylinder => &[("Radius", 0), ("Height", 1)],

// 3 parameters:
Self::Box => &[("Width", 0), ("Height", 1), ("Depth", 2)],

// No parameters:
Self::Plane => &[],
```

## WGSL Function Pattern (`src/shaders/primitives.wgsl`)

```wgsl
// Simplest — sphere:
fn sdf_sphere(p: vec3f, s: vec3f) -> f32 {
    return length(p) - s.x;
}

// 3-param — box:
fn sdf_box(p: vec3f, s: vec3f) -> f32 {
    let q = abs(p) - s;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}
```

## CPU Eval Pattern (`src/graph/voxel.rs`)

```rust
fn evaluate_sdf(kind: &SdfPrimitive, p: Vec3, s: Vec3) -> f32 {
    match kind {
        SdfPrimitive::Sphere => p.length() - s.x,
        SdfPrimitive::Box => {
            let q = p.abs() - s;
            q.max(Vec3::ZERO).length() + q.x.max(q.y.max(q.z)).min(0.0)
        }
        // ... other variants
    }
}
```

## Default Colors (avoid duplicates)

| Variant | Color (RGB) |
|---------|-------------|
| Sphere | (0.8, 0.3, 0.2) — red |
| Box | (0.2, 0.5, 0.8) — blue |
| Cylinder | (0.2, 0.8, 0.3) — green |
| Torus | (0.8, 0.6, 0.2) — orange |
| Cone | (0.7, 0.3, 0.7) — purple |
| Capsule | (0.3, 0.7, 0.7) — teal |
| Plane | (0.5, 0.5, 0.5) — gray |
| Ellipsoid | (0.9, 0.5, 0.3) — peach |
| HexPrism | (0.4, 0.6, 0.8) — light blue |
| Pyramid | (0.8, 0.7, 0.3) — gold |
