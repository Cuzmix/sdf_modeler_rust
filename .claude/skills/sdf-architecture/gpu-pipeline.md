# GPU Pipeline Reference

## Bind Group Layout

| Group | Binding | Resource | Size |
|-------|---------|----------|------|
| @group(0) | @binding(0) | `var<uniform> camera: Camera` | 368 bytes |
| @group(1) | @binding(0) | `var<storage, read> nodes: array<SdfNode>` | 128 bytes/node |
| @group(1) | @binding(1) | `var<storage, read> voxel_data: array<f32>` | varies |
| @group(2) | @binding(0) | `var voxel_sampler: sampler` | - |
| @group(2) | @binding(1+) | `var voxel_tex_N: texture_3d<f32>` | per sculpt |

## CameraUniform Layout (368 bytes, `#[repr(C)]`)

| Field | Type | Bytes | Notes |
|-------|------|-------|-------|
| `inv_view_proj` | mat4x4 | 64 | Inverse view-projection matrix |
| `eye` | vec4f | 16 | Camera world position |
| `viewport` | vec4f | 16 | [width, height, 0, 0] |
| `time` | f32 | 4 | Elapsed time |
| `quality_mode` | f32 | 4 | 0=full, 1=interactive |
| `grid_enabled` | f32 | 4 | Ground grid toggle |
| `selected_idx` | f32 | 4 | Selected node index (-1 = none) |
| `scene_min` | vec4f | 16 | AABB min, w = shading_mode |
| `scene_max` | vec4f | 16 | AABB max |
| `brush_pos` | vec4f | 16 | [x, y, z, radius] |
| `cross_section` | vec4f | 16 | [axis, position, 0, 0] |
| `ambient_info` | vec4f | 16 | [ambient_intensity, 0, 0, 0] |
| `scene_light_info` | vec4f | 16 | [light_count, 0, 0, 0] |
| `scene_lights` | 8 lights | 512 | 4 vec4f per light (pos_type, dir_intensity, color_range, params) |

## SdfNodeGpu Layout (128 bytes = 8 x vec4f, `#[repr(C)]`)

| Field | Contents | Notes |
|-------|----------|-------|
| `type_op` | [type_id, smooth_k/intensity, metallic/range, roughness/spot_angle] | Type determines field interpretation |
| `position` | [x, y, z, layer_intensity] | World position or modifier value |
| `rotation` | [rx, ry, rz, 0] | Euler rotation or modifier extra |
| `scale` | [sx, sy, sz, light_mask] | Scale or light mask (u8 in .w) |
| `color` | [r, g, b, is_selected] | Material color + selection flag |
| `extra0` | [voxel_offset, resolution, emissive.x, emissive.y] | Sculpt voxel data |
| `extra1` | [bounds_min.xyz, emissive.z] | Bounding volume |
| `extra2` | [bounds_max.xyz, fresnel] | Bounding volume + material |

## GPU Type IDs

| Range | Node Type |
|-------|-----------|
| 0-9 | Primitives (Sphere=0 through Pyramid=9) |
| 10-22 | CSG Operations |
| 20 | Sculpt |
| 21 | Transform |
| 30-41 | Modifiers (Twist=30 through Noise=41) |
| 50-53 | Lights (Point=50, Spot=51, Directional=52, Ambient=53) |

## SceneLightGpu Layout (64 bytes)

| Field | Contents |
|-------|----------|
| `position_type` | [pos.x, pos.y, pos.z, type (0=point, 1=spot, 2=dir)] |
| `direction_intensity` | [dir.x, dir.y, dir.z, intensity] |
| `color_range` | [r, g, b, range] |
| `params` | [cos_half_spot_angle, 0, 0, 0] |

## Shader Module Assembly

Render prelude (in order): bindings → voxel_sampling → vertex → transforms → primitives → modifiers → noise → operations

Compute prelude: same but without vertex shader (compute shaders error with @vertex)

## Raymarching Configuration (compile-time placeholders)

| Placeholder | Default | Interactive |
|-------------|---------|-------------|
| `MARCH_MAX_STEPS` | 96 | 48 |
| `MARCH_EPSILON` | 0.0005 | 0.0005 |
| `MARCH_STEP_MULT` | 0.9 | 0.9 |
| `MARCH_MAX_DIST` | 100.0 | 100.0 |

## Pick Pass

1x1 Rgba8Unorm texture with separate pipeline:
- 0 = background (miss)
- 1 = ground plane
- 2+ = scene node (index = value - 2, mapped via topo_order)
- 253/254/255 = gizmo X/Y/Z axes

Staging buffer: 256 bytes minimum (wgpu alignment). Async readback (1-frame delay, no CPU stall).

## Blit Pass

Edge detection for selection outline:
- Alpha channel encodes selection: 0.0 = selected, 1.0 = not selected
- 4-cardinal-direction edge detection loop (configurable 1-8px width)
- `textureLoad` for exact pixel values (not sampled)
- `BlitParams` uniform: viewport dimensions + outline_color (rgb) + outline_width
