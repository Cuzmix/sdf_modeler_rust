use glam::Vec3;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

use super::voxel::VoxelGrid;

pub type NodeId = u64;

/// Maximum number of scene lights the GPU can handle simultaneously.
/// When a scene has more lights than this, the nearest ones to the camera are active.
pub const MAX_SCENE_LIGHTS: usize = 8;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SdfPrimitive {
    Sphere,
    Box,
    Cylinder,
    Torus,
    Plane,
    Cone,
    Capsule,
    Ellipsoid,
    HexPrism,
    Pyramid,
}

impl SdfPrimitive {
    pub const ALL: &[Self] = &[
        Self::Sphere,
        Self::Box,
        Self::Cylinder,
        Self::Torus,
        Self::Plane,
        Self::Cone,
        Self::Capsule,
        Self::Ellipsoid,
        Self::HexPrism,
        Self::Pyramid,
    ];

    pub fn base_name(&self) -> &'static str {
        match self {
            Self::Sphere => "Sphere",
            Self::Box => "Box",
            Self::Cylinder => "Cylinder",
            Self::Torus => "Torus",
            Self::Plane => "Plane",
            Self::Cone => "Cone",
            Self::Capsule => "Capsule",
            Self::Ellipsoid => "Ellipsoid",
            Self::HexPrism => "HexPrism",
            Self::Pyramid => "Pyramid",
        }
    }

    pub fn default_position(&self) -> Vec3 {
        Vec3::ZERO
    }

    pub fn default_scale(&self) -> Vec3 {
        match self {
            Self::Sphere | Self::Box | Self::Plane => Vec3::ONE,
            Self::Torus => Vec3::new(1.0, 0.3, 1.0),
            Self::Cylinder | Self::Cone => Vec3::new(0.5, 1.0, 0.5),
            Self::Capsule => Vec3::new(0.3, 1.0, 0.3),
            Self::Ellipsoid => Vec3::new(1.0, 0.6, 0.4),
            Self::HexPrism => Vec3::new(0.5, 0.5, 0.5),
            Self::Pyramid => Vec3::new(1.0, 1.0, 1.0),
        }
    }

    pub fn default_color(&self) -> Vec3 {
        match self {
            Self::Sphere => Vec3::new(0.8, 0.3, 0.2),
            Self::Box => Vec3::new(0.2, 0.5, 0.8),
            Self::Cylinder => Vec3::new(0.2, 0.8, 0.3),
            Self::Torus => Vec3::new(0.8, 0.6, 0.2),
            Self::Cone => Vec3::new(0.7, 0.3, 0.7),
            Self::Capsule => Vec3::new(0.3, 0.7, 0.7),
            Self::Plane => Vec3::new(0.5, 0.5, 0.5),
            Self::Ellipsoid => Vec3::new(0.9, 0.5, 0.3),
            Self::HexPrism => Vec3::new(0.4, 0.6, 0.8),
            Self::Pyramid => Vec3::new(0.8, 0.7, 0.3),
        }
    }

    pub fn gpu_type_id(&self) -> f32 {
        match self {
            Self::Sphere => 0.0,
            Self::Box => 1.0,
            Self::Cylinder => 2.0,
            Self::Torus => 3.0,
            Self::Plane => 4.0,
            Self::Cone => 5.0,
            Self::Capsule => 6.0,
            Self::Ellipsoid => 7.0,
            Self::HexPrism => 8.0,
            Self::Pyramid => 9.0,
        }
    }

    /// Returns (label, axis_index) for each scale parameter this primitive uses.
    pub fn scale_params(&self) -> &'static [(&'static str, usize)] {
        match self {
            Self::Sphere => &[("Radius", 0)],
            Self::Box => &[("Width", 0), ("Height", 1), ("Depth", 2)],
            Self::Cylinder => &[("Radius", 0), ("Height", 1)],
            Self::Torus => &[("Major R", 0), ("Tube R", 1)],
            Self::Plane => &[],
            Self::Cone => &[("Radius", 0), ("Height", 1)],
            Self::Capsule => &[("Radius", 0), ("Half H", 1)],
            Self::Ellipsoid => &[("Radius X", 0), ("Radius Y", 1), ("Radius Z", 2)],
            Self::HexPrism => &[("Radius", 0), ("Height", 1)],
            Self::Pyramid => &[("Base", 0), ("Height", 1)],
        }
    }

    pub fn sdf_function_name(&self) -> &'static str {
        match self {
            Self::Sphere => "sdf_sphere",
            Self::Box => "sdf_box",
            Self::Cylinder => "sdf_cylinder",
            Self::Torus => "sdf_torus",
            Self::Plane => "sdf_plane",
            Self::Cone => "sdf_cone",
            Self::Capsule => "sdf_capsule",
            Self::Ellipsoid => "sdf_ellipsoid",
            Self::HexPrism => "sdf_hex_prism",
            Self::Pyramid => "sdf_pyramid",
        }
    }

    pub fn badge(&self) -> &'static str {
        match self {
            Self::Sphere => "[Sph]",
            Self::Box => "[Box]",
            Self::Cylinder => "[Cyl]",
            Self::Torus => "[Tor]",
            Self::Plane => "[Pln]",
            Self::Cone => "[Con]",
            Self::Capsule => "[Cap]",
            Self::Ellipsoid => "[Ell]",
            Self::HexPrism => "[Hex]",
            Self::Pyramid => "[Pyr]",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ModifierKind {
    // Domain deformations (modify point before child eval)
    Twist,
    Bend,
    Taper,
    // Unary modifiers (modify distance after child eval)
    Round,
    Onion,
    Elongate,
    // Repetition (modify point before child eval)
    Mirror,
    Repeat,
    FiniteRepeat,
    RadialRepeat,
    // Distance offset (modify distance after child eval)
    Offset,
    // Domain warp (modify point before child eval via noise displacement)
    Noise,
}

impl ModifierKind {
    pub const ALL: &[Self] = &[
        Self::Twist,
        Self::Bend,
        Self::Taper,
        Self::Round,
        Self::Onion,
        Self::Elongate,
        Self::Mirror,
        Self::Repeat,
        Self::FiniteRepeat,
        Self::RadialRepeat,
        Self::Offset,
        Self::Noise,
    ];

    pub fn base_name(&self) -> &'static str {
        match self {
            Self::Twist => "Twist",
            Self::Bend => "Bend",
            Self::Taper => "Taper",
            Self::Round => "Round",
            Self::Onion => "Onion",
            Self::Elongate => "Elongate",
            Self::Mirror => "Mirror",
            Self::Repeat => "Repeat",
            Self::FiniteRepeat => "Finite Repeat",
            Self::RadialRepeat => "Radial Repeat",
            Self::Offset => "Offset",
            Self::Noise => "Noise",
        }
    }

    pub fn badge(&self) -> &'static str {
        match self {
            Self::Twist => "[Twi]",
            Self::Bend => "[Bnd]",
            Self::Taper => "[Tpr]",
            Self::Round => "[Rnd]",
            Self::Onion => "[Oni]",
            Self::Elongate => "[Elo]",
            Self::Mirror => "[Mir]",
            Self::Repeat => "[Rep]",
            Self::FiniteRepeat => "[FRp]",
            Self::RadialRepeat => "[Rad]",
            Self::Offset => "[Ofs]",
            Self::Noise => "[Nse]",
        }
    }

    pub fn default_value(&self) -> Vec3 {
        match self {
            Self::Twist => Vec3::new(1.0, 0.0, 0.0),
            Self::Bend => Vec3::new(1.0, 0.0, 0.0),
            Self::Taper => Vec3::new(0.5, 0.0, 0.0),
            Self::Round => Vec3::new(0.1, 0.0, 0.0),
            Self::Onion => Vec3::new(0.1, 0.0, 0.0),
            Self::Elongate => Vec3::new(0.0, 0.5, 0.0),
            Self::Mirror => Vec3::new(1.0, 0.0, 0.0),
            Self::Repeat => Vec3::new(2.0, 0.0, 0.0),
            Self::FiniteRepeat => Vec3::new(2.0, 0.0, 0.0),
            Self::RadialRepeat => Vec3::new(6.0, 1.0, 0.0), // 6 copies, Y axis
            Self::Offset => Vec3::new(0.1, 0.0, 0.0),
            Self::Noise => Vec3::new(2.0, 0.1, 3.0), // (frequency, amplitude, octaves)
        }
    }

    pub fn default_extra(&self) -> Vec3 {
        match self {
            Self::FiniteRepeat => Vec3::new(2.0, 0.0, 0.0),
            _ => Vec3::ZERO,
        }
    }

    pub fn gpu_type_id(&self) -> f32 {
        match self {
            Self::Twist => 30.0,
            Self::Bend => 31.0,
            Self::Taper => 32.0,
            Self::Round => 33.0,
            Self::Onion => 34.0,
            Self::Elongate => 35.0,
            Self::Mirror => 36.0,
            Self::Repeat => 37.0,
            Self::FiniteRepeat => 38.0,
            Self::RadialRepeat => 39.0,
            Self::Offset => 40.0,
            Self::Noise => 41.0,
        }
    }

    /// Point modifiers modify `p` before child evaluation (integrate into transform chain).
    /// Distance modifiers modify the distance after child evaluation.
    pub fn is_point_modifier(&self) -> bool {
        match self {
            Self::Twist
            | Self::Bend
            | Self::Taper
            | Self::Elongate
            | Self::Mirror
            | Self::Repeat
            | Self::FiniteRepeat
            | Self::RadialRepeat
            | Self::Noise => true,
            Self::Round | Self::Onion | Self::Offset => false,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum CsgOp {
    Union,
    SmoothUnion,
    Subtract,
    Intersect,
    SmoothSubtract,
    SmoothIntersect,
    ChamferUnion,
    ChamferSubtract,
    ChamferIntersect,
    StairsUnion,
    StairsSubtract,
    ColumnsUnion,
    ColumnsSubtract,
}

impl CsgOp {
    pub const ALL: &[Self] = &[
        Self::Union,
        Self::SmoothUnion,
        Self::Subtract,
        Self::Intersect,
        Self::SmoothSubtract,
        Self::SmoothIntersect,
        Self::ChamferUnion,
        Self::ChamferSubtract,
        Self::ChamferIntersect,
        Self::StairsUnion,
        Self::StairsSubtract,
        Self::ColumnsUnion,
        Self::ColumnsSubtract,
    ];

    pub fn base_name(&self) -> &'static str {
        match self {
            Self::Union => "Union",
            Self::SmoothUnion => "Smooth Union",
            Self::Subtract => "Subtract",
            Self::Intersect => "Intersect",
            Self::SmoothSubtract => "Smooth Subtract",
            Self::SmoothIntersect => "Smooth Intersect",
            Self::ChamferUnion => "Chamfer Union",
            Self::ChamferSubtract => "Chamfer Subtract",
            Self::ChamferIntersect => "Chamfer Intersect",
            Self::StairsUnion => "Stairs Union",
            Self::StairsSubtract => "Stairs Subtract",
            Self::ColumnsUnion => "Columns Union",
            Self::ColumnsSubtract => "Columns Subtract",
        }
    }

    pub fn default_smooth_k(&self) -> f32 {
        match self {
            Self::SmoothUnion => 0.5,
            Self::SmoothSubtract | Self::SmoothIntersect => 0.3,
            Self::ChamferUnion | Self::ChamferSubtract | Self::ChamferIntersect => 0.2,
            Self::StairsUnion | Self::StairsSubtract => 0.2,
            Self::ColumnsUnion | Self::ColumnsSubtract => 0.2,
            _ => 0.0,
        }
    }

    pub fn gpu_op_id(&self) -> f32 {
        match self {
            Self::Union => 10.0,
            Self::SmoothUnion => 11.0,
            Self::Subtract => 12.0,
            Self::Intersect => 13.0,
            Self::SmoothSubtract => 14.0,
            Self::SmoothIntersect => 15.0,
            Self::ChamferUnion => 16.0,
            Self::ChamferSubtract => 17.0,
            Self::ChamferIntersect => 18.0,
            Self::StairsUnion => 19.0,
            Self::StairsSubtract => 20.0,
            Self::ColumnsUnion => 21.0,
            Self::ColumnsSubtract => 22.0,
        }
    }

    pub fn wgsl_function_name(&self) -> &'static str {
        match self {
            Self::Union => "op_union",
            Self::SmoothUnion => "op_smooth_union",
            Self::Subtract | Self::SmoothSubtract => "op_subtract",
            Self::Intersect | Self::SmoothIntersect => "op_intersect",
            Self::ChamferUnion => "op_chamfer_union",
            Self::ChamferSubtract => "op_chamfer_subtract",
            Self::ChamferIntersect => "op_chamfer_intersect",
            Self::StairsUnion => "op_stairs_union",
            Self::StairsSubtract => "op_stairs_subtract",
            Self::ColumnsUnion => "op_columns_union",
            Self::ColumnsSubtract => "op_columns_subtract",
        }
    }

    pub fn badge(&self) -> &'static str {
        match self {
            Self::Union => "[Uni]",
            Self::SmoothUnion => "[SmU]",
            Self::Subtract => "[Sub]",
            Self::Intersect => "[Int]",
            Self::SmoothSubtract => "[S-]",
            Self::SmoothIntersect => "[S∩]",
            Self::ChamferUnion => "[C∪]",
            Self::ChamferSubtract => "[C-]",
            Self::ChamferIntersect => "[C∩]",
            Self::StairsUnion => "[St∪]",
            Self::StairsSubtract => "[St-]",
            Self::ColumnsUnion => "[Co∪]",
            Self::ColumnsSubtract => "[Co-]",
        }
    }

    /// Whether this operation requires an extra step/column count parameter.
    pub fn has_steps_param(&self) -> bool {
        matches!(
            self,
            Self::StairsUnion | Self::StairsSubtract | Self::ColumnsUnion | Self::ColumnsSubtract
        )
    }

    /// Whether this operation supports an independent color blend parameter.
    /// Hard union has no blending, so color_blend is irrelevant.
    pub fn has_color_blend_param(&self) -> bool {
        !matches!(self, Self::Union)
    }

    /// Default step/column count for operations that support it.
    pub fn default_steps(&self) -> f32 {
        match self {
            Self::StairsUnion | Self::StairsSubtract => 4.0,
            Self::ColumnsUnion | Self::ColumnsSubtract => 4.0,
            _ => 0.0,
        }
    }
}

/// Type of light source in the scene.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LightType {
    Point,
    Spot,
    Directional,
    /// Global ambient light — uniform illumination from all directions.
    /// Has color and intensity but no position, direction, or falloff.
    Ambient,
    /// Procedural light array — spawns N point lights in a geometric pattern.
    Array,
}

impl LightType {
    pub const ALL: &[Self] = &[
        Self::Point,
        Self::Spot,
        Self::Directional,
        Self::Ambient,
        Self::Array,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Point => "Point",
            Self::Spot => "Spot",
            Self::Directional => "Directional",
            Self::Ambient => "Ambient",
            Self::Array => "Array",
        }
    }

    pub fn badge(&self) -> &'static str {
        match self {
            Self::Point => "[Pt]",
            Self::Spot => "[Sp]",
            Self::Directional => "[Dir]",
            Self::Ambient => "[Amb]",
            Self::Array => "[Arr]",
        }
    }
}

/// Geometric pattern for a Light Array.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum ArrayPattern {
    /// Lights evenly spaced on a circle in the local XZ plane.
    #[default]
    Ring,
    /// Lights evenly spaced along the local X axis.
    Line,
    /// Lights arranged in a grid in the local XZ plane.
    Grid,
    /// Lights arranged in an Archimedean spiral in the local XZ plane.
    Spiral,
}

impl ArrayPattern {
    pub const ALL: &[Self] = &[Self::Ring, Self::Line, Self::Grid, Self::Spiral];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Ring => "Ring",
            Self::Line => "Line",
            Self::Grid => "Grid",
            Self::Spiral => "Spiral",
        }
    }
}

/// Configuration for a procedural Light Array node.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LightArrayConfig {
    /// Geometric pattern of the array.
    #[serde(default)]
    pub pattern: ArrayPattern,
    /// Number of lights in the array (2–32).
    #[serde(default = "default_array_count")]
    pub count: u32,
    /// Overall size of the pattern (radius for Ring/Spiral, half-length for Line, extent for Grid).
    #[serde(default = "default_array_radius")]
    pub radius: f32,
    /// Hue variation across instances (0.0 = uniform, 1.0 = full rainbow spread).
    #[serde(default)]
    pub color_variation: f32,
}

fn default_array_count() -> u32 {
    6
}
fn default_array_radius() -> f32 {
    2.0
}

impl Default for LightArrayConfig {
    fn default() -> Self {
        Self {
            pattern: ArrayPattern::Ring,
            count: default_array_count(),
            radius: default_array_radius(),
            color_variation: 0.0,
        }
    }
}

fn default_roughness() -> f32 {
    0.5
}
fn default_reflectance_f0() -> f32 {
    0.04
}
fn default_clearcoat_roughness() -> f32 {
    0.2
}
fn default_sheen_roughness() -> f32 {
    0.5
}
fn default_thickness() -> f32 {
    1.0
}
fn default_ior() -> f32 {
    1.5
}
fn default_anisotropy_direction_local() -> Vec3 {
    Vec3::X
}
fn default_layer_intensity() -> f32 {
    1.0
}
fn default_scale() -> Vec3 {
    Vec3::ONE
}
fn default_color_blend() -> f32 {
    -1.0
}
fn default_light_intensity() -> f32 {
    1.0
}
fn default_light_range() -> f32 {
    10.0
}
fn default_spot_angle() -> f32 {
    45.0
}
fn default_light_color() -> Vec3 {
    Vec3::ONE
}
fn default_shadow_softness() -> f32 {
    8.0
}
fn default_cast_shadows_true() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialParams {
    #[serde(default = "default_material_base_color")]
    pub base_color: Vec3,
    #[serde(default = "default_roughness")]
    pub roughness: f32,
    #[serde(default)]
    pub metallic: f32,
    #[serde(default)]
    pub emissive_color: Vec3,
    #[serde(default)]
    pub emissive_intensity: f32,
    #[serde(default = "default_reflectance_f0", alias = "fresnel")]
    pub reflectance_f0: f32,
    #[serde(default)]
    pub clearcoat: f32,
    #[serde(default = "default_clearcoat_roughness")]
    pub clearcoat_roughness: f32,
    #[serde(default)]
    pub sheen_color: Vec3,
    #[serde(default = "default_sheen_roughness")]
    pub sheen_roughness: f32,
    #[serde(default)]
    pub transmission: f32,
    #[serde(default = "default_thickness")]
    pub thickness: f32,
    #[serde(default = "default_ior")]
    pub ior: f32,
    #[serde(default)]
    pub anisotropy_strength: f32,
    #[serde(default = "default_anisotropy_direction_local")]
    pub anisotropy_direction_local: Vec3,
}

fn default_material_base_color() -> Vec3 {
    Vec3::ONE
}

impl Default for MaterialParams {
    fn default() -> Self {
        Self {
            base_color: default_material_base_color(),
            roughness: default_roughness(),
            metallic: 0.0,
            emissive_color: Vec3::ZERO,
            emissive_intensity: 0.0,
            reflectance_f0: default_reflectance_f0(),
            clearcoat: 0.0,
            clearcoat_roughness: default_clearcoat_roughness(),
            sheen_color: Vec3::ZERO,
            sheen_roughness: default_sheen_roughness(),
            transmission: 0.0,
            thickness: default_thickness(),
            ior: default_ior(),
            anisotropy_strength: 0.0,
            anisotropy_direction_local: default_anisotropy_direction_local(),
        }
    }
}

impl MaterialParams {
    pub fn with_base_color(base_color: Vec3) -> Self {
        Self {
            base_color,
            ..Self::default()
        }
    }
}

fn hash_vec3<H: Hasher>(value: Vec3, hasher: &mut H) {
    value.x.to_bits().hash(hasher);
    value.y.to_bits().hash(hasher);
    value.z.to_bits().hash(hasher);
}

fn hash_material_params<H: Hasher>(material: &MaterialParams, hasher: &mut H) {
    hash_vec3(material.base_color, hasher);
    material.roughness.to_bits().hash(hasher);
    material.metallic.to_bits().hash(hasher);
    hash_vec3(material.emissive_color, hasher);
    material.emissive_intensity.to_bits().hash(hasher);
    material.reflectance_f0.to_bits().hash(hasher);
    material.clearcoat.to_bits().hash(hasher);
    material.clearcoat_roughness.to_bits().hash(hasher);
    hash_vec3(material.sheen_color, hasher);
    material.sheen_roughness.to_bits().hash(hasher);
    material.transmission.to_bits().hash(hasher);
    material.thickness.to_bits().hash(hasher);
    material.ior.to_bits().hash(hasher);
    material.anisotropy_strength.to_bits().hash(hasher);
    hash_vec3(material.anisotropy_direction_local, hasher);
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum NodeData {
    Primitive {
        kind: SdfPrimitive,
        position: Vec3,
        rotation: Vec3,
        scale: Vec3,
        #[serde(default)]
        material: MaterialParams,
        /// Legacy: kept for v2 save file migration only. Always None at runtime.
        #[serde(default, skip_serializing)]
        voxel_grid: Option<VoxelGrid>,
    },
    Operation {
        op: CsgOp,
        smooth_k: f32,
        #[serde(default)]
        steps: f32,
        /// Independent color blend radius. -1.0 means "follow smooth_k".
        #[serde(default = "default_color_blend")]
        color_blend: f32,
        left: Option<NodeId>,
        right: Option<NodeId>,
    },
    Sculpt {
        input: Option<NodeId>,
        position: Vec3,
        rotation: Vec3,
        #[serde(default)]
        material: MaterialParams,
        #[serde(default = "default_layer_intensity")]
        layer_intensity: f32,
        voxel_grid: VoxelGrid,
        #[serde(default = "crate::graph::voxel::default_resolution")]
        desired_resolution: u32,
    },
    Transform {
        input: Option<NodeId>,
        #[serde(default)]
        translation: Vec3,
        #[serde(default)]
        rotation: Vec3,
        #[serde(default = "default_scale")]
        scale: Vec3,
    },
    Modifier {
        kind: ModifierKind,
        input: Option<NodeId>,
        value: Vec3,
        #[serde(default)]
        extra: Vec3,
    },
    Light {
        light_type: LightType,
        #[serde(default = "default_light_color")]
        color: Vec3,
        #[serde(default = "default_light_intensity")]
        intensity: f32,
        #[serde(default = "default_light_range")]
        range: f32,
        #[serde(default = "default_spot_angle")]
        spot_angle: f32,
        /// Whether this light casts shadows. Default true for Directional/Spot, false for Point.
        #[serde(default = "default_cast_shadows_true")]
        cast_shadows: bool,
        /// Shadow softness (k parameter for soft_shadow). Higher = sharper. Range 1.0–64.0.
        #[serde(default = "default_shadow_softness")]
        shadow_softness: f32,
        /// Shadow color (tint for shadowed regions). Default black = normal shadows.
        #[serde(default)]
        shadow_color: Vec3,
        /// Whether this light emits volumetric scattering (god rays).
        #[serde(default)]
        volumetric: bool,
        /// Density of volumetric scattering (0.01–1.0). Higher = more opaque beams.
        #[serde(default = "default_volumetric_density")]
        volumetric_density: f32,
        /// Optional SDF cookie: reference to a Primitive or Operation node whose SDF
        /// shapes the light's beam. Inside the cookie shape = full light, outside = no light.
        #[serde(default)]
        cookie_node: Option<NodeId>,
        /// Proximity modulation mode — modulates intensity based on distance to SDF surfaces.
        #[serde(default)]
        proximity_mode: ProximityMode,
        /// Distance over which the proximity effect ramps (0.1–10.0).
        #[serde(default = "default_proximity_range")]
        proximity_range: f32,
        /// Configuration for Array light type (pattern, count, radius, color variation).
        /// Only meaningful when light_type is Array.
        #[serde(default)]
        array_config: Option<LightArrayConfig>,
        /// Optional expression for animating intensity over time. Overrides static intensity.
        #[serde(default)]
        intensity_expr: Option<String>,
        /// Optional expression for animating color hue over time (result = degrees of hue shift).
        #[serde(default)]
        color_hue_expr: Option<String>,
    },
}

fn default_volumetric_density() -> f32 {
    0.15
}
fn default_proximity_range() -> f32 {
    2.0
}

/// Proximity light modulation mode — SDF-native feature where light intensity
/// is modulated by the light's distance to the nearest SDF surface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub enum ProximityMode {
    /// No proximity modulation — intensity unchanged.
    #[default]
    Off,
    /// Light brightens as it approaches surfaces (factor = 1.0 + smoothstep effect).
    Brighten,
    /// Light dims as it approaches surfaces (factor = smoothstep away from surface).
    Dim,
}

impl ProximityMode {
    pub const ALL: &[Self] = &[Self::Off, Self::Brighten, Self::Dim];

    pub fn label(&self) -> &'static str {
        match self {
            Self::Off => "Off",
            Self::Brighten => "Brighten",
            Self::Dim => "Dim",
        }
    }
}

impl NodeData {
    pub fn material(&self) -> Option<&MaterialParams> {
        match self {
            NodeData::Primitive { material, .. } | NodeData::Sculpt { material, .. } => {
                Some(material)
            }
            _ => None,
        }
    }

    pub fn material_mut(&mut self) -> Option<&mut MaterialParams> {
        match self {
            NodeData::Primitive { material, .. } | NodeData::Sculpt { material, .. } => {
                Some(material)
            }
            _ => None,
        }
    }

    /// Iterate over child node IDs (0-2 children depending on variant).
    pub fn children(&self) -> impl Iterator<Item = NodeId> {
        let (a, b) = match self {
            NodeData::Operation { left, right, .. } => (*left, *right),
            NodeData::Sculpt { input, .. }
            | NodeData::Transform { input, .. }
            | NodeData::Modifier { input, .. } => (*input, None),
            NodeData::Primitive { .. } | NodeData::Light { .. } => (None, None),
        };
        a.into_iter().chain(b)
    }

    /// For geometry nodes (Primitive/Sculpt), return local bounding sphere (center, radius).
    /// Returns None for non-geometry nodes.
    pub fn geometry_local_sphere(&self) -> Option<([f32; 3], f32)> {
        match self {
            NodeData::Primitive {
                position, scale, ..
            } => {
                let r = scale.x.max(scale.y).max(scale.z);
                Some(([position.x, position.y, position.z], r))
            }
            NodeData::Sculpt {
                position,
                voxel_grid,
                ..
            } => {
                let mid = [
                    position.x + (voxel_grid.bounds_min.x + voxel_grid.bounds_max.x) * 0.5,
                    position.y + (voxel_grid.bounds_min.y + voxel_grid.bounds_max.y) * 0.5,
                    position.z + (voxel_grid.bounds_min.z + voxel_grid.bounds_max.z) * 0.5,
                ];
                let r = ((voxel_grid.bounds_max.x - voxel_grid.bounds_min.x) * 0.5)
                    .max((voxel_grid.bounds_max.y - voxel_grid.bounds_min.y) * 0.5)
                    .max((voxel_grid.bounds_max.z - voxel_grid.bounds_min.z) * 0.5);
                Some((mid, r))
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneNode {
    pub id: NodeId,
    pub name: String,
    pub data: NodeData,
    #[serde(default)]
    pub locked: bool,
}

/// Node counts broken down by type.
#[derive(Default, Debug, Clone)]
pub struct NodeTypeCounts {
    pub total: usize,
    pub visible: usize,
    pub primitives: usize,
    pub operations: usize,
    pub transforms: usize,
    pub modifiers: usize,
    pub sculpts: usize,
    pub lights: usize,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Scene {
    pub nodes: HashMap<NodeId, SceneNode>,
    pub(crate) next_id: u64,
    pub(crate) name_counters: HashMap<String, u32>,
    #[serde(default)]
    pub hidden_nodes: HashSet<NodeId>,
    /// Per-node light linking bitmask. Bit N = light slot N affects this node.
    /// Default `0xFF` (all lights) for nodes not in the map.
    #[serde(default)]
    pub light_masks: HashMap<NodeId, u8>,
}

impl Scene {
    pub fn new() -> Self {
        let mut scene = Self {
            nodes: HashMap::new(),
            next_id: 0,
            name_counters: HashMap::new(),
            hidden_nodes: HashSet::new(),
            light_masks: HashMap::new(),
        };
        scene.create_primitive(SdfPrimitive::Sphere);
        scene.create_default_lights();
        scene
    }

    /// Get the light linking bitmask for a node. Returns `0xFF` (all lights)
    /// if no custom mask is set.
    pub fn get_light_mask(&self, id: NodeId) -> u8 {
        self.light_masks.get(&id).copied().unwrap_or(0xFF)
    }

    /// Set the light linking bitmask for a node.
    /// Used by light linking actions and UI (tasks 5-7).
    pub fn set_light_mask(&mut self, id: NodeId, mask: u8) {
        if mask == 0xFF {
            self.light_masks.remove(&id);
        } else {
            self.light_masks.insert(id, mask);
        }
    }

    /// Returns true if any Light node in the scene has an active animation expression.
    /// Used to determine if continuous repaint is needed.
    pub fn has_light_expressions(&self) -> bool {
        self.nodes.values().any(|n| {
            if let NodeData::Light {
                intensity_expr,
                color_hue_expr,
                ..
            } = &n.data
            {
                intensity_expr.is_some() || color_hue_expr.is_some()
            } else {
                false
            }
        })
    }

    pub fn next_name(&mut self, base: &str) -> String {
        let counter = self.name_counters.entry(base.to_string()).or_insert(0);
        *counter += 1;
        if *counter == 1 {
            base.to_string()
        } else {
            format!("{} {}", base, counter)
        }
    }

    pub fn add_node(&mut self, name: String, data: NodeData) -> NodeId {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(
            id,
            SceneNode {
                id,
                name,
                data,
                locked: false,
            },
        );
        id
    }

    pub fn is_hidden(&self, id: NodeId) -> bool {
        self.hidden_nodes.contains(&id)
    }

    pub fn toggle_visibility(&mut self, id: NodeId) {
        if !self.hidden_nodes.remove(&id) {
            self.hidden_nodes.insert(id);
        }
    }

    pub fn remove_node(&mut self, id: NodeId) -> Option<SceneNode> {
        self.hidden_nodes.remove(&id);
        self.light_masks.remove(&id);
        let node = self.nodes.remove(&id);
        // Null out any references to this node (instead of cascade-deleting)
        let to_patch: Vec<(NodeId, bool, bool, bool)> = self
            .nodes
            .values()
            .filter_map(|n| match &n.data {
                NodeData::Operation { left, right, .. } => {
                    let is_left = *left == Some(id);
                    let is_right = *right == Some(id);
                    if is_left || is_right {
                        Some((n.id, is_left, is_right, false))
                    } else {
                        None
                    }
                }
                NodeData::Sculpt { input, .. } if *input == Some(id) => {
                    Some((n.id, false, false, true))
                }
                NodeData::Transform { input, .. } if *input == Some(id) => {
                    Some((n.id, false, false, true))
                }
                NodeData::Modifier { input, .. } if *input == Some(id) => {
                    Some((n.id, false, false, true))
                }
                _ => None,
            })
            .collect();
        for (parent_id, is_left, is_right, is_single_input) in to_patch {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                match &mut parent.data {
                    NodeData::Operation { left, right, .. } => {
                        if is_left {
                            *left = None;
                        }
                        if is_right {
                            *right = None;
                        }
                    }
                    NodeData::Sculpt { input, .. }
                    | NodeData::Transform { input, .. }
                    | NodeData::Modifier { input, .. }
                        if is_single_input =>
                    {
                        *input = None;
                    }
                    _ => {}
                }
            }
        }
        node
    }

    /// Remove an entire subtree rooted at `root_id`.
    /// Returns the removed node IDs as a stable, sorted list.
    pub fn remove_subtree(&mut self, root_id: NodeId) -> Vec<NodeId> {
        if !self.nodes.contains_key(&root_id) {
            return Vec::new();
        }

        let removed_ids = self.collect_subtree_nodes(root_id);
        self.detach_from_parent(root_id);

        for id in &removed_ids {
            self.hidden_nodes.remove(id);
            self.light_masks.remove(id);
        }
        for id in &removed_ids {
            self.nodes.remove(id);
        }

        removed_ids
    }

    // --- Factories ---

    pub fn create_primitive(&mut self, kind: SdfPrimitive) -> NodeId {
        let name = self.next_name(kind.base_name());
        self.add_node(
            name,
            NodeData::Primitive {
                position: kind.default_position(),
                rotation: Vec3::ZERO,
                scale: kind.default_scale(),
                material: MaterialParams::with_base_color(kind.default_color()),
                kind,
                voxel_grid: None,
            },
        )
    }

    pub fn create_operation(
        &mut self,
        op: CsgOp,
        left: Option<NodeId>,
        right: Option<NodeId>,
    ) -> NodeId {
        let name = self.next_name(op.base_name());
        let steps = op.default_steps();
        self.add_node(
            name,
            NodeData::Operation {
                smooth_k: op.default_smooth_k(),
                steps,
                color_blend: -1.0,
                op,
                left,
                right,
            },
        )
    }

    pub fn create_sculpt(
        &mut self,
        input: NodeId,
        position: Vec3,
        rotation: Vec3,
        color: Vec3,
        voxel_grid: VoxelGrid,
    ) -> NodeId {
        let name = self.next_name("Sculpt");
        let desired_resolution = voxel_grid.resolution;
        self.add_node(
            name,
            NodeData::Sculpt {
                input: Some(input),
                position,
                rotation,
                material: MaterialParams::with_base_color(color),
                layer_intensity: 1.0,
                voxel_grid,
                desired_resolution,
            },
        )
    }

    pub fn create_transform(&mut self, input: Option<NodeId>) -> NodeId {
        let name = self.next_name("Transform");
        self.add_node(
            name,
            NodeData::Transform {
                input,
                translation: Vec3::ZERO,
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
            },
        )
    }

    pub fn create_reroute(&mut self, input: Option<NodeId>) -> NodeId {
        let name = self.next_name("Reroute");
        self.add_node(
            name,
            NodeData::Transform {
                input,
                translation: Vec3::ZERO,
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
            },
        )
    }

    pub fn create_modifier(&mut self, kind: ModifierKind, input: Option<NodeId>) -> NodeId {
        let name = self.next_name(kind.base_name());
        let value = kind.default_value();
        let extra = kind.default_extra();
        self.add_node(
            name,
            NodeData::Modifier {
                kind,
                input,
                value,
                extra,
            },
        )
    }

    pub fn create_light(&mut self, light_type: LightType) -> (NodeId, NodeId) {
        let is_array = matches!(light_type, LightType::Array);
        let light_name = self.next_name(if is_array {
            "Light Array"
        } else {
            light_type.label()
        });
        let light_id = self.add_node(
            light_name,
            NodeData::Light {
                light_type: light_type.clone(),
                color: Vec3::ONE,
                intensity: 1.0,
                range: if is_array { 5.0 } else { 10.0 },
                spot_angle: 45.0,
                cast_shadows: !matches!(light_type, LightType::Point | LightType::Array),
                shadow_softness: 8.0,
                shadow_color: Vec3::ZERO,
                volumetric: false,
                volumetric_density: 0.15,
                cookie_node: None,
                proximity_mode: ProximityMode::Off,
                proximity_range: 2.0,
                array_config: if is_array {
                    Some(LightArrayConfig::default())
                } else {
                    None
                },
                intensity_expr: None,
                color_hue_expr: None,
            },
        );
        // Create a Transform parent positioned above and to the side of the origin
        // so the light billboard is immediately visible (not buried inside geometry).
        let transform_name = self.next_name("Transform");
        let transform_id = self.add_node(
            transform_name,
            NodeData::Transform {
                input: Some(light_id),
                translation: Vec3::new(2.0, 3.0, 2.0),
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
            },
        );
        (light_id, transform_id)
    }

    /// Create default Key, Fill, and Ambient lights for a new scene.
    /// Key Light: warm white directional, intensity 1.5, direction (-0.5, -1.0, -0.3).
    /// Fill Light: cool blue-white directional, intensity 0.4, direction (0.3, -0.5, 0.6).
    /// Ambient Light: neutral white, intensity 0.05 (minimum base illumination).
    pub fn create_default_lights(&mut self) {
        // Key Light
        let key_light_id = self.add_node(
            "Key Light".to_string(),
            NodeData::Light {
                light_type: LightType::Directional,
                color: Vec3::new(1.0, 0.98, 0.95),
                intensity: 1.5,
                range: 10.0,
                spot_angle: 45.0,
                cast_shadows: true,
                shadow_softness: 8.0,
                shadow_color: Vec3::ZERO,
                volumetric: false,
                volumetric_density: 0.15,
                cookie_node: None,
                proximity_mode: ProximityMode::Off,
                proximity_range: 2.0,
                array_config: None,
                intensity_expr: None,
                color_hue_expr: None,
            },
        );
        let _key_transform_id = self.add_node(
            "Key Light Transform".to_string(),
            NodeData::Transform {
                input: Some(key_light_id),
                translation: Vec3::new(2.0, 4.0, 3.0),
                rotation: Vec3::new(0.9593, 0.5990, 0.8957),
                scale: Vec3::ONE,
            },
        );

        // Fill Light
        let fill_light_id = self.add_node(
            "Fill Light".to_string(),
            NodeData::Light {
                light_type: LightType::Directional,
                color: Vec3::new(0.85, 0.9, 1.0),
                intensity: 0.4,
                range: 10.0,
                spot_angle: 45.0,
                cast_shadows: false,
                shadow_softness: 8.0,
                shadow_color: Vec3::ZERO,
                volumetric: false,
                volumetric_density: 0.15,
                cookie_node: None,
                proximity_mode: ProximityMode::Off,
                proximity_range: 2.0,
                array_config: None,
                intensity_expr: None,
                color_hue_expr: None,
            },
        );
        let _fill_transform_id = self.add_node(
            "Fill Light Transform".to_string(),
            NodeData::Transform {
                input: Some(fill_light_id),
                translation: Vec3::new(-2.0, 2.0, -3.0),
                rotation: Vec3::new(-0.5400, 0.2128, -0.7362),
                scale: Vec3::ONE,
            },
        );

        // Ambient Light
        let ambient_light_id = self.add_node(
            "Ambient Light".to_string(),
            NodeData::Light {
                light_type: LightType::Ambient,
                color: Vec3::ONE,
                intensity: 0.05,
                range: 10.0,
                spot_angle: 45.0,
                cast_shadows: false,
                shadow_softness: 8.0,
                shadow_color: Vec3::ZERO,
                volumetric: false,
                volumetric_density: 0.15,
                cookie_node: None,
                proximity_mode: ProximityMode::Off,
                proximity_range: 2.0,
                array_config: None,
                intensity_expr: None,
                color_hue_expr: None,
            },
        );
        let _ambient_transform_id = self.add_node(
            "Ambient Light Transform".to_string(),
            NodeData::Transform {
                input: Some(ambient_light_id),
                translation: Vec3::ZERO,
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
            },
        );
    }

    fn rewire_parents_to(&mut self, target_id: NodeId, replacement_id: NodeId) {
        let parents: Vec<(NodeId, bool, bool)> = self
            .nodes
            .values()
            .filter_map(|node| {
                if node.id == replacement_id {
                    return None;
                }
                match &node.data {
                    NodeData::Operation { left, right, .. } => {
                        let is_left = *left == Some(target_id);
                        let is_right = *right == Some(target_id);
                        if is_left || is_right {
                            Some((node.id, is_left, is_right))
                        } else {
                            None
                        }
                    }
                    NodeData::Sculpt { input, .. }
                    | NodeData::Transform { input, .. }
                    | NodeData::Modifier { input, .. }
                        if *input == Some(target_id) =>
                    {
                        Some((node.id, true, false))
                    }
                    _ => None,
                }
            })
            .collect();

        for (parent_id, is_left, is_right) in parents {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                match &mut parent.data {
                    NodeData::Operation { left, right, .. } => {
                        if is_left {
                            *left = Some(replacement_id);
                        }
                        if is_right {
                            *right = Some(replacement_id);
                        }
                    }
                    NodeData::Sculpt { input, .. }
                    | NodeData::Transform { input, .. }
                    | NodeData::Modifier { input, .. } => {
                        *input = Some(replacement_id);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Insert a Modifier above `target_id`.
    /// Creates a Modifier node with `input = target_id` and rewires all parents.
    pub fn insert_modifier_above(&mut self, target_id: NodeId, kind: ModifierKind) -> NodeId {
        let modifier_id = self.create_modifier(kind, Some(target_id));
        self.rewire_parents_to(target_id, modifier_id);
        modifier_id
    }

    /// Insert a Transform modifier above `target_id`.
    /// Creates a Transform node with `input = target_id` and rewires all parents.
    pub fn insert_transform_above(&mut self, target_id: NodeId) -> NodeId {
        let transform_id = self.create_transform(Some(target_id));
        self.rewire_parents_to(target_id, transform_id);
        transform_id
    }

    /// Create a new primitive operand and wrap `target_id` in a boolean operation.
    /// Returns `(operation_id, operand_id)` when `target_id` exists.
    pub fn create_guided_boolean_primitive(
        &mut self,
        target_id: NodeId,
        op: CsgOp,
        primitive: SdfPrimitive,
    ) -> Option<(NodeId, NodeId)> {
        if !self.nodes.contains_key(&target_id) {
            return None;
        }

        let operand_id = self.create_primitive(primitive);
        let operation_id = self.create_operation(op, Some(target_id), Some(operand_id));
        self.rewire_parents_to(target_id, operation_id);
        Some((operation_id, operand_id))
    }

    /// Insert a Sculpt modifier above `target_id`.
    /// Creates a Sculpt node with `input = target_id` and rewires all parents.
    pub fn insert_sculpt_above(
        &mut self,
        target_id: NodeId,
        position: Vec3,
        rotation: Vec3,
        color: Vec3,
        voxel_grid: VoxelGrid,
    ) -> NodeId {
        let sculpt_id = self.create_sculpt(target_id, position, rotation, color, voxel_grid);

        // Rewire all parents that referenced target_id
        let parents: Vec<(NodeId, bool, bool)> = self
            .nodes
            .values()
            .filter_map(|n| {
                if n.id == sculpt_id {
                    return None;
                }
                match &n.data {
                    NodeData::Operation { left, right, .. } => {
                        let is_left = *left == Some(target_id);
                        let is_right = *right == Some(target_id);
                        if is_left || is_right {
                            Some((n.id, is_left, is_right))
                        } else {
                            None
                        }
                    }
                    NodeData::Sculpt { input, .. } if *input == Some(target_id) => {
                        Some((n.id, true, false))
                    }
                    NodeData::Transform { input, .. } if *input == Some(target_id) => {
                        Some((n.id, true, false))
                    }
                    NodeData::Modifier { input, .. } if *input == Some(target_id) => {
                        Some((n.id, true, false))
                    }
                    _ => None,
                }
            })
            .collect();

        for (parent_id, is_left, is_right) in parents {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                match &mut parent.data {
                    NodeData::Operation { left, right, .. } => {
                        if is_left {
                            *left = Some(sculpt_id);
                        }
                        if is_right {
                            *right = Some(sculpt_id);
                        }
                    }
                    NodeData::Sculpt { input, .. }
                    | NodeData::Transform { input, .. }
                    | NodeData::Modifier { input, .. } => {
                        *input = Some(sculpt_id);
                    }
                    _ => {}
                }
            }
        }

        sculpt_id
    }

    /// Insert a differential sculpt layer above `target_id` and wrap the
    /// resulting sculpt in an outer transform so object-level transforms move
    /// the analytical base and sculpt layer together.
    pub fn insert_sculpt_layer_above(
        &mut self,
        target_id: NodeId,
        position: Vec3,
        rotation: Vec3,
        color: Vec3,
        voxel_grid: VoxelGrid,
    ) -> (NodeId, NodeId) {
        let sculpt_id = self.insert_sculpt_above(target_id, position, rotation, color, voxel_grid);
        let transform_id = self.insert_transform_above(sculpt_id);
        (transform_id, sculpt_id)
    }

    /// Remove the first differential sculpt wrapper attached above `host_id`.
    /// Reconnects the sculpt's parents to its input child and preserves the rest of the object chain.
    pub fn remove_attached_sculpt(&mut self, host_id: NodeId) -> Option<NodeId> {
        if !self.nodes.contains_key(&host_id) {
            return None;
        }

        let parent_map = self.build_parent_map();
        let mut current = host_id;
        let mut sculpt_id = None;
        let mut sculpt_input_child = None;

        while let Some(&parent_id) = parent_map.get(&current) {
            let Some(parent) = self.nodes.get(&parent_id) else {
                break;
            };
            match &parent.data {
                NodeData::Transform {
                    input: Some(input_id),
                    ..
                }
                | NodeData::Modifier {
                    input: Some(input_id),
                    ..
                } if *input_id == current => {
                    current = parent_id;
                }
                NodeData::Sculpt {
                    input: Some(input_id),
                    ..
                } if *input_id == current => {
                    sculpt_id = Some(parent_id);
                    sculpt_input_child = Some(current);
                    break;
                }
                _ => break,
            }
        }

        let sculpt_id = sculpt_id?;
        let sculpt_input_child = sculpt_input_child?;
        self.rewire_parents_to(sculpt_id, sculpt_input_child);
        self.remove_node(sculpt_id);
        Some(sculpt_id)
    }

    /// Remove a single-input passthrough node and reconnect its parents to the child input.
    pub fn remove_passthrough_node(&mut self, node_id: NodeId) -> Option<NodeId> {
        let child_id = match self.nodes.get(&node_id).map(|node| &node.data) {
            Some(NodeData::Sculpt {
                input: Some(child_id),
                ..
            })
            | Some(NodeData::Transform {
                input: Some(child_id),
                ..
            })
            | Some(NodeData::Modifier {
                input: Some(child_id),
                ..
            }) => *child_id,
            _ => return None,
        };

        self.rewire_parents_to(node_id, child_id);
        self.remove_node(node_id);
        Some(child_id)
    }

    // --- Topology mutation ---

    pub fn set_left_child(&mut self, op_id: NodeId, child_id: Option<NodeId>) {
        if let Some(node) = self.nodes.get_mut(&op_id) {
            if let NodeData::Operation { left, .. } = &mut node.data {
                *left = child_id;
            }
        }
    }

    pub fn set_right_child(&mut self, op_id: NodeId, child_id: Option<NodeId>) {
        if let Some(node) = self.nodes.get_mut(&op_id) {
            if let NodeData::Operation { right, .. } = &mut node.data {
                *right = child_id;
            }
        }
    }

    pub fn set_sculpt_input(&mut self, node_id: NodeId, child_id: Option<NodeId>) {
        if let Some(node) = self.nodes.get_mut(&node_id) {
            match &mut node.data {
                NodeData::Sculpt { input, .. }
                | NodeData::Transform { input, .. }
                | NodeData::Modifier { input, .. } => {
                    *input = child_id;
                }
                _ => {}
            }
        }
    }

    pub fn swap_children(&mut self, op_id: NodeId) {
        if let Some(node) = self.nodes.get_mut(&op_id) {
            if let NodeData::Operation { left, right, .. } = &mut node.data {
                std::mem::swap(left, right);
            }
        }
    }

    // --- Reparenting ---

    /// Returns true if `candidate` is a descendant of `ancestor`.
    pub fn is_descendant(&self, candidate: NodeId, ancestor: NodeId) -> bool {
        let Some(node) = self.nodes.get(&ancestor) else {
            return false;
        };
        for child_id in node.data.children() {
            if child_id == candidate || self.is_descendant(candidate, child_id) {
                return true;
            }
        }
        false
    }

    /// Check if `target_id` is a valid drop target for `dragged_id`.
    /// Returns false if: same node, target is a descendant of dragged (cycle),
    /// target is a primitive, or target has no free child slot.
    pub fn is_valid_drop_target(&self, target_id: NodeId, dragged_id: NodeId) -> bool {
        if target_id == dragged_id {
            return false;
        }
        if self.is_descendant(target_id, dragged_id) {
            return false;
        }
        let Some(target_node) = self.nodes.get(&target_id) else {
            return false;
        };
        match &target_node.data {
            NodeData::Primitive { .. } | NodeData::Light { .. } => false,
            NodeData::Operation { left, right, .. } => left.is_none() || right.is_none(),
            NodeData::Sculpt { input, .. }
            | NodeData::Transform { input, .. }
            | NodeData::Modifier { input, .. } => input.is_none(),
        }
    }

    /// Detach a node from its parent (null out the parent's reference to it).
    /// If the node is already top-level, this is a no-op.
    pub fn detach_from_parent(&mut self, child_id: NodeId) {
        let parent_map = self.build_parent_map();
        let Some(&parent_id) = parent_map.get(&child_id) else {
            return;
        };
        if let Some(parent) = self.nodes.get_mut(&parent_id) {
            match &mut parent.data {
                NodeData::Operation { left, right, .. } => {
                    if *left == Some(child_id) {
                        *left = None;
                    }
                    if *right == Some(child_id) {
                        *right = None;
                    }
                }
                NodeData::Sculpt { input, .. }
                | NodeData::Transform { input, .. }
                | NodeData::Modifier { input, .. } => {
                    if *input == Some(child_id) {
                        *input = None;
                    }
                }
                _ => {}
            }
        }
    }

    /// Reparent: detach from old parent, attach to first free slot of new parent.
    pub fn reparent(&mut self, dragged_id: NodeId, target_id: NodeId) {
        self.detach_from_parent(dragged_id);
        if let Some(target) = self.nodes.get_mut(&target_id) {
            match &mut target.data {
                NodeData::Operation { left, right, .. } => {
                    if left.is_none() {
                        *left = Some(dragged_id);
                    } else if right.is_none() {
                        *right = Some(dragged_id);
                    }
                }
                NodeData::Sculpt { input, .. }
                | NodeData::Transform { input, .. }
                | NodeData::Modifier { input, .. } => {
                    if input.is_none() {
                        *input = Some(dragged_id);
                    }
                }
                _ => {}
            }
        }
    }

    // --- Graph analysis ---

    /// Hash of graph topology only (not parameter values).
    pub fn structure_key(&self) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.nodes.len().hash(&mut hasher);
        // Hash hidden_nodes so visibility changes trigger shader regen
        self.hidden_nodes.len().hash(&mut hasher);
        let mut hidden_sorted: Vec<NodeId> = self.hidden_nodes.iter().cloned().collect();
        hidden_sorted.sort();
        for id in &hidden_sorted {
            id.hash(&mut hasher);
        }
        let mut ids: Vec<NodeId> = self.nodes.keys().cloned().collect();
        ids.sort();
        for id in ids {
            let node = &self.nodes[&id];
            id.hash(&mut hasher);
            match &node.data {
                NodeData::Primitive { kind, .. } => {
                    0u8.hash(&mut hasher);
                    std::mem::discriminant(kind).hash(&mut hasher);
                }
                NodeData::Operation {
                    op, left, right, ..
                } => {
                    1u8.hash(&mut hasher);
                    std::mem::discriminant(op).hash(&mut hasher);
                    left.hash(&mut hasher);
                    right.hash(&mut hasher);
                }
                NodeData::Sculpt {
                    input, voxel_grid, ..
                } => {
                    2u8.hash(&mut hasher);
                    input.hash(&mut hasher);
                    voxel_grid.resolution.hash(&mut hasher);
                }
                NodeData::Transform { input, .. } => {
                    3u8.hash(&mut hasher);
                    input.hash(&mut hasher);
                }
                NodeData::Modifier { kind, input, .. } => {
                    4u8.hash(&mut hasher);
                    std::mem::discriminant(kind).hash(&mut hasher);
                    input.hash(&mut hasher);
                }
                NodeData::Light {
                    light_type,
                    cookie_node,
                    array_config,
                    ..
                } => {
                    5u8.hash(&mut hasher);
                    std::mem::discriminant(light_type).hash(&mut hasher);
                    cookie_node.hash(&mut hasher);
                    // Array count affects how many GPU lights are generated
                    if let Some(cfg) = array_config {
                        cfg.count.hash(&mut hasher);
                    }
                }
            }
        }
        hasher.finish()
    }

    /// Lightweight hash of all mutable node parameters (positions, rotations,
    /// scales, colors, smooth_k, etc.). Skips voxel data — voxel changes are
    /// tracked explicitly via dirty flags at brush/undo/redo sites.
    pub fn data_fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        let mut ids: Vec<NodeId> = self.nodes.keys().cloned().collect();
        ids.sort();
        for id in ids {
            let node = &self.nodes[&id];
            id.hash(&mut hasher);
            node.name.hash(&mut hasher);
            match &node.data {
                NodeData::Primitive {
                    position,
                    rotation,
                    scale,
                    material,
                    ..
                } => {
                    hash_vec3(*position, &mut hasher);
                    hash_vec3(*rotation, &mut hasher);
                    hash_vec3(*scale, &mut hasher);
                    hash_material_params(material, &mut hasher);
                }
                NodeData::Operation {
                    smooth_k,
                    steps,
                    color_blend,
                    ..
                } => {
                    smooth_k.to_bits().hash(&mut hasher);
                    steps.to_bits().hash(&mut hasher);
                    color_blend.to_bits().hash(&mut hasher);
                }
                NodeData::Sculpt {
                    position,
                    rotation,
                    material,
                    desired_resolution,
                    ..
                } => {
                    hash_vec3(*position, &mut hasher);
                    hash_vec3(*rotation, &mut hasher);
                    hash_material_params(material, &mut hasher);
                    desired_resolution.hash(&mut hasher);
                }
                NodeData::Transform {
                    translation,
                    rotation,
                    scale,
                    ..
                } => {
                    translation.x.to_bits().hash(&mut hasher);
                    translation.y.to_bits().hash(&mut hasher);
                    translation.z.to_bits().hash(&mut hasher);
                    rotation.x.to_bits().hash(&mut hasher);
                    rotation.y.to_bits().hash(&mut hasher);
                    rotation.z.to_bits().hash(&mut hasher);
                    scale.x.to_bits().hash(&mut hasher);
                    scale.y.to_bits().hash(&mut hasher);
                    scale.z.to_bits().hash(&mut hasher);
                }
                NodeData::Modifier { value, extra, .. } => {
                    value.x.to_bits().hash(&mut hasher);
                    value.y.to_bits().hash(&mut hasher);
                    value.z.to_bits().hash(&mut hasher);
                    extra.x.to_bits().hash(&mut hasher);
                    extra.y.to_bits().hash(&mut hasher);
                    extra.z.to_bits().hash(&mut hasher);
                }
                NodeData::Light {
                    color,
                    intensity,
                    range,
                    spot_angle,
                    cast_shadows,
                    shadow_softness,
                    shadow_color,
                    volumetric,
                    volumetric_density,
                    cookie_node,
                    proximity_mode,
                    proximity_range,
                    array_config,
                    intensity_expr,
                    color_hue_expr,
                    ..
                } => {
                    color.x.to_bits().hash(&mut hasher);
                    color.y.to_bits().hash(&mut hasher);
                    color.z.to_bits().hash(&mut hasher);
                    intensity.to_bits().hash(&mut hasher);
                    range.to_bits().hash(&mut hasher);
                    spot_angle.to_bits().hash(&mut hasher);
                    cast_shadows.hash(&mut hasher);
                    shadow_softness.to_bits().hash(&mut hasher);
                    shadow_color.x.to_bits().hash(&mut hasher);
                    shadow_color.y.to_bits().hash(&mut hasher);
                    shadow_color.z.to_bits().hash(&mut hasher);
                    volumetric.hash(&mut hasher);
                    volumetric_density.to_bits().hash(&mut hasher);
                    cookie_node.hash(&mut hasher);
                    std::mem::discriminant(proximity_mode).hash(&mut hasher);
                    proximity_range.to_bits().hash(&mut hasher);
                    if let Some(cfg) = array_config {
                        std::mem::discriminant(&cfg.pattern).hash(&mut hasher);
                        cfg.count.hash(&mut hasher);
                        cfg.radius.to_bits().hash(&mut hasher);
                        cfg.color_variation.to_bits().hash(&mut hasher);
                    }
                    intensity_expr.hash(&mut hasher);
                    color_hue_expr.hash(&mut hasher);
                }
            }
            // Include light mask in fingerprint
            let mask = self.get_light_mask(id);
            mask.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Returns nodes not referenced as a child by any other node.
    pub fn top_level_nodes(&self) -> Vec<NodeId> {
        let referenced: HashSet<NodeId> = self
            .nodes
            .values()
            .flat_map(|n| n.data.children())
            .collect();
        let mut top: Vec<NodeId> = self
            .nodes
            .keys()
            .filter(|id| !referenced.contains(id))
            .cloned()
            .collect();
        top.sort();
        top
    }

    /// Post-order traversal that skips hidden nodes and their entire subtrees.
    /// Used by codegen and buffer upload — hidden geometry should not appear in the shader.
    pub fn visible_topo_order(&self) -> Vec<NodeId> {
        let tops = self.top_level_nodes();
        if tops.is_empty() {
            return Vec::new();
        }
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        for &root in &tops {
            self.visible_topo_visit(root, &mut visited, &mut result);
        }
        result
    }

    fn visible_topo_visit(
        &self,
        id: NodeId,
        visited: &mut HashSet<NodeId>,
        result: &mut Vec<NodeId>,
    ) {
        if !visited.insert(id) {
            return;
        }
        if self.hidden_nodes.contains(&id) {
            return;
        }
        let Some(node) = self.nodes.get(&id) else {
            return;
        };
        for child_id in node.data.children() {
            self.visible_topo_visit(child_id, visited, result);
        }
        result.push(id);
    }

    // --- Tree traversal helpers ---

    /// Build a parent map: child_id → parent_id.
    pub fn build_parent_map(&self) -> HashMap<NodeId, NodeId> {
        let mut map = HashMap::new();
        for node in self.nodes.values() {
            for child_id in node.data.children() {
                map.insert(child_id, node.id);
            }
        }
        map
    }

    /// Walk upward from `start` through the parent map. Return the first
    /// ancestor whose `NodeData` is `Sculpt` and that has `start` somewhere
    /// in its input chain.
    pub fn find_sculpt_parent(
        &self,
        start: NodeId,
        parent_map: &HashMap<NodeId, NodeId>,
    ) -> Option<NodeId> {
        let mut current = start;
        while let Some(&parent_id) = parent_map.get(&current) {
            if let Some(parent_node) = self.nodes.get(&parent_id) {
                if matches!(parent_node.data, NodeData::Sculpt { .. }) {
                    return Some(parent_id);
                }
            }
            current = parent_id;
        }
        None
    }

    /// Walk up from a leaf through ancestor transforms and compute world-space
    /// bounding sphere (center, radius).
    pub fn walk_transforms_sphere(
        &self,
        center: [f32; 3],
        extent: f32,
        leaf_id: NodeId,
        parent_map: &HashMap<NodeId, NodeId>,
    ) -> ([f32; 3], f32) {
        let mut wc = center;
        let mut wr = extent;
        let mut current = leaf_id;
        while let Some(&pid) = parent_map.get(&current) {
            if let Some(parent) = self.nodes.get(&pid) {
                if let NodeData::Transform {
                    translation,
                    rotation,
                    scale,
                    ..
                } = &parent.data
                {
                    // Scale: expand radius, scale center
                    let s = scale.x.abs().max(scale.y.abs()).max(scale.z.abs());
                    wr *= s;
                    wc[0] *= scale.x;
                    wc[1] *= scale.y;
                    wc[2] *= scale.z;
                    // Rotate: conservative sphere expansion
                    if rotation.length_squared() > 1e-12 {
                        let dist = (wc[0] * wc[0] + wc[1] * wc[1] + wc[2] * wc[2]).sqrt();
                        wr += dist;
                        wc = [0.0, 0.0, 0.0];
                    }
                    // Translate: offset center
                    wc[0] += translation.x;
                    wc[1] += translation.y;
                    wc[2] += translation.z;
                }
            }
            current = pid;
        }
        (wc, wr)
    }

    /// Check if any node in the subtree rooted at `root` is a Sculpt node.
    pub fn subtree_has_sculpt(&self, root: NodeId) -> bool {
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if let Some(node) = self.nodes.get(&id) {
                if matches!(node.data, NodeData::Sculpt { .. }) {
                    return true;
                }
                stack.extend(node.data.children());
            }
        }
        false
    }

    /// Collect all node IDs in the subtree rooted at `root`.
    pub fn collect_subtree(&self, root: NodeId) -> HashSet<NodeId> {
        let mut set = HashSet::new();
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            if !set.insert(id) {
                continue;
            }
            if let Some(node) = self.nodes.get(&id) {
                stack.extend(node.data.children());
            }
        }
        set
    }

    /// Collect all node IDs in the subtree rooted at `root` as a stable, sorted list.
    pub fn collect_subtree_nodes(&self, root: NodeId) -> Vec<NodeId> {
        let mut nodes: Vec<NodeId> = self.collect_subtree(root).into_iter().collect();
        nodes.sort_unstable();
        nodes
    }

    /// Duplicate an entire subtree rooted at `root_id`.
    /// Returns the new root ID, or None if `root_id` doesn't exist.
    /// Sculpt nodes get their voxel grids deep-cloned. Names get " Copy" appended.
    pub fn duplicate_subtree(&mut self, root_id: NodeId) -> Option<NodeId> {
        let subtree = self.collect_subtree(root_id);
        if subtree.is_empty() || !self.nodes.contains_key(&root_id) {
            return None;
        }

        // Allocate new IDs
        let mut id_map: HashMap<NodeId, NodeId> = HashMap::new();
        for &old_id in &subtree {
            let new_id = self.next_id;
            self.next_id += 1;
            id_map.insert(old_id, new_id);
        }

        // Clone nodes with remapped references
        let remap = |opt: &Option<NodeId>| -> Option<NodeId> {
            opt.and_then(|id| {
                if subtree.contains(&id) {
                    id_map.get(&id).copied()
                } else {
                    Some(id) // external reference, keep as-is
                }
            })
        };

        let cloned_nodes: Vec<SceneNode> = subtree
            .iter()
            .filter_map(|&old_id| {
                let node = self.nodes.get(&old_id)?;
                let new_id = *id_map.get(&old_id)?;
                let new_data = match &node.data {
                    NodeData::Primitive {
                        kind,
                        position,
                        rotation,
                        scale,
                        material,
                        ..
                    } => NodeData::Primitive {
                        kind: kind.clone(),
                        position: *position,
                        rotation: *rotation,
                        scale: *scale,
                        material: material.clone(),
                        voxel_grid: None,
                    },
                    NodeData::Operation {
                        op,
                        smooth_k,
                        steps,
                        color_blend,
                        left,
                        right,
                    } => NodeData::Operation {
                        op: op.clone(),
                        smooth_k: *smooth_k,
                        steps: *steps,
                        color_blend: *color_blend,
                        left: remap(left),
                        right: remap(right),
                    },
                    NodeData::Sculpt {
                        input,
                        position,
                        rotation,
                        material,
                        layer_intensity,
                        voxel_grid,
                        desired_resolution,
                    } => NodeData::Sculpt {
                        input: remap(input),
                        position: *position,
                        rotation: *rotation,
                        material: material.clone(),
                        layer_intensity: *layer_intensity,
                        voxel_grid: voxel_grid.clone(),
                        desired_resolution: *desired_resolution,
                    },
                    NodeData::Transform {
                        input,
                        translation,
                        rotation,
                        scale,
                    } => NodeData::Transform {
                        input: remap(input),
                        translation: *translation,
                        rotation: *rotation,
                        scale: *scale,
                    },
                    NodeData::Modifier {
                        kind,
                        input,
                        value,
                        extra,
                    } => NodeData::Modifier {
                        kind: kind.clone(),
                        input: remap(input),
                        value: *value,
                        extra: *extra,
                    },
                    NodeData::Light {
                        light_type,
                        color,
                        intensity,
                        range,
                        spot_angle,
                        cast_shadows,
                        shadow_softness,
                        shadow_color,
                        volumetric,
                        volumetric_density,
                        cookie_node,
                        proximity_mode,
                        proximity_range,
                        array_config,
                        intensity_expr,
                        color_hue_expr,
                    } => NodeData::Light {
                        light_type: light_type.clone(),
                        color: *color,
                        intensity: *intensity,
                        range: *range,
                        spot_angle: *spot_angle,
                        cast_shadows: *cast_shadows,
                        shadow_softness: *shadow_softness,
                        shadow_color: *shadow_color,
                        volumetric: *volumetric,
                        volumetric_density: *volumetric_density,
                        cookie_node: *cookie_node,
                        proximity_mode: proximity_mode.clone(),
                        proximity_range: *proximity_range,
                        array_config: array_config.clone(),
                        intensity_expr: intensity_expr.clone(),
                        color_hue_expr: color_hue_expr.clone(),
                    },
                };
                Some(SceneNode {
                    id: new_id,
                    name: format!("{} Copy", node.name),
                    data: new_data,
                    locked: false,
                })
            })
            .collect();

        for node in cloned_nodes {
            self.nodes.insert(node.id, node);
        }

        // Copy light masks for duplicated nodes
        for (&old_id, &new_id) in &id_map {
            if let Some(&mask) = self.light_masks.get(&old_id) {
                self.light_masks.insert(new_id, mask);
            }
        }

        id_map.get(&root_id).copied()
    }

    /// Compute world-space bounding sphere (center, radius) for a subtree.
    /// `parent_map` should be pre-built via `build_parent_map()`.
    pub fn compute_subtree_sphere(
        &self,
        root: NodeId,
        parent_map: &HashMap<NodeId, NodeId>,
    ) -> ([f32; 3], f32) {
        let subtree_nodes = self.collect_subtree(root);
        let mut bmin = [f32::MAX; 3];
        let mut bmax = [f32::MIN; 3];
        let mut has_geom = false;
        let mut max_smooth_k: f32 = 0.0;

        for &nid in &subtree_nodes {
            if let Some(node) = self.nodes.get(&nid) {
                if let NodeData::Operation { smooth_k, .. } = &node.data {
                    max_smooth_k = max_smooth_k.max(*smooth_k);
                }
                if let Some((center, extent)) = node.data.geometry_local_sphere() {
                    let (wc, wr) = self.walk_transforms_sphere(center, extent, nid, parent_map);
                    for i in 0..3 {
                        bmin[i] = bmin[i].min(wc[i] - wr);
                        bmax[i] = bmax[i].max(wc[i] + wr);
                    }
                    has_geom = true;
                }
            }
        }

        if !has_geom {
            return ([0.0; 3], 1.0);
        }

        let pad = max_smooth_k + 0.5;
        let center = [
            (bmin[0] + bmax[0]) * 0.5,
            (bmin[1] + bmax[1]) * 0.5,
            (bmin[2] + bmax[2]) * 0.5,
        ];
        let half = [
            (bmax[0] - bmin[0]) * 0.5 + pad,
            (bmax[1] - bmin[1]) * 0.5 + pad,
            (bmax[2] - bmin[2]) * 0.5 + pad,
        ];
        let radius = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt();
        (center, radius)
    }

    /// Flatten a subtree into a single standalone Sculpt node.
    /// Replaces the entire subtree rooted at `subtree_root` with a new Sculpt node
    /// containing the pre-baked `voxel_grid`. Returns the new node's ID.
    pub fn flatten_subtree(
        &mut self,
        subtree_root: NodeId,
        voxel_grid: VoxelGrid,
        center: Vec3,
        color: Vec3,
    ) -> NodeId {
        // 1. Collect all nodes in the subtree (to delete later)
        let subtree_ids = self.collect_subtree(subtree_root);

        // 2. Find parents that reference subtree_root and record their rewiring info
        let parents: Vec<(NodeId, bool, bool)> = self
            .nodes
            .values()
            .filter_map(|n| {
                if subtree_ids.contains(&n.id) {
                    return None; // Skip nodes that are part of the subtree
                }
                match &n.data {
                    NodeData::Operation { left, right, .. } => {
                        let is_left = *left == Some(subtree_root);
                        let is_right = *right == Some(subtree_root);
                        if is_left || is_right {
                            Some((n.id, is_left, is_right))
                        } else {
                            None
                        }
                    }
                    NodeData::Sculpt { input, .. } if *input == Some(subtree_root) => {
                        Some((n.id, true, false))
                    }
                    NodeData::Transform { input, .. } if *input == Some(subtree_root) => {
                        Some((n.id, true, false))
                    }
                    NodeData::Modifier { input, .. } if *input == Some(subtree_root) => {
                        Some((n.id, true, false))
                    }
                    _ => None,
                }
            })
            .collect();

        // 3. Add the new standalone Sculpt node (input: None)
        let desired_resolution = voxel_grid.resolution;
        let name = self.next_name("Sculpt");
        let new_id = self.add_node(
            name,
            NodeData::Sculpt {
                input: None,
                position: center,
                rotation: Vec3::ZERO,
                material: MaterialParams::with_base_color(color),
                layer_intensity: 1.0,
                voxel_grid,
                desired_resolution,
            },
        );

        // 4. Rewire parents: subtree_root → new_id
        for (parent_id, is_left, is_right) in parents {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                match &mut parent.data {
                    NodeData::Operation { left, right, .. } => {
                        if is_left {
                            *left = Some(new_id);
                        }
                        if is_right {
                            *right = Some(new_id);
                        }
                    }
                    NodeData::Sculpt { input, .. }
                    | NodeData::Transform { input, .. }
                    | NodeData::Modifier { input, .. } => {
                        *input = Some(new_id);
                    }
                    _ => {}
                }
            }
        }

        // 5. Delete all subtree nodes (now orphaned)
        for id in subtree_ids {
            self.nodes.remove(&id);
        }

        new_id
    }

    /// Compute a conservative world-space AABB encompassing all scene geometry.
    /// Walks the tree to properly compose transforms with their children's bounds.
    pub fn compute_bounds(&self) -> ([f32; 3], [f32; 3]) {
        let parent_map = self.build_parent_map();
        let mut bmin = [f32::MAX; 3];
        let mut bmax = [f32::MIN; 3];
        let mut has_geometry = false;

        for node in self.nodes.values() {
            if let Some((center, extent)) = node.data.geometry_local_sphere() {
                let (wc, wr) = self.walk_transforms_sphere(center, extent, node.id, &parent_map);
                for i in 0..3 {
                    bmin[i] = bmin[i].min(wc[i] - wr);
                    bmax[i] = bmax[i].max(wc[i] + wr);
                }
                has_geometry = true;
            }
        }

        if !has_geometry {
            return ([-5.0; 3], [5.0; 3]);
        }

        for i in 0..3 {
            bmin[i] -= 1.5;
            bmax[i] += 1.5;
        }
        (bmin, bmax)
    }

    // -----------------------------------------------------------------------
    // Scene statistics helpers
    // -----------------------------------------------------------------------

    /// Count of nodes by type.
    pub fn node_type_counts(&self) -> NodeTypeCounts {
        let mut counts = NodeTypeCounts::default();
        for node in self.nodes.values() {
            counts.total += 1;
            if !self.hidden_nodes.contains(&node.id) {
                counts.visible += 1;
            }
            match &node.data {
                NodeData::Primitive { .. } => counts.primitives += 1,
                NodeData::Operation { .. } => counts.operations += 1,
                NodeData::Transform { .. } => counts.transforms += 1,
                NodeData::Modifier { .. } => counts.modifiers += 1,
                NodeData::Sculpt { .. } => counts.sculpts += 1,
                NodeData::Light { .. } => counts.lights += 1,
            }
        }
        counts
    }

    /// Total memory used by all VoxelGrid allocations, in bytes.
    pub fn voxel_memory_bytes(&self) -> usize {
        let mut total = 0;
        for node in self.nodes.values() {
            if let NodeData::Sculpt { voxel_grid, .. } = &node.data {
                // Each voxel is an f32 (4 bytes)
                total += voxel_grid.data.len() * std::mem::size_of::<f32>();
            }
        }
        total
    }

    /// Estimate SDF evaluation complexity: count nodes in visible_topo_order
    /// that contribute to the SDF (primitives, operations, modifiers, transforms, sculpts).
    pub fn sdf_eval_complexity(&self) -> usize {
        self.visible_topo_order()
            .iter()
            .filter(|id| {
                if let Some(node) = self.nodes.get(id) {
                    !matches!(&node.data, NodeData::Light { .. })
                } else {
                    false
                }
            })
            .count()
    }

    /// Deep equality check (topology + parameters). Used by undo system.
    pub fn content_eq(&self, other: &Scene) -> bool {
        if self.hidden_nodes != other.hidden_nodes {
            return false;
        }
        if self.light_masks != other.light_masks {
            return false;
        }
        if self.nodes.len() != other.nodes.len() {
            return false;
        }
        for (id, node) in &self.nodes {
            let Some(other_node) = other.nodes.get(id) else {
                return false;
            };
            if node.name != other_node.name {
                return false;
            }
            match (&node.data, &other_node.data) {
                (
                    NodeData::Primitive {
                        kind: k1,
                        position: p1,
                        rotation: r1,
                        scale: s1,
                        material: m1,
                        ..
                    },
                    NodeData::Primitive {
                        kind: k2,
                        position: p2,
                        rotation: r2,
                        scale: s2,
                        material: m2,
                        ..
                    },
                ) => {
                    if std::mem::discriminant(k1) != std::mem::discriminant(k2)
                        || p1 != p2
                        || r1 != r2
                        || s1 != s2
                        || m1 != m2
                    {
                        return false;
                    }
                }
                (
                    NodeData::Operation {
                        op: o1,
                        smooth_k: k1,
                        steps: s1,
                        color_blend: cb1,
                        left: l1,
                        right: r1,
                    },
                    NodeData::Operation {
                        op: o2,
                        smooth_k: k2,
                        steps: s2,
                        color_blend: cb2,
                        left: l2,
                        right: r2,
                    },
                ) => {
                    if std::mem::discriminant(o1) != std::mem::discriminant(o2)
                        || k1 != k2
                        || s1 != s2
                        || cb1 != cb2
                        || l1 != l2
                        || r1 != r2
                    {
                        return false;
                    }
                }
                (
                    NodeData::Sculpt {
                        input: i1,
                        position: p1,
                        rotation: r1,
                        material: m1,
                        layer_intensity: li1,
                        voxel_grid: v1,
                        desired_resolution: dr1,
                    },
                    NodeData::Sculpt {
                        input: i2,
                        position: p2,
                        rotation: r2,
                        material: m2,
                        layer_intensity: li2,
                        voxel_grid: v2,
                        desired_resolution: dr2,
                    },
                ) => {
                    if i1 != i2
                        || p1 != p2
                        || r1 != r2
                        || m1 != m2
                        || li1 != li2
                        || dr1 != dr2
                        || !v1.content_eq(v2)
                    {
                        return false;
                    }
                }
                (
                    NodeData::Transform {
                        input: i1,
                        translation: t1,
                        rotation: r1,
                        scale: s1,
                    },
                    NodeData::Transform {
                        input: i2,
                        translation: t2,
                        rotation: r2,
                        scale: s2,
                    },
                ) => {
                    if i1 != i2 || t1 != t2 || r1 != r2 || s1 != s2 {
                        return false;
                    }
                }
                (
                    NodeData::Modifier {
                        kind: k1,
                        input: i1,
                        value: v1,
                        extra: e1,
                    },
                    NodeData::Modifier {
                        kind: k2,
                        input: i2,
                        value: v2,
                        extra: e2,
                    },
                ) => {
                    if std::mem::discriminant(k1) != std::mem::discriminant(k2)
                        || i1 != i2
                        || v1 != v2
                        || e1 != e2
                    {
                        return false;
                    }
                }
                (
                    NodeData::Light {
                        light_type: lt1,
                        color: c1,
                        intensity: int1,
                        range: r1,
                        spot_angle: sa1,
                        cast_shadows: cs1,
                        shadow_softness: ss1,
                        shadow_color: sc1,
                        volumetric: vol1,
                        volumetric_density: vd1,
                        cookie_node: ck1,
                        proximity_mode: pm1,
                        proximity_range: pr1,
                        array_config: ac1,
                        intensity_expr: ie1,
                        color_hue_expr: ce1,
                    },
                    NodeData::Light {
                        light_type: lt2,
                        color: c2,
                        intensity: int2,
                        range: r2,
                        spot_angle: sa2,
                        cast_shadows: cs2,
                        shadow_softness: ss2,
                        shadow_color: sc2,
                        volumetric: vol2,
                        volumetric_density: vd2,
                        cookie_node: ck2,
                        proximity_mode: pm2,
                        proximity_range: pr2,
                        array_config: ac2,
                        intensity_expr: ie2,
                        color_hue_expr: ce2,
                    },
                ) => {
                    if std::mem::discriminant(lt1) != std::mem::discriminant(lt2)
                        || c1 != c2
                        || int1 != int2
                        || r1 != r2
                        || sa1 != sa2
                        || cs1 != cs2
                        || ss1 != ss2
                        || sc1 != sc2
                        || vol1 != vol2
                        || vd1 != vd2
                        || ck1 != ck2
                        || pm1 != pm2
                        || pr1 != pr2
                        || ac1 != ac2
                        || ie1 != ie2
                        || ce1 != ce2
                    {
                        return false;
                    }
                }
                _ => return false,
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create an empty scene (no default sphere) for predictable testing.
    fn empty_scene() -> Scene {
        Scene {
            nodes: HashMap::new(),
            next_id: 0,
            name_counters: HashMap::new(),
            hidden_nodes: HashSet::new(),
            light_masks: HashMap::new(),
        }
    }

    // ── Scene::new ──────────────────────────────────────────────────

    #[test]
    fn new_scene_has_sphere_and_default_lights() {
        let scene = Scene::new();
        // 1 sphere + 3 lights (Key, Fill, Ambient) + 3 light transforms = 7 nodes
        assert_eq!(scene.nodes.len(), 7);
        let has_sphere = scene.nodes.values().any(|n| {
            matches!(
                n.data,
                NodeData::Primitive {
                    kind: SdfPrimitive::Sphere,
                    ..
                }
            )
        });
        assert!(has_sphere);
        let light_count = scene
            .nodes
            .values()
            .filter(|n| matches!(n.data, NodeData::Light { .. }))
            .count();
        assert_eq!(light_count, 3);
    }

    // ── add_node / create factories ─────────────────────────────────

    #[test]
    fn add_node_returns_incrementing_ids() {
        let mut scene = empty_scene();
        let id_a = scene.create_primitive(SdfPrimitive::Sphere);
        let id_b = scene.create_primitive(SdfPrimitive::Box);
        assert_eq!(id_b, id_a + 1);
        assert_eq!(scene.nodes.len(), 2);
    }

    #[test]
    fn create_operation_links_children() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        let op = scene.create_operation(CsgOp::Union, Some(left), Some(right));
        match &scene.nodes[&op].data {
            NodeData::Operation {
                left: l, right: r, ..
            } => {
                assert_eq!(*l, Some(left));
                assert_eq!(*r, Some(right));
            }
            _ => panic!("expected Operation"),
        }
    }

    #[test]
    fn guided_boolean_primitive_wraps_top_level_target() {
        let mut scene = empty_scene();
        let target = scene.create_primitive(SdfPrimitive::Sphere);

        let (operation_id, operand_id) = scene
            .create_guided_boolean_primitive(target, CsgOp::Union, SdfPrimitive::Box)
            .expect("target should exist");

        match &scene.nodes[&operation_id].data {
            NodeData::Operation { left, right, op, .. } => {
                assert_eq!(*op, CsgOp::Union);
                assert_eq!(*left, Some(target));
                assert_eq!(*right, Some(operand_id));
            }
            _ => panic!("expected Operation"),
        }

        assert_eq!(scene.top_level_nodes(), vec![operation_id]);
    }

    #[test]
    fn guided_boolean_primitive_rewires_existing_parent_chain() {
        let mut scene = empty_scene();
        let target = scene.create_primitive(SdfPrimitive::Sphere);
        let transform = scene.create_transform(Some(target));

        let (operation_id, operand_id) = scene
            .create_guided_boolean_primitive(target, CsgOp::Subtract, SdfPrimitive::Cylinder)
            .expect("target should exist");

        match &scene.nodes[&transform].data {
            NodeData::Transform { input, .. } => assert_eq!(*input, Some(operation_id)),
            _ => panic!("expected Transform"),
        }
        match &scene.nodes[&operation_id].data {
            NodeData::Operation { left, right, op, .. } => {
                assert_eq!(*op, CsgOp::Subtract);
                assert_eq!(*left, Some(target));
                assert_eq!(*right, Some(operand_id));
            }
            _ => panic!("expected Operation"),
        }
    }

    #[test]
    fn create_transform_links_input() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Cylinder);
        let xform = scene.create_transform(Some(prim));
        match &scene.nodes[&xform].data {
            NodeData::Transform { input, .. } => assert_eq!(*input, Some(prim)),
            _ => panic!("expected Transform"),
        }
    }

    #[test]
    fn create_modifier_links_input() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let modifier = scene.create_modifier(ModifierKind::Round, Some(prim));
        match &scene.nodes[&modifier].data {
            NodeData::Modifier { input, kind, .. } => {
                assert_eq!(*input, Some(prim));
                assert_eq!(*kind, ModifierKind::Round);
            }
            _ => panic!("expected Modifier"),
        }
    }

    #[test]
    fn create_reroute_links_input() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let reroute = scene.create_reroute(Some(prim));
        match &scene.nodes[&reroute].data {
            NodeData::Transform { input, .. } => assert_eq!(*input, Some(prim)),
            _ => panic!("expected Transform"),
        }
        assert!(scene.nodes[&reroute].name.starts_with("Reroute"));
    }
    // ── next_name ───────────────────────────────────────────────────

    #[test]
    fn next_name_increments_counter() {
        let mut scene = empty_scene();
        assert_eq!(scene.next_name("Sphere"), "Sphere");
        assert_eq!(scene.next_name("Sphere"), "Sphere 2");
        assert_eq!(scene.next_name("Sphere"), "Sphere 3");
        assert_eq!(scene.next_name("Box"), "Box"); // independent counter
    }

    // ── remove_node ─────────────────────────────────────────────────

    #[test]
    fn remove_node_patches_operation_parent() {
        let mut scene = empty_scene();
        let child = scene.create_primitive(SdfPrimitive::Sphere);
        let op = scene.create_operation(CsgOp::Union, Some(child), None);

        let removed = scene.remove_node(child);
        assert!(removed.is_some());
        assert!(!scene.nodes.contains_key(&child));

        // Parent's left slot should be nulled
        match &scene.nodes[&op].data {
            NodeData::Operation { left, .. } => assert_eq!(*left, None),
            _ => panic!("expected Operation"),
        }
    }

    #[test]
    fn remove_node_patches_transform_parent() {
        let mut scene = empty_scene();
        let child = scene.create_primitive(SdfPrimitive::Box);
        let xform = scene.create_transform(Some(child));

        scene.remove_node(child);
        match &scene.nodes[&xform].data {
            NodeData::Transform { input, .. } => assert_eq!(*input, None),
            _ => panic!("expected Transform"),
        }
    }

    #[test]
    fn remove_nonexistent_node_returns_none() {
        let mut scene = empty_scene();
        assert!(scene.remove_node(999).is_none());
    }

    #[test]
    fn remove_subtree_deletes_entire_branch() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right_leaf = scene.create_primitive(SdfPrimitive::Box);
        let right = scene.insert_transform_above(right_leaf);
        let root = scene.create_operation(CsgOp::Union, Some(left), Some(right));

        let removed = scene.remove_subtree(right);

        assert_eq!(removed, vec![right_leaf, right]);
        assert!(scene.nodes.contains_key(&left));
        assert!(scene.nodes.contains_key(&root));
        let NodeData::Operation { right: root_right, .. } = &scene.nodes[&root].data else {
            panic!("expected operation");
        };
        assert_eq!(*root_right, None);
    }

    // ── visibility ──────────────────────────────────────────────────

    #[test]
    fn toggle_visibility_hides_and_unhides() {
        let mut scene = empty_scene();
        let id = scene.create_primitive(SdfPrimitive::Sphere);
        assert!(!scene.is_hidden(id));
        scene.toggle_visibility(id);
        assert!(scene.is_hidden(id));
        scene.toggle_visibility(id);
        assert!(!scene.is_hidden(id));
    }

    // ── top_level_nodes ─────────────────────────────────────────────

    #[test]
    fn top_level_nodes_excludes_children() {
        let mut scene = empty_scene();
        let child_a = scene.create_primitive(SdfPrimitive::Sphere);
        let child_b = scene.create_primitive(SdfPrimitive::Box);
        let op = scene.create_operation(CsgOp::Union, Some(child_a), Some(child_b));

        let tops = scene.top_level_nodes();
        assert_eq!(tops, vec![op]);
    }

    #[test]
    fn top_level_nodes_includes_orphans() {
        let mut scene = empty_scene();
        let a = scene.create_primitive(SdfPrimitive::Sphere);
        let b = scene.create_primitive(SdfPrimitive::Box);

        let mut tops = scene.top_level_nodes();
        tops.sort();
        assert_eq!(tops, vec![a, b]);
    }

    // ── is_descendant ───────────────────────────────────────────────

    #[test]
    fn is_descendant_detects_children() {
        let mut scene = empty_scene();
        let child = scene.create_primitive(SdfPrimitive::Sphere);
        let parent = scene.create_operation(CsgOp::Union, Some(child), None);
        assert!(scene.is_descendant(child, parent));
        assert!(!scene.is_descendant(parent, child));
    }

    #[test]
    fn is_descendant_detects_grandchildren() {
        let mut scene = empty_scene();
        let grandchild = scene.create_primitive(SdfPrimitive::Sphere);
        let child = scene.create_transform(Some(grandchild));
        let root = scene.create_operation(CsgOp::Union, Some(child), None);
        assert!(scene.is_descendant(grandchild, root));
        assert!(!scene.is_descendant(root, grandchild));
    }

    // ── is_valid_drop_target ────────────────────────────────────────

    #[test]
    fn drop_target_rejects_same_node() {
        let mut scene = empty_scene();
        let id = scene.create_primitive(SdfPrimitive::Sphere);
        assert!(!scene.is_valid_drop_target(id, id));
    }

    #[test]
    fn drop_target_rejects_descendant_cycle() {
        let mut scene = empty_scene();
        let child = scene.create_primitive(SdfPrimitive::Sphere);
        let parent = scene.create_operation(CsgOp::Union, Some(child), None);
        // Dropping parent onto child would create a cycle
        assert!(!scene.is_valid_drop_target(child, parent));
    }

    #[test]
    fn drop_target_rejects_primitive() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let dragged = scene.create_primitive(SdfPrimitive::Box);
        assert!(!scene.is_valid_drop_target(prim, dragged));
    }

    #[test]
    fn drop_target_accepts_operation_with_free_slot() {
        let mut scene = empty_scene();
        let op = scene.create_operation(CsgOp::Union, None, None);
        let dragged = scene.create_primitive(SdfPrimitive::Sphere);
        assert!(scene.is_valid_drop_target(op, dragged));
    }

    #[test]
    fn drop_target_rejects_full_operation() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        let op = scene.create_operation(CsgOp::Union, Some(left), Some(right));
        let dragged = scene.create_primitive(SdfPrimitive::Cylinder);
        assert!(!scene.is_valid_drop_target(op, dragged));
    }

    // ── reparent / detach ───────────────────────────────────────────

    #[test]
    fn reparent_moves_node_to_new_parent() {
        let mut scene = empty_scene();
        let child = scene.create_primitive(SdfPrimitive::Sphere);
        let old_parent = scene.create_operation(CsgOp::Union, Some(child), None);
        let new_parent = scene.create_operation(CsgOp::Subtract, None, None);

        scene.reparent(child, new_parent);

        // Old parent's slot should be empty
        match &scene.nodes[&old_parent].data {
            NodeData::Operation { left, .. } => assert_eq!(*left, None),
            _ => panic!(),
        }
        // New parent's left slot should have child
        match &scene.nodes[&new_parent].data {
            NodeData::Operation { left, .. } => assert_eq!(*left, Some(child)),
            _ => panic!(),
        }
    }

    #[test]
    fn detach_from_parent_nulls_reference() {
        let mut scene = empty_scene();
        let child = scene.create_primitive(SdfPrimitive::Sphere);
        let parent = scene.create_transform(Some(child));

        scene.detach_from_parent(child);
        match &scene.nodes[&parent].data {
            NodeData::Transform { input, .. } => assert_eq!(*input, None),
            _ => panic!(),
        }
    }

    // ── swap_children ───────────────────────────────────────────────

    #[test]
    fn swap_children_swaps_left_and_right() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        let op = scene.create_operation(CsgOp::Subtract, Some(left), Some(right));

        scene.swap_children(op);
        match &scene.nodes[&op].data {
            NodeData::Operation {
                left: l, right: r, ..
            } => {
                assert_eq!(*l, Some(right));
                assert_eq!(*r, Some(left));
            }
            _ => panic!(),
        }
    }

    // ── set_left_child / set_right_child / set_sculpt_input ─────────

    #[test]
    fn set_left_right_child() {
        let mut scene = empty_scene();
        let op = scene.create_operation(CsgOp::Union, None, None);
        let prim = scene.create_primitive(SdfPrimitive::Sphere);

        scene.set_left_child(op, Some(prim));
        match &scene.nodes[&op].data {
            NodeData::Operation { left, right, .. } => {
                assert_eq!(*left, Some(prim));
                assert_eq!(*right, None);
            }
            _ => panic!(),
        }

        let prim2 = scene.create_primitive(SdfPrimitive::Box);
        scene.set_right_child(op, Some(prim2));
        match &scene.nodes[&op].data {
            NodeData::Operation { right, .. } => assert_eq!(*right, Some(prim2)),
            _ => panic!(),
        }
    }

    #[test]
    fn set_sculpt_input_updates_transform() {
        let mut scene = empty_scene();
        let xform = scene.create_transform(None);
        let prim = scene.create_primitive(SdfPrimitive::Sphere);

        scene.set_sculpt_input(xform, Some(prim));
        match &scene.nodes[&xform].data {
            NodeData::Transform { input, .. } => assert_eq!(*input, Some(prim)),
            _ => panic!(),
        }
    }

    // ── insert_modifier_above ───────────────────────────────────────

    #[test]
    fn insert_modifier_above_rewires_parent() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let op = scene.create_operation(CsgOp::Union, Some(prim), None);

        let mod_id = scene.insert_modifier_above(prim, ModifierKind::Round);

        // Operation's left should now point to modifier
        match &scene.nodes[&op].data {
            NodeData::Operation { left, .. } => assert_eq!(*left, Some(mod_id)),
            _ => panic!(),
        }
        // Modifier's input should point to prim
        match &scene.nodes[&mod_id].data {
            NodeData::Modifier { input, .. } => assert_eq!(*input, Some(prim)),
            _ => panic!(),
        }
    }

    // ── insert_transform_above ──────────────────────────────────────

    #[test]
    fn insert_transform_above_rewires_parent() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let op = scene.create_operation(CsgOp::Union, Some(prim), None);

        let xform_id = scene.insert_transform_above(prim);

        match &scene.nodes[&op].data {
            NodeData::Operation { left, .. } => assert_eq!(*left, Some(xform_id)),
            _ => panic!(),
        }
        match &scene.nodes[&xform_id].data {
            NodeData::Transform { input, .. } => assert_eq!(*input, Some(prim)),
            _ => panic!(),
        }
    }

    // ── build_parent_map ────────────────────────────────────────────

    #[test]
    fn build_parent_map_maps_children_to_parents() {
        let mut scene = empty_scene();
        let child = scene.create_primitive(SdfPrimitive::Sphere);
        let parent = scene.create_transform(Some(child));
        let root = scene.create_operation(CsgOp::Union, Some(parent), None);

        let parent_map = scene.build_parent_map();
        assert_eq!(parent_map.get(&child), Some(&parent));
        assert_eq!(parent_map.get(&parent), Some(&root));
        assert_eq!(parent_map.get(&root), None);
    }

    // ── collect_subtree ─────────────────────────────────────────────

    #[test]
    fn collect_subtree_gathers_all_descendants() {
        let mut scene = empty_scene();
        let leaf_a = scene.create_primitive(SdfPrimitive::Sphere);
        let leaf_b = scene.create_primitive(SdfPrimitive::Box);
        let root = scene.create_operation(CsgOp::Union, Some(leaf_a), Some(leaf_b));

        let subtree = scene.collect_subtree(root);
        assert_eq!(subtree.len(), 3);
        assert!(subtree.contains(&root));
        assert!(subtree.contains(&leaf_a));
        assert!(subtree.contains(&leaf_b));
    }

    #[test]
    fn collect_subtree_nodes_returns_sorted_list() {
        let mut scene = empty_scene();
        let leaf_a = scene.create_primitive(SdfPrimitive::Sphere);
        let leaf_b = scene.create_primitive(SdfPrimitive::Box);
        let root = scene.create_operation(CsgOp::Union, Some(leaf_a), Some(leaf_b));

        let nodes = scene.collect_subtree_nodes(root);
        assert_eq!(nodes, vec![leaf_a, leaf_b, root]);
    }

    // ── visible_topo_order ──────────────────────────────────────────

    #[test]
    fn visible_topo_order_is_post_order() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        let root = scene.create_operation(CsgOp::Union, Some(left), Some(right));

        let order = scene.visible_topo_order();
        // Post-order: children before parent
        let root_pos = order.iter().position(|&id| id == root).unwrap();
        let left_pos = order.iter().position(|&id| id == left).unwrap();
        let right_pos = order.iter().position(|&id| id == right).unwrap();
        assert!(left_pos < root_pos);
        assert!(right_pos < root_pos);
    }

    #[test]
    fn visible_topo_order_skips_hidden_subtrees() {
        let mut scene = empty_scene();
        let left = scene.create_primitive(SdfPrimitive::Sphere);
        let right = scene.create_primitive(SdfPrimitive::Box);
        let root = scene.create_operation(CsgOp::Union, Some(left), Some(right));

        scene.toggle_visibility(left);
        let order = scene.visible_topo_order();
        assert!(order.contains(&root));
        assert!(order.contains(&right));
        assert!(!order.contains(&left));
    }

    // ── duplicate_subtree ───────────────────────────────────────────

    #[test]
    fn duplicate_subtree_creates_independent_copy() {
        let mut scene = empty_scene();
        let leaf = scene.create_primitive(SdfPrimitive::Sphere);
        let root = scene.create_operation(CsgOp::Union, Some(leaf), None);

        let new_root = scene.duplicate_subtree(root).unwrap();
        assert_ne!(new_root, root);

        // New root should have a remapped child, not the original leaf
        match &scene.nodes[&new_root].data {
            NodeData::Operation { left, .. } => {
                let new_leaf = left.unwrap();
                assert_ne!(new_leaf, leaf);
                assert!(scene.nodes.contains_key(&new_leaf));
            }
            _ => panic!(),
        }

        // Original tree is unchanged
        match &scene.nodes[&root].data {
            NodeData::Operation { left, .. } => assert_eq!(*left, Some(leaf)),
            _ => panic!(),
        }

        // Names have " Copy" appended
        assert!(scene.nodes[&new_root].name.ends_with(" Copy"));
    }

    #[test]
    fn duplicate_nonexistent_returns_none() {
        let mut scene = empty_scene();
        assert!(scene.duplicate_subtree(999).is_none());
    }

    // ── structure_key ───────────────────────────────────────────────

    #[test]
    fn structure_key_changes_on_topology_change() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let key_before = scene.structure_key();

        scene.create_operation(CsgOp::Union, Some(prim), None);
        let key_after = scene.structure_key();
        assert_ne!(key_before, key_after);
    }

    #[test]
    fn structure_key_stable_on_parameter_change() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let key_before = scene.structure_key();

        // Change position (parameter, not topology)
        if let Some(node) = scene.nodes.get_mut(&prim) {
            if let NodeData::Primitive { position, .. } = &mut node.data {
                *position = Vec3::new(5.0, 5.0, 5.0);
            }
        }
        let key_after = scene.structure_key();
        assert_eq!(key_before, key_after);
    }

    // ── data_fingerprint ────────────────────────────────────────────

    #[test]
    fn data_fingerprint_changes_on_parameter_change() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let fp_before = scene.data_fingerprint();

        if let Some(node) = scene.nodes.get_mut(&prim) {
            if let NodeData::Primitive { position, .. } = &mut node.data {
                *position = Vec3::new(5.0, 5.0, 5.0);
            }
        }
        let fp_after = scene.data_fingerprint();
        assert_ne!(fp_before, fp_after);
    }

    // ── compute_bounds ──────────────────────────────────────────────

    #[test]
    fn compute_bounds_default_for_empty_scene() {
        let scene = empty_scene();
        let (bmin, bmax) = scene.compute_bounds();
        assert_eq!(bmin, [-5.0; 3]);
        assert_eq!(bmax, [5.0; 3]);
    }

    #[test]
    fn compute_bounds_encloses_sphere_at_origin() {
        let scene = Scene::new(); // has a unit sphere at origin
        let (bmin, bmax) = scene.compute_bounds();
        // Sphere radius=1 at origin, padding=1.5 → bounds should be at least [-2.5, 2.5]
        for i in 0..3 {
            assert!(
                bmin[i] <= -2.5,
                "bmin[{}] = {} should be <= -2.5",
                i,
                bmin[i]
            );
            assert!(bmax[i] >= 2.5, "bmax[{}] = {} should be >= 2.5", i, bmax[i]);
        }
    }

    #[test]
    fn compute_bounds_accounts_for_translation() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let xform = scene.create_transform(Some(prim));
        if let Some(node) = scene.nodes.get_mut(&xform) {
            if let NodeData::Transform { translation, .. } = &mut node.data {
                *translation = Vec3::new(10.0, 0.0, 0.0);
            }
        }

        let (_bmin, bmax) = scene.compute_bounds();
        // Sphere at x=10, radius=1, padding=1.5 → bmax.x >= 10+1+1.5=12.5
        assert!(bmax[0] >= 12.5, "bmax[0] = {} should be >= 12.5", bmax[0]);
    }

    // ── content_eq ──────────────────────────────────────────────────

    #[test]
    fn content_eq_identical_scenes() {
        let scene = Scene::new();
        let clone = scene.clone();
        assert!(scene.content_eq(&clone));
    }

    #[test]
    fn content_eq_detects_position_change() {
        let scene = Scene::new();
        let mut modified = scene.clone();
        let id = *modified
            .nodes
            .iter()
            .find(|(_, n)| matches!(n.data, NodeData::Primitive { .. }))
            .unwrap()
            .0;
        if let Some(node) = modified.nodes.get_mut(&id) {
            if let NodeData::Primitive { position, .. } = &mut node.data {
                *position = Vec3::new(99.0, 0.0, 0.0);
            }
        }
        assert!(!scene.content_eq(&modified));
    }

    #[test]
    fn content_eq_detects_hidden_node_difference() {
        let scene = Scene::new();
        let mut modified = scene.clone();
        let id = *modified.nodes.keys().next().unwrap();
        modified.hidden_nodes.insert(id);
        assert!(!scene.content_eq(&modified));
    }

    // ── NodeData::children ──────────────────────────────────────────

    #[test]
    fn node_data_children_primitive_has_none() {
        let data = NodeData::Primitive {
            kind: SdfPrimitive::Sphere,
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::ONE,
            material: MaterialParams::with_base_color(Vec3::ONE),
            voxel_grid: None,
        };
        assert_eq!(data.children().count(), 0);
    }

    #[test]
    fn node_data_children_operation_has_two() {
        let data = NodeData::Operation {
            op: CsgOp::Union,
            smooth_k: 0.0,
            steps: 0.0,
            color_blend: -1.0,
            left: Some(1),
            right: Some(2),
        };
        let children: Vec<_> = data.children().collect();
        assert_eq!(children, vec![1, 2]);
    }

    #[test]
    fn node_data_children_operation_partial() {
        let data = NodeData::Operation {
            op: CsgOp::Union,
            smooth_k: 0.0,
            steps: 0.0,
            color_blend: -1.0,
            left: Some(1),
            right: None,
        };
        let children: Vec<_> = data.children().collect();
        assert_eq!(children, vec![1]);
    }

    // ── find_sculpt_parent ──────────────────────────────────────────

    #[test]
    fn find_sculpt_parent_returns_none_without_sculpt() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let xform = scene.create_transform(Some(prim));
        let _root = scene.create_operation(CsgOp::Union, Some(xform), None);

        let parent_map = scene.build_parent_map();
        assert_eq!(scene.find_sculpt_parent(prim, &parent_map), None);
    }

    // ── differential sculpt (insert_sculpt_above) ────────────────────

    #[test]
    fn insert_sculpt_above_creates_parent_child_hierarchy() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sculpt_id =
            scene.insert_sculpt_above(prim, Vec3::ZERO, Vec3::ZERO, Vec3::new(0.5, 0.5, 0.5), grid);

        // Sculpt's input should be the original primitive
        match &scene.nodes[&sculpt_id].data {
            NodeData::Sculpt { input, .. } => assert_eq!(*input, Some(prim)),
            _ => panic!("expected Sculpt node"),
        }

        // Primitive should be a descendant of sculpt
        assert!(scene.is_descendant(prim, sculpt_id));
    }

    #[test]
    fn insert_sculpt_above_rewires_operation_parent() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let cube = scene.create_primitive(SdfPrimitive::Box);
        let union = scene.create_operation(CsgOp::Union, Some(sphere), Some(cube));

        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sculpt_id = scene.insert_sculpt_above(
            sphere,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.5, 0.5, 0.5),
            grid,
        );

        // Operation's left child should now be the sculpt, not the original sphere
        match &scene.nodes[&union].data {
            NodeData::Operation { left, right, .. } => {
                assert_eq!(*left, Some(sculpt_id));
                assert_eq!(*right, Some(cube));
            }
            _ => panic!("expected Operation node"),
        }

        // Sculpt wraps the sphere
        match &scene.nodes[&sculpt_id].data {
            NodeData::Sculpt { input, .. } => assert_eq!(*input, Some(sphere)),
            _ => panic!("expected Sculpt node"),
        }
    }

    #[test]
    fn insert_sculpt_layer_above_wraps_sculpt_with_outer_transform() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let (transform_id, sculpt_id) = scene.insert_sculpt_layer_above(
            prim,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.5, 0.5, 0.5),
            VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0)),
        );

        let NodeData::Transform { input, .. } = &scene.nodes[&transform_id].data else {
            panic!("expected transform");
        };
        assert_eq!(*input, Some(sculpt_id));

        let NodeData::Sculpt { input, .. } = &scene.nodes[&sculpt_id].data else {
            panic!("expected sculpt");
        };
        assert_eq!(*input, Some(prim));
        assert!(scene.is_descendant(prim, transform_id));
    }

    #[test]
    fn find_sculpt_parent_finds_differential_sculpt() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sculpt_id =
            scene.insert_sculpt_above(prim, Vec3::ZERO, Vec3::ZERO, Vec3::new(0.5, 0.5, 0.5), grid);

        let parent_map = scene.build_parent_map();
        // Starting from the child primitive, should find the sculpt parent
        assert_eq!(scene.find_sculpt_parent(prim, &parent_map), Some(sculpt_id));
    }

    #[test]
    fn is_descendant_detects_child_of_differential_sculpt() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sculpt_id =
            scene.insert_sculpt_above(prim, Vec3::ZERO, Vec3::ZERO, Vec3::new(0.5, 0.5, 0.5), grid);

        // This is the critical check for the differential sculpt brush fix:
        // when the pick shader returns the child primitive's material ID,
        // is_descendant confirms it belongs to the active sculpt node.
        assert!(scene.is_descendant(prim, sculpt_id));
        assert!(!scene.is_descendant(sculpt_id, prim));
    }

    #[test]
    fn is_descendant_through_sculpt_in_operation() {
        let mut scene = empty_scene();
        let sphere = scene.create_primitive(SdfPrimitive::Sphere);
        let cube = scene.create_primitive(SdfPrimitive::Box);
        let union = scene.create_operation(CsgOp::Union, Some(sphere), Some(cube));

        let grid = VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0));
        let sculpt_id = scene.insert_sculpt_above(
            sphere,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.5, 0.5, 0.5),
            grid,
        );

        // Sphere is a child of sculpt, sculpt is a child of union
        assert!(scene.is_descendant(sphere, sculpt_id));
        assert!(scene.is_descendant(sculpt_id, union));
        assert!(scene.is_descendant(sphere, union));

        // Cube is NOT a child of sculpt (it's a sibling in the union)
        assert!(!scene.is_descendant(cube, sculpt_id));
    }

    #[test]
    fn remove_attached_sculpt_reconnects_parent_chain() {
        let mut scene = empty_scene();
        let primitive_id = scene.create_primitive(SdfPrimitive::Sphere);
        let inner_transform_id = scene.insert_transform_above(primitive_id);
        let sculpt_id = scene.insert_sculpt_above(
            inner_transform_id,
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.8, 0.2, 0.2),
            VoxelGrid::new_displacement(8, Vec3::splat(-1.0), Vec3::splat(1.0)),
        );
        let outer_transform_id = scene.insert_transform_above(sculpt_id);

        let removed = scene.remove_attached_sculpt(primitive_id);

        assert_eq!(removed, Some(sculpt_id));
        assert!(!scene.nodes.contains_key(&sculpt_id));
        let NodeData::Transform { input, .. } = &scene.nodes[&outer_transform_id].data else {
            panic!("expected transform");
        };
        assert_eq!(*input, Some(inner_transform_id));
    }

    // ── subtree_has_sculpt ──────────────────────────────────────────

    #[test]
    fn subtree_has_sculpt_false_without_sculpt() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let root = scene.create_operation(CsgOp::Union, Some(prim), None);
        assert!(!scene.subtree_has_sculpt(root));
    }

    // ── Light nodes ─────────────────────────────────────────────────

    #[test]
    fn create_light_returns_light_and_transform_ids() {
        let mut scene = empty_scene();
        let (light_id, transform_id) = scene.create_light(LightType::Point);
        assert!(scene.nodes.contains_key(&light_id));
        assert!(scene.nodes.contains_key(&transform_id));
        assert!(matches!(
            scene.nodes[&light_id].data,
            NodeData::Light { .. }
        ));
        match &scene.nodes[&transform_id].data {
            NodeData::Transform { input, .. } => assert_eq!(*input, Some(light_id)),
            _ => panic!("expected Transform parent"),
        }
    }

    #[test]
    fn light_node_has_no_children() {
        let data = NodeData::Light {
            light_type: LightType::Point,
            color: Vec3::ONE,
            intensity: 1.0,
            range: 10.0,
            spot_angle: 45.0,
            cast_shadows: false,
            shadow_softness: 8.0,
            shadow_color: Vec3::ZERO,
            volumetric: false,
            volumetric_density: 0.15,
            cookie_node: None,
            proximity_mode: ProximityMode::Off,
            proximity_range: 2.0,
            array_config: None,
            intensity_expr: None,
            color_hue_expr: None,
        };
        assert_eq!(data.children().count(), 0);
    }

    #[test]
    fn light_node_has_no_geometry_sphere() {
        let data = NodeData::Light {
            light_type: LightType::Spot,
            color: Vec3::ONE,
            intensity: 2.0,
            range: 5.0,
            spot_angle: 30.0,
            cast_shadows: true,
            shadow_softness: 8.0,
            shadow_color: Vec3::ZERO,
            volumetric: false,
            volumetric_density: 0.15,
            cookie_node: None,
            proximity_mode: ProximityMode::Off,
            proximity_range: 2.0,
            array_config: None,
            intensity_expr: None,
            color_hue_expr: None,
        };
        assert!(data.geometry_local_sphere().is_none());
    }

    #[test]
    fn light_node_not_valid_drop_target() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Directional);
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        assert!(!scene.is_valid_drop_target(light_id, prim));
    }

    #[test]
    fn light_node_appears_in_topo_order() {
        let mut scene = empty_scene();
        let (light_id, transform_id) = scene.create_light(LightType::Point);
        let order = scene.visible_topo_order();
        assert!(order.contains(&light_id));
        assert!(order.contains(&transform_id));
    }

    #[test]
    fn light_node_content_eq_detects_intensity_change() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        let clone = scene.clone();
        if let Some(node) = scene.nodes.get_mut(&light_id) {
            if let NodeData::Light { intensity, .. } = &mut node.data {
                *intensity = 5.0;
            }
        }
        assert!(!scene.content_eq(&clone));
    }

    #[test]
    fn light_node_data_fingerprint_changes_on_color_change() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        let fp_before = scene.data_fingerprint();
        if let Some(node) = scene.nodes.get_mut(&light_id) {
            if let NodeData::Light { color, .. } = &mut node.data {
                *color = Vec3::new(1.0, 0.0, 0.0);
            }
        }
        let fp_after = scene.data_fingerprint();
        assert_ne!(fp_before, fp_after);
    }

    #[test]
    fn light_node_structure_key_changes_on_type_change() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        let key_before = scene.structure_key();
        if let Some(node) = scene.nodes.get_mut(&light_id) {
            if let NodeData::Light { light_type, .. } = &mut node.data {
                *light_type = LightType::Spot;
            }
        }
        let key_after = scene.structure_key();
        assert_ne!(key_before, key_after);
    }

    #[test]
    fn duplicate_light_subtree() {
        let mut scene = empty_scene();
        let (_light_id, transform_id) = scene.create_light(LightType::Point);
        let new_root = scene.duplicate_subtree(transform_id).unwrap();
        assert_ne!(new_root, transform_id);
        // The duplicated tree should have a Transform containing a Light
        match &scene.nodes[&new_root].data {
            NodeData::Transform {
                input: Some(child_id),
                ..
            } => {
                assert!(matches!(scene.nodes[child_id].data, NodeData::Light { .. }));
            }
            _ => panic!("expected Transform with Light child"),
        }
    }

    // -----------------------------------------------------------------------
    // CsgOp default_smooth_k values
    // -----------------------------------------------------------------------

    #[test]
    fn csg_op_smooth_subtract_default_k() {
        assert!((CsgOp::SmoothSubtract.default_smooth_k() - 0.3).abs() < 1e-5);
    }

    #[test]
    fn csg_op_smooth_intersect_default_k() {
        assert!((CsgOp::SmoothIntersect.default_smooth_k() - 0.3).abs() < 1e-5);
    }

    #[test]
    fn csg_op_chamfer_default_k() {
        assert!((CsgOp::ChamferUnion.default_smooth_k() - 0.2).abs() < 1e-5);
        assert!((CsgOp::ChamferSubtract.default_smooth_k() - 0.2).abs() < 1e-5);
        assert!((CsgOp::ChamferIntersect.default_smooth_k() - 0.2).abs() < 1e-5);
    }

    #[test]
    fn csg_op_hard_ops_have_zero_k() {
        assert_eq!(CsgOp::Union.default_smooth_k(), 0.0);
        assert_eq!(CsgOp::Subtract.default_smooth_k(), 0.0);
        assert_eq!(CsgOp::Intersect.default_smooth_k(), 0.0);
    }

    #[test]
    fn csg_op_stairs_columns_have_steps_param() {
        assert!(CsgOp::StairsUnion.has_steps_param());
        assert!(CsgOp::StairsSubtract.has_steps_param());
        assert!(CsgOp::ColumnsUnion.has_steps_param());
        assert!(CsgOp::ColumnsSubtract.has_steps_param());
        // All other ops should NOT have steps param
        assert!(!CsgOp::Union.has_steps_param());
        assert!(!CsgOp::SmoothUnion.has_steps_param());
        assert!(!CsgOp::ChamferUnion.has_steps_param());
    }

    #[test]
    fn csg_op_stairs_columns_default_steps() {
        assert_eq!(CsgOp::StairsUnion.default_steps(), 4.0);
        assert_eq!(CsgOp::ColumnsSubtract.default_steps(), 4.0);
        assert_eq!(CsgOp::Union.default_steps(), 0.0);
    }

    #[test]
    fn csg_op_all_has_13_variants() {
        assert_eq!(CsgOp::ALL.len(), 13);
    }

    #[test]
    fn csg_op_wgsl_function_names() {
        assert_eq!(CsgOp::SmoothSubtract.wgsl_function_name(), "op_subtract");
        assert_eq!(CsgOp::SmoothIntersect.wgsl_function_name(), "op_intersect");
        assert_eq!(CsgOp::ChamferUnion.wgsl_function_name(), "op_chamfer_union");
        assert_eq!(CsgOp::StairsUnion.wgsl_function_name(), "op_stairs_union");
        assert_eq!(
            CsgOp::ColumnsSubtract.wgsl_function_name(),
            "op_columns_subtract"
        );
    }

    // -----------------------------------------------------------------------
    // LightType properties
    // -----------------------------------------------------------------------

    #[test]
    fn light_type_all_has_5_variants() {
        assert_eq!(LightType::ALL.len(), 5);
    }

    #[test]
    fn light_type_labels() {
        assert_eq!(LightType::Point.label(), "Point");
        assert_eq!(LightType::Spot.label(), "Spot");
        assert_eq!(LightType::Directional.label(), "Directional");
        assert_eq!(LightType::Ambient.label(), "Ambient");
    }

    #[test]
    fn light_type_badges() {
        assert_eq!(LightType::Point.badge(), "[Pt]");
        assert_eq!(LightType::Spot.badge(), "[Sp]");
        assert_eq!(LightType::Directional.badge(), "[Dir]");
        assert_eq!(LightType::Ambient.badge(), "[Amb]");
    }

    #[test]
    fn create_light_sets_default_properties() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Spot);
        match &scene.nodes[&light_id].data {
            NodeData::Light {
                light_type,
                color,
                intensity,
                range,
                spot_angle,
                cast_shadows,
                shadow_softness,
                shadow_color,
                volumetric,
                volumetric_density,
                cookie_node,
                proximity_mode,
                proximity_range,
                array_config,
                intensity_expr,
                color_hue_expr,
            } => {
                assert_eq!(*light_type, LightType::Spot);
                assert_eq!(*color, Vec3::ONE);
                assert!((intensity - 1.0).abs() < 1e-5);
                assert!((range - 10.0).abs() < 1e-5);
                assert!((spot_angle - 45.0).abs() < 1e-5);
                // Spot lights default to cast_shadows=true
                assert!(*cast_shadows);
                assert!((shadow_softness - 8.0).abs() < 1e-5);
                assert_eq!(*shadow_color, Vec3::ZERO);
                assert!(!volumetric);
                assert!((volumetric_density - 0.15).abs() < 1e-5);
                assert!(cookie_node.is_none());
                assert_eq!(*proximity_mode, ProximityMode::Off);
                assert!((proximity_range - 2.0).abs() < 1e-5);
                assert!(array_config.is_none());
                assert!(intensity_expr.is_none());
                assert!(color_hue_expr.is_none());
            }
            _ => panic!("expected Light node"),
        }
    }

    #[test]
    fn create_light_wraps_in_transform() {
        let mut scene = empty_scene();
        let (light_id, transform_id) = scene.create_light(LightType::Directional);
        match &scene.nodes[&transform_id].data {
            NodeData::Transform {
                input,
                translation,
                rotation,
                scale,
            } => {
                assert_eq!(*input, Some(light_id));
                assert_eq!(*translation, Vec3::new(2.0, 3.0, 2.0));
                assert_eq!(*rotation, Vec3::ZERO);
                assert_eq!(*scale, Vec3::ONE);
            }
            _ => panic!("expected Transform parent"),
        }
    }

    #[test]
    fn create_light_array_has_default_config() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Array);
        match &scene.nodes[&light_id].data {
            NodeData::Light {
                light_type,
                array_config,
                range,
                ..
            } => {
                assert_eq!(*light_type, LightType::Array);
                let cfg = array_config
                    .as_ref()
                    .expect("Array light must have array_config");
                assert_eq!(cfg.pattern, ArrayPattern::Ring);
                assert_eq!(cfg.count, 6);
                assert!((cfg.radius - 2.0).abs() < 1e-5);
                assert!((cfg.color_variation - 0.0).abs() < 1e-5);
                // Array lights default to range=5 (not 10)
                assert!((range - 5.0).abs() < 1e-5);
            }
            _ => panic!("expected Light node"),
        }
    }

    // ── light_masks ────────────────────────────────────────────────

    #[test]
    fn light_mask_default_is_all_lights() {
        let scene = Scene::new();
        let id = *scene.nodes.keys().next().unwrap();
        assert_eq!(scene.get_light_mask(id), 0xFF);
    }

    #[test]
    fn light_mask_set_and_get() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        scene.set_light_mask(prim, 0b0000_0101);
        assert_eq!(scene.get_light_mask(prim), 0b0000_0101);
    }

    #[test]
    fn light_mask_reset_to_default_removes_entry() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        scene.set_light_mask(prim, 0x0F);
        assert!(scene.light_masks.contains_key(&prim));
        scene.set_light_mask(prim, 0xFF);
        assert!(!scene.light_masks.contains_key(&prim));
    }

    #[test]
    fn light_mask_cleaned_up_on_remove_node() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        scene.set_light_mask(prim, 0x0F);
        assert!(scene.light_masks.contains_key(&prim));
        scene.remove_node(prim);
        assert!(!scene.light_masks.contains_key(&prim));
    }

    #[test]
    fn light_mask_included_in_content_eq() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let clone = scene.clone();
        scene.set_light_mask(prim, 0x01);
        assert!(!scene.content_eq(&clone));
    }

    #[test]
    fn light_mask_included_in_data_fingerprint() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        let fp_before = scene.data_fingerprint();
        scene.set_light_mask(prim, 0x01);
        let fp_after = scene.data_fingerprint();
        assert_ne!(fp_before, fp_after);
    }

    #[test]
    fn light_mask_duplicated_with_subtree() {
        let mut scene = empty_scene();
        let prim = scene.create_primitive(SdfPrimitive::Sphere);
        scene.set_light_mask(prim, 0b0000_0011);
        let new_root = scene.duplicate_subtree(prim).unwrap();
        assert_eq!(scene.get_light_mask(new_root), 0b0000_0011);
    }

    // ── Advanced lighting: cookie_node serialization ─────────────────

    #[test]
    fn cookie_node_serialization_roundtrip() {
        let mut scene = empty_scene();
        let prim_id = scene.create_primitive(SdfPrimitive::Sphere);
        let (light_id, _) = scene.create_light(LightType::Spot);
        if let NodeData::Light {
            ref mut cookie_node,
            ..
        } = scene.nodes.get_mut(&light_id).unwrap().data
        {
            *cookie_node = Some(prim_id);
        }
        // Serialize and deserialize the light node's data
        let node = &scene.nodes[&light_id];
        let json = serde_json::to_string(&node.data).expect("serialize NodeData");
        let deserialized: NodeData = serde_json::from_str(&json).expect("deserialize NodeData");
        match deserialized {
            NodeData::Light { cookie_node, .. } => {
                assert_eq!(
                    cookie_node,
                    Some(prim_id),
                    "cookie_node should survive serialization roundtrip"
                );
            }
            _ => panic!("expected Light variant"),
        }
    }

    #[test]
    fn cookie_node_none_serialization_roundtrip() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        let node = &scene.nodes[&light_id];
        let json = serde_json::to_string(&node.data).expect("serialize");
        let deserialized: NodeData = serde_json::from_str(&json).expect("deserialize");
        match deserialized {
            NodeData::Light { cookie_node, .. } => {
                assert!(
                    cookie_node.is_none(),
                    "cookie_node should be None after roundtrip"
                );
            }
            _ => panic!("expected Light variant"),
        }
    }

    // ── Advanced lighting: expression serialization ──────────────────

    #[test]
    fn light_expression_serialization_roundtrip() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        if let NodeData::Light {
            ref mut intensity_expr,
            ref mut color_hue_expr,
            ..
        } = scene.nodes.get_mut(&light_id).unwrap().data
        {
            *intensity_expr = Some("sin(t * 3.0) * 0.5 + 0.5".to_string());
            *color_hue_expr = Some("fract(t * 0.1) * 360.0".to_string());
        }
        let json = serde_json::to_string(&scene.nodes[&light_id].data).expect("serialize");
        let deserialized: NodeData = serde_json::from_str(&json).expect("deserialize");
        match deserialized {
            NodeData::Light {
                intensity_expr,
                color_hue_expr,
                ..
            } => {
                assert_eq!(intensity_expr.as_deref(), Some("sin(t * 3.0) * 0.5 + 0.5"));
                assert_eq!(color_hue_expr.as_deref(), Some("fract(t * 0.1) * 360.0"));
            }
            _ => panic!("expected Light variant"),
        }
    }

    // ── Advanced lighting: volumetric serialization ──────────────────

    #[test]
    fn volumetric_fields_serialization_roundtrip() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Spot);
        if let NodeData::Light {
            ref mut volumetric,
            ref mut volumetric_density,
            ..
        } = scene.nodes.get_mut(&light_id).unwrap().data
        {
            *volumetric = true;
            *volumetric_density = 0.73;
        }
        let json = serde_json::to_string(&scene.nodes[&light_id].data).expect("serialize");
        let deserialized: NodeData = serde_json::from_str(&json).expect("deserialize");
        match deserialized {
            NodeData::Light {
                volumetric,
                volumetric_density,
                ..
            } => {
                assert!(volumetric, "volumetric flag should survive roundtrip");
                assert!(
                    (volumetric_density - 0.73).abs() < 1e-5,
                    "density should survive roundtrip"
                );
            }
            _ => panic!("expected Light variant"),
        }
    }

    // ── Advanced lighting: shadow fields serialization ───────────────

    #[test]
    fn shadow_fields_serialization_roundtrip() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Directional);
        if let NodeData::Light {
            ref mut cast_shadows,
            ref mut shadow_softness,
            ref mut shadow_color,
            ..
        } = scene.nodes.get_mut(&light_id).unwrap().data
        {
            *cast_shadows = true;
            *shadow_softness = 24.0;
            *shadow_color = Vec3::new(0.1, 0.2, 0.3);
        }
        let json = serde_json::to_string(&scene.nodes[&light_id].data).expect("serialize");
        let deserialized: NodeData = serde_json::from_str(&json).expect("deserialize");
        match deserialized {
            NodeData::Light {
                cast_shadows,
                shadow_softness,
                shadow_color,
                ..
            } => {
                assert!(cast_shadows, "cast_shadows should survive roundtrip");
                assert!((shadow_softness - 24.0).abs() < 1e-5);
                assert!((shadow_color.x - 0.1).abs() < 1e-5);
                assert!((shadow_color.y - 0.2).abs() < 1e-5);
                assert!((shadow_color.z - 0.3).abs() < 1e-5);
            }
            _ => panic!("expected Light variant"),
        }
    }

    // ── Advanced lighting: proximity fields serialization ────────────

    #[test]
    fn proximity_fields_serialization_roundtrip() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        if let NodeData::Light {
            ref mut proximity_mode,
            ref mut proximity_range,
            ..
        } = scene.nodes.get_mut(&light_id).unwrap().data
        {
            *proximity_mode = ProximityMode::Brighten;
            *proximity_range = 5.5;
        }
        let json = serde_json::to_string(&scene.nodes[&light_id].data).expect("serialize");
        let deserialized: NodeData = serde_json::from_str(&json).expect("deserialize");
        match deserialized {
            NodeData::Light {
                proximity_mode,
                proximity_range,
                ..
            } => {
                assert_eq!(proximity_mode, ProximityMode::Brighten);
                assert!((proximity_range - 5.5).abs() < 1e-5);
            }
            _ => panic!("expected Light variant"),
        }
    }

    // ── Advanced lighting: structure_key changes with cookie ─────────

    #[test]
    fn structure_key_changes_when_cookie_added() {
        let mut scene = empty_scene();
        let prim_id = scene.create_primitive(SdfPrimitive::Sphere);
        let (light_id, _) = scene.create_light(LightType::Spot);

        let key_before = scene.structure_key();
        if let NodeData::Light {
            ref mut cookie_node,
            ..
        } = scene.nodes.get_mut(&light_id).unwrap().data
        {
            *cookie_node = Some(prim_id);
        }
        let key_after = scene.structure_key();
        assert_ne!(
            key_before, key_after,
            "structure_key should change when cookie is added"
        );
    }

    // ── Advanced lighting: has_light_expressions ─────────────────────

    #[test]
    fn has_light_expressions_false_when_no_expressions() {
        let mut scene = empty_scene();
        scene.create_light(LightType::Point);
        assert!(!scene.has_light_expressions(), "no expressions set");
    }

    #[test]
    fn has_light_expressions_true_when_intensity_expr_set() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        if let NodeData::Light {
            ref mut intensity_expr,
            ..
        } = scene.nodes.get_mut(&light_id).unwrap().data
        {
            *intensity_expr = Some("sin(t)".to_string());
        }
        assert!(
            scene.has_light_expressions(),
            "should detect intensity expression"
        );
    }

    #[test]
    fn has_light_expressions_true_when_color_hue_expr_set() {
        let mut scene = empty_scene();
        let (light_id, _) = scene.create_light(LightType::Point);
        if let NodeData::Light {
            ref mut color_hue_expr,
            ..
        } = scene.nodes.get_mut(&light_id).unwrap().data
        {
            *color_hue_expr = Some("t * 60.0".to_string());
        }
        assert!(
            scene.has_light_expressions(),
            "should detect color hue expression"
        );
    }

    // ── Scene statistics helpers ────────────────────────────────────

    #[test]
    fn node_type_counts_empty_scene() {
        let scene = empty_scene();
        let counts = scene.node_type_counts();
        assert_eq!(counts.total, 0);
        assert_eq!(counts.visible, 0);
    }

    #[test]
    fn node_type_counts_default_scene() {
        let scene = Scene::new();
        let counts = scene.node_type_counts();
        // 1 sphere + 3 lights + 3 light transforms = 7
        assert_eq!(counts.total, 7);
        assert_eq!(counts.primitives, 1);
        assert_eq!(counts.lights, 3);
        assert_eq!(counts.transforms, 3);
        assert_eq!(counts.operations, 0);
        assert_eq!(counts.modifiers, 0);
        assert_eq!(counts.sculpts, 0);
        assert_eq!(counts.visible, 7);
    }

    #[test]
    fn node_type_counts_hidden_nodes_excluded_from_visible() {
        let mut scene = Scene::new();
        let sphere_id = scene
            .nodes
            .values()
            .find(|n| matches!(n.data, NodeData::Primitive { .. }))
            .unwrap()
            .id;
        scene.hidden_nodes.insert(sphere_id);
        let counts = scene.node_type_counts();
        assert_eq!(counts.total, 7);
        assert_eq!(counts.visible, 6);
    }

    #[test]
    fn voxel_memory_bytes_empty() {
        let scene = Scene::new();
        assert_eq!(scene.voxel_memory_bytes(), 0);
    }

    #[test]
    fn sdf_eval_complexity_excludes_lights() {
        let scene = Scene::new();
        let complexity = scene.sdf_eval_complexity();
        // Only the sphere should count (lights + light transforms excluded)
        // Actually visible_topo_order includes transforms too, but lights are excluded
        // Let's just assert lights don't count
        let counts = scene.node_type_counts();
        assert!(complexity <= counts.total - counts.lights);
    }
}
