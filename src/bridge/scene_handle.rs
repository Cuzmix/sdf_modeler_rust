use crate::graph::scene::Scene;
use crate::settings::RenderConfig;

/// Opaque scene handle for Flutter — owns the SDF scene tree and render config.
/// Tracks structure_key to detect when shaders need regeneration.
#[cfg(feature = "flutter_ui")]
#[flutter_rust_bridge::frb(opaque)]
pub struct SceneHandle {
    pub(crate) scene: Scene,
    pub(crate) config: RenderConfig,
    pub(crate) current_structure_key: u64,
}

#[cfg(feature = "flutter_ui")]
impl SceneHandle {
    pub fn new() -> Self {
        let scene = Scene::new();
        let config = RenderConfig::default();
        Self {
            current_structure_key: scene.structure_key(),
            scene,
            config,
        }
    }
}
