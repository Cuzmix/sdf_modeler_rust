use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::ui::viewport::ViewportResources;

#[derive(Clone)]
pub struct AppRenderContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter: Arc<wgpu::Adapter>,
    pub target_format: wgpu::TextureFormat,
}

impl AppRenderContext {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        adapter: Arc<wgpu::Adapter>,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        Self {
            device,
            queue,
            adapter,
            target_format,
        }
    }
}

#[derive(Clone)]
pub struct ViewportResourceHandle {
    inner: Arc<RwLock<ViewportResources>>,
}

impl ViewportResourceHandle {
    pub fn new(resources: ViewportResources) -> Self {
        Self {
            inner: Arc::new(RwLock::new(resources)),
        }
    }

    pub fn read(&self) -> RwLockReadGuard<'_, ViewportResources> {
        self.inner
            .read()
            .expect("viewport resource lock should not be poisoned")
    }

    pub fn write(&self) -> RwLockWriteGuard<'_, ViewportResources> {
        self.inner
            .write()
            .expect("viewport resource lock should not be poisoned")
    }
}

#[derive(Clone)]
pub struct WakeHandle {
    wake_fn: Arc<dyn Fn() + Send + Sync>,
}

impl WakeHandle {
    pub fn new<F>(wake_fn: F) -> Self
    where
        F: Fn() + Send + Sync + 'static,
    {
        Self {
            wake_fn: Arc::new(wake_fn),
        }
    }

    pub fn wake(&self) {
        (self.wake_fn)();
    }
}
