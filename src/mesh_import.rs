use std::io::{BufRead, Read};
use std::path::Path;

use glam::Vec3;

use crate::compat::maybe_par_iter;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::graph::voxel::VoxelGrid;

// ---------------------------------------------------------------------------
// Triangle mesh representation
// ---------------------------------------------------------------------------

pub struct TriMesh {
    pub vertices: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
}

// ---------------------------------------------------------------------------
// OBJ loader (vertices + faces only)
// ---------------------------------------------------------------------------

pub fn load_obj(path: &Path) -> Result<TriMesh, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let reader = std::io::BufReader::new(file);

    let mut vertices = Vec::new();
    let mut triangles = Vec::new();

    for line in reader.lines() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim();
        if line.starts_with("v ") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                let x: f32 = parts[1]
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?;
                let y: f32 = parts[2]
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?;
                let z: f32 = parts[3]
                    .parse()
                    .map_err(|e: std::num::ParseFloatError| e.to_string())?;
                vertices.push(Vec3::new(x, y, z));
            }
        } else if line.starts_with("f ") {
            let parts: Vec<&str> = line.split_whitespace().skip(1).collect();
            // Parse face vertex indices (OBJ is 1-indexed, may have v/vt/vn format)
            let indices: Vec<u32> = parts
                .iter()
                .filter_map(|s| {
                    s.split('/')
                        .next()
                        .and_then(|idx| idx.parse::<u32>().ok())
                        .map(|i| i - 1) // Convert to 0-indexed
                })
                .collect();
            // Triangulate polygon (fan from first vertex)
            if indices.len() >= 3 {
                for i in 1..indices.len() - 1 {
                    triangles.push([indices[0], indices[i], indices[i + 1]]);
                }
            }
        }
    }

    if vertices.is_empty() || triangles.is_empty() {
        return Err("No geometry found in OBJ file".to_string());
    }

    Ok(TriMesh {
        vertices,
        triangles,
    })
}

// ---------------------------------------------------------------------------
// STL binary loader
// ---------------------------------------------------------------------------

pub fn load_stl(path: &Path) -> Result<TriMesh, String> {
    let mut file = std::fs::File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut data = Vec::new();
    file.read_to_end(&mut data).map_err(|e| e.to_string())?;

    if data.len() < 84 {
        return Err("STL file too small".to_string());
    }

    // Skip 80-byte header
    let num_tris = u32::from_le_bytes([data[80], data[81], data[82], data[83]]) as usize;
    let expected = 84 + num_tris * 50;
    if data.len() < expected {
        return Err(format!(
            "STL file truncated: expected {} bytes, got {}",
            expected,
            data.len()
        ));
    }

    let mut vertices = Vec::with_capacity(num_tris * 3);
    let mut triangles = Vec::with_capacity(num_tris);

    for i in 0..num_tris {
        let offset = 84 + i * 50;
        // Skip normal (12 bytes), read 3 vertices (36 bytes)
        for v in 0..3 {
            let vo = offset + 12 + v * 12;
            let x = f32::from_le_bytes([data[vo], data[vo + 1], data[vo + 2], data[vo + 3]]);
            let y = f32::from_le_bytes([data[vo + 4], data[vo + 5], data[vo + 6], data[vo + 7]]);
            let z = f32::from_le_bytes([data[vo + 8], data[vo + 9], data[vo + 10], data[vo + 11]]);
            vertices.push(Vec3::new(x, y, z));
        }
        let base = (i * 3) as u32;
        triangles.push([base, base + 1, base + 2]);
    }

    if triangles.is_empty() {
        return Err("No triangles found in STL file".to_string());
    }

    Ok(TriMesh {
        vertices,
        triangles,
    })
}

// ---------------------------------------------------------------------------
// Load mesh by extension
// ---------------------------------------------------------------------------

pub fn load_mesh(path: &Path) -> Result<TriMesh, String> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("stl") => load_stl(path),
        Some("obj") => load_obj(path),
        _ => Err("Unsupported mesh format (use OBJ or STL)".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Point-to-triangle distance
// ---------------------------------------------------------------------------

/// Compute the unsigned distance from point `p` to triangle (a, b, c).
fn point_triangle_dist_sq(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return ap.length_squared();
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return bp.length_squared();
    }

    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let proj = a + ab * v;
        return (p - proj).length_squared();
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return cp.length_squared();
    }

    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let proj = a + ac * w;
        return (p - proj).length_squared();
    }

    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let proj = b + (c - b) * w;
        return (p - proj).length_squared();
    }

    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let proj = a + ab * v + ac * w;
    (p - proj).length_squared()
}

// ---------------------------------------------------------------------------
// Sign determination via ray casting (parity)
// ---------------------------------------------------------------------------

/// Determine if point `p` is inside the mesh by casting a ray along +X
/// and counting triangle intersections (odd = inside).
fn is_inside(p: Vec3, mesh: &TriMesh) -> bool {
    let mut crossings = 0u32;
    let ray_dir = Vec3::X;

    for tri in &mesh.triangles {
        let a = mesh.vertices[tri[0] as usize];
        let b = mesh.vertices[tri[1] as usize];
        let c = mesh.vertices[tri[2] as usize];

        // Möller–Trumbore ray-triangle intersection
        let edge1 = b - a;
        let edge2 = c - a;
        let h = ray_dir.cross(edge2);
        let det = edge1.dot(h);

        if det.abs() < 1e-8 {
            continue; // Parallel
        }

        let inv_det = 1.0 / det;
        let s = p - a;
        let u = inv_det * s.dot(h);
        if !(0.0..=1.0).contains(&u) {
            continue;
        }

        let q = s.cross(edge1);
        let v = inv_det * ray_dir.dot(q);
        if v < 0.0 || u + v > 1.0 {
            continue;
        }

        let t = inv_det * edge2.dot(q);
        if t > 1e-6 {
            crossings += 1;
        }
    }

    crossings % 2 == 1
}

// ---------------------------------------------------------------------------
// Mesh to SDF conversion
// ---------------------------------------------------------------------------

/// Convert a triangle mesh to a VoxelGrid SDF.
/// Brute-force: for each voxel, find nearest triangle distance and determine sign.
pub fn mesh_to_sdf(
    mesh: &TriMesh,
    resolution: u32,
    progress: &std::sync::atomic::AtomicU32,
) -> (VoxelGrid, Vec3) {
    // Compute mesh AABB
    let mut mesh_min = Vec3::splat(f32::MAX);
    let mut mesh_max = Vec3::splat(f32::MIN);
    for v in &mesh.vertices {
        mesh_min = mesh_min.min(*v);
        mesh_max = mesh_max.max(*v);
    }

    // Add padding (10% of diagonal)
    let diag = (mesh_max - mesh_min).length();
    let padding = diag * 0.1;
    mesh_min -= Vec3::splat(padding);
    mesh_max += Vec3::splat(padding);

    let center = (mesh_min + mesh_max) * 0.5;
    let half_extent = (mesh_max - mesh_min) * 0.5;

    // Create grid centered at mesh center
    let bounds_min = -half_extent;
    let bounds_max = half_extent;
    let res = resolution;
    let step = (bounds_max - bounds_min) / (res - 1) as f32;

    // Compute SDF in parallel by z-slice
    let slices: Vec<Vec<f32>> = maybe_par_iter!(0..res)
        .map(|z| {
            let mut slice = vec![0.0f32; (res * res) as usize];
            for y in 0..res {
                for x in 0..res {
                    let local_p = bounds_min + Vec3::new(x as f32, y as f32, z as f32) * step;
                    let world_p = center + local_p;

                    // Find nearest triangle distance
                    let mut min_dist_sq = f32::MAX;
                    for tri in &mesh.triangles {
                        let a = mesh.vertices[tri[0] as usize];
                        let b = mesh.vertices[tri[1] as usize];
                        let c = mesh.vertices[tri[2] as usize];
                        let d_sq = point_triangle_dist_sq(world_p, a, b, c);
                        min_dist_sq = min_dist_sq.min(d_sq);
                    }

                    let dist = min_dist_sq.sqrt();
                    let sign = if is_inside(world_p, mesh) { -1.0 } else { 1.0 };
                    slice[(y * res + x) as usize] = dist * sign;
                }
            }
            progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            slice
        })
        .collect();

    // Assemble grid data
    let mut data = vec![0.0f32; (res * res * res) as usize];
    for (z, slice) in slices.into_iter().enumerate() {
        let offset = z * (res * res) as usize;
        data[offset..offset + (res * res) as usize].copy_from_slice(&slice);
    }

    let grid = VoxelGrid {
        resolution: res,
        bounds_min,
        bounds_max,
        is_displacement: false,
        data,
    };

    (grid, center)
}
