---
name: add-export-format
description: Add a new mesh export format like OBJ, STL, PLY, glTF, USDA, FBX, or any new 3D file format writer for mesh export.
---

# Add Export Format

Step-by-step guide to add a new mesh export format writer.

## Constraints

- **Pure Rust + `std::io` only** — no external crate dependencies for export writers
- All writers use `std::io::Write` trait for output
- The `ExportMesh` struct provides all mesh data

## ExportMesh Structure (`src/export.rs`)

```rust
pub struct ExportMesh {
    pub vertices: Vec<[f32; 3]>,      // Vertex positions
    pub triangles: Vec<[u32; 3]>,     // Triangle indices (3 per face)
    pub vertex_colors: Vec<[f32; 3]>, // Per-vertex RGB (0.0–1.0)
}
```

## Step 1: Writer Function (`src/export.rs`)

Add the writer function before `write_mesh()`:

```rust
pub fn write_<format>(mesh: &ExportMesh, path: &Path) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut writer = std::io::BufWriter::new(file);

    // Write header, vertices, faces, etc.

    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}
```

### Pattern for text formats (like OBJ, PLY):
```rust
writeln!(writer, "header line").map_err(|e| e.to_string())?;
for vertex in &mesh.vertices {
    writeln!(writer, "v {} {} {}", vertex[0], vertex[1], vertex[2])
        .map_err(|e| e.to_string())?;
}
```

### Pattern for binary formats (like STL, glTF):
```rust
writer.write_all(&header_bytes).map_err(|e| e.to_string())?;
writer.write_all(&(mesh.triangles.len() as u32).to_le_bytes())
    .map_err(|e| e.to_string())?;
```

## Step 2: Dispatcher (`src/export.rs`)

Add match arm in `write_mesh()`:

```rust
pub fn write_mesh(mesh: &ExportMesh, path: &Path) -> Result<(), String> {
    match path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("stl") => write_stl(mesh, path),
        Some("ply") => write_ply(mesh, path),
        Some("glb") => write_glb(mesh, path),
        Some("usda") => write_usda(mesh, path),
        Some("<ext>") => write_<format>(mesh, path),  // ADD THIS
        _ => write_obj(mesh, path),
    }
}
```

## Step 3: File Dialog Filter (`src/ui/export_dialog.rs`)

Add the format to the file dialog filter list so users can select it.

## Step 4: Tests (`src/export.rs`)

Add tests following the naming pattern `write_<format>_<aspect>()`:

```rust
#[test]
fn write_<format>_produces_valid_output() {
    let mesh = ExportMesh {
        vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        triangles: vec![[0, 1, 2]],
        vertex_colors: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    let dir = std::env::temp_dir();
    let path = dir.join("test_export.<ext>");
    write_<format>(&mesh, &path).unwrap();
    // Verify file contents
    let data = std::fs::read(&path).unwrap();
    assert!(!data.is_empty());
    std::fs::remove_file(&path).ok();
}

#[test]
fn write_<format>_empty_mesh() {
    let mesh = ExportMesh {
        vertices: vec![],
        triangles: vec![],
        vertex_colors: vec![],
    };
    let dir = std::env::temp_dir();
    let path = dir.join("test_empty.<ext>");
    write_<format>(&mesh, &path).unwrap();
    std::fs::remove_file(&path).ok();
}
```

## Step 5: Verify

```bash
cargo check && cargo clippy -- -D warnings && cargo test && cargo build
```

## Existing Formats (for reference)

| Format | Function | Type | Notes |
|--------|----------|------|-------|
| OBJ | `write_obj()` | Text | Wavefront, simplest format |
| STL | `write_stl()` | Binary | 80-byte header + per-triangle data |
| PLY | `write_ply()` | Text (ASCII) | Stanford, vertex colors in header |
| glTF | `write_glb()` | Binary | Structured: JSON chunk + BIN chunk |
| USDA | `write_usda()` | Text | USD ASCII, Pixar format |

## Checklist

- [ ] Writer function added to `src/export.rs` (pure Rust, std::io only)
- [ ] Match arm added in `write_mesh()` dispatcher
- [ ] File dialog filter updated in `export_dialog.rs`
- [ ] Tests: valid output + empty mesh handling
- [ ] All 4 verification steps pass
