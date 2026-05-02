# SDF Modeler — Performance Techniques

Catalogue of optimizations used to scale the WebGPU SDF marcher to
60 fps with 200+ visible leaves on tablet-class hardware. Most are
language-agnostic and transfer directly to a Rust + wgpu / Rust + egui
SDF modeler; some are WebGPU-specific (noted inline).

## TL;DR — biggest wins

If you only have time for three:

1. **Per-pixel BVH cull.** O(N) scene SDF evaluation collapses to ~O(log N) per ray. Single biggest win once the scene exceeds ~30 leaves.
2. **Bisection refinement at the marcher's hit.** Eliminates per-pixel depth banding without raising step count; ~5% frame cost for surface pixels.
3. **Half-res raymarch + edge-aware upscale.** 4× fewer marcher pixels at near-full perceived quality, stacked with adaptive quality during interaction.

Everything else is icing.

---

## 1. Spatial culling — get the scene SDF cost off the per-pixel hot path

### 1.1 Per-pixel BVH traversal

**Problem:** Naive sphere tracing evaluates `sceneSdf(p)` at every step, which folds *every* leaf in the scene. With N leaves and S steps, per-ray cost is O(N·S). At 1080p × 50 leaves × 64 steps = 3.3 billion leaf evaluations/frame.

**Technique:** Build a BVH over the scene's bounding spheres on CPU each frame. At each pixel, walk the BVH stacklessly; skip whole subtrees whose AABB doesn't intersect the ray segment. Collect the surviving leaves into a per-pixel array (`pixel_ids[]`). The marcher folds *only* that subset, in scene order.

**Key insights:**
- Use **bounding spheres**, not AABBs, for the cull primitive — closest-point-on-segment to centre is just one `clamp` + `dot`, no axis tests.
- For rotated boxes, conservatively use the outer sphere over the rotated AABB; rebuilding the BVH every frame means you don't need a fancier bound.
- The skip-pointer DFS layout (each inner node stores the index of its post-subtree sibling) gives stackless GPU traversal — crucial because most GPU shading languages don't have native stack types.

**Trade-off:** Per-pixel array has a hard cap (`MAX_PIXEL_LEAVES = 64` here) — a workgroup memory tradeoff. Beyond that you'd need tiling.

**Language-agnostic:** Yes. The BVH builder + skip-pointer layout port directly. WGSL ↔ WGSL/SPIR-V/MSL.

### 1.2 Tile cull pre-pass

**Problem:** Even with per-pixel BVH, every pixel pays for the BVH walk. For 1080p that's 2M walks per frame.

**Technique:** Run a coarser cull at tile granularity (32×32 px tiles) in a separate compute pass *before* the marcher. Each tile gets a list of "leaves whose bounding sphere projects into this tile." The per-pixel BVH descends from the same root but exits earlier because most subtrees are already out-of-tile.

**Trade-off:** Adds one compute pass per frame, but it's much cheaper than the savings on the marcher. Net positive once the scene has ~30+ leaves.

**Language-agnostic:** Yes — frustum / tile projection math is standard.

### 1.3 BVH bound widening for smooth blends

**Problem:** A leaf inside a group with `smoothness = k` doesn't contribute only within its tight bound — its smin influence extends outward by ~`k`. If the BVH cull uses the tight bound, the marcher silently drops the leaf in regions where it should still be smin'ing.

**Technique:** When packing leaves for the GPU, widen `bound_radius` by the **worst-case k along the leaf's group ancestry** (walk up the parent chain, take max). The cull then conservatively includes any pixel where the smin can still bulge.

**Key insight:** This only matters because subtraction/intersection operations let "far" geometry affect "near" pixels. For pure unions you can use the tight bound.

**Language-agnostic:** Yes.

### 1.4 Force-include intersect ops

**Problem:** Spatial cull works for `min` (union, subtract) — a leaf far from the ray can't lower the accumulator. But for `max` (intersect), a far leaf *can* clamp the accumulator. Spatial cull silently breaks intersect.

**Technique:** After BVH traversal, run a small force-include loop that adds every intersect-op leaf to `pixel_ids[]` regardless of spatial position. Cheap because most scenes have zero intersect leaves.

**Language-agnostic:** Yes.

---

## 2. Sphere tracing — make each pixel converge with fewer / better steps

### 2.1 Adaptive hit threshold

**Problem:** A constant hit threshold wastes steps near distant surfaces (sub-pixel precision is wasted there) and may not converge near close ones.

**Technique:**
```
threshold(t) = base * (1 + t * growth)
```
Larger threshold at large travel; tighter threshold at small travel. Distant rays exit a few steps earlier.

**Trade-off:** Loose threshold at depth means hit *position* is less precise. Mitigated by bisection refinement (next).

**Tuning:** `base = 0.001`, `growth = 0.02` in this codebase. At max raymarch distance (24), threshold is `~0.0015` — still well below pixel scale at typical view distance.

**Language-agnostic:** Yes.

### 2.2 Bisection refinement at the hit

**Problem:** Plain sphere tracing terminates when `d < threshold`. The actual zero crossing is somewhere inside the last step's span. Adjacent pixels that converge in different step counts land at different depths → visible **concentric ring artifacts** on smooth carved (subtract / intersect) surfaces.

**Technique:** After the marcher signals a hit, binary-search between the previous step (`d > threshold`, outside) and the current step (`d ≤ threshold`, at or past surface):

```
lo = prev_travel; hi = travel
for k in 0..N:
  mid = (lo + hi) / 2
  d_mid = sceneSdf(at mid)
  if d_mid > 0: lo = mid
  else:        hi = mid
return hi  // tightly straddles the surface
```

Each iteration halves the depth uncertainty. 13 iterations gives `2¹³ = 8192×` tightening.

**Cost:** Exactly `N` extra `sceneSdf` calls **per surface pixel** (background pixels — rays that never hit — are unaffected). At a typical 25–40 marcher steps + 13 bisect = ~40–50% more work *per surface pixel*, ~5% net frame cost.

**Why it matters:** Without this, you'd need to push step count to 200+ and threshold to 1e-5 to get equivalent quality. Bisection achieves it ~30× cheaper because it only refines *at the surface*, not at every step along the ray.

**Key insight:** This is rarely seen in ShaderToy demos but is standard in production raymarchers (Inigo Quilez's tools use it). Most demos brute-force precision via more steps + tighter threshold.

**Language-agnostic:** Yes — pure shader code, ports directly.

### 2.3 Step count + threshold as live tunables

**Problem:** Different scenes / cameras need different step counts. Hard-coded constants force conservative defaults.

**Technique:** Expose `RAYMARCH_STEPS`, `BISECT_ITERATIONS`, `HIT_THRESHOLD`, `THRESHOLD_GROWTH`, etc. as runtime uniforms (UBO fields). Wire them through a settings panel for live tuning.

**Trade-off:** Slight cost from UBO read inside the loop. Worth it for the iteration speed during perf tuning. Compile-time constants for things that genuinely don't change (array sizes).

**Language-agnostic:** Yes.

### 2.4 Conservative-distance floor at piecewise-SDF boundaries

**Problem:** A voxel-baked SDF returns the *baked* distance inside the bbox, but at the bbox boundary we can only return a conservative bbox-distance. If the floor is below the marcher's hit threshold, distant rays trigger false hits at the bbox face — the voxel renders as its own bounding cube.

**Technique:** Floor the conservative bbox-distance just above the marcher's *worst-case* (max-travel) adaptive threshold. Math:

```
max_threshold = base * (1 + RAYMARCH_MAX_DIST * growth)
voxel_bbox_floor = max_threshold + safety_margin
```

In this codebase: `0.0055` floors the worst case of `~0.0049` at max travel.

**Language-agnostic:** Yes.

---

## 3. Memory layout — minimise GPU bandwidth + register pressure

### 3.1 Tight leaf record packing

**Problem:** Each per-pixel marcher iteration loads the leaf record from storage. At 1080p × 64 steps × 30 leaves, that's billions of struct loads/frame. Every byte of struct size matters.

**Technique:** Pack the leaf into a 64-byte record using `vec3 + scalar` slots (the `vec3` alignment padding fits a `u32` or `f32`):

```
0..15   header (kind, bool_op, voxel_slot, id)  4 × u32
16..31  position (vec3) + uniform_scale (f32)
32..47  dims (vec4)                              kind-specific
48..63  bbox_min (vec3) + bound_radius (f32) | bbox_max (vec3) + smoothness (f32)
```

64 bytes = one cache line on most GPUs. Started at 96, dropped to 80 (rotation moved to side buffer), then 64 (smoothness moved to group records).

**Language-agnostic:** Yes — the `std140` / `std430` layout rules are roughly the same across WGSL, GLSL, HLSL.

### 3.2 Inverse-rotation side buffer

**Problem:** Each leaf-eval applied the inverse rotation as 6 trig + 9 multiplies. At billions of evals/frame, that's a lot of `sin`/`cos`.

**Technique:** Pre-compute the inverse rotation matrix on CPU, stash it in a side buffer parallel to the leaf buffer. Marcher reads the matrix and does a 9-multiply matrix-vector instead.

**Trade-off:** +48 bytes per leaf in a side buffer; CPU does the matrix construction.

**Language-agnostic:** Yes.

### 3.3 Pre-allocated scratch buffers

**Problem:** During gizmo drag at 120Hz pointer rate, the scene compiler was allocating ~200 KB of buffers per frame = 24 MB/s of allocation. GC stutters appeared at 1500-leaf scenes.

**Technique:** Renderer holds a `compileScratch` reference passed into `compileScene`; the compiler reuses the scratch arrays unless the scene grew beyond their capacity (rare).

**Language-agnostic:** Yes — this is a generic "object pool" pattern. In Rust you'd use `Vec::clear() + extend()` or pre-sized `Box<[T]>` with manual length tracking.

### 3.4 Voxel slot indexing (fixed-size pool)

**Problem:** WGSL/SPIR-V doesn't support runtime-indexed texture arrays portably. You can't do `voxel_textures[leaf.voxel_slot]` directly.

**Technique:** Cap voxel-leaf count (`MAX_VOXEL_LEAVES = 16`); declare 16 individual texture bindings; dispatch via `switch` on `leaf.voxel_slot`. Slots assigned densely on CPU when packing.

**WebGPU-specific:** This is mostly a WebGPU/Vulkan limitation; Metal and modern Vulkan support runtime-indexed binding arrays. If you're targeting wgpu the same constraint applies.

---

## 4. CPU↔GPU pipeline — avoid per-frame waste

### 4.1 Skip BVH rebuild during transient edits

**Problem:** O(N log N) sort for the BVH build dominates per-frame cost during gizmo drag at 120Hz on a 1500-leaf scene (5–10 ms/frame). That's 30+% of the frame budget on what's essentially translation.

**Technique:** During a transient edit (gizmo drag, slider scrub), `setScene(scene, { skipBvh: true })`. The marcher uses the *stale* BVH for the drag duration. On `endTransientEdit` (drag-release), do one final rebuild.

**Trade-off:** A moved leaf can briefly fall outside its old subtree's AABB and not render mid-drag. Visually acceptable at 120Hz — recovers within one frame on release.

**Language-agnostic:** Yes — the "transient state" concept maps cleanly. egui has its own "is dragging" signal you can hook.

### 4.2 RAF-coalesced setScene

**Problem:** 120Hz pointer events fire faster than the renderer can render. Multiple `setScene` calls per frame queue up and most are wasted work.

**Technique:** Subscribe to the scene store with a `requestAnimationFrame`-debounced handler: at most one compile/upload per frame, picking up the *latest* scene state.

**Language-agnostic:** Yes. In a Rust GUI, hook `eframe::App::update` (which runs at frame rate) — read the latest scene state there, not from a callback per pointer event.

### 4.3 Skip-on-transient subscription pattern (UI)

**Problem:** A scene-graph hierarchy panel subscribed to scene mutations rebuilds 1500 rows on every gizmo-drag pointermove. React reconciler stutters.

**Technique:** Heavy UI subscribers explicitly skip transient updates: subscribe at the store level, check `lastChangeWasTransient` flag, return early if true. Refresh once on the matching commit.

**Language-agnostic:** Yes — egui's immediate-mode model is friendlier here (re-render is cheap by default), but if you're caching laid-out widgets the same trick applies.

### 4.4 Lazy compute pipeline init

**Problem:** Compiling all compute pipelines at startup adds 50–200 ms to first paint.

**Technique:** Wrap each pipeline in a `LazyComputePipeline` that compiles on first `ensure()` call. Pipelines for unused features (paint brush, mesh export) never compile.

**Language-agnostic:** Yes.

---

## 5. CSG-specific structure — keep the SDF tree well-shaped

### 5.1 Smoothness on groups, not leaves

**Problem:** Per-leaf smoothness with left-to-right fold causes asymmetric blends — `smin(A, B, k_B)` uses the second leaf's k for the pair, ignoring A's. Two adjacent leaves with mismatched k produce a visible kink.

**Technique:** Move smoothness onto the **enclosing group** (per MagicaCSG / Womp). Every child of a group blends with the same `k` → no asymmetric pair, by construction.

**Bonus:** Reduces the leaf BoolOp enum from 6 (union, subtract, intersect, smooth-union, smooth-subtract, smooth-intersect) to 3. The smooth dispatch happens in the combine site based on `k`.

**Language-agnostic:** Yes — pure data-model decision.

### 5.2 Single combine seam dispatched on k

**Problem:** Two-track design (`combineHard(op, a, b)` + `combineSmooth(op, a, b, k)`) makes every call site know which to use.

**Technique:** Single function:
```
combine(op, acc, leaf, k):
  if k < EPSILON: hard min/max/-max
  else:           smin/smax of op
```
One dispatch, both regimes. Caller never picks between hard and smooth.

**Language-agnostic:** Yes.

### 5.3 Seed CSG fold with `+∞`, fold *every* leaf

**Problem:** A naive marcher might initialise the accumulator with the first leaf's SDF, treating it as a union regardless of its actual op. Result: a subtract-only or intersect-only scene **leaks** the leaf's SDF as a visible shape.

**Technique:** Seed `d = +∞` (empty set's SDF). Every leaf — including the first — folds through `combineBoolOp`. Subtract from `+∞` is `+∞` (correct: nothing to carve). Intersect with `+∞` is the leaf (correct: bounded by the leaf alone for that pixel).

**Language-agnostic:** Yes. This is a common shape error in tutorial-grade SDF code.

### 5.4 Group-ancestry walking for bound widening

**Problem:** A leaf's smoothing influence is bounded by the *worst* k along its group ancestry, not just its parent's k.

**Technique:** When packing for the GPU, walk parent chain and take `max(k_parent, k_grandparent, ...)`.

**Language-agnostic:** Yes.

---

## 6. Half-resolution rendering — pay 4× less for what you can't see

### 6.1 Half-res raymarch + edge-aware upscale

**Problem:** 1080p is overkill on most surfaces; the eye doesn't see sub-pixel SDF detail.

**Technique:** Render the marcher at half resolution (¼ the pixels) into an intermediate target. A second pass upscales to swap-chain resolution using **joint-bilateral filtering** keyed on world position:

- For each output pixel, sample the 2×2 nearest half-res neighbours.
- Pick the nearest-by-bilinear-weight as the reference.
- Reject neighbours whose `worldPos` differs from the reference by more than `0.05 × distance + 0.05`.
- Blend remaining neighbours by their bilinear weights.

This preserves silhouette edges (where one half-res pixel is foreground, the next background) instead of bleeding them.

**Trade-off:** Mild blur on smooth shading. Mostly invisible at 50% scale on a tablet.

**Cost:** ~4× fewer marcher pixels. Practically a free 4× speedup on raymarch-bound frames.

**Language-agnostic:** Yes — the upscale shader is tiny and ports trivially.

### 6.2 Adaptive quality during interaction

**Problem:** The user *needs* low latency during orbit / drag (they're feedback-bound) but *wants* full quality at rest.

**Technique:** A `lastInteractionAt` timestamp; the half-res toggle reads it each frame. Half-res while `now() - lastInteractionAt < 150ms`, full-res otherwise. Smooth transition because half-res still shows recognisable shapes.

**Language-agnostic:** Yes.

---

## 7. Untouched but plausibly higher-leverage

Things this codebase doesn't do but that would likely give further wins on a node-based modeler:

- **Hierarchical BVH culling** — currently the BVH is over leaves; you could lift it to operate on groups, culling whole subtrees per ray. Saves more on deep nesting.
- **JIT-compile per-scene shader** — emit a hand-tailored WGSL for the current scene structure (no dynamic op dispatch, just inlined `min`/`max`). Trades compile latency for inner-loop speed. Useful for scenes that stay stable for many frames.
- **Voxelize on-the-fly for complex regions** — bake high-cost subtrees into voxel SDFs lazily; switch the marcher to texture sampling for those regions.
- **Sparse voxel grids** — `r32float` 256³ = 64 MB per leaf. A sparse hash-grid or VDB-style structure scales to bigger fields.
- **Jittered sampling + temporal accumulation** — for soft shadows / AO. Spread the cost over frames.

---

## How these interact

Order of impact when scaling N (leaves) and S (steps):

| N=10, S=64 | N=200, S=64 | N=1500, S=128 |
|---|---|---|
| Naïve fold dominates | BVH cull is essential | BVH cull + tile cull + skip-BVH-on-transient |
| Step count is the lever | Adaptive threshold + bisection | Half-res by default; full only at rest |
| Layout barely matters | Tight leaf records help | Layout + scratch reuse + RAF coalesce critical |

For a Rust + egui + wgpu node-based modeler, I'd land them in roughly this priority:

1. Per-pixel BVH cull (§1.1) + skip-pointer DFS layout
2. Bisection refinement (§2.2) + adaptive threshold (§2.1)
3. Half-res render + edge-aware upscale (§6.1)
4. CSG correctness: seed-with-infinity (§5.3), single combine seam (§5.2)
5. Tight leaf record packing (§3.1) + inverse-rotation side buffer (§3.2)
6. Skip-BVH-on-transient (§4.1) + scratch buffers (§4.3)
7. Tile cull pre-pass (§1.2) — once you've crossed ~30 leaves
8. Adaptive quality (§6.2) — once the half-res path works

Most of these are 50–500 lines each. None require WebGPU specifically — wgpu / Vulkan / Metal all have the storage textures, compute pipelines, and uniform buffers needed.
