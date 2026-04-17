use std::collections::HashMap;
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::compat::maybe_par_iter;
use glam::Vec3;
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

use crate::graph::scene::{NodeData, Scene};
use crate::graph::voxel;

// ---------------------------------------------------------------------------
// Marching cubes tables (Paul Bourke)
// ---------------------------------------------------------------------------

/// For each of 256 cube configurations, a bitmask of which edges are intersected.
#[rustfmt::skip]
const EDGE_TABLE: [u16; 256] = [
    0x000, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x099, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x033, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0x0aa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x066, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0x0ff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x055, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0x0cc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0x0cc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x055, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0x0ff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x066, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0x0aa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x033, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x099, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x000,
];

/// For each of 256 cube configurations, up to 5 triangles (15 edge indices, -1 terminated).
#[rustfmt::skip]
const TRI_TABLE: [[i8; 16]; 256] = [
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  1,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  8,  3,  9,  8,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  8,  3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  2, 10,  0,  2,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 2,  8,  3,  2, 10,  8, 10,  9,  8, -1, -1, -1, -1, -1, -1, -1],
    [ 3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0, 11,  2,  8, 11,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  9,  0,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1, 11,  2,  1,  9, 11,  9,  8, 11, -1, -1, -1, -1, -1, -1, -1],
    [ 3, 10,  1, 11, 10,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0, 10,  1,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 3,  9,  0,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 4,  3,  0,  7,  3,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  1,  9,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 4,  1,  9,  4,  7,  1,  7,  3,  1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  2, 10,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 3,  4,  7,  3,  0,  4,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  2, 10,  9,  0,  2,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1],
    [ 2, 10,  9,  2,  9,  7,  2,  7,  3,  7,  9,  4, -1, -1, -1, -1],
    [ 8,  4,  7,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11,  4,  7, 11,  2,  4,  2,  0,  4, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  0,  1,  8,  4,  7,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1],
    [ 4,  7, 11,  9,  4, 11,  9, 11,  2,  9,  2,  1, -1, -1, -1, -1],
    [ 3, 10,  1,  3, 11, 10,  7,  8,  4, -1, -1, -1, -1, -1, -1, -1],
    [ 1, 11, 10,  1,  4, 11,  1,  0,  4,  7, 11,  4, -1, -1, -1, -1],
    [ 4,  7,  8,  9,  0, 11,  9, 11, 10, 11,  0,  3, -1, -1, -1, -1],
    [ 4,  7, 11,  4, 11,  9,  9, 11, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  5,  4,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  5,  4,  1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 8,  5,  4,  8,  3,  5,  3,  1,  5, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 3,  0,  8,  1,  2, 10,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1],
    [ 5,  2, 10,  5,  4,  2,  4,  0,  2, -1, -1, -1, -1, -1, -1, -1],
    [ 2, 10,  5,  3,  2,  5,  3,  5,  4,  3,  4,  8, -1, -1, -1, -1],
    [ 9,  5,  4,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0, 11,  2,  0,  8, 11,  4,  9,  5, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  5,  4,  0,  1,  5,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1],
    [ 2,  1,  5,  2,  5,  8,  2,  8, 11,  4,  8,  5, -1, -1, -1, -1],
    [10,  3, 11, 10,  1,  3,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1],
    [ 4,  9,  5,  0,  8,  1,  8, 10,  1,  8, 11, 10, -1, -1, -1, -1],
    [ 5,  4,  0,  5,  0, 11,  5, 11, 10, 11,  0,  3, -1, -1, -1, -1],
    [ 5,  4,  8,  5,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  7,  8,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  3,  0,  9,  5,  3,  5,  7,  3, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  7,  8,  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  7,  8,  9,  5,  7, 10,  1,  2, -1, -1, -1, -1, -1, -1, -1],
    [10,  1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3, -1, -1, -1, -1],
    [ 8,  0,  2,  8,  2,  5,  8,  5,  7, 10,  5,  2, -1, -1, -1, -1],
    [ 2, 10,  5,  2,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1],
    [ 7,  9,  5,  7,  8,  9,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7, 11, -1, -1, -1, -1],
    [ 2,  3, 11,  0,  1,  8,  1,  7,  8,  1,  5,  7, -1, -1, -1, -1],
    [11,  2,  1, 11,  1,  7,  7,  1,  5, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  5,  8,  8,  5,  7, 10,  1,  3, 10,  3, 11, -1, -1, -1, -1],
    [ 5,  7,  0,  5,  0,  9,  7, 11,  0,  1,  0, 10, 11, 10,  0, -1],
    [11, 10,  0, 11,  0,  3, 10,  5,  0,  8,  0,  7,  5,  7,  0, -1],
    [11, 10,  5,  7, 11,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  8,  3,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  0,  1,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  8,  3,  1,  9,  8,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  6,  5,  2,  6,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  6,  5,  1,  2,  6,  3,  0,  8, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1, -1, -1, -1],
    [ 5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8, -1, -1, -1, -1],
    [ 2,  3, 11, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11,  0,  8, 11,  2,  0, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  1,  9,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1],
    [ 5, 10,  6,  1,  9,  2,  9, 11,  2,  9,  8, 11, -1, -1, -1, -1],
    [ 6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, -1, -1, -1, -1],
    [ 3, 11,  6,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1, -1, -1, -1],
    [ 6,  5,  9,  6,  9, 11, 11,  9,  8, -1, -1, -1, -1, -1, -1, -1],
    [ 5, 10,  6,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 4,  3,  0,  4,  7,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  9,  0,  5, 10,  6,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1],
    [10,  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4, -1, -1, -1, -1],
    [ 6,  1,  2,  6,  5,  1,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7, -1, -1, -1, -1],
    [ 8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6, -1, -1, -1, -1],
    [ 7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9, -1],
    [ 3, 11,  2,  7,  8,  4, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1],
    [ 5, 10,  6,  4,  7,  2,  4,  2,  0,  2,  7, 11, -1, -1, -1, -1],
    [ 0,  1,  9,  4,  7,  8,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1],
    [ 9,  2,  1,  9, 11,  2,  9,  4, 11,  7, 11,  4,  5, 10,  6, -1],
    [ 8,  4,  7,  3, 11,  5,  3,  5,  1,  5, 11,  6, -1, -1, -1, -1],
    [ 5,  1, 11,  5, 11,  6,  1,  0, 11,  7, 11,  4,  0,  4, 11, -1],
    [ 0,  5,  9,  0,  6,  5,  0,  3,  6, 11,  6,  3,  8,  4,  7, -1],
    [ 6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, -1, -1, -1, -1],
    [10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 4, 10,  6,  4,  9, 10,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1],
    [10,  0,  1, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1, -1, -1, -1],
    [ 8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1, 10, -1, -1, -1, -1],
    [ 1,  4,  9,  1,  2,  4,  2,  6,  4, -1, -1, -1, -1, -1, -1, -1],
    [ 3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4, -1, -1, -1, -1],
    [ 0,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 8,  3,  2,  8,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1],
    [10,  4,  9, 10,  6,  4, 11,  2,  3, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  8,  2,  2,  8, 11,  4,  9, 10,  4, 10,  6, -1, -1, -1, -1],
    [ 3, 11,  2,  0,  1,  6,  0,  6,  4,  6,  1, 10, -1, -1, -1, -1],
    [ 6,  4,  1,  6,  1, 10,  4,  8,  1,  2,  1, 11,  8, 11,  1, -1],
    [ 9,  6,  4,  9,  3,  6,  9,  1,  3, 11,  6,  3, -1, -1, -1, -1],
    [ 8, 11,  1,  8,  1,  0, 11,  6,  1,  9,  1,  4,  6,  4,  1, -1],
    [ 3, 11,  6,  3,  6,  0,  0,  6,  4, -1, -1, -1, -1, -1, -1, -1],
    [ 6,  4,  8, 11,  6,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 7, 10,  6,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  7,  3,  0, 10,  7,  0,  9, 10,  6,  7, 10, -1, -1, -1, -1],
    [10,  6,  7,  1, 10,  7,  1,  7,  8,  1,  8,  0, -1, -1, -1, -1],
    [10,  6,  7, 10,  7,  1,  1,  7,  3, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7, -1, -1, -1, -1],
    [ 2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9, -1],
    [ 7,  8,  0,  7,  0,  6,  6,  0,  2, -1, -1, -1, -1, -1, -1, -1],
    [ 7,  3,  2,  6,  7,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 2,  3, 11, 10,  6,  8, 10,  8,  9,  8,  6,  7, -1, -1, -1, -1],
    [ 2,  0,  7,  2,  7, 11,  0,  9,  7,  6,  7, 10,  9, 10,  7, -1],
    [ 1,  8,  0,  1,  7,  8,  1, 10,  7,  6,  7, 10,  2,  3, 11, -1],
    [11,  2,  1, 11,  1,  7, 10,  6,  1,  6,  7,  1, -1, -1, -1, -1],
    [ 8,  9,  6,  8,  6,  7,  9,  1,  6, 11,  6,  3,  1,  3,  6, -1],
    [ 0,  9,  1, 11,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 7,  8,  0,  7,  0,  6,  3, 11,  0, 11,  6,  0, -1, -1, -1, -1],
    [ 7, 11,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 3,  0,  8, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  1,  9, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 8,  1,  9,  8,  3,  1, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1],
    [10,  1,  2,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  2, 10,  3,  0,  8,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1],
    [ 2,  9,  0,  2, 10,  9,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1],
    [ 6, 11,  7,  2, 10,  3, 10,  8,  3, 10,  9,  8, -1, -1, -1, -1],
    [ 7,  2,  3,  6,  2,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 7,  0,  8,  7,  6,  0,  6,  2,  0, -1, -1, -1, -1, -1, -1, -1],
    [ 2,  7,  6,  2,  3,  7,  0,  1,  9, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6, -1, -1, -1, -1],
    [10,  7,  6, 10,  1,  7,  1,  3,  7, -1, -1, -1, -1, -1, -1, -1],
    [10,  7,  6,  1,  7, 10,  1,  8,  7,  1,  0,  8, -1, -1, -1, -1],
    [ 0,  3,  7,  0,  7, 10,  0, 10,  9,  6, 10,  7, -1, -1, -1, -1],
    [ 7,  6, 10,  7, 10,  8,  8, 10,  9, -1, -1, -1, -1, -1, -1, -1],
    [ 6,  8,  4, 11,  8,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 3,  6, 11,  3,  0,  6,  0,  4,  6, -1, -1, -1, -1, -1, -1, -1],
    [ 8,  6, 11,  8,  4,  6,  9,  0,  1, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  4,  6,  9,  6,  3,  9,  3,  1, 11,  3,  6, -1, -1, -1, -1],
    [ 6,  8,  4,  6, 11,  8,  2, 10,  1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  2, 10,  3,  0, 11,  0,  6, 11,  0,  4,  6, -1, -1, -1, -1],
    [ 4, 11,  8,  4,  6, 11,  0,  2,  9,  2, 10,  9, -1, -1, -1, -1],
    [10,  9,  3, 10,  3,  2,  9,  4,  3, 11,  3,  6,  4,  6,  3, -1],
    [ 8,  2,  3,  8,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8, -1, -1, -1, -1],
    [ 1,  9,  4,  1,  4,  2,  2,  4,  6, -1, -1, -1, -1, -1, -1, -1],
    [ 8,  1,  3,  8,  6,  1,  8,  4,  6,  6, 10,  1, -1, -1, -1, -1],
    [10,  1,  0, 10,  0,  6,  6,  0,  4, -1, -1, -1, -1, -1, -1, -1],
    [ 4,  6,  3,  4,  3,  8,  6, 10,  3,  0,  3,  9, 10,  9,  3, -1],
    [10,  9,  4,  6, 10,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 4,  9,  5,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  8,  3,  4,  9,  5, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1],
    [ 5,  0,  1,  5,  4,  0,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1],
    [11,  7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5, -1, -1, -1, -1],
    [ 9,  5,  4, 10,  1,  2,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1],
    [ 6, 11,  7,  1,  2, 10,  0,  8,  3,  4,  9,  5, -1, -1, -1, -1],
    [ 7,  6, 11,  5,  4, 10,  4,  2, 10,  4,  0,  2, -1, -1, -1, -1],
    [ 3,  4,  8,  3,  5,  4,  3,  2,  5, 10,  5,  2, 11,  7,  6, -1],
    [ 7,  2,  3,  7,  6,  2,  5,  4,  9, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7, -1, -1, -1, -1],
    [ 3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0, -1, -1, -1, -1],
    [ 6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8, -1],
    [ 9,  5,  4, 10,  1,  6,  1,  7,  6,  1,  3,  7, -1, -1, -1, -1],
    [ 1,  6, 10,  1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4, -1],
    [ 4,  0, 10,  4, 10,  5,  0,  3, 10,  6, 10,  7,  3,  7, 10, -1],
    [ 7,  6, 10,  7, 10,  8,  5,  4, 10,  4,  8, 10, -1, -1, -1, -1],
    [ 6,  9,  5,  6, 11,  9, 11,  8,  9, -1, -1, -1, -1, -1, -1, -1],
    [ 3,  6, 11,  0,  6,  3,  0,  5,  6,  0,  9,  5, -1, -1, -1, -1],
    [ 0, 11,  8,  0,  5, 11,  0,  1,  5,  5,  6, 11, -1, -1, -1, -1],
    [ 6, 11,  3,  6,  3,  5,  5,  3,  1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  2, 10,  9,  5, 11,  9, 11,  8, 11,  5,  6, -1, -1, -1, -1],
    [ 0, 11,  3,  0,  6, 11,  0,  9,  6,  5,  6,  9,  1,  2, 10, -1],
    [11,  8,  5, 11,  5,  6,  8,  0,  5, 10,  5,  2,  0,  2,  5, -1],
    [ 6, 11,  3,  6,  3,  5,  2, 10,  3, 10,  5,  3, -1, -1, -1, -1],
    [ 5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2, -1, -1, -1, -1],
    [ 9,  5,  6,  9,  6,  0,  0,  6,  2, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8, -1],
    [ 1,  5,  6,  2,  1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  3,  6,  1,  6, 10,  3,  8,  6,  5,  6,  9,  8,  9,  6, -1],
    [10,  1,  0, 10,  0,  6,  9,  5,  0,  5,  6,  0, -1, -1, -1, -1],
    [ 0,  3,  8,  5,  6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [10,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11,  5, 10,  7,  5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [11,  5, 10, 11,  7,  5,  8,  3,  0, -1, -1, -1, -1, -1, -1, -1],
    [ 5, 11,  7,  5, 10, 11,  1,  9,  0, -1, -1, -1, -1, -1, -1, -1],
    [10, 7,   5, 10, 11,  7,  9,  8,  1,  8,  3,  1, -1, -1, -1, -1],
    [11,  1,  2, 11,  7,  1,  7,  5,  1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2, 11, -1, -1, -1, -1],
    [ 9,  7,  5,  9,  2,  7,  9,  0,  2,  2, 11,  7, -1, -1, -1, -1],
    [ 7,  5,  2,  7,  2, 11,  5,  9,  2,  3,  2,  8,  9,  8,  2, -1],
    [ 2,  5, 10,  2,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1],
    [ 8,  2,  0,  8,  5,  2,  8,  7,  5, 10,  2,  5, -1, -1, -1, -1],
    [ 9,  0,  1, 10,  2,  5,  2,  7,  5,  2,  3,  7, -1, -1, -1, -1],
    [ 9,  8,  2,  9,  2,  1,  8,  7,  2, 10,  2,  5,  7,  5,  2, -1],
    [ 1,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  8,  7,  0,  7,  1,  1,  7,  5, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  0,  3,  9,  3,  5,  5,  3,  7, -1, -1, -1, -1, -1, -1, -1],
    [ 9,  8,  7,  5,  9,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 5,  8,  4,  5, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1],
    [ 5,  0,  4,  5, 11,  0,  5, 10, 11, 11,  3,  0, -1, -1, -1, -1],
    [ 0,  1,  9,  8,  4, 10,  8, 10, 11, 10,  4,  5, -1, -1, -1, -1],
    [10, 11,  4, 10,  4,  5, 11,  3,  4,  9,  4,  1,  3,  1,  4, -1],
    [ 2,  5,  1,  2,  8,  5,  2, 11,  8,  4,  5,  8, -1, -1, -1, -1],
    [ 0,  4, 11,  0, 11,  3,  4,  5, 11,  2, 11,  1,  5,  1, 11, -1],
    [ 0,  2,  5,  0,  5,  9,  2, 11,  5,  4,  5,  8, 11,  8,  5, -1],
    [ 9,  4,  5,  2, 11,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 2,  5, 10,  3,  5,  2,  3,  4,  5,  3,  8,  4, -1, -1, -1, -1],
    [ 5, 10,  2,  5,  2,  4,  4,  2,  0, -1, -1, -1, -1, -1, -1, -1],
    [ 3, 10,  2,  3,  5, 10,  3,  8,  5,  4,  5,  8,  0,  1,  9, -1],
    [ 5, 10,  2,  5,  2,  4,  1,  9,  2,  9,  4,  2, -1, -1, -1, -1],
    [ 8,  4,  5,  8,  5,  3,  3,  5,  1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  4,  5,  1,  0,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5, -1, -1, -1, -1],
    [ 9,  4,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 4, 11,  7,  4,  9, 11,  9, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  8,  3,  4,  9,  7,  9, 11,  7,  9, 10, 11, -1, -1, -1, -1],
    [ 1, 10, 11,  1, 11,  4,  1,  4,  0,  7,  4, 11, -1, -1, -1, -1],
    [ 3,  1,  4,  3,  4,  8,  1, 10,  4,  7,  4, 11, 10, 11,  4, -1],
    [ 4, 11,  7,  9, 11,  4,  9,  2, 11,  9,  1,  2, -1, -1, -1, -1],
    [ 9,  7,  4,  9, 11,  7,  9,  1, 11,  2, 11,  1,  0,  8,  3, -1],
    [11,  7,  4,  11,  4,  2,  2,  4,  0, -1, -1, -1, -1, -1, -1, -1],
    [11,  7,  4, 11,  4,  2,  8,  3,  4,  3,  2,  4, -1, -1, -1, -1],
    [ 2,  9, 10,  2,  7,  9,  2,  3,  7,  7,  4,  9, -1, -1, -1, -1],
    [ 9, 10,  7,  9,  7,  4, 10,  2,  7,  8,  7,  0,  2,  0,  7, -1],
    [ 3,  7, 10,  3, 10,  2,  7,  4, 10,  1, 10,  0,  4,  0, 10, -1],
    [ 1, 10,  2,  8,  7,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 4,  9,  1,  4,  1,  7,  7,  1,  3, -1, -1, -1, -1, -1, -1, -1],
    [ 4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1, -1, -1, -1, -1],
    [ 4,  0,  3,  7,  4,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 4,  8,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 9, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 3,  0,  9,  3,  9, 11, 11,  9, 10, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  1, 10,  0, 10,  8,  8, 10, 11, -1, -1, -1, -1, -1, -1, -1],
    [ 3,  1, 10, 11,  3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  2, 11,  1, 11,  9,  9, 11,  8, -1, -1, -1, -1, -1, -1, -1],
    [ 3,  0,  9,  3,  9, 11,  1,  2,  9,  2, 11,  9, -1, -1, -1, -1],
    [ 0,  2, 11,  8,  0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 3,  2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 2,  3,  8,  2,  8, 10, 10,  8,  9, -1, -1, -1, -1, -1, -1, -1],
    [ 9, 10,  2,  0,  9,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 2,  3,  8,  2,  8, 10,  0,  1,  8,  1, 10,  8, -1, -1, -1, -1],
    [ 1, 10,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 1,  3,  8,  9,  1,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  9,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [ 0,  3,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
];

// ---------------------------------------------------------------------------
// Marching cubes implementation
// ---------------------------------------------------------------------------

pub struct ExportMesh {
    pub vertices: Vec<[f32; 3]>,
    pub triangles: Vec<[u32; 3]>,
    pub vertex_colors: Vec<[f32; 3]>,
}

/// Corner offsets for a unit cube (matches edge/tri table convention).
const CORNERS: [[u32; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

/// Each edge connects two corners (indices into CORNERS).
const EDGE_CORNERS: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

/// Interpolate vertex position along an edge where the SDF crosses zero.
fn interp_vertex(p0: Vec3, v0: f32, p1: Vec3, v1: f32) -> Vec3 {
    if (v1 - v0).abs() < 1e-10 {
        return p0;
    }
    let t = (-v0) / (v1 - v0);
    let t = t.clamp(0.0, 1.0);
    p0 + t * (p1 - p0)
}

/// Quantize a vertex position for deduplication.
fn quantize(v: &[f32; 3]) -> (i32, i32, i32) {
    // Quantize to ~0.0001 unit precision
    (
        (v[0] * 10000.0).round() as i32,
        (v[1] * 10000.0).round() as i32,
        (v[2] * 10000.0).round() as i32,
    )
}

/// Collect all leaf (Primitive/Sculpt) node IDs and their colors from the scene.
fn collect_leaf_colors(scene: &Scene) -> Vec<(crate::graph::scene::NodeId, [f32; 3])> {
    let order = scene.visible_topo_order();
    let mut leaves = Vec::new();
    for &id in &order {
        if let Some(node) = scene.nodes.get(&id) {
            match &node.data {
                NodeData::Primitive { material, .. } | NodeData::Sculpt { material, .. } => {
                    let color = material.base_color;
                    leaves.push((id, [color.x, color.y, color.z]));
                }
                _ => {}
            }
        }
    }
    leaves
}

/// Find the color of the closest leaf node at a given point.
fn sample_color_at(
    scene: &Scene,
    p: Vec3,
    leaves: &[(crate::graph::scene::NodeId, [f32; 3])],
) -> [f32; 3] {
    let mut best_dist = f32::MAX;
    let mut best_color = [0.5, 0.5, 0.5];
    for &(id, color) in leaves {
        let d = voxel::evaluate_sdf_tree(scene, id, p).abs();
        if d < best_dist {
            best_dist = d;
            best_color = color;
        }
    }
    best_color
}

pub fn marching_cubes(
    scene: &Scene,
    resolution: u32,
    bounds_min: Vec3,
    bounds_max: Vec3,
    progress: &AtomicU32,
    adaptive: bool,
    cancelled: &AtomicBool,
) -> Option<ExportMesh> {
    let res = resolution as usize;
    let grid_size = res + 1; // Number of sample points per axis
    let step = (bounds_max - bounds_min) / resolution as f32;

    // Find root nodes to evaluate
    let roots = scene.top_level_nodes();

    // Build surface mask if adaptive mode is on.
    // Coarse pass identifies which fine-grid cells are near the surface,
    // so we can skip SDF evaluation for cells deep inside or far outside.
    let surface_mask: Option<Vec<bool>> = if adaptive && resolution >= 32 {
        let coarse_div = (resolution / 4).max(8) as usize;
        let coarse_gs = coarse_div + 1;
        let coarse_step = (bounds_max - bounds_min) / coarse_div as f32;

        // Sample coarse grid
        let coarse_slices: Vec<Vec<f32>> = maybe_par_iter!(0..coarse_gs)
            .map(|z| {
                let mut slice = vec![0.0f32; coarse_gs * coarse_gs];
                for y in 0..coarse_gs {
                    for x in 0..coarse_gs {
                        let p = bounds_min + Vec3::new(x as f32, y as f32, z as f32) * coarse_step;
                        let mut d = f32::MAX;
                        for &root_id in &roots {
                            d = d.min(voxel::evaluate_sdf_tree(scene, root_id, p));
                        }
                        slice[y * coarse_gs + x] = d;
                    }
                }
                slice
            })
            .collect();

        // Identify coarse cells with sign changes (surface intersections)
        let mut coarse_active = vec![false; coarse_div * coarse_div * coarse_div];
        for z in 0..coarse_div {
            for y in 0..coarse_div {
                for x in 0..coarse_div {
                    let mut has_neg = false;
                    let mut has_pos = false;
                    for c in &CORNERS {
                        let gx = x + c[0] as usize;
                        let gy = y + c[1] as usize;
                        let gz = z + c[2] as usize;
                        let v = coarse_slices[gz][gy * coarse_gs + gx];
                        if v < 0.0 {
                            has_neg = true;
                        } else {
                            has_pos = true;
                        }
                    }
                    if has_neg && has_pos {
                        coarse_active[z * coarse_div * coarse_div + y * coarse_div + x] = true;
                    }
                }
            }
        }

        // Expand mask by 1 cell to catch boundary regions
        let mut expanded = vec![false; coarse_div * coarse_div * coarse_div];
        for z in 0..coarse_div {
            for y in 0..coarse_div {
                for x in 0..coarse_div {
                    if coarse_active[z * coarse_div * coarse_div + y * coarse_div + x] {
                        for dz in 0..3usize {
                            for dy in 0..3usize {
                                for dx in 0..3usize {
                                    let nx = (x + dx).wrapping_sub(1);
                                    let ny = (y + dy).wrapping_sub(1);
                                    let nz = (z + dz).wrapping_sub(1);
                                    if nx < coarse_div && ny < coarse_div && nz < coarse_div {
                                        expanded
                                            [nz * coarse_div * coarse_div + ny * coarse_div + nx] =
                                            true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Map fine-grid points to coarse mask
        let scale = coarse_div as f32 / res as f32;
        let mut fine_mask = vec![false; grid_size * grid_size * grid_size];
        for z in 0..grid_size {
            for y in 0..grid_size {
                for x in 0..grid_size {
                    let cx = ((x as f32 * scale) as usize).min(coarse_div - 1);
                    let cy = ((y as f32 * scale) as usize).min(coarse_div - 1);
                    let cz = ((z as f32 * scale) as usize).min(coarse_div - 1);
                    if expanded[cz * coarse_div * coarse_div + cy * coarse_div + cx] {
                        fine_mask[z * grid_size * grid_size + y * grid_size + x] = true;
                    }
                }
            }
        }
        Some(fine_mask)
    } else {
        None
    };

    // Phase 1: Sample SDF at all grid points (parallelized by z-slice)
    let mut field = vec![0.0f32; grid_size * grid_size * grid_size];

    // Sample in parallel by z-slice
    let slices: Vec<Vec<f32>> = maybe_par_iter!(0..grid_size)
        .map(|z| {
            let mut slice = vec![0.0f32; grid_size * grid_size];
            for y in 0..grid_size {
                for x in 0..grid_size {
                    // Skip points not near the surface in adaptive mode
                    if let Some(ref mask) = surface_mask {
                        if !mask[z * grid_size * grid_size + y * grid_size + x] {
                            slice[y * grid_size + x] = f32::MAX;
                            continue;
                        }
                    }
                    let p = bounds_min + Vec3::new(x as f32, y as f32, z as f32) * step;
                    let mut d = f32::MAX;
                    for &root_id in &roots {
                        let rd = voxel::evaluate_sdf_tree(scene, root_id, p);
                        d = d.min(rd);
                    }
                    slice[y * grid_size + x] = d;
                }
            }
            progress.fetch_add(1, Ordering::Relaxed);
            slice
        })
        .collect();

    // Copy slices into contiguous field
    for (z, slice) in slices.into_iter().enumerate() {
        let offset = z * grid_size * grid_size;
        field[offset..offset + grid_size * grid_size].copy_from_slice(&slice);
    }

    if cancelled.load(Ordering::Relaxed) {
        return None;
    }

    // Phase 2: Process cells to extract triangles (parallelized by z-slice)
    #[allow(clippy::type_complexity)]
    let cell_slices: Vec<(Vec<[f32; 3]>, Vec<[u32; 3]>)> = maybe_par_iter!(0..res)
        .map(|z| {
            let mut local_verts: Vec<[f32; 3]> = Vec::new();
            let mut local_tris: Vec<[u32; 3]> = Vec::new();
            let mut local_dedup: HashMap<(i32, i32, i32), u32> = HashMap::new();

            for y in 0..res {
                for x in 0..res {
                    // Get corner values
                    let mut vals = [0.0f32; 8];
                    for (i, c) in CORNERS.iter().enumerate() {
                        let gx = x + c[0] as usize;
                        let gy = y + c[1] as usize;
                        let gz = z + c[2] as usize;
                        vals[i] = field[gz * grid_size * grid_size + gy * grid_size + gx];
                    }

                    // Compute cube index
                    let mut cube_index: u8 = 0;
                    for (i, val) in vals.iter().enumerate() {
                        if *val < 0.0 {
                            cube_index |= 1 << i;
                        }
                    }

                    if EDGE_TABLE[cube_index as usize] == 0 {
                        continue;
                    }

                    // Compute edge vertices
                    let mut edge_verts = [Vec3::ZERO; 12];
                    let edges = EDGE_TABLE[cube_index as usize];
                    for e in 0..12 {
                        if edges & (1 << e) != 0 {
                            let [c0, c1] = EDGE_CORNERS[e];
                            let p0 = bounds_min
                                + Vec3::new(
                                    (x + CORNERS[c0][0] as usize) as f32,
                                    (y + CORNERS[c0][1] as usize) as f32,
                                    (z + CORNERS[c0][2] as usize) as f32,
                                ) * step;
                            let p1 = bounds_min
                                + Vec3::new(
                                    (x + CORNERS[c1][0] as usize) as f32,
                                    (y + CORNERS[c1][1] as usize) as f32,
                                    (z + CORNERS[c1][2] as usize) as f32,
                                ) * step;
                            edge_verts[e] = interp_vertex(p0, vals[c0], p1, vals[c1]);
                        }
                    }

                    // Emit triangles
                    let tri_row = &TRI_TABLE[cube_index as usize];
                    let mut t = 0;
                    while t < 15 && tri_row[t] >= 0 {
                        let mut tri = [0u32; 3];
                        for j in 0..3 {
                            let edge_idx = tri_row[t + j] as usize;
                            let v = edge_verts[edge_idx];
                            let vf = [v.x, v.y, v.z];
                            let key = quantize(&vf);
                            let idx = if let Some(&existing) = local_dedup.get(&key) {
                                existing
                            } else {
                                let new_idx = local_verts.len() as u32;
                                local_verts.push(vf);
                                local_dedup.insert(key, new_idx);
                                new_idx
                            };
                            tri[j] = idx;
                        }
                        local_tris.push(tri);
                        t += 3;
                    }
                }
            }
            progress.fetch_add(1, Ordering::Relaxed);
            (local_verts, local_tris)
        })
        .collect();

    if cancelled.load(Ordering::Relaxed) {
        return None;
    }

    // Phase 3: Merge per-slice results with global dedup
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut global_dedup: HashMap<(i32, i32, i32), u32> = HashMap::new();

    for (local_verts, local_tris) in cell_slices {
        // Build index remap for this slice
        let mut remap: Vec<u32> = Vec::with_capacity(local_verts.len());
        for vf in &local_verts {
            let key = quantize(vf);
            let idx = if let Some(&existing) = global_dedup.get(&key) {
                existing
            } else {
                let new_idx = vertices.len() as u32;
                vertices.push(*vf);
                global_dedup.insert(key, new_idx);
                new_idx
            };
            remap.push(idx);
        }

        for tri in &local_tris {
            triangles.push([
                remap[tri[0] as usize],
                remap[tri[1] as usize],
                remap[tri[2] as usize],
            ]);
        }
    }

    // Phase 4: Sample vertex colors (parallelized)
    let leaves = collect_leaf_colors(scene);
    let vertex_colors: Vec<[f32; 3]> = if leaves.is_empty() {
        vec![[0.5, 0.5, 0.5]; vertices.len()]
    } else {
        maybe_par_iter!(&vertices)
            .map(|v| sample_color_at(scene, Vec3::new(v[0], v[1], v[2]), &leaves))
            .collect()
    };

    if cancelled.load(Ordering::Relaxed) {
        return None;
    }

    Some(ExportMesh {
        vertices,
        triangles,
        vertex_colors,
    })
}

pub fn write_obj_to(mesh: &ExportMesh, writer: &mut impl Write) -> Result<(), String> {
    writeln!(writer, "# SDF Modeler Export").map_err(|e| e.to_string())?;
    writeln!(
        writer,
        "# Vertices: {}, Triangles: {}",
        mesh.vertices.len(),
        mesh.triangles.len()
    )
    .map_err(|e| e.to_string())?;

    for v in &mesh.vertices {
        writeln!(writer, "v {} {} {}", v[0], v[1], v[2]).map_err(|e| e.to_string())?;
    }

    // OBJ uses 1-indexed faces
    for t in &mesh.triangles {
        writeln!(writer, "f {} {} {}", t[0] + 1, t[1] + 1, t[2] + 1).map_err(|e| e.to_string())?;
    }

    writer.flush().map_err(|e| e.to_string())?;
    Ok(())
}

pub fn write_obj(mesh: &ExportMesh, path: &Path) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut writer = std::io::BufWriter::new(file);
    write_obj_to(mesh, &mut writer)
}

pub fn write_stl(mesh: &ExportMesh, path: &Path) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut w = std::io::BufWriter::new(file);

    // 80-byte header
    let mut header = [0u8; 80];
    let tag = b"SDF Modeler Export";
    header[..tag.len()].copy_from_slice(tag);
    w.write_all(&header).map_err(|e| e.to_string())?;

    // Triangle count (u32 LE)
    let num_tris = mesh.triangles.len() as u32;
    w.write_all(&num_tris.to_le_bytes())
        .map_err(|e| e.to_string())?;

    for tri in &mesh.triangles {
        let v0 = Vec3::from(mesh.vertices[tri[0] as usize]);
        let v1 = Vec3::from(mesh.vertices[tri[1] as usize]);
        let v2 = Vec3::from(mesh.vertices[tri[2] as usize]);
        let normal = (v1 - v0).cross(v2 - v0).normalize_or_zero();

        // Normal
        for c in [normal.x, normal.y, normal.z] {
            w.write_all(&c.to_le_bytes()).map_err(|e| e.to_string())?;
        }
        // 3 vertices
        for v in [v0, v1, v2] {
            for c in [v.x, v.y, v.z] {
                w.write_all(&c.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }
        // Attribute byte count
        w.write_all(&0u16.to_le_bytes())
            .map_err(|e| e.to_string())?;
    }

    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

pub fn write_ply(mesh: &ExportMesh, path: &Path) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut w = std::io::BufWriter::new(file);

    // PLY ASCII header
    writeln!(w, "ply").map_err(|e| e.to_string())?;
    writeln!(w, "format ascii 1.0").map_err(|e| e.to_string())?;
    writeln!(w, "comment SDF Modeler Export").map_err(|e| e.to_string())?;
    writeln!(w, "element vertex {}", mesh.vertices.len()).map_err(|e| e.to_string())?;
    writeln!(w, "property float x").map_err(|e| e.to_string())?;
    writeln!(w, "property float y").map_err(|e| e.to_string())?;
    writeln!(w, "property float z").map_err(|e| e.to_string())?;
    let has_colors = mesh.vertex_colors.len() == mesh.vertices.len();
    if has_colors {
        writeln!(w, "property uchar red").map_err(|e| e.to_string())?;
        writeln!(w, "property uchar green").map_err(|e| e.to_string())?;
        writeln!(w, "property uchar blue").map_err(|e| e.to_string())?;
    }
    writeln!(w, "element face {}", mesh.triangles.len()).map_err(|e| e.to_string())?;
    writeln!(w, "property list uchar uint vertex_indices").map_err(|e| e.to_string())?;
    writeln!(w, "end_header").map_err(|e| e.to_string())?;

    for (i, v) in mesh.vertices.iter().enumerate() {
        if has_colors {
            let c = &mesh.vertex_colors[i];
            let r = (c[0].clamp(0.0, 1.0) * 255.0) as u8;
            let g = (c[1].clamp(0.0, 1.0) * 255.0) as u8;
            let b = (c[2].clamp(0.0, 1.0) * 255.0) as u8;
            writeln!(w, "{} {} {} {} {} {}", v[0], v[1], v[2], r, g, b)
                .map_err(|e| e.to_string())?;
        } else {
            writeln!(w, "{} {} {}", v[0], v[1], v[2]).map_err(|e| e.to_string())?;
        }
    }

    for t in &mesh.triangles {
        writeln!(w, "3 {} {} {}", t[0], t[1], t[2]).map_err(|e| e.to_string())?;
    }

    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

pub fn write_glb(mesh: &ExportMesh, path: &Path) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut w = std::io::BufWriter::new(file);

    let has_colors = mesh.vertex_colors.len() == mesh.vertices.len();

    // --- Build binary buffer: [positions] [colors?] [indices] ---
    let pos_bytes = mesh.vertices.len() * 12; // 3 x f32
    let col_bytes = if has_colors {
        mesh.vertex_colors.len() * 12
    } else {
        0
    }; // 3 x f32
    let idx_bytes = mesh.triangles.len() * 12; // 3 x u32
    let bin_len = pos_bytes + col_bytes + idx_bytes;

    // Compute position min/max for accessor bounds
    let mut pos_min = [f32::MAX; 3];
    let mut pos_max = [f32::MIN; 3];
    for v in &mesh.vertices {
        for i in 0..3 {
            pos_min[i] = pos_min[i].min(v[i]);
            pos_max[i] = pos_max[i].max(v[i]);
        }
    }

    // --- Build JSON ---
    let json = if has_colors {
        let idx_offset = pos_bytes + col_bytes;
        format!(
            r#"{{"asset":{{"version":"2.0","generator":"SDF Modeler"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0,"COLOR_0":2}},"indices":1}}]}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":{},"type":"VEC3","min":[{},{},{}],"max":[{},{},{}]}},{{"bufferView":2,"componentType":5125,"count":{},"type":"SCALAR"}},{{"bufferView":1,"componentType":5126,"count":{},"type":"VEC3"}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{},"target":34962}},{{"buffer":0,"byteOffset":{},"byteLength":{},"target":34962}},{{"buffer":0,"byteOffset":{},"byteLength":{},"target":34963}}],"buffers":[{{"byteLength":{}}}]}}"#,
            mesh.vertices.len(),
            pos_min[0],
            pos_min[1],
            pos_min[2],
            pos_max[0],
            pos_max[1],
            pos_max[2],
            mesh.triangles.len() * 3,
            mesh.vertex_colors.len(),
            pos_bytes,
            pos_bytes,
            col_bytes,
            idx_offset,
            idx_bytes,
            bin_len,
        )
    } else {
        format!(
            r#"{{"asset":{{"version":"2.0","generator":"SDF Modeler"}},"scene":0,"scenes":[{{"nodes":[0]}}],"nodes":[{{"mesh":0}}],"meshes":[{{"primitives":[{{"attributes":{{"POSITION":0}},"indices":1}}]}}],"accessors":[{{"bufferView":0,"componentType":5126,"count":{},"type":"VEC3","min":[{},{},{}],"max":[{},{},{}]}},{{"bufferView":1,"componentType":5125,"count":{},"type":"SCALAR"}}],"bufferViews":[{{"buffer":0,"byteOffset":0,"byteLength":{},"target":34962}},{{"buffer":0,"byteOffset":{},"byteLength":{},"target":34963}}],"buffers":[{{"byteLength":{}}}]}}"#,
            mesh.vertices.len(),
            pos_min[0],
            pos_min[1],
            pos_min[2],
            pos_max[0],
            pos_max[1],
            pos_max[2],
            mesh.triangles.len() * 3,
            pos_bytes,
            pos_bytes,
            idx_bytes,
            bin_len,
        )
    };

    let json_bytes = json.as_bytes();
    // Pad JSON to 4-byte alignment with spaces (0x20)
    let json_pad = (4 - (json_bytes.len() % 4)) % 4;
    let json_chunk_len = json_bytes.len() + json_pad;
    // Pad BIN to 4-byte alignment with zeros
    let bin_pad = (4 - (bin_len % 4)) % 4;
    let bin_chunk_len = bin_len + bin_pad;

    let total_len = 12 + 8 + json_chunk_len + 8 + bin_chunk_len;

    // --- GLB Header ---
    w.write_all(b"glTF").map_err(|e| e.to_string())?; // magic
    w.write_all(&2u32.to_le_bytes())
        .map_err(|e| e.to_string())?; // version
    w.write_all(&(total_len as u32).to_le_bytes())
        .map_err(|e| e.to_string())?; // length

    // --- JSON Chunk ---
    w.write_all(&(json_chunk_len as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(&0x4E4F534Au32.to_le_bytes())
        .map_err(|e| e.to_string())?; // "JSON"
    w.write_all(json_bytes).map_err(|e| e.to_string())?;
    for _ in 0..json_pad {
        w.write_all(b" ").map_err(|e| e.to_string())?;
    }

    // --- BIN Chunk ---
    w.write_all(&(bin_chunk_len as u32).to_le_bytes())
        .map_err(|e| e.to_string())?;
    w.write_all(&0x004E4942u32.to_le_bytes())
        .map_err(|e| e.to_string())?; // "BIN\0"

    // Positions
    for v in &mesh.vertices {
        for c in v {
            w.write_all(&c.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    // Colors
    if has_colors {
        for c in &mesh.vertex_colors {
            for ch in c {
                w.write_all(&ch.to_le_bytes()).map_err(|e| e.to_string())?;
            }
        }
    }
    // Indices
    for t in &mesh.triangles {
        for i in t {
            w.write_all(&i.to_le_bytes()).map_err(|e| e.to_string())?;
        }
    }
    for _ in 0..bin_pad {
        w.write_all(&[0u8]).map_err(|e| e.to_string())?;
    }

    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

pub fn write_usda(mesh: &ExportMesh, path: &Path) -> Result<(), String> {
    let file = std::fs::File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
    let mut w = std::io::BufWriter::new(file);

    writeln!(w, "#usda 1.0").map_err(|e| e.to_string())?;
    writeln!(w, "(").map_err(|e| e.to_string())?;
    writeln!(w, "    defaultPrim = \"SdfExport\"").map_err(|e| e.to_string())?;
    writeln!(w, "    metersPerUnit = 1").map_err(|e| e.to_string())?;
    writeln!(w, "    upAxis = \"Y\"").map_err(|e| e.to_string())?;
    writeln!(w, ")").map_err(|e| e.to_string())?;
    writeln!(w).map_err(|e| e.to_string())?;
    writeln!(w, "def Mesh \"SdfExport\"").map_err(|e| e.to_string())?;
    writeln!(w, "{{").map_err(|e| e.to_string())?;

    // points
    write!(w, "    point3f[] points = [").map_err(|e| e.to_string())?;
    for (i, v) in mesh.vertices.iter().enumerate() {
        if i > 0 {
            write!(w, ", ").map_err(|e| e.to_string())?;
        }
        write!(w, "({}, {}, {})", v[0], v[1], v[2]).map_err(|e| e.to_string())?;
    }
    writeln!(w, "]").map_err(|e| e.to_string())?;

    // faceVertexCounts — all triangles
    write!(w, "    int[] faceVertexCounts = [").map_err(|e| e.to_string())?;
    for i in 0..mesh.triangles.len() {
        if i > 0 {
            write!(w, ", ").map_err(|e| e.to_string())?;
        }
        write!(w, "3").map_err(|e| e.to_string())?;
    }
    writeln!(w, "]").map_err(|e| e.to_string())?;

    // faceVertexIndices
    write!(w, "    int[] faceVertexIndices = [").map_err(|e| e.to_string())?;
    let mut first = true;
    for t in &mesh.triangles {
        for &idx in t {
            if !first {
                write!(w, ", ").map_err(|e| e.to_string())?;
            }
            write!(w, "{}", idx).map_err(|e| e.to_string())?;
            first = false;
        }
    }
    writeln!(w, "]").map_err(|e| e.to_string())?;

    // Vertex colors
    if mesh.vertex_colors.len() == mesh.vertices.len() {
        write!(w, "    color3f[] primvars:displayColor = [").map_err(|e| e.to_string())?;
        for (i, c) in mesh.vertex_colors.iter().enumerate() {
            if i > 0 {
                write!(w, ", ").map_err(|e| e.to_string())?;
            }
            write!(w, "({}, {}, {})", c[0], c[1], c[2]).map_err(|e| e.to_string())?;
        }
        writeln!(w, "]").map_err(|e| e.to_string())?;
        writeln!(
            w,
            "    uniform token primvars:displayColor:interpolation = \"vertex\""
        )
        .map_err(|e| e.to_string())?;
    }

    writeln!(w, "}}").map_err(|e| e.to_string())?;

    w.flush().map_err(|e| e.to_string())?;
    Ok(())
}

/// Write an ExportMesh to the given path, choosing format by file extension.
pub fn write_mesh(mesh: &ExportMesh, path: &Path) -> Result<(), String> {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("stl") => write_stl(mesh, path),
        Some("ply") => write_ply(mesh, path),
        Some("glb") => write_glb(mesh, path),
        Some("usda") => write_usda(mesh, path),
        _ => write_obj(mesh, path),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::collections::HashSet;
    use std::sync::atomic::AtomicU32;

    use glam::Vec3;

    use crate::graph::scene::{CsgOp, NodeData, NodeId, Scene, SdfPrimitive};

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn empty_scene() -> Scene {
        Scene {
            nodes: HashMap::new(),
            next_id: 0,
            name_counters: HashMap::new(),
            hidden_nodes: HashSet::new(),
            light_masks: HashMap::new(),
            structure_version: 0,
            data_version: 0,
        }
    }

    fn scene_with_sphere() -> (Scene, NodeId) {
        let mut scene = empty_scene();
        let name = scene.next_name("Sphere");
        let id = scene.add_node(
            name,
            NodeData::Primitive {
                kind: SdfPrimitive::Sphere,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
                material: crate::graph::scene::MaterialParams {
                    base_color: Vec3::new(1.0, 0.0, 0.0),
                    roughness: 0.5,
                    metallic: 0.0,
                    emissive_color: Vec3::ZERO,
                    emissive_intensity: 0.0,
                    reflectance_f0: 0.04,
                    ..crate::graph::scene::MaterialParams::default()
                },
                voxel_grid: None,
            },
        );
        (scene, id)
    }

    fn scene_with_box() -> (Scene, NodeId) {
        let mut scene = empty_scene();
        let name = scene.next_name("Box");
        let id = scene.add_node(
            name,
            NodeData::Primitive {
                kind: SdfPrimitive::Box,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
                material: crate::graph::scene::MaterialParams {
                    base_color: Vec3::new(0.0, 1.0, 0.0),
                    roughness: 0.5,
                    metallic: 0.0,
                    emissive_color: Vec3::ZERO,
                    emissive_intensity: 0.0,
                    reflectance_f0: 0.04,
                    ..crate::graph::scene::MaterialParams::default()
                },
                voxel_grid: None,
            },
        );
        (scene, id)
    }

    /// Build a simple mesh for format writer tests.
    fn sample_mesh() -> ExportMesh {
        ExportMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
            vertex_colors: vec![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        }
    }

    /// Build a mesh with no vertex colors.
    fn sample_mesh_no_colors() -> ExportMesh {
        ExportMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
            vertex_colors: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // interp_vertex tests
    // -----------------------------------------------------------------------

    #[test]
    fn interp_vertex_midpoint_when_symmetric() {
        let result = interp_vertex(
            Vec3::new(0.0, 0.0, 0.0),
            -1.0,
            Vec3::new(2.0, 0.0, 0.0),
            1.0,
        );
        assert!((result.x - 1.0).abs() < 1e-6);
        assert!(result.y.abs() < 1e-6);
        assert!(result.z.abs() < 1e-6);
    }

    #[test]
    fn interp_vertex_at_p0_when_v0_is_zero() {
        let result = interp_vertex(Vec3::new(0.0, 0.0, 0.0), 0.0, Vec3::new(1.0, 0.0, 0.0), 1.0);
        assert!((result.x - 0.0).abs() < 1e-6);
    }

    #[test]
    fn interp_vertex_at_p1_when_v1_is_zero() {
        let result = interp_vertex(
            Vec3::new(0.0, 0.0, 0.0),
            -1.0,
            Vec3::new(1.0, 0.0, 0.0),
            0.0,
        );
        assert!((result.x - 1.0).abs() < 1e-6);
    }

    #[test]
    fn interp_vertex_quarter_way() {
        // v0=-1, v1=3 → t = 1/4
        let result = interp_vertex(
            Vec3::new(0.0, 0.0, 0.0),
            -1.0,
            Vec3::new(4.0, 0.0, 0.0),
            3.0,
        );
        assert!((result.x - 1.0).abs() < 1e-5);
    }

    #[test]
    fn interp_vertex_equal_values_returns_p0() {
        let result = interp_vertex(Vec3::new(0.0, 0.0, 0.0), 1.0, Vec3::new(1.0, 0.0, 0.0), 1.0);
        assert!((result.x - 0.0).abs() < 1e-6);
    }

    #[test]
    fn interp_vertex_3d_interpolation() {
        let result = interp_vertex(
            Vec3::new(0.0, 0.0, 0.0),
            -1.0,
            Vec3::new(1.0, 2.0, 3.0),
            1.0,
        );
        assert!((result.x - 0.5).abs() < 1e-5);
        assert!((result.y - 1.0).abs() < 1e-5);
        assert!((result.z - 1.5).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // quantize tests
    // -----------------------------------------------------------------------

    #[test]
    fn quantize_zero() {
        assert_eq!(quantize(&[0.0, 0.0, 0.0]), (0, 0, 0));
    }

    #[test]
    fn quantize_positive_values() {
        assert_eq!(quantize(&[1.0, 2.0, 3.0]), (10000, 20000, 30000));
    }

    #[test]
    fn quantize_negative_values() {
        assert_eq!(quantize(&[-0.5, -1.0, -2.0]), (-5000, -10000, -20000));
    }

    #[test]
    fn quantize_precision_boundary() {
        // Values within 0.0001 should map to the same quantized value
        let a = quantize(&[1.00004, 0.0, 0.0]);
        let b = quantize(&[1.00006, 0.0, 0.0]);
        // Both round to 10000 or 10001 — they should be close
        assert!((a.0 - b.0).abs() <= 1);
    }

    #[test]
    fn quantize_different_values_differ() {
        let a = quantize(&[0.0, 0.0, 0.0]);
        let b = quantize(&[0.01, 0.0, 0.0]);
        assert_ne!(a, b);
    }

    // -----------------------------------------------------------------------
    // collect_leaf_colors tests
    // -----------------------------------------------------------------------

    #[test]
    fn collect_leaf_colors_empty_scene() {
        let scene = empty_scene();
        let leaves = collect_leaf_colors(&scene);
        assert!(leaves.is_empty());
    }

    #[test]
    fn collect_leaf_colors_single_primitive() {
        let (scene, _id) = scene_with_sphere();
        let leaves = collect_leaf_colors(&scene);
        assert_eq!(leaves.len(), 1);
        assert!((leaves[0].1[0] - 1.0).abs() < 1e-6); // red channel
    }

    #[test]
    fn collect_leaf_colors_skips_operations() {
        let mut scene = empty_scene();
        let name = scene.next_name("Union");
        scene.add_node(
            name,
            NodeData::Operation {
                op: CsgOp::Union,
                left: None,
                right: None,
                smooth_k: 0.0,
                steps: 0.0,
                color_blend: -1.0,
            },
        );
        let leaves = collect_leaf_colors(&scene);
        assert!(leaves.is_empty());
    }

    // -----------------------------------------------------------------------
    // sample_color_at tests
    // -----------------------------------------------------------------------

    #[test]
    fn sample_color_at_returns_closest_leaf_color() {
        let (scene, _id) = scene_with_sphere();
        let leaves = collect_leaf_colors(&scene);
        let color = sample_color_at(&scene, Vec3::ZERO, &leaves);
        // Sphere color is (1, 0, 0)
        assert!((color[0] - 1.0).abs() < 1e-6);
        assert!(color[1].abs() < 1e-6);
        assert!(color[2].abs() < 1e-6);
    }

    #[test]
    fn sample_color_at_picks_closer_of_two() {
        let mut scene = empty_scene();
        // Sphere at origin — red
        let name1 = scene.next_name("Sphere");
        scene.add_node(
            name1,
            NodeData::Primitive {
                kind: SdfPrimitive::Sphere,
                position: Vec3::ZERO,
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
                material: crate::graph::scene::MaterialParams {
                    base_color: Vec3::new(1.0, 0.0, 0.0),
                    roughness: 0.5,
                    metallic: 0.0,
                    emissive_color: Vec3::ZERO,
                    emissive_intensity: 0.0,
                    reflectance_f0: 0.04,
                    ..crate::graph::scene::MaterialParams::default()
                },
                voxel_grid: None,
            },
        );
        // Sphere at (5,0,0) — blue
        let name2 = scene.next_name("Sphere");
        scene.add_node(
            name2,
            NodeData::Primitive {
                kind: SdfPrimitive::Sphere,
                position: Vec3::new(5.0, 0.0, 0.0),
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
                material: crate::graph::scene::MaterialParams {
                    base_color: Vec3::new(0.0, 0.0, 1.0),
                    roughness: 0.5,
                    metallic: 0.0,
                    emissive_color: Vec3::ZERO,
                    emissive_intensity: 0.0,
                    reflectance_f0: 0.04,
                    ..crate::graph::scene::MaterialParams::default()
                },
                voxel_grid: None,
            },
        );
        let leaves = collect_leaf_colors(&scene);
        // Point at origin should be closest to red sphere
        let color = sample_color_at(&scene, Vec3::ZERO, &leaves);
        assert!((color[0] - 1.0).abs() < 1e-6);
        // Point at (5,0,0) should be closest to blue sphere
        let color2 = sample_color_at(&scene, Vec3::new(5.0, 0.0, 0.0), &leaves);
        assert!((color2[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sample_color_at_default_when_no_leaves() {
        let scene = empty_scene();
        let leaves: Vec<(NodeId, [f32; 3])> = vec![];
        let color = sample_color_at(&scene, Vec3::ZERO, &leaves);
        // Default gray
        assert!((color[0] - 0.5).abs() < 1e-6);
        assert!((color[1] - 0.5).abs() < 1e-6);
        assert!((color[2] - 0.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // marching_cubes tests
    // -----------------------------------------------------------------------

    #[test]
    fn marching_cubes_empty_scene_produces_no_triangles() {
        let scene = empty_scene();
        let progress = AtomicU32::new(0);
        let mesh = marching_cubes(
            &scene,
            8,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &progress,
            false,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(mesh.triangles.is_empty());
        assert!(mesh.vertices.is_empty());
    }

    #[test]
    fn marching_cubes_sphere_produces_triangles() {
        let (scene, _id) = scene_with_sphere();
        let progress = AtomicU32::new(0);
        let mesh = marching_cubes(
            &scene,
            16,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &progress,
            false,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(
            !mesh.triangles.is_empty(),
            "sphere should produce triangles"
        );
        assert!(!mesh.vertices.is_empty(), "sphere should produce vertices");
    }

    #[test]
    fn marching_cubes_sphere_vertices_near_unit_sphere() {
        let (scene, _id) = scene_with_sphere();
        let progress = AtomicU32::new(0);
        let mesh = marching_cubes(
            &scene,
            32,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &progress,
            false,
            &AtomicBool::new(false),
        )
        .unwrap();
        // All vertices should be approximately on the unit sphere surface
        for v in &mesh.vertices {
            let dist = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            assert!(
                (dist - 1.0).abs() < 0.15,
                "vertex at [{}, {}, {}] has radius {}, expected ~1.0",
                v[0],
                v[1],
                v[2],
                dist
            );
        }
    }

    #[test]
    fn marching_cubes_box_produces_triangles() {
        let (scene, _id) = scene_with_box();
        let progress = AtomicU32::new(0);
        let mesh = marching_cubes(
            &scene,
            16,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &progress,
            false,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(!mesh.triangles.is_empty(), "box should produce triangles");
    }

    #[test]
    fn marching_cubes_vertex_colors_populated() {
        let (scene, _id) = scene_with_sphere();
        let progress = AtomicU32::new(0);
        let mesh = marching_cubes(
            &scene,
            16,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &progress,
            false,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert_eq!(mesh.vertex_colors.len(), mesh.vertices.len());
        // Sphere color is red (1,0,0)
        for c in &mesh.vertex_colors {
            assert!((c[0] - 1.0).abs() < 1e-6, "expected red vertex color");
        }
    }

    #[test]
    fn marching_cubes_progress_counter_advances() {
        let (scene, _id) = scene_with_sphere();
        let progress = AtomicU32::new(0);
        let _mesh = marching_cubes(
            &scene,
            8,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &progress,
            false,
            &AtomicBool::new(false),
        )
        .unwrap();
        // Progress should have been incremented (grid_size + res times)
        assert!(progress.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn marching_cubes_triangle_indices_valid() {
        let (scene, _id) = scene_with_sphere();
        let progress = AtomicU32::new(0);
        let mesh = marching_cubes(
            &scene,
            16,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &progress,
            false,
            &AtomicBool::new(false),
        )
        .unwrap();
        let vertex_count = mesh.vertices.len() as u32;
        for tri in &mesh.triangles {
            for &idx in tri {
                assert!(
                    idx < vertex_count,
                    "triangle index {} out of bounds ({})",
                    idx,
                    vertex_count
                );
            }
        }
    }

    #[test]
    fn marching_cubes_deduplicates_shared_vertices() {
        let (scene, _id) = scene_with_sphere();
        let progress = AtomicU32::new(0);
        let mesh = marching_cubes(
            &scene,
            16,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &progress,
            false,
            &AtomicBool::new(false),
        )
        .unwrap();
        // For a closed surface, vertices should be shared — # vertices << # triangle*3
        let total_vertex_refs = mesh.triangles.len() * 3;
        assert!(
            mesh.vertices.len() < total_vertex_refs,
            "expected deduplication: {} vertices vs {} references",
            mesh.vertices.len(),
            total_vertex_refs
        );
    }

    #[test]
    fn marching_cubes_adaptive_produces_similar_result() {
        let (scene, _id) = scene_with_sphere();
        let progress1 = AtomicU32::new(0);
        let mesh_normal = marching_cubes(
            &scene,
            32,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &progress1,
            false,
            &AtomicBool::new(false),
        )
        .unwrap();
        let progress2 = AtomicU32::new(0);
        let mesh_adaptive = marching_cubes(
            &scene,
            32,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &progress2,
            true,
            &AtomicBool::new(false),
        )
        .unwrap();
        // Adaptive should still produce triangles
        assert!(!mesh_adaptive.triangles.is_empty());
        // Should produce roughly similar triangle counts (within 2x)
        let ratio = mesh_adaptive.triangles.len() as f64 / mesh_normal.triangles.len() as f64;
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "adaptive ratio {} seems wrong (normal={}, adaptive={})",
            ratio,
            mesh_normal.triangles.len(),
            mesh_adaptive.triangles.len()
        );
    }

    #[test]
    fn marching_cubes_higher_resolution_more_triangles() {
        let (scene, _id) = scene_with_sphere();
        let p1 = AtomicU32::new(0);
        let mesh_low = marching_cubes(
            &scene,
            8,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &p1,
            false,
            &AtomicBool::new(false),
        )
        .unwrap();
        let p2 = AtomicU32::new(0);
        let mesh_high = marching_cubes(
            &scene,
            16,
            Vec3::splat(-2.0),
            Vec3::splat(2.0),
            &p2,
            false,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(
            mesh_high.triangles.len() > mesh_low.triangles.len(),
            "higher resolution should produce more triangles: {} vs {}",
            mesh_high.triangles.len(),
            mesh_low.triangles.len()
        );
    }

    #[test]
    fn marching_cubes_bounds_outside_geometry_no_mesh() {
        let (scene, _id) = scene_with_sphere();
        let progress = AtomicU32::new(0);
        // Bounds far from origin (sphere is at origin with radius 1)
        let mesh = marching_cubes(
            &scene,
            8,
            Vec3::new(10.0, 10.0, 10.0),
            Vec3::new(12.0, 12.0, 12.0),
            &progress,
            false,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(
            mesh.triangles.is_empty(),
            "no geometry in bounds → no triangles"
        );
    }

    // -----------------------------------------------------------------------
    // OBJ writer tests
    // -----------------------------------------------------------------------

    #[test]
    fn write_obj_to_header_and_vertices() {
        let mesh = sample_mesh();
        let mut buf = Vec::new();
        write_obj_to(&mesh, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("# SDF Modeler Export"));
        assert!(output.contains("# Vertices: 3, Triangles: 1"));
        assert!(output.contains("v 0 0 0"));
        assert!(output.contains("v 1 0 0"));
        assert!(output.contains("v 0 1 0"));
    }

    #[test]
    fn write_obj_to_faces_are_1_indexed() {
        let mesh = sample_mesh();
        let mut buf = Vec::new();
        write_obj_to(&mesh, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        // OBJ face indices are 1-based
        assert!(output.contains("f 1 2 3"));
    }

    #[test]
    fn write_obj_to_multiple_triangles() {
        let mesh = ExportMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [1, 3, 2]],
            vertex_colors: vec![],
        };
        let mut buf = Vec::new();
        write_obj_to(&mesh, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("f 1 2 3"));
        assert!(output.contains("f 2 4 3"));
    }

    #[test]
    fn write_obj_to_empty_mesh() {
        let mesh = ExportMesh {
            vertices: vec![],
            triangles: vec![],
            vertex_colors: vec![],
        };
        let mut buf = Vec::new();
        write_obj_to(&mesh, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("# Vertices: 0, Triangles: 0"));
    }

    // -----------------------------------------------------------------------
    // STL writer tests
    // -----------------------------------------------------------------------

    #[test]
    fn write_stl_header_and_triangle_count() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_export.stl");
        write_stl(&mesh, &tmp).unwrap();
        let data = std::fs::read(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        // 80-byte header starts with "SDF Modeler Export"
        assert!(data.len() >= 84);
        assert!(data[..18] == *b"SDF Modeler Export");
        // Triangle count at offset 80
        let tri_count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]);
        assert_eq!(tri_count, 1);
    }

    #[test]
    fn write_stl_correct_total_size() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_export_size.stl");
        write_stl(&mesh, &tmp).unwrap();
        let data = std::fs::read(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        // STL binary: 80 header + 4 count + 50*num_tris
        let expected = 80 + 4 + 50 * mesh.triangles.len();
        assert_eq!(data.len(), expected);
    }

    #[test]
    fn write_stl_multiple_triangles() {
        let mesh = ExportMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [1, 3, 2]],
            vertex_colors: vec![],
        };
        let tmp = std::env::temp_dir().join("test_export_multi.stl");
        write_stl(&mesh, &tmp).unwrap();
        let data = std::fs::read(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        let tri_count = u32::from_le_bytes([data[80], data[81], data[82], data[83]]);
        assert_eq!(tri_count, 2);
        assert_eq!(data.len(), 80 + 4 + 50 * 2);
    }

    // -----------------------------------------------------------------------
    // PLY writer tests
    // -----------------------------------------------------------------------

    #[test]
    fn write_ply_header_structure() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_export.ply");
        write_ply(&mesh, &tmp).unwrap();
        let output = std::fs::read_to_string(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        assert!(output.starts_with("ply\n"));
        assert!(output.contains("format ascii 1.0"));
        assert!(output.contains("element vertex 3"));
        assert!(output.contains("element face 1"));
        assert!(output.contains("end_header"));
    }

    #[test]
    fn write_ply_with_colors() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_export_color.ply");
        write_ply(&mesh, &tmp).unwrap();
        let output = std::fs::read_to_string(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        assert!(output.contains("property uchar red"));
        assert!(output.contains("property uchar green"));
        assert!(output.contains("property uchar blue"));
        // First vertex (0,0,0) with color (1,0,0) → 255 0 0
        assert!(output.contains("0 0 0 255 0 0"));
    }

    #[test]
    fn write_ply_without_colors() {
        let mesh = sample_mesh_no_colors();
        let tmp = std::env::temp_dir().join("test_export_nocolor.ply");
        write_ply(&mesh, &tmp).unwrap();
        let output = std::fs::read_to_string(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        assert!(!output.contains("property uchar red"));
    }

    #[test]
    fn write_ply_face_format() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_export_face.ply");
        write_ply(&mesh, &tmp).unwrap();
        let output = std::fs::read_to_string(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        // PLY uses 0-indexed and prefixes face with vertex count
        assert!(output.contains("3 0 1 2"));
    }

    // -----------------------------------------------------------------------
    // GLB writer tests
    // -----------------------------------------------------------------------

    #[test]
    fn write_glb_magic_and_version() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_export.glb");
        write_glb(&mesh, &tmp).unwrap();
        let data = std::fs::read(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        // GLB magic: "glTF"
        assert_eq!(&data[0..4], b"glTF");
        // Version 2
        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        assert_eq!(version, 2);
    }

    #[test]
    fn write_glb_total_length_matches_file() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_export_len.glb");
        write_glb(&mesh, &tmp).unwrap();
        let data = std::fs::read(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        let total_length = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        assert_eq!(total_length, data.len());
    }

    #[test]
    fn write_glb_json_chunk_type() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_export_json.glb");
        write_glb(&mesh, &tmp).unwrap();
        let data = std::fs::read(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        // Chunk type at offset 16 should be 0x4E4F534A ("JSON")
        let chunk_type = u32::from_le_bytes([data[16], data[17], data[18], data[19]]);
        assert_eq!(chunk_type, 0x4E4F534A);
    }

    #[test]
    fn write_glb_contains_asset_version() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_export_asset.glb");
        write_glb(&mesh, &tmp).unwrap();
        let data = std::fs::read(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        // The JSON content should contain asset version 2.0
        let json_str = String::from_utf8_lossy(&data);
        assert!(json_str.contains("\"version\":\"2.0\""));
        assert!(json_str.contains("\"generator\":\"SDF Modeler\""));
    }

    #[test]
    fn write_glb_no_colors_fewer_accessors() {
        let mesh = sample_mesh_no_colors();
        let tmp = std::env::temp_dir().join("test_export_noc.glb");
        write_glb(&mesh, &tmp).unwrap();
        let data = std::fs::read(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        let json_str = String::from_utf8_lossy(&data);
        // Without colors, should NOT have COLOR_0
        assert!(!json_str.contains("COLOR_0"));
    }

    #[test]
    fn write_glb_with_colors_has_color_accessor() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_export_wc.glb");
        write_glb(&mesh, &tmp).unwrap();
        let data = std::fs::read(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        let json_str = String::from_utf8_lossy(&data);
        assert!(json_str.contains("COLOR_0"));
    }

    // -----------------------------------------------------------------------
    // USDA writer tests
    // -----------------------------------------------------------------------

    #[test]
    fn write_usda_header() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_export.usda");
        write_usda(&mesh, &tmp).unwrap();
        let output = std::fs::read_to_string(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        assert!(output.starts_with("#usda 1.0"));
        assert!(output.contains("defaultPrim = \"SdfExport\""));
        assert!(output.contains("upAxis = \"Y\""));
    }

    #[test]
    fn write_usda_points() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_export_pts.usda");
        write_usda(&mesh, &tmp).unwrap();
        let output = std::fs::read_to_string(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        assert!(output.contains("point3f[] points = ["));
        assert!(output.contains("(0, 0, 0)"));
        assert!(output.contains("(1, 0, 0)"));
        assert!(output.contains("(0, 1, 0)"));
    }

    #[test]
    fn write_usda_face_vertex_counts_all_3() {
        let mesh = ExportMesh {
            vertices: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            triangles: vec![[0, 1, 2], [1, 3, 2]],
            vertex_colors: vec![],
        };
        let tmp = std::env::temp_dir().join("test_export_fvc.usda");
        write_usda(&mesh, &tmp).unwrap();
        let output = std::fs::read_to_string(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        assert!(output.contains("int[] faceVertexCounts = [3, 3]"));
    }

    #[test]
    fn write_usda_vertex_colors() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_export_vc.usda");
        write_usda(&mesh, &tmp).unwrap();
        let output = std::fs::read_to_string(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        assert!(output.contains("color3f[] primvars:displayColor = ["));
        assert!(output.contains("interpolation = \"vertex\""));
    }

    #[test]
    fn write_usda_no_colors_omits_display_color() {
        let mesh = sample_mesh_no_colors();
        let tmp = std::env::temp_dir().join("test_export_nc.usda");
        write_usda(&mesh, &tmp).unwrap();
        let output = std::fs::read_to_string(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();

        assert!(!output.contains("primvars:displayColor"));
    }

    // -----------------------------------------------------------------------
    // write_mesh dispatch tests
    // -----------------------------------------------------------------------

    #[test]
    fn write_mesh_dispatches_stl() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_dispatch.stl");
        write_mesh(&mesh, &tmp).unwrap();
        let data = std::fs::read(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();
        // STL starts with 80-byte header
        assert!(data.len() >= 84);
    }

    #[test]
    fn write_mesh_dispatches_ply() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_dispatch.ply");
        write_mesh(&mesh, &tmp).unwrap();
        let output = std::fs::read_to_string(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();
        assert!(output.starts_with("ply\n"));
    }

    #[test]
    fn write_mesh_dispatches_glb() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_dispatch.glb");
        write_mesh(&mesh, &tmp).unwrap();
        let data = std::fs::read(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();
        assert_eq!(&data[0..4], b"glTF");
    }

    #[test]
    fn write_mesh_dispatches_usda() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_dispatch.usda");
        write_mesh(&mesh, &tmp).unwrap();
        let output = std::fs::read_to_string(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();
        assert!(output.starts_with("#usda 1.0"));
    }

    #[test]
    fn write_mesh_defaults_to_obj() {
        let mesh = sample_mesh();
        let tmp = std::env::temp_dir().join("test_dispatch.xyz");
        write_mesh(&mesh, &tmp).unwrap();
        let output = std::fs::read_to_string(&tmp).unwrap();
        std::fs::remove_file(&tmp).ok();
        assert!(output.contains("# SDF Modeler Export"));
    }

    // -----------------------------------------------------------------------
    // Edge table / tri table sanity checks
    // -----------------------------------------------------------------------

    #[test]
    fn edge_table_all_inside_is_zero() {
        // cube_index 0xFF = all corners inside → EDGE_TABLE should be 0
        // Actually, 0xFF means all 8 corners are inside the surface.
        // The Paul Bourke table for index 0xFF is 0x000.
        assert_eq!(EDGE_TABLE[0xFF], 0x000);
    }

    #[test]
    fn edge_table_all_outside_is_zero() {
        assert_eq!(EDGE_TABLE[0x00], 0x000);
    }

    #[test]
    fn tri_table_all_inside_empty() {
        assert_eq!(TRI_TABLE[0xFF][0], -1);
    }

    #[test]
    fn tri_table_all_outside_empty() {
        assert_eq!(TRI_TABLE[0x00][0], -1);
    }

    #[test]
    fn tri_table_single_corner_has_one_triangle() {
        // cube_index 1 = only corner 0 inside → one triangle (3 edges)
        let row = &TRI_TABLE[1];
        assert!(row[0] >= 0);
        assert!(row[1] >= 0);
        assert!(row[2] >= 0);
        assert_eq!(row[3], -1); // only one triangle
    }

    // -----------------------------------------------------------------------
    // Marching cubes with CSG
    // -----------------------------------------------------------------------

    #[test]
    fn marching_cubes_csg_union_produces_mesh() {
        let mut scene = empty_scene();
        let name1 = scene.next_name("Sphere");
        let s1 = scene.add_node(
            name1,
            NodeData::Primitive {
                kind: SdfPrimitive::Sphere,
                position: Vec3::new(-0.5, 0.0, 0.0),
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
                material: crate::graph::scene::MaterialParams {
                    base_color: Vec3::ONE,
                    roughness: 0.5,
                    metallic: 0.0,
                    emissive_color: Vec3::ZERO,
                    emissive_intensity: 0.0,
                    reflectance_f0: 0.04,
                    ..crate::graph::scene::MaterialParams::default()
                },
                voxel_grid: None,
            },
        );
        let name2 = scene.next_name("Sphere");
        let s2 = scene.add_node(
            name2,
            NodeData::Primitive {
                kind: SdfPrimitive::Sphere,
                position: Vec3::new(0.5, 0.0, 0.0),
                rotation: Vec3::ZERO,
                scale: Vec3::ONE,
                material: crate::graph::scene::MaterialParams {
                    base_color: Vec3::ONE,
                    roughness: 0.5,
                    metallic: 0.0,
                    emissive_color: Vec3::ZERO,
                    emissive_intensity: 0.0,
                    reflectance_f0: 0.04,
                    ..crate::graph::scene::MaterialParams::default()
                },
                voxel_grid: None,
            },
        );
        let name_op = scene.next_name("Union");
        scene.add_node(
            name_op,
            NodeData::Operation {
                op: CsgOp::Union,
                left: Some(s1),
                right: Some(s2),
                smooth_k: 0.0,
                steps: 0.0,
                color_blend: -1.0,
            },
        );

        let progress = AtomicU32::new(0);
        let mesh = marching_cubes(
            &scene,
            16,
            Vec3::splat(-3.0),
            Vec3::splat(3.0),
            &progress,
            false,
            &AtomicBool::new(false),
        )
        .unwrap();
        assert!(!mesh.triangles.is_empty(), "CSG union should produce mesh");
    }
}
