pub(super) fn axis_value(axis: i32, values: [f32; 3]) -> f32 {
    match axis {
        1 => values[1],
        2 => values[2],
        _ => values[0],
    }
}
