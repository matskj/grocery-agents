use smallvec::SmallVec;

pub fn neighbor_distances(values: &[f32]) -> SmallVec<[f32; 8]> {
    values
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .collect::<SmallVec<[f32; 8]>>()
}
