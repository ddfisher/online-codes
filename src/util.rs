use std::collections::HashSet;
use rand_xoshiro::Xoshiro256StarStar;
use rand_core::SeedableRng;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use crate::types::{StreamId, CheckBlockId, BlockIndex};

// TODO: optimize
pub fn xor_block(dest: &mut [u8], src: &[u8], block_size: usize) {
    for i in 0..block_size {
        dest[i] ^= src[i];
    }
}

// TODO: don't lose bits when combining the stream id and block id
pub fn seed_block_rng(stream_id: StreamId, check_block_id: CheckBlockId) -> Xoshiro256StarStar {
    // Make sure the seed is a good, even mix of 0's and 1's.
    Xoshiro256StarStar::seed_from_u64(check_block_id.wrapping_add(stream_id))
}

pub fn get_adjacent_blocks(
    check_block_id: CheckBlockId,
    stream_id: StreamId,
    degree_distribution: &WeightedIndex<f64>,
    num_blocks: usize,
) -> Vec<BlockIndex> {
    let mut rng = seed_block_rng(stream_id, check_block_id);
    let degree = 1 + degree_distribution.sample(&mut rng);
    sample_with_exclusive_repeats(&mut rng, num_blocks, degree)
}

pub fn sample_with_exclusive_repeats(
    rng: &mut Xoshiro256StarStar,
    high_exclusive: usize,
    num: usize,
) -> Vec<usize> {
    let mut selected = HashSet::with_capacity(num);
    let distribution = Uniform::new(0, high_exclusive);
    for _ in 0..num {
        let sample = distribution.sample(rng);
        if !selected.insert(sample) {
            selected.remove(&sample);
        }
    }

    selected.into_iter().collect()
}
