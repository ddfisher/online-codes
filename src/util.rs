use crate::types::{BlockIndex, CheckBlockId, StreamId};
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::collections::{HashMap, HashSet};

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

pub fn seed_stream_rng(stream_id: StreamId) -> Xoshiro256StarStar {
    seed_block_rng(stream_id, 0)
}

pub fn num_aux_blocks(num_blocks: usize, epsilon: f64, q: usize) -> usize {
    (0.55_f64 * q as f64 * epsilon * num_blocks as f64).ceil() as usize
}

pub fn get_aux_block_adjacencies(
    stream_id: StreamId,
    num_blocks: usize,
    num_auxiliary_blocks: usize,
    q: usize,
) -> HashMap<BlockIndex, (usize, Vec<BlockIndex>)> {
    let mut mapping: HashMap<BlockIndex, (usize, Vec<BlockIndex>)> = HashMap::new();
    let mut rng = seed_stream_rng(stream_id);
    for i in 0..num_blocks {
        for aux_index in sample_with_exclusive_repeats(&mut rng, num_auxiliary_blocks, q) {
            // TODO: clean up a bit
            let (num, ids) = &mut mapping.entry(aux_index + num_blocks).or_default();
            *num += 1;
            ids.push(i);
        }
    }
    mapping
}

pub fn make_degree_distribution(epsilon: f64) -> WeightedIndex<f64> {
    // See section 3.2 of the Maymounkov-Mazières paper.
    let f = ((f64::ln(epsilon * epsilon / 4.0)) / f64::ln(1.0 - epsilon / 2.0)).ceil() as usize;
    let mut p = Vec::with_capacity(f);
    let p1 = 1.0 - ((1.0 + 1.0 / f as f64) / (1.0 + epsilon));
    p.push(p1);
    // Extracted unchanging constant from p_i's.
    let c = (1.0 - p1) * f as f64 / (f - 1) as f64;
    for i in 2..=f {
        p.push(c / (i * (i - 1)) as f64);
    }
    WeightedIndex::new(&p).expect("serious probability calculation error")
}
