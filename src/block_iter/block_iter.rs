use rand::distributions::{Distribution, Uniform, WeightedIndex};
use rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;
use std::collections::HashSet;

// TODO: use larger seeds for the PRNG
// TODO: allow specification of starting block_id
// TODO: write tests with proptest
// TODO: write benchmarks with criterion
// TODO: profile and fix low-hanging fruit
// TODO: reorder functions
// TODO: make a minor code cleanup pass
// TODO: write docs
// TODO: remove main.rs


pub struct BlockIter<'a> {
    data: &'a [u8],
    aux_data: Vec<u8>,
    block_size: usize,
    degree_distribution: WeightedIndex<f64>,
    check_block_id: CheckBlockId,
    stream_id: StreamId,
}

impl<'a> Iterator for BlockIter<'a> {
    type Item = Vec<u8>;
    fn next(&mut self) -> Option<Vec<u8>> {
        let num_blocks = self.data.len() / self.block_size;
        let num_aux_blocks = self.aux_data.len() / self.block_size;
        let mut check_block = vec![0; self.block_size];
        let adjacent_blocks = get_adjacent_blocks(
            self.check_block_id,
            self.stream_id,
            &self.degree_distribution,
            num_blocks + num_aux_blocks,
        );
        for block_index in adjacent_blocks {
            if block_index < num_blocks {
                xor_block(
                    &mut check_block,
                    &self.data[block_index * self.block_size..],
                    self.block_size,
                );
            } else {
                // Aux block.
                xor_block(
                    &mut check_block,
                    &self.aux_data[(block_index - num_blocks) * self.block_size..],
                    self.block_size,
                );
            }
        }

        self.check_block_id += 1;
        Some(check_block)
    }
}


fn sample_with_exclusive_repeats(
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

    return selected.into_iter().collect();
}

// TODO: optimize
fn xor_block(dest: &mut [u8], src: &[u8], block_size: usize) {
    for i in 0..block_size {
        dest[i] ^= src[i];
    }
}


// TODO: don't lose bits when combining the stream id and block id
fn seed_block_rng(stream_id: StreamId, check_block_id: CheckBlockId) -> Xoshiro256StarStar {
    // Make sure the seed is a good, even mix of 0's and 1's.
    Xoshiro256StarStar::seed_from_u64(check_block_id.wrapping_add(stream_id))
}

fn get_adjacent_blocks(
    check_block_id: CheckBlockId,
    stream_id: StreamId,
    degree_distribution: &WeightedIndex<f64>,
    num_blocks: usize,
) -> Vec<BlockIndex> {
    let mut rng = seed_block_rng(stream_id, check_block_id);
    let degree = 1 + degree_distribution.sample(&mut rng);
    sample_with_exclusive_repeats(&mut rng, num_blocks, degree)
}

