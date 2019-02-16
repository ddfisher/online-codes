use rand::distributions::{Distribution, Uniform, WeightedIndex};
use rand_core::{RngCore, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use std::collections::HashSet;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}

// TODO: allow user to change these
const q: usize = 3;
const epsilon: f64 = 0.01;

// TODO: optimize
fn sample_without_replacement(
    rng: &mut Xoshiro256StarStar,
    high_exclusive: usize,
    num: usize,
) -> Vec<usize> {
    // TODO: this might have bad behavior when called by inner_encode with a high degree
    let mut selected = HashSet::with_capacity(num);
    let distribution = Uniform::new(0, high_exclusive);
    let mut sample = distribution.sample(rng);
    while selected.contains(&sample) {
        sample = distribution.sample(rng);
    }
    selected.insert(sample);

    selected.into_iter().collect()
}

// TODO: optimize
fn xor_block(dest: &mut [u8], src: &[u8], block_size: usize) {
    for i in 0..block_size {
        dest[i] ^= src[i];
    }
}

pub fn encode<'a>(data: &'a mut Vec<u8>, block_size: usize, seed: u64) -> BlockIter<'a> {
    outer_encode(data, block_size, seed);
    inner_encode(data, block_size)
}

fn outer_encode(data: &mut Vec<u8>, block_size: usize, seed: u64) {
    assert!(data.len() % block_size == 0);
    let num_blocks = data.len() / block_size;
    let num_auxiliary_blocks = (0.55f64 * q as f64 * epsilon * num_blocks as f64).ceil() as usize;
    data.resize((num_blocks + num_auxiliary_blocks) * block_size, 0u8);
    let (blocks, aux_blocks) = data.split_at_mut(num_blocks * block_size);
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    for block in blocks.chunks_exact(block_size) {
        for aux_index in sample_without_replacement(&mut rng, num_auxiliary_blocks, q) {
            xor_block(
                &mut aux_blocks[aux_index * block_size..(aux_index + 1) * block_size],
                block,
                block_size,
            );
        }
    }
}

pub struct BlockIter<'a> {
    data: &'a Vec<u8>,
    block_size: usize,
    degree_distribution: WeightedIndex<f64>,
    block_id: u64, // TODO: this should be a larger size type
}

impl<'a> Iterator for BlockIter<'a> {
    type Item = Vec<u8>;
    fn next(&mut self) -> Option<Vec<u8>> {
        let mut rng = Xoshiro256StarStar::seed_from_u64(self.block_id);
        let degree = 1 + self.degree_distribution.sample(&mut rng);
        let mut check_block = vec![0; self.block_size];
        let num_blocks = self.data.len() / self.block_size;
        for block_index in sample_without_replacement(&mut rng, num_blocks, degree) {
            xor_block(
                &mut check_block,
                &self.data[block_index * self.block_size..(block_index + 1) * self.block_size],
                self.block_size,
            );
        }

        self.block_id += 1;
        Some(check_block)
    }
}

fn inner_encode<'a>(data: &'a mut Vec<u8>, block_size: usize) -> BlockIter<'a> {
    assert!(data.len() % block_size == 0);
    // TODO: only compute this once
    // See section 3.2 of the Maymounkov-Mazières paper.
    let f = ((f64::ln(epsilon * epsilon / 4.0)) / f64::ln(1.0 - epsilon / 2.0)).ceil() as usize;
    let mut p = Vec::with_capacity(f);
    let p1 = 1.0 - ((1.0 + 1.0 / f as f64) / 1.0 + epsilon);
    p.push(p1);
    // Extracted unchanging constant from p_i's.
    let c = (1.0 - p1) * f as f64 / (f - 1) as f64;
    for i in 2..=f {
        p.push(c / (i * (i - 1)) as f64);
    }
    let degree_distribution =
        WeightedIndex::new(&p).expect("serious probability calculation error");

    BlockIter {
        data,
        block_size,
        degree_distribution,
        block_id: 0,
    }
}
